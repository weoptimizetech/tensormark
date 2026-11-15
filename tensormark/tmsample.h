// tensormark/tmsample.h — one sampling policy for every decoder.
//
// The two chat CLIs carried their own top-k/top-p implementation, differing
// only in their default knobs (llama_chat 0.7/40/0.90, gpt2_chat 0.8/40/0.95):
// ~20 duplicated lines with a subtly different tie-break waiting to happen.
// This header is the single implementation; the CLIs keep their own defaults
// and pass them in.
//
// The policy, unchanged from llama_chat.cpp's sample_token (which is what the
// acceptance runs measured): greedy argmax, else divide by temperature, take
// the top_k logits (nth_element + sort of the prefix — NOT a full sort, the
// rest of the array is deliberately left unordered), softmax over them, cut at
// top_p cumulative mass, discrete_distribution over the survivors.
//
// Determinism note for tests: the returned token depends only on (logits,
// knobs, the rng state) — same seed, same logits, same token — but the
// distribution is an implementation detail of the standard library, so a
// transcript test must pin BOTH the seed and the libc++ version it was
// recorded with.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace tmsample {

struct Opts {
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
};

// Repetition penalty over the last `window` emitted tokens, applied to the
// logits in place BEFORE selection.
//
// It exists because the served path runs GREEDY (the shim pins /greedy, and the
// cache-reuse machinery is built on replaying token-for-token), and greedy
// decoding has one signature failure: a list-shaped answer enters a cycle and
// repeats it. Observed live through the Ollama endpoint on qwen38_4b_q40 — asked
// for the authors of "Attention Is All You Need" the model gave eight correct
// names, drifted into inventions, and then restarted the list and looped.
//
// This is a DETERMINISTIC transform of the logits given the token history: no
// RNG, so it composes with greedy decoding and does not touch the byte-identical
// replay contract the reuse path depends on. Sign convention follows llama.cpp:
// a positive logit is divided by the penalty, a negative one multiplied, so the
// transform always pushes a repeated token DOWN without changing sign.
inline void apply_repeat_penalty(std::vector<float>& logits, const int* recent, int n_recent,
                                 int window, float penalty) {
    if (penalty <= 1.0f || window <= 0 || n_recent <= 0) return;
    const int from = n_recent > window ? n_recent - window : 0;
    const int V = static_cast<int>(logits.size());
    for (int i = from; i < n_recent; ++i) {
        const int t = recent[i];
        if (t < 0 || t >= V) continue;
        float& v = logits[static_cast<std::size_t>(t)];
        v = v > 0.f ? v / penalty : v * penalty;
    }
}

// No-repeat N-gram blocking: no N-token sequence may be emitted twice.
//
// This is the TARGETED remedy for the loop `apply_repeat_penalty` above is a blunt
// instrument against, and it exists because that penalty was measured to buy the loop
// fix at an unacceptable price: a unigram penalty over a 64-token window "punishes
// exactly the numerals the answer needs", so on a counting task every penalised arm
// (1.05 / 1.15 / 1.30) stopped committing at all while plain greedy answered correctly
// (tensormark/README.md, the three-range gaps table). A counting answer legitimately
// repeats single tokens; a LOOP is a repeated N-gram. Banning only the latter separates
// the two, so a list-shaped answer cannot cycle while numerals stay free.
//
// Mechanics, following the usual no_repeat_ngram_size: the last (N-1) tokens are the
// context; every earlier occurrence of that same context bans whatever token FOLLOWED
// it, so no N-gram can occur twice. Deterministic given the token history — no RNG —
// so it composes with greedy decoding and leaves the byte-identical replay contract the
// cache-reuse path depends on untouched.
inline void apply_no_repeat_ngram(std::vector<float>& logits, const int* recent,
                                  int n_recent, int ngram) {
    if (ngram < 2 || n_recent < ngram) return;
    const int V = static_cast<int>(logits.size());
    const int m = ngram - 1;                 // context length
    const int s = n_recent - m;              // where the current context starts
    std::vector<std::pair<int, float>> banned;
    for (int i = 0; i <= s; ++i) {           // i == s is the context itself, no successor
        if (i == s) break;                   // only EARLIER occurrences have a successor
        if (i + m >= n_recent) break;
        bool same = true;
        for (int k = 0; k < m; ++k)
            if (recent[i + k] != recent[s + k]) { same = false; break; }
        if (!same) continue;
        const int t = recent[i + m];
        if (t < 0 || t >= V) continue;
        // Already banned by an earlier occurrence: do NOT capture again. The captured
        // value is what the restore pass puts back, so a second capture would record
        // the -inf the first ban just wrote and undo the restore.
        if (logits[static_cast<std::size_t>(t)] == -std::numeric_limits<float>::infinity())
            continue;
        banned.emplace_back(t, logits[static_cast<std::size_t>(t)]);
        logits[static_cast<std::size_t>(t)] = -std::numeric_limits<float>::infinity();
    }
    if (banned.empty()) return;
    // Never leave the decoder with nothing legal. Under a grammar the banned token can
    // be the ONLY one the grammar allows, and an all -inf row would make argmax return
    // token 0 — an unconstrained token emitted from a constrained decoder.
    float best = -std::numeric_limits<float>::infinity();
    for (float v : logits) best = std::max(best, v);
    if (best == -std::numeric_limits<float>::infinity())
        for (const auto& [t, v] : banned) logits[static_cast<std::size_t>(t)] = v;
}

// DRY ("Don't Repeat Yourself") — the SEQUENCE-level repetition control, and the
// modern replacement for both knobs above.
//
// A unigram penalty punishes isolated tokens, which is why it wrecks a counting
// answer (it hits the numerals a count must reuse) and syntax alike. A hard N-gram
// ban is better but still blunt: forbidding one token makes the decoder snap to the
// NEXT-BEST token, which is how a degenerate numeric continuation survives as a
// DIFFERENT wrong number. DRY instead measures how far a sequence is already
// REPEATING and charges a penalty that grows EXPONENTIALLY with that length, so a
// degeneration is pushed off early and gently rather than forbidden late.
//
// For every earlier position i whose sequence agrees with the tail for L tokens:
//   if L >= dry_allowed_length, penalise the token that FOLLOWED the earlier run
//   (s[i+1] — what continuing the repeat would emit) by
//       multiplier * base^(L - allowed_length)
// applied as a logit shift. Sequences up to dry_allowed_length are therefore free
// (short repeats are legitimate: counting, formatting, emphasis), and the cost
// escalates steeply from there.
//
// `is_breaker` (may be null) marks sequence-breaker tokens — matches never extend
// across one, and a breaker is never itself penalised — so structural repetition
// (newlines, list markers) is not charged, which is what keeps syntax intact.
//
// Deterministic given the token history: no RNG, so it composes with greedy decoding
// and leaves the byte-identical replay contract untouched. Parameter names and
// defaults follow llama.cpp's DRY (multiplier 0 = disabled there, as here).
struct DryOpts {
    float multiplier = 0.0f;   // 0 = disabled
    float base = 1.75f;
    int allowed_length = 2;
    int penalty_last_n = 256;
};

inline void apply_dry(std::vector<float>& logits, const int* recent, int n_recent,
                      const DryOpts& o, const unsigned char* is_breaker = nullptr) {
    if (o.multiplier <= 0.0f || o.allowed_length < 1) return;
    if (n_recent < o.allowed_length + 1) return;
    const int n = std::min(n_recent, o.penalty_last_n);
    if (n < o.allowed_length + 1) return;
    const int* s = recent + (n_recent - n);      // window: the last n tokens
    const int V = static_cast<int>(logits.size());
    const int last = n - 1;
    for (int i = 0; i < last; ++i) {
        // how far s[..i] and s[..last] agree looking backwards
        int L = 0;
        while (L <= i && s[i - L] == s[last - L] &&
               !(is_breaker && (is_breaker[s[i - L]] || is_breaker[s[last - L]])))
            ++L;
        if (L < o.allowed_length) continue;
        const int nxt = s[i + 1];                // what followed the earlier run
        if (nxt < 0 || nxt >= V) continue;
        if (is_breaker && is_breaker[nxt]) continue;
        logits[static_cast<std::size_t>(nxt)] -=
            o.multiplier * std::pow(o.base, static_cast<float>(L - o.allowed_length));
    }
}

// Greedy argmax, or top-k/top-p sampling over `logits[0..V)`.
[[nodiscard]] inline int sample(const std::vector<float>& logits, int V, const Opts& o,
                                bool greedy, std::mt19937& rng) {
    if (greedy)
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<float> lg = logits;
    for (float& v : lg) v /= o.temperature;
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::nth_element(idx.begin(), idx.begin() + o.top_k - 1, idx.end(),
                     [&](int a, int b) { return lg[a] > lg[b]; });
    std::sort(idx.begin(), idx.begin() + o.top_k, [&](int a, int b) { return lg[a] > lg[b]; });
    const float mx = lg[static_cast<std::size_t>(idx[0])];
    std::vector<double> pr(static_cast<std::size_t>(o.top_k));
    double z = 0;
    for (int k = 0; k < o.top_k; ++k) {
        pr[static_cast<std::size_t>(k)] =
            std::exp(static_cast<double>(lg[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])]) - mx);
        z += pr[static_cast<std::size_t>(k)];
    }
    double cum = 0;
    int keep = o.top_k;
    for (int k = 0; k < o.top_k; ++k) {
        cum += pr[static_cast<std::size_t>(k)] / z;
        if (cum >= o.top_p) { keep = k + 1; break; }
    }
    std::discrete_distribution<int> dist(pr.begin(), pr.begin() + keep);
    return idx[static_cast<std::size_t>(dist(rng))];
}

}  // namespace tmsample
