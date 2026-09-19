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
#include <random>
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
