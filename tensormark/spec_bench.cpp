// spec_bench.cpp — what does the MTP head buy on a real checkpoint?
//
// tensormark/test_mtp.cpp gates the MTP plumbing on a synthetic fixture, and says why in its
// own header: "No released qwen35 checkpoint carries the head ... equivalence with llama.cpp
// needs a checkpoint that has the head." Qwen3.8-4B-Distill (Q6_K) IS such a checkpoint — it
// carries mtp.eh_proj and friends — so this measures the head on real weights with the only
// assertion that matters: the speculative loop must emit the SAME tokens as the plain loop,
// or it is not an optimization, it is a different model.
//
// It reports both halves, because either alone is misleading:
//   * acceptance rate — how often the trunk agreed with the head's draft;
//   * wall-clock — whether advancing two tokens on the rounds it agreed costs less than the
//     bookkeeping a rejected draft needs.
//
// MEASURED (2026-09-17, M1 8 GB, qwen38_4b_q6k.tmq, TM_CTX=4096, 64 greedy tokens):
//   plain 6.18 s (10.2 tok/s) vs spec 8.31 s (7.6 tok/s) = 0.74x — a SLOWDOWN — at 90.9%
//   acceptance (33 drafted, 30 accepted, 3 rejected) with a bit-identical token stream.
//
// The round is now DECOMPOSED rather than reasoned about (TM_SPEC_TIMING=1), and the cause
// this header used to record — "a round snapshots the Gated-DeltaNet recurrent state of every
// linear layer ... that snapshot is not free" — is REFUTED by it:
//   verify  forward({t,d})  T=2     6824.7 ms   82.1%
//   draft   mtp_step                 617.2 ms    7.4%
//   replay  3 rejects                384.0 ms    4.6%
//   snap    snap_states              345.5 ms    4.2%
//   other                            140.4 ms    1.7%
// snap_states copies linear_layers_ * (lv_heads_*lk_dim_*lv_dim_ + (conv_kernel_-1)*
// conv_channels) floats — ~50 MB on this model, ~10 ms a round. Eliminating it entirely
// would move 0.74x to roughly 0.78x, so it was never the reason to stay opt-in.
//
// The real blocker is the position-scaling table printed below, and it is structural: one
// verify pass over k positions costs 0.84x (k=2) down to 0.71x (k=16) PER POSITION of k
// separate passes. Batching buys 1.4x at absolute best, so a round that yields ~1.9 tokens
// for ~1.68 token-equivalents of verify cannot close the gap once draft, snap and replay are
// added — and drafting more tokens does not rescue it either, because at k=16 the head must
// itself run 16 times to produce the 16 drafts. Speculation is not a switch this architecture
// is missing; it is a trade this architecture loses. The path stays opt-in, and the plain
// loop is what a chat runs.
//
// usage: spec_bench <model.tmq> <tokenizer.vocab.json> [n_tokens] [prompt]
#include "tmtok.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static int argmax_of(const std::vector<float>& v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
}

// Unbuffered stage trace: an abort here loses stdout with it (the prints below are buffered),
// so every step reports itself on stderr first. Without this, a `map::at` deep inside a step
// is indistinguishable from one in the loader — which is exactly how the qk_ alias gap below
// was found rather than guessed at.
static void stage(const char* what) {
    std::fprintf(stderr, "[stage] %s\n", what);
    std::fflush(stderr);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: spec_bench <model.tmq> <tokenizer.vocab.json> [n_tokens] [prompt]\n");
        return 2;
    }
    const std::string tmq = argv[1], tokm = argv[2];
    const int N = argc > 3 ? std::atoi(argv[3]) : 64;
    const std::string prompt = argc > 4 ? argv[4] : "The capital of France is";

    stage("loading tokenizer");
    // One construction point (tmtok.h): the suffix decides the format and the
    // sidecars come from the same base — identical to what the chat CLI does
    // with this path.
    const tmtok::Tokenizer tok = tmtok::open_for_model(tokm);
    const std::vector<int> ids = tok.encode(prompt);
    stage("tokenizer ok");

    tmllama::Llama m;
    std::fprintf(stderr, "loading %s ...\n", tmq.c_str());
    m.load_quant(tmq);
    std::fprintf(stderr, "ctx %d | mtp_ready %d | prompt %zu tok\n", m.ctx, (int)m.mtp_ready(),
                 ids.size());
    // The position-scaling phase below is meaningful without a head, and is the control that
    // separates "batching positions is expensive in this engine" from "it is expensive in this
    // MODEL's linear-attention layers". A dense model runs it; the A/B needs the head.
    const bool has_head = m.mtp_ready();
    if (!has_head)
        std::fprintf(stderr, "no MTP head here — position-scaling phase only\n");

    // ---- plain greedy loop: the reference, one forward per token ----
    auto plain = [&]() {
        stage("plain: reset+prefill");
        m.reset_cache();
        const std::vector<float> lg = m.forward(ids);        // copy: forward returns an alias
        const int first = argmax_of(lg);
        std::vector<int> out{first};
        const auto t0 = clk::now();
        for (int i = 1; i < N; ++i) {
            const std::vector<float>& l = m.forward({out.back()});
            out.push_back(argmax_of(l));
        }
        stage("plain: done");
        return std::make_pair(out, secs(t0, clk::now()));
    };

    // ---- speculative loop: one batched forward over two positions, two tokens when accepted ----
    auto spec = [&](int* drafted, int* accepted, int* rejected) {
        stage("spec: reset+prefill");
        m.reset_cache();
        const std::vector<float> lg = m.forward(ids);
        const int first = argmax_of(lg);
        const auto t0 = clk::now();
        stage("spec: entering spec_generate_greedy");
        std::vector<int> out = m.spec_generate_greedy(first, N, drafted, accepted, rejected);
        stage("spec: done");
        return std::make_pair(out, secs(t0, clk::now()));
    };

    // Plain first (which warms the weights into the page cache), spec second, plain again: the
    // plain figure quoted is the better of its two runs, so the comparison is not a
    // warm-vs-cold artifact.
    bool same = true;
    std::vector<int> p1;
    if (has_head) {
        auto [q1, t_p1] = plain();
        int d = 0, a = 0, r = 0;
        auto [s, t_s] = spec(&d, &a, &r);
        auto [q2, t_p2] = plain();
        p1 = q1;

        const double t_p = std::min(t_p1, t_p2);
        const int n = (int)p1.size();
        same = (p1 == s) && (p1 == q2);

        std::printf("\nprompt   %s\n", prompt.c_str());
        std::printf("tokens   %d\n", n);
        std::printf("plain    %.2f s (%.1f tok/s)  [runs %.2f / %.2f]\n", t_p, (n - 1) / t_p,
                    t_p1, t_p2);
        std::printf("spec     %.2f s (%.1f tok/s)  speedup %.2fx\n", t_s, (n - 1) / t_s,
                    t_p / t_s);
        std::printf("drafted  %d   accepted %d   rejected %d   acceptance %.1f%%\n", d, a, r,
                    d ? 100.0 * a / d : 0.0);
        std::printf("identical token stream: %s\n",
                    same ? "YES" : "*** NO — NOT AN OPTIMIZATION ***");
        if (!same) {
            for (int i = 0; i < n && i < (int)s.size(); ++i) {
                if (p1[i] != s[i]) {
                    std::printf("  first divergence at %d: plain %d vs spec %d\n", i, p1[i], s[i]);
                    break;
                }
            }
        }
    }

    // ---- position scaling: what a k-position verify pass costs per position ----
    // This is what decides whether the speculative loop can ever pay on this architecture, and
    // it is the measurement the 0.82x above demands. A verify pass over k positions replaces k
    // plain steps only if the per-position cost FALLS with k — i.e. if decode is dominated by
    // weight traffic, a fixed cost per pass, rather than by the per-position recurrence in the
    // Gated-DeltaNet layers. Flat means one batched pass costs exactly what k sequential steps
    // cost and no amount of drafting can win; falling means the loop should draft more than one
    // token per round, which is a fix rather than a verdict.
    std::printf("\nposition scaling (prefill %zu tok, then %d passes of k positions, best of 2):\n",
                ids.size(), 4);
    const int R = 4;
    const int probe_tok = 13;                       // any single token id; the cost is the point
    double best_for_k1 = 0;
    // TM_POSK selects the k list: one pass costs R reps times two, so measuring
    // the AMX crossover up to k=32 means minutes per leg and the k list is the
    // only knob worth having.
    std::vector<int> ks;
    if (const char* p = std::getenv("TM_POSK"); p && *p) {
        const char* s = p;
        while (*s) {
            ks.push_back((int)std::strtol(s, nullptr, 10));
            while (*s && *s != ',') ++s;
            if (*s == ',') ++s;
        }
    } else {
        ks = {1, 2, 4, 8, 16, 32};
    }
    for (int k : ks) {
        double best = 1e9;
        for (int rep = 0; rep < 2; ++rep) {
            m.reset_cache();
            (void)m.forward(ids);
            const std::vector<int> batch((std::size_t)k, probe_tok);
            const auto t0 = clk::now();
            for (int i = 0; i < R; ++i) (void)m.forward(batch);
            best = std::min(best, secs(t0, clk::now()));
        }
        std::printf("  k=%2d  %7.1f ms/pass  %6.1f ms/position  (%.2fx a 1-position pass)\n", k,
                    1000.0 * best / R, 1000.0 * best / (k * R),
                    (best / k) / (best_for_k1 > 0 ? best_for_k1 : best / k));
        if (k == 1) best_for_k1 = best;
    }
    if (!has_head) return 0;
    std::printf("\ndecoded plain: %s\n", tok.decode(p1).c_str());
    return same ? 0 : 1;
}
