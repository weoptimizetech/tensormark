// ANE execution/finite-output gate, not an exact-equivalence or quality gate.
// TM_ANE=1 TM_ANE_PATH=<package> ./build/bench_ane_split <model.tmq> 1024 3
// Hidden-only experiments additionally require TM_ANE_APPROX_KV=1. Their
// projected cache is approximate; throughput must not be labeled exact-lane.
#include "llama.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
int positive_int(std::string_view text) {
    int result = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size() || result < 1)
        throw std::invalid_argument("expected a positive integer");
    return result;
}
void require_finite(const std::vector<float>& values) {
    if (values.empty() || !std::all_of(values.begin(), values.end(),
                                     [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("empty or nonfinite logits");
}
int top1(const std::vector<float>& values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}
}

int main(int argc, char** argv) {
    try {
        const std::string tmq = argc > 1 ? argv[1] : "data/tinyllama/tinyllama_q40.tmq";
        const int T = argc > 2 ? positive_int(argv[2]) : 1024;
        const int rounds = argc > 3 ? positive_int(argv[3]) : 5;
        const int seed = argc > 4 ? positive_int(argv[4]) : 7;
        const int decode_steps = argc > 5 ? positive_int(argv[5]) : 0;
        if (T < 2 || argc > 6)
            throw std::invalid_argument("usage: bench_ane_split <model.tmq> <T >= 2> <rounds> [seed] [decode_steps]");
        const char* enabled = std::getenv("TM_ANE");
        if (!enabled || std::string_view(enabled) != "1")
            throw std::invalid_argument("this ANE gate requires TM_ANE=1 and TM_ANE_PATH");

        // Setup reads TM_ANE at load time, not during forward(). Disable it
        // BEFORE loading, otherwise the alleged reference can itself use ANE.
        if (setenv("TM_ANE", "0", 1)) throw std::runtime_error("setenv failed");
        tmllama::Llama m;
        m.load_quant(tmq);
        if (T > m.ctx || decode_steps > m.ctx - T)
            throw std::invalid_argument("prefill plus decode steps exceeds context");
        std::mt19937 rng(seed);
        std::vector<int> ids(T);
        if (const char* tokens = std::getenv("TM_BENCH_TOKENS")) {
            std::ifstream input(tokens);
            if (!input) throw std::invalid_argument("cannot open TM_BENCH_TOKENS");
            for (auto& id : ids)
                if (!(input >> id) || id < 0 || id >= m.V)
                    throw std::invalid_argument("TM_BENCH_TOKENS needs T valid token IDs");
            std::string extra;
            if (input >> extra)
                throw std::invalid_argument("TM_BENCH_TOKENS has more than T token IDs");
        } else {
            for (auto& id : ids) id = 1 + static_cast<int>(rng() % (m.V - 1));
        }
        const std::vector<float> ref = m.forward(ids);
        require_finite(ref);
        if (m.ane_prefills() != 0) throw std::runtime_error("reference unexpectedly used ANE");
        // Teacher forcing compares identical token histories even if the ANE
        // lane would choose a different token. Run separately from prefill timing.
        std::vector<int> teacher_ids;
        std::vector<std::vector<float>> decode_ref;
        int next = top1(ref);
        for (int step = 0; step < decode_steps; ++step) {
            teacher_ids.push_back(next);
            decode_ref.push_back(m.forward({next}));
            require_finite(decode_ref.back());
            next = top1(decode_ref.back());
        }
        if (setenv("TM_ANE", "1", 1)) throw std::runtime_error("setenv failed");
        m.set_ane_enabled();
        if (!m.ane_ready()) throw std::runtime_error("ANE setup refused; not benchmarking fallback");

        for (int r = 0; r < rounds; ++r) {
            m.reset_cache();
            const auto before = m.ane_prefills();
            const auto start = std::chrono::steady_clock::now();
            const auto& logits = m.forward(ids);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (m.ane_prefills() != before + 1)
                throw std::runtime_error("ANE did not execute; fallback is not a passing gate");
            require_finite(logits);
            if (logits.size() != ref.size() || !std::isfinite(ms) || ms <= 0)
                throw std::runtime_error("invalid logits shape or timing");
            double maxdiff = 0, error2 = 0, ref2 = 0;
            for (std::size_t i = 0; i < logits.size(); ++i) {
                const double diff = static_cast<double>(logits[i]) - ref[i];
                maxdiff = std::max(maxdiff, std::fabs(diff));
                error2 += diff * diff;
                ref2 += static_cast<double>(ref[i]) * ref[i];
            }
            const bool top1_match = top1(logits) == top1(ref);
            std::printf("{\"round\":%d,\"ane_executed\":true,\"wall_ms\":%.1f,\"tok_per_s\":%.1f,"
                        "\"last_logits_maxdiff_vs_engine\":%.6f,\"relative_l2\":%.6f,"
                        "\"last_token_top1_match\":%s}\n",
                        r, ms, T / ms * 1000, maxdiff,
                        std::sqrt(error2 / std::max(ref2, 1e-30)), top1_match ? "true" : "false");
            std::fflush(stdout);
            if (decode_steps && r == 0) {
                double decode_error2 = 0, decode_ref2 = 0, decode_maxdiff = 0;
                int matching = 0;
                for (int step = 0; step < decode_steps; ++step) {
                    const auto& actual = m.forward({teacher_ids[step]});
                    const auto& expected = decode_ref[step];
                    require_finite(actual);
                    if (actual.size() != expected.size())
                        throw std::runtime_error("decode logits shape mismatch");
                    matching += top1(actual) == top1(expected);
                    for (std::size_t i = 0; i < actual.size(); ++i) {
                        const double diff = double(actual[i]) - expected[i];
                        decode_error2 += diff * diff;
                        decode_ref2 += double(expected[i]) * expected[i];
                        decode_maxdiff = std::max(decode_maxdiff, std::fabs(diff));
                    }
                }
                std::printf("{\"phase\":\"decode_quality\",\"seed\":%d,\"steps\":%d,"
                            "\"top1_matches\":%d,\"maxdiff\":%.6f,\"relative_l2\":%.6f}\n",
                            seed, decode_steps, matching, decode_maxdiff,
                            std::sqrt(decode_error2 / std::max(decode_ref2, 1e-30)));
                std::fflush(stdout);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ANE gate failed: %s\n", error.what());
        return 1;
    }
}
