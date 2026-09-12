// Quantized Llama decode benchmark: bounded warmup, steady-state tokens/s,
// latency distribution and greedy sequence hash. Ambient CPU load is part
// of the workload, never a reason to discard a run.
#include "llama.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <vector>

int main(int argc, char** argv) {
    try {
        auto integer = [&](int arg, int fallback, int lo, int hi) {
            if (argc <= arg) return fallback;
            char* end = nullptr;
            const long v = std::strtol(argv[arg], &end, 10);
            if (!*argv[arg] || *end || v < lo || v > hi)
                throw std::invalid_argument("invalid benchmark count/context");
            return (int)v;
        };
        const std::string path = argc > 1 ? argv[1] : "data/tinyllama/tinyllama_q40.tmq";
        const int tokens = integer(2, 64, 1, 32768);
        const int warmup = integer(3, 4, 0, 1024);
        const int context = integer(4, 2048, 8, 32768);
        if (tokens + warmup + 6 > context)
            throw std::invalid_argument("prompt + warmup + tokens exceeds context");
        tmllama::Llama model;
        model.ctx = context;
        model.load_quant(path);
        std::vector<int> prompt{1, 450, 7483, 310, 3444, 338};
        auto* logits = &model.forward(prompt);
        auto next = [&] {
            int best = 0;
            for (int j = 0; j < model.V; ++j) {
                if (!std::isfinite((*logits)[j])) throw std::runtime_error("non-finite logits");
                if ((*logits)[j] > (*logits)[best]) best = j;
            }
            return best;
        };
        for (int i = 0; i < warmup; ++i) logits = &model.forward({next()});
        std::vector<double> latencies;
        latencies.reserve(tokens);
        uint64_t hash = 14695981039346656037ULL;
        int last = 0;
        using clock = std::chrono::steady_clock;
        const auto start = clock::now();
        for (int i = 0; i < tokens; ++i) {
            const auto step = clock::now();
            last = next();
            hash = (hash ^ (uint64_t)last) * 1099511628211ULL;
            logits = &model.forward({last});
            latencies.push_back(std::chrono::duration<double>(clock::now() - step).count());
        }
        const double seconds = std::chrono::duration<double>(clock::now() - start).count();
        std::sort(latencies.begin(), latencies.end());
        const double median = latencies[latencies.size() / 2];
        const double p95 = latencies[(size_t)std::ceil(0.95 * latencies.size()) - 1];
        struct rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        printf("decode %d tokens in %.6f s = %.3f t/s | peak RSS %.3f GB | last id %d | "
               "median_ms %.3f p95_ms %.3f | hash %016llx | warmup %d ctx %d\n",
               tokens, seconds, tokens / seconds, ru.ru_maxrss / 1073741824.0,
               last, median * 1e3, p95 * 1e3, (unsigned long long)hash, warmup, context);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
