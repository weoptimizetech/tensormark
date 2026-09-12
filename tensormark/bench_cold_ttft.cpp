// bench_cold_ttft.cpp — cold-process time to first token: process start ->
// load_quant (mmap) -> prompt prefill -> first decode token, all wall-timed.
// This is the one-shot CLI niche (llama_chat_metal --oneshot).
//
// Usage: ./build/bench_cold_ttft <tmq> [ctx]
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <vector>

using clock_ = std::chrono::steady_clock;
static double ms(clock_::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "data/tinyllama/tinyllama_q40.tmq";
        tmllama::Llama model;
        model.ctx = argc > 2 ? std::atoi(argv[2]) : 512;
        const auto t0 = clock_::now();
        model.load_quant(path);
        const auto t1 = clock_::now();
        std::vector<int> prompt{1, 450, 7483, 310, 3444, 338};
        auto& lg = model.forward(prompt);
        const auto t2 = clock_::now();
        int best = 0;
        for (int j = 1; j < model.V; ++j) if (lg[j] > lg[best]) best = j;
        const auto t3 = clock_::now();
        struct rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        printf("cold TTFT: load %.0f ms + prefill %.0f ms + first token %.3f ms = %.0f ms | peak RSS %.2f GB\n",
               ms(t1 - t0), ms(t2 - t1), ms(t3 - t2), ms(t3 - t0), ru.ru_maxrss / 1073741824.0);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
