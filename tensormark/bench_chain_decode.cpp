// bench_chain_decode.cpp — apples-to-apples decode bench for the GPU-resident
// greedy chain: like Ollama's protocol, tokens are chosen ON the GPU and
// command buffers are committed back-to-back, with no host roundtrip per
// token. Compare against bench_llama_decode (host-synchronized per-token).
//
// Usage: ./build/bench_chain_decode <tmq> [tokens] [warmup] [context]
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "data/tinyllama/tinyllama_q40.tmq";
        const int tokens = argc > 2 ? std::atoi(argv[2]) : 128;
        const int warmup = argc > 3 ? std::atoi(argv[3]) : 4;
        const int context = argc > 4 ? std::atoi(argv[4]) : 512;
        tmllama::Llama model;
        model.ctx = context;
        model.load_quant(path);
        std::vector<int> prompt{1, 450, 7483, 310, 3444, 338};
        model.forward(prompt);
        if (!model.gpu_greedy_chain_ok())
            throw std::runtime_error("chain unavailable (need TM_DECODE_GPU=1 or auto-fit)");
        const int chunk = 32;
        for (int i = 0; i < warmup; ++i) model.generate_greedy_gpu(109, chunk);
        model.reset_cache();
        model.forward(prompt);
        using clock = std::chrono::steady_clock;
        const auto start = clock::now();
        int produced = 0, feed = 109;
        while (produced < tokens) {
            const int n = std::min(chunk, tokens - produced);
            // chain of n: n-1 tokens fed internally; the last is returned as
            // the next step's input (keep semantics in gpu_chain_end).
            auto toks = model.generate_greedy_gpu(feed, n + 1);
            feed = toks.back();
            produced += n;
        }
        const double seconds = std::chrono::duration<double>(clock::now() - start).count();
        printf("chain decode %d tokens in %.6f s = %.3f t/s | ctx %d\n",
               tokens, seconds, tokens / seconds, context);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
