// hybrid_bench.cpp — real-host prefill bench for the hybrid split controller.
//
//   clang++ -std=c++23 -O2 -DTM_HAVE_METAL -march=native -fobjc-arc \
//       hybrid_bench.cpp metal_shim.mm -o build/hybrid_bench \
//       -framework Metal -framework Foundation -framework Accelerate
//
//   TM_LLAMA_HYBRID=<frac|auto> TM_PREFILL_GPU=1 TM_DECODE_GPU=0 \
//       ./build/hybrid_bench data/tinyllama/tinyllama_q40.tmq 2000 4
//
// Runs `rounds` T-token prefills (first one discarded as warmup: it compiles
// the Metal library), printing one JSON line per round on stdout. With
// TM_HYBRID_ID=1 the controller's per-prefill gpu_ms/cpu_ms lines go to
// stderr; under a pinned fraction they are still emitted, so a sweep can
// compute e(frac) per round. Pinned rounds never touch tmtune state; an auto
// run does, subject to the quiet-regime persistence gate.
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string tmq = argc > 1 ? argv[1] : "data/tinyllama/tinyllama_q40.tmq";
    const int T = argc > 2 ? std::atoi(argv[2]) : 2000;
    const int rounds = argc > 3 ? std::atoi(argv[3]) : 4;
    if (T < 512 || T > 2048 || rounds < 2) {
        std::fprintf(stderr, "need 512 <= T <= 2048 and rounds >= 2\n");
        return 2;
    }
    tmllama::Llama m;
    m.load_quant(tmq);
    std::mt19937 rng(7);
    std::vector<int> ids(T);
    for (auto& id : ids) id = 1 + (int)(rng() % (m.V - 1));
    // Run one extra prefill and drop it: the first forward compiles the Metal
    // library and warms the ESC controller + pipelines, so its wall time is not
    // representative. The header above promises exactly this but the loop used
    // to print round 0 too, dragging the prefill median down ~15% in the paired
    // tables (690 vs 943 tok/s cold-vs-warm in the same run).
    for (int r = 0; r < rounds + 1; ++r) {
        m.reset_cache();
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float>& logits = m.forward(ids);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        (void)logits;
        if (r == 0) continue;   // warmup round: not measured
        std::printf("{\"round\":%d,\"wall_ms\":%.1f,\"tok_per_s\":%.1f}\n",
                    r, ms, T / ms * 1000.0);
    }
    return 0;
}
