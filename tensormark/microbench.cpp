// tensormark/microbench.cpp — M3 spec criterion 2: engine-vs-PyTorch microbench.
//
// Two-tier bar (specs/TENSORMARK_PERF_SPEC.md): composite/fused paths
// (attention block, dispatch-heavy chains) >= 1.3x PyTorch eager median;
// raw single-op GEMM parity (0.9-1.1x); nothing below 0.8x. All timings
// paired same-session: the torch runner (microbench_torch.py) prints the
// same shape labels; this binary prints JSON both sides can be joined on.
//
//   ./build.sh microbench.cpp && ./build/microbench            # engine side
//   python3 microbench_torch.py                                # torch side
//
// Shapes mirror real workloads: GEMM 64-4096 (decoder/MLP range), decode
// attention ctx 128-2048 (12 heads x dh 64), 1x1 + 3x3-dw convs (BlazeFace).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <Accelerate/Accelerate.h>
#include "autograd.h"   // tm_gelu_range — the shipped gelu kernel

using clk = std::chrono::steady_clock;
static double ms_now() {
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// median of reps after warmup; callable re-runs the same op in place
template <class F>
static double bench(F&& op, int warmup = 3, int reps = 20) {
    for (int i = 0; i < warmup; ++i) op();
    std::vector<double> t(reps);
    for (int i = 0; i < reps; ++i) { double a = ms_now(); op(); t[i] = ms_now() - a; }
    std::sort(t.begin(), t.end());
    return t[reps / 2];
}

int main() {
    std::mt19937 rng(7);
    auto rnd = [&](std::vector<float>& v) {
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        for (auto& x : v) x = u(rng);
    };
    printf("[");
    bool first = true;
    auto emit = [&](const char* name, double ms, double chk) {
        printf("%s{\"name\":\"%s\",\"ms\":%.4f,\"chk\":%.6f}",
               first ? "" : ",", name, ms, chk);
        first = false;
    };
    // checksum of the candidate's output buffer: the joiner asserts both
    // sides computed the SAME math before trusting any speed-up.
    auto chk = [](const std::vector<float>& v) {
        double s = 0;
        for (float x : v) s += std::fabs((double)x);
        return s;
    };
    // Shared inputs (build/bench_inputs.bin, written by gen_bench_inputs.py):
    // both sides compute on byte-identical data so output checksums prove the
    // same math. Fall back to random values (chk validation off) if absent.
    std::vector<float> file_inputs;
    const float* cur = nullptr;
    if (FILE* f = fopen("build/bench_inputs.bin", "rb")) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f) / 4;
        fseek(f, 0, SEEK_SET);
        file_inputs.resize((std::size_t)sz);
        if (fread(file_inputs.data(), 4, (std::size_t)sz, f) == (std::size_t)sz)
            cur = file_inputs.data();
        fclose(f);
    }
    auto fill = [&](std::vector<float>& v) {
        if (cur) { std::copy(cur, cur + v.size(), v.begin()); cur += v.size(); }
        else rnd(v);
    };

    // ---- tier 2: raw GEMM parity (same Accelerate BLAS as PyTorch/Accelerate
    //      backend; expect 0.9-1.1x) ----
    for (int n : {64, 256, 1024, 4096}) {
        // decoder-style: (T, K)x(K, N) with T=1 (decode GEMV-ish) and T=256
        std::vector<float> a1((std::size_t)n), w((std::size_t)n * n);
        fill(a1); fill(w);
        std::vector<float> a256((std::size_t)256 * n);
        fill(a256);
        for (int T : {1, 256}) {
            const float* a = T == 1 ? a1.data() : a256.data();
            std::vector<float> c((std::size_t)T * n);
            char name[64];
            snprintf(name, sizeof name, "gemm_T%d_K%d_N%d", T, n, n);
            double ms = bench([&] {
                // T=1 dispatches as gemv — what PyTorch's `a @ w` does
                            // (mv path), and what the decode path should use.
                if (T == 1)
                    cblas_sgemv(CblasRowMajor, CblasTrans, n, n, 1.f, w.data(),
                                n, a, 1, 0.f, c.data(), 1);
                else
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, n, n,
                                1.f, a, n, w.data(), n, 0.f, c.data(), n);
            });
            emit(name, ms, chk(c));
        }
    }

    // ---- tier 1: fused decode attention block (scores+mask+softmax+@V in
    //      one pass, no materialized upper triangle) ctx 128-2048 ----
    for (int S : {128, 512, 2048}) {
        const int H = 12, dh = 64, D = H * dh;
        std::vector<float> q((std::size_t)D), K((std::size_t)S * D), V((std::size_t)S * D),
            o((std::size_t)D), sc(S);
        fill(q); fill(K); fill(V);
        char name[64];
        snprintf(name, sizeof name, "attn_decode_H12_dh64_S%d", S);
        const float scale = 1.f / std::sqrt((float)dh);
        double ms = bench([&] {
            for (int hh = 0; hh < H; ++hh) {
                const float* qh = q.data() + hh * dh;
                float m = -1e30f;
                for (int s = 0; s < S; ++s) {
                    sc[s] = cblas_sdot(dh, qh, 1, &K[(std::size_t)s * D + hh * dh], 1) * scale;
                    m = std::max(m, sc[s]);
                }
                float z = 0.f;
                for (int s = 0; s < S; ++s) { sc[s] = std::exp(sc[s] - m); z += sc[s]; }
                float* acc = o.data() + hh * dh;
                // overwrite (not saxpy-accumulate): the benchmark re-runs
                // this op 23x; accumulating would grow chk by exactly the
                // rep count and fail the same-math gate.
                std::fill(acc, acc + dh, 0.f);
                for (int s = 0; s < S; ++s)
                    cblas_saxpy(dh, sc[s] / z, &V[(std::size_t)s * D + hh * dh], 1, acc, 1);
            }
        });
        emit(name, ms, chk(o));
    }

    // ---- tier 1: dispatch-heavy elementwise chain (the M3 profile's other
    //      cost: many small ops back-to-back) ----
    {
        const int N = 256 * 768;
        std::vector<float> a(N), b(N), c(N);
        fill(a); fill(b);
        static volatile float sink;   // volatile: printf("") is optimizable, this is not
        // Engine gelu op (tm_gelu_range: vForce vvtanhf over TM_VBLK chunks) —
        // the shipped kernel, not a scalar restatement of the formula.
        std::vector<float> v(N);
        double ms = bench([&] {
            for (int i = 0; i < N; ++i) v[i] = a[i] + b[i];
            tmg::tm_gelu_range(v.data(), c.data(), N);
            sink = c[0] + c[N / 2] + c[N - 1];   // defeat dead-code elimination
        });
        emit("gelu_fused_N196608", ms, chk(c));
    }

    // ---- tier 1: 1x1 conv as GEMM + 3x3 depthwise (BlazeFace mix) ----
    {
        const int C = 96, HW = 32 * 32;
        std::vector<float> x((std::size_t)C * HW), w((std::size_t)C * C), y((std::size_t)C * HW);
        fill(x); fill(w);
        double ms = bench([&] {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C, HW, C,
                        1.f, w.data(), C, x.data(), HW, 0.f, y.data(), HW);
        });
        emit("conv1x1_C96_32x32", ms, chk(y));
    }
    {
        const int C = 96, Wd = 32;
        std::vector<float> x((std::size_t)C * (Wd + 2) * (Wd + 2)), out((std::size_t)C * Wd * Wd);
        fill(x);
        // depthwise weights come from the shared input file too (96 x 3x3) —
        // hardcoded constants made this candidate compute different math
        // from the torch side and fail the same-math gate.
        std::vector<float> w((std::size_t)C * 9);
        fill(w);
        static volatile float dsink;
        double ms = bench([&] {
            for (int c = 0; c < C; ++c)
                for (int r = 0; r < Wd; ++r)
                    for (int col = 0; col < Wd; ++col) {
                        float s = 0.f;
                        for (int dr = 0; dr < 3; ++dr)
                            for (int dc = 0; dc < 3; ++dc)
                                s += x[((std::size_t)c * (Wd + 2) + r + dr) * (Wd + 2) + col + dc] * w[(std::size_t)c * 9 + dr * 3 + dc];
                        out[((std::size_t)c * Wd + r) * Wd + col] = s;
                    }
            dsink = out[0];   // defeat dead-code elimination
        });
        emit("dw3x3_C96_32x32", ms, chk(out));
    }

    printf("]\n");
    return 0;
}
