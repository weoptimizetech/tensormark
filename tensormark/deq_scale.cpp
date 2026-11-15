// deq_scale.cpp — does the K-quant dequantize scale across threads?
//
// This is the premise behind row-parallelising the AMX prefill lane's dequant, and
// the reason to measure it BEFORE touching that lane: gemm_w's own note records
// that a pool-parallel dequant for the q4/q8 lane measured no faster AND corrupts
// the heap when interleaved with batched-attention sgemms, root cause never
// isolated. So: if this workload does not scale here, the whole idea is dead on
// throughput alone and the corruption question never has to be reopened.
//
// The workload is the real thing, not a proxy: every K-quant tensor of a real
// checkpoint, dequantized row by row into one reused fp32 buffer, exactly as the
// K-quant branch of gemm_w does it — once serially and once dispatched over the
// same GemvPool the GEMV path uses.
//
//   clang++ -std=c++23 -O3 -mcpu=apple-m1 -I. -DACCELERATE_NEW_LAPACK \
//       deq_scale.cpp -o build/deq_scale -framework Accelerate
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Load is part of the result, not context around it: this pool competes with
// whatever else is on the box, and the first attempt at this measurement landed in
// a window carrying 3.7-5.5 on four cores, which is not the machine the prefill
// lane would meet. Every arm therefore prints the load it ran under.
static void print_load(const char* tag) {
    double l[3] = {0, 0, 0};
    getloadavg(l, 3);
    std::printf("  [load %-24s %5.2f %5.2f %5.2f]\n", tag, l[0], l[1], l[2]);
    std::fflush(stdout);
}

namespace {

// Row-range dequant job: the shape the GemvPool already splits (o0, o1), with the
// tensor descriptor and the destination travelling in the weight pointer.
struct DeqJob {
    const void* blocks;
    int dtype;
    int K;
    float* dst;
};

void deq_kern(const tmllama::GemvArgs& a, int o0, int o1) {
    const auto& d = *static_cast<const DeqJob*>(a.w);
    for (int o = o0; o < o1; ++o)
        tmllama::Llama::dequant_row_kq(d.blocks, d.dtype, d.K, o,
                                d.dst + (std::size_t)o * d.K);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: deq_scale <model.tmq> [reps] [all|one]\n");
        return 1;
    }
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    tmllama::Llama m;
    m.ctx = 256;
    m.load_quant(argv[1]);
    if (m.qk_.empty()) {
        std::printf("no K-quant tensors in this model\n");
        return 0;
    }
    // "one" restricts the pass to the largest tensor: the same comparison at a
    // twentieth of the wall time, which is what lets it be re-run in a window that
    // is actually quiet instead of once per model load.
    using KV = typename std::remove_reference_t<decltype(m.qk_)>::value_type;
    std::vector<const KV*> work;
    for (const auto& kv : m.qk_) work.push_back(&kv);
    const bool only_largest = argc > 3 && std::string(argv[3]) == "one";
    if (only_largest) {
        std::sort(work.begin(), work.end(), [&](const KV* a, const KV* b) {
            const auto [ao, ai] = m.qshape_.at(a->first);
            const auto [bo, bi] = m.qshape_.at(b->first);
            return (std::size_t)ao * (std::size_t)ai > (std::size_t)bo * (std::size_t)bi;
        });
        work.resize(1);
    }
    std::size_t values = 0, max_bytes = 0;
    for (const KV* p : work) {
        const auto [n_out, n_in] = m.qshape_.at(p->first);
        values += (std::size_t)n_out * (std::size_t)n_in;
        max_bytes = std::max(max_bytes, (std::size_t)n_out * (std::size_t)n_in * 4);
    }
    const double fp32_mb = (double)values * 4 / 1e6;
    std::printf("%zu K-quant tensors%s, %.3f Gvalues, %.1f MB of fp32 written per pass "
                "(largest tensor %.1f MB), TM_THREADS=%s\n",
                work.size(), only_largest ? " (largest only)" : "",
                (double)values / 1e9, fp32_mb, (double)max_bytes / 1e6,
                getenv("TM_THREADS") ? getenv("TM_THREADS") : "auto");

    std::vector<float> buf(max_bytes / 4);
    auto one_pass_serial = [&] {
        const auto t0 = clk::now();
        for (const KV* p : work) {
            const auto& kq = p->second;
            const auto [n_out, n_in] = m.qshape_.at(p->first);
            for (int o = 0; o < n_out; ++o)
                tmllama::Llama::dequant_row_kq(kq.blocks, kq.dtype, n_in, o,
                                        buf.data() + (std::size_t)o * n_in);
        }
        return secs(t0, clk::now());
    };
    auto one_pass_pool = [&] {
        const auto t0 = clk::now();
        for (const KV* p : work) {
            const auto& kq = p->second;
            const auto [n_out, n_in] = m.qshape_.at(p->first);
            DeqJob d{kq.blocks, (int)kq.dtype, n_in, buf.data()};
            tmllama::GemvJob j;
            j.kern = &deq_kern;
            j.a.w = &d;
            j.a.n_in = n_in;
            j.n_out = n_out;
            tmllama::gemv_pool().dispatch(j);
        }
        return secs(t0, clk::now());
    };
    // Third arm: the same split, on plain threads at default QoS rather than through
    // the pool. The pool reports `qos 33` (= QOS_CLASS_USER_INTERACTIVE, its default
    // "interactive" band), and it also raises the CALLING thread into that band, so
    // the two arms differ in launcher, chunking (GemvJob::chunk, 128 rows) and QoS
    // together. Separating "the parallel split does not pay" from "this launcher does
    // not" is the difference between killing the idea and fixing how it is launched.
    auto one_pass_threads = [&] {
        const int nt = std::max(1, getenv("TM_THREADS") ? std::atoi(getenv("TM_THREADS")) : 4);
        const auto t0 = clk::now();
        for (const KV* p : work) {
            const auto& kq = p->second;
            const auto [n_out, n_in] = m.qshape_.at(p->first);
            DeqJob d{kq.blocks, (int)kq.dtype, n_in, buf.data()};
            std::vector<std::thread> ts;
            const int per = (n_out + nt - 1) / nt;
            for (int t = 0; t < nt; ++t) {
                const int o0 = t * per, o1 = std::min(n_out, o0 + per);
                if (o0 >= o1) break;
                ts.emplace_back([&d, n_in, o0, o1] {
                    tmllama::GemvArgs a{};
                    a.w = &d;
                    a.n_in = n_in;
                    deq_kern(a, o0, o1);
                });
            }
            for (auto& th : ts) th.join();
        }
        return secs(t0, clk::now());
    };

    double best_serial = 1e9, best_pool = 1e9, best_thr = 1e9;
    print_load("at start");
    for (int rep = 0; rep < reps; ++rep) {
        for (int arm = 0; arm < 3; ++arm) {
            if (rep == 0)
                print_load(arm == 0   ? "before arm A (serial)"
                           : arm == 1 ? "before arm B (pool)"
                                      : "before arm C (plain threads)");
            const double dt = arm == 0   ? one_pass_serial()
                              : arm == 1 ? one_pass_pool()
                                         : one_pass_threads();
            double& best = arm == 0 ? best_serial : arm == 1 ? best_pool : best_thr;
            best = std::min(best, dt);
        }
    }
    print_load("at end");
    std::printf("\nbest of %d, arms interleaved A B C:\n", reps);
    std::printf("  serial        %7.3f s/pass  %6.1f Mvalues/s fp32\n", best_serial,
                values / 1e6 / best_serial);
    std::printf("  pool          %7.3f s/pass  %6.1f Mvalues/s fp32   speedup %.2fx\n",
                best_pool, values / 1e6 / best_pool, best_serial / best_pool);
    std::printf("  plain threads %7.3f s/pass  %6.1f Mvalues/s fp32   speedup %.2fx\n",
                best_thr, values / 1e6 / best_thr, best_serial / best_thr);
    return 0;
}
