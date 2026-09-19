// tensormark/llama.h — M5: Llama-family (Llama-2 arch) fp32 inference.
//
// Port of the VERIFIED numpy reference (llama_ref.py, max abs 1.8e-4 vs HF
// fp32 logits on TinyLlama-1.1B). Config: hidden 2048, heads 32 (dh 64),
// kv heads 4 (GQA rep 8), layers 22, vocab 32000, ctx 2048, rope_theta
// 10000, RMSNorm eps 1e-5, lm_head NOT tied, separate q/k/v/o projections,
// MLP gate/up/down (SwiGLU).
//
// RoPE: the HF rotate_half (split halves x[:32]|x[32:]) form applied to
// (x*c + rotate(x)*s) equals, per pair (x[j], x[j+32]):
//     out[j]    = x[j]*cos - x[j+32]*sin
//     out[j+32] = x[j+32]*cos + x[j]*sin
// with angle theta^(-2j) — the C++ implements this directly.
//
// KV cache: PER-HEAD CONTIGUOUS per KV head (KVH blocks of (ctx, dh)),
// M3 layout. T==1 decode uses gemv; T>1 prefill uses gemm.
#pragma once

#include <mach/mach.h>   // host_statistics: ambient-load sample (below)
#include <chrono>
#include <cfenv>
#include <thread>

// Fraction of cores currently busy system-wide (idle excluded), from two
// host_statistics samples ~150 ms apart — same semantics as
// benchguard.py busy_cores (ticks layout: user, system, IDLE, nice).
// Used ONLY for the pool-width default; a transient burst at first
// dispatch pins the session width, which matches how decode sessions
// actually run (ambient load persists over minutes, not ms).
// M6: optional Metal batched prefill. tm_metal_sgemm_q4 is defined only in
// ObjC++ TUs that include metal.h; weak_import keeps pure-C++ builds and
// builds without the Metal TU linkable (symbol resolves to null -> CPU path).
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
#include "llama_gpu_api.h"
extern "C" [[nodiscard]] bool tm_metal_sgemm_q4(const float* A, const void* Wq, float* C,
                                  unsigned M, unsigned N, unsigned K)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_q4_prefill(const float* A, const void* const* Wqs,
                                    const unsigned* Ns, unsigned nparts,
                                    float* C, unsigned M, unsigned K)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_q4_gemv(const float* x, const void* Wq, float* y,
                                 unsigned N, unsigned K)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_begin(void) __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_flush(void) __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_reserve(int slot, size_t bytes)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_end(int outslot, float* dst, unsigned n)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_upload(int slot, size_t byte_off,
                                    const void* src, size_t bytes)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_end_async(void) __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_wait_pending(void) __attribute__((weak_import));
extern "C" void tm_metal_tok_argmax(int xslot, int tokslot, unsigned idx, unsigned n)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_embed_q4(int tokslot, unsigned idx, int wid, uint64_t woff,
                                      int xslot, unsigned D)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_sample(int xslot, int tokslot, unsigned idx, unsigned n, float inv_temp, unsigned seed)
    __attribute__((weak_import));
extern "C" [[nodiscard]] unsigned tm_metal_tok_peek(int tokslot, unsigned idx) __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_download(int slot, size_t byte_off,
                                      void* dst, size_t bytes)
    __attribute__((weak_import));
// K/V cache rows in the slot's element type (fp32 or half per the Metal
// policy); offsets and counts in elements.
extern "C" [[nodiscard]] bool tm_metal_tok_kv_reserve(int slot, size_t n)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_kv_upload(int slot, size_t elem_off,
                                       const float* src, size_t n)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_kv_download(int slot, size_t elem_off,
                                         float* dst, size_t n)
    __attribute__((weak_import));
extern "C" [[nodiscard]] bool tm_metal_tok_probe_copy(int srcslot, size_t src_off,
                                        int dstslot, unsigned n)
    __attribute__((weak_import));
extern "C" [[nodiscard]] int tm_metal_tok_wbuf(const void* tensor_ptr, size_t bytes,
                                 const void* map_base, size_t map_size,
                                 int zero_copy, uint64_t& woff_out)
    __attribute__((weak_import));
extern "C" bool tm_metal_tok_gemv_seg_supported(unsigned dtype)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_gemv_seg_dt(int xslot, int wid, int yslot,
                                         size_t yoff, unsigned N, unsigned K,
                                         unsigned dtype, const uint64_t* seg_woff,
                                         const unsigned* seg_rows, unsigned nseg)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_gemv(int xslot, int wid, uint64_t woff,
                                  int yslot, size_t yoff, unsigned N,
                                  unsigned K)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_gemv_h(int xslot, int wid, uint64_t woff,
                                    int yslot, size_t yoff, unsigned N,
                                    unsigned K)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_gemv_seg(int xslot, int wid, int yslot,
                                      size_t yoff, unsigned N, unsigned K,
                                      const uint64_t* seg_woff,
                                      const unsigned* seg_rows, unsigned nseg)
    __attribute__((weak_import));
    // K-quant weights (Q4_1/Q5_K/Q6_K): the token lane's GEMV for the dtypes a GGUF
    // K-quant source keeps in the .tmq, which is what the hybrid lane needs to admit
    // the shipped qwen35 checkpoints at all.
extern "C" void tm_metal_tok_gemv_kq(int xslot, int wid, uint64_t woff,
                                         int yslot, size_t yoff, unsigned N,
                                         unsigned K, unsigned dtype)
        __attribute__((weak_import));
        // The K-quant embedding gather. Without it a Q6_K embedding refuses the greedy chain
        // (gpu_greedy_chain_ok tests the embed against gpu_h_/gpu_w_), and an unchained lane
        // commits once per token instead of chaining — which is where its advantage lives.
        extern "C" void tm_metal_tok_embed_kq(int tokslot, unsigned idx, int wid,
                                          uint64_t woff, int xslot, unsigned D,
                                          unsigned dtype)
        __attribute__((weak_import));
extern "C" void tm_metal_tok_embed_h(int tokslot, unsigned idx, int wid,
                                     uint64_t woff, int xslot, unsigned D)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_bias_qkv(int yslot, int biasslot, size_t bias_off,
                                      unsigned N)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_rmsnorm(int xslot, const void* w,
                                     int outslot, unsigned D, float eps)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_add(int a, int b, unsigned n)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_silu_mul(int g, size_t goff, int u,
                                      size_t uoff, unsigned n)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_rope_q(int qslot, unsigned H, unsigned dh,
                                    int pos, float theta)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_rope_k_cache(int ksrcslot, size_t ksrc_off,
                                          int cacheslot, unsigned KVH,
                                          unsigned dh, unsigned ctx,
                                          int pos, float theta)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_v_cache(int vsrcslot, size_t vsrc_off,
                                     int cacheslot, unsigned KVH,
                                     unsigned dh, unsigned ctx, int pos)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_attn(int qslot, int kcslot, int vcslot,
                                  int probslot, int outslot, unsigned QD,
                                  unsigned H, unsigned KVH, unsigned dh,
                                  unsigned ctx, int allow, float scale,
                                  unsigned REP)
    __attribute__((weak_import));
#endif

#include "quant.h"
#include "gdn.h"   // qwen35 Gated-DeltaNet kernels
#include "tmq.h"
#include "safetensors.h"
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <limits>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <Accelerate/Accelerate.h>
#include "gemv_pool.h"
#include "tm_tune.h"

namespace tmllama {

// --- numeric configuration accessors ---------------------------------------
//
// Two shapes, deliberately different names because they take different things
// and mixing them is a silent wrong answer:
//
//   env_var_*(name)          reads the environment VARIABLE called `name`
//   parse_*_value(value)     parses an already-resolved VALUE (an argv token)
//
// The 2026-09-16 defect was exactly that confusion: a member
// `Llama::env_double(const char* name, double fallback)` shadowed this
// namespace's value-taking `env_double(const char* value, double fallback)`, so
// llama.h:1529/:1610 — which pass `getenv("TM_GPU_BUDGET_GB")` — compiled
// against the name-taking overload, called getenv() on the *value* (and
// getenv(nullptr) when the variable was unset), and left TM_GPU_BUDGET_GB dead
// while the spec documented it as live. There is now exactly one accessor per
// type, it returns std::optional (no fallback to hide a miss), and it is named
// for the variable so a value argument reads wrong at the call site.
//
// Parsing is strict: trailing garbage is a failure, so "128abc" is nullopt
// instead of silently meaning 128. Built on strtol/strtod (not from_chars):
// the CI runner's libc++ lacks the floating-point overloads.
[[nodiscard]] inline std::optional<double> env_var_double(const char* name) {
    const char* e = name ? std::getenv(name) : nullptr;
    if (!e || !*e) return std::nullopt;
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if (end == e || *end) return std::nullopt;
    return v;
}
[[nodiscard]] inline std::optional<long> env_var_long(const char* name) {
    const char* e = name ? std::getenv(name) : nullptr;
    if (!e || !*e) return std::nullopt;
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || *end) return std::nullopt;
    return v;
}
// Load-time configuration that must be LOUD when it is wrong: unset or empty
// takes `fallback`, but a present-and-malformed value throws instead of
// degrading. A silently wrong memory budget or device gate is worse than a
// failed load (the GPU paths below are the watchdog-panic remedy).
inline double env_var_double_checked(const char* name, double fallback) {
    const char* e = name ? std::getenv(name) : nullptr;
    if (!e || !*e) return fallback;
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if (end == e || *end || !(v >= 0.0))
        throw std::invalid_argument(std::string(name) + " must be a non-negative number");
    return v;
}
[[nodiscard]] inline std::optional<long> parse_long_value(const char* value) {
    if (!value || !*value) return std::nullopt;
    char* end = nullptr;
    const long v = std::strtol(value, &end, 10);
    if (end == value || *end) return std::nullopt;
    return v;
}
// Historical name, kept for the CLI lanes (llama_chat.cpp, gpt2_chat.cpp),
// which pass argv tokens and chat-command text rather than a variable name.
// It never reads the environment.
[[nodiscard]] inline long env_long(const char* value, long fallback) {
    return parse_long_value(value).value_or(fallback);
}

// Physical unified memory in GiB. The auto device gates are sized as a
// *fraction* of this, not a fixed GiB bar, so the engine harvests a bigger
// Apple chip automatically: the same 0.25/0.5 fractions that keep a 7B on
// the CPU of the 8 GB host let a 27B run on the GPU of a 128 GB host.
inline double physical_ram_gb() {
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long psize = ::sysconf(_SC_PAGESIZE);
    if (pages <= 0 || psize <= 0) return 8.0;   // unknown: assume 8 GB
    return (double)pages * (double)psize / (double)(1u << 30);
}

// Page-aligned, page-multiple allocation for buffers the GPU views zero-copy
// (the KV caches): a small context makes std::vector hand back a 16-byte
// aligned heap block, and the GPU layer stack then refused to start
// (2026-09-07, seen as "KV cache not page-shareable" on short benches).
template <class T>
struct PageAlloc {
    using value_type = T;
    PageAlloc() = default;
    template <class U> PageAlloc(const PageAlloc<U>&) {}
    T* allocate(std::size_t n) {
        const std::size_t pg = 16384;
        const std::size_t bytes = ((n * sizeof(T)) + pg - 1) / pg * pg;
        void* p = nullptr;
        if (posix_memalign(&p, pg, bytes ? bytes : pg) != 0) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) { std::free(p); }
    template <class U> bool operator==(const PageAlloc<U>&) const { return true; }
    template <class U> bool operator!=(const PageAlloc<U>&) const { return false; }
};
using PageVec = std::vector<float, PageAlloc<float>>;


inline float silu_f(float x) { return x / (1.f + std::exp(-x)); }


// Opt-in per-section wall clock for block() (TM_LLAMA_PROFILE=1): totals to
// stderr at exit. ~20 ns per mark; skipped entirely when off.
struct LlamaProf {
    // LMHEAD is outside block(): it is the output projection plus the argmax
    // over the whole vocabulary, once per token. It is marked because the
    // remainder after the block sections is otherwise attributed to nothing,
    // which made the biggest single tensor in decode invisible in the trace.
    // GEMV_CORE is a SUBSET measure, not a sibling of the section marks: it
    // accumulates time inside gemv_w only, so it overlaps qkv_gemm, gate_up,
    // o_proj and down_gemm. It exists to answer one question — of the ~15 ms a
    // body spends in the GEMV-bearing sections, how much is the kernel and how
    // much is the glue between dispatches? Comparing it against the sum of
    // those sections is the point; adding it to the total is not.
    // QUANT_X is the second SUBSET measure: the int8 activation encoding that
    // every GEMV does before it touches a weight. Several consecutive
    // projections read the SAME activation (the qkv/gate/beta/alpha block all
    // take h_), so if this share is material the fix is to encode once and
    // reuse, not to touch a kernel.
    // Scoped so a Sec cannot silently convert into the array index it feeds:
    // mark()/add_ms() do the conversion once, here.
    // GDN_* break the Gated-DeltaNet layers (24 of this model's 32) out of the
    // marks they used to share with full attention. Without them a GDN layer's
    // four projections charged to qkv_gemm and its conv+recurrence to attention,
    // so the trace could say what a decode costs but not who spends it -- and
    // "who" is the whole question for a hybrid model.
    enum class Sec { RMS1, QKV, QKV_SPLIT, ROPE, KV_APPEND, ATTN, OPROJ, RES1, RMS2, GATE_UP, SILU, DOWN, RES2, GPU_STACK, LMHEAD, GEMV_CORE, QUANT_X, GDN_PROJ, GDN_CONV, GDN_RECUR, GDN_OUT, N };
    static constexpr int N = std::to_underlying(Sec::N);
    static constexpr const char* names[N] = {"rmsnorm1", "qkv_gemm", "qkv_split", "rope", "kv_append", "attention",
                                             "o_proj", "residual1", "rmsnorm2", "gate_up_gemm", "silu_mul", "down_gemm", "residual2",
                                             "gpu_stack", "lm_head", "gemv_core(subset)", "quant_x(subset)",
                                             "gdn_proj", "gdn_conv", "gdn_recur", "gdn_out"};
    const bool on;
    double acc[N] = {};
    std::chrono::steady_clock::time_point t0;
    LlamaProf() : on([] { const char* e = std::getenv("TM_LLAMA_PROFILE"); return e && e[0] == '1'; }()) {}
    ~LlamaProf() {
        if (!on) return;
        double tot = 0; for (double a : acc) tot += a;
        std::fprintf(stderr, "TM_LLAMA_PROFILE (ms, %% of %.1f ms accounted)\n", tot);
        for (int i = 0; i < N; ++i) std::fprintf(stderr, "  %-13s %9.2f  %5.1f%%\n", names[i], acc[i], tot > 0 ? 100 * acc[i] / tot : 0);
    }
    void start() { if (on) t0 = std::chrono::steady_clock::now(); }
    void mark(Sec s) {
        if (!on) return;
        const auto t = std::chrono::steady_clock::now();
        acc[std::to_underlying(s)] +=
            std::chrono::duration<double, std::milli>(t - t0).count();
        t0 = t;
    }
    // Accumulate into a slot WITHOUT touching t0, so a subset measure can be
    // taken from inside code that an enclosing section is already timing.
    void add_ms(Sec s, double ms) { if (on) acc[std::to_underlying(s)] += ms; }
};
inline LlamaProf& llama_prof() { static LlamaProf p; return p; }

// Scoped subset timer: adds its duration to one slot on exit and leaves the
// enclosing section's t0 alone.
struct ProfGuard {
    LlamaProf::Sec s;
    std::chrono::steady_clock::time_point t0;
    const bool on;
    explicit ProfGuard(LlamaProf::Sec sec) : s(sec), on(llama_prof().on) {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~ProfGuard() {
        if (on)
            llama_prof().add_ms(s, std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0).count());
    }
    ProfGuard(const ProfGuard&) = delete;
    ProfGuard& operator=(const ProfGuard&) = delete;
};

    // Cross-engine tensor dump (TM_DEBUG_DUMP=1). Names match llama.cpp's
    // `llama-debug -m model.gguf -p PROMPT -n 1`, whose per-tensor lines print
    // a sum; the sum is invariant under the layout differences between our
    // buffers and ggml's, so the two traces are directly diffable.
    inline bool tm_dump_on() {
        static const bool on = [] { const char* e = std::getenv("TM_DEBUG_DUMP");
                                     return e && e[0] == '1'; }();
        return on;
    }
    // Row-level dump for element-by-element cross-engine checks (TM_DUMP_ROW=<layer>):
    // writes the first `n` floats of a checkpoint to /tmp/tm_<tag>_l<layer>.bin so a
    // numpy recomputation from the GGUF can be compared value by value. Sums hide
    // compensating errors; single elements and a numpy redo of the same matmul do not.
    inline void tm_dump_row(const char* tag, int layer, const float* p, std::size_t n) {
        if (!tm_dump_on() || !p) return;
        const char* e = std::getenv("TM_DUMP_ROW");
        if (!e || std::atoi(e) != layer) return;
        std::string f = std::string("/tmp/tm_") + tag + "_l" + std::to_string(layer) + ".bin";
        if (FILE* fp = std::fopen(f.c_str(), "wb")) {
            std::fwrite(p, sizeof(float), n, fp);
            std::fclose(fp);
            std::printf("TMROW %-14s l=%-3d n=%zu -> %s\n", tag, layer, n, f.c_str());
            std::fflush(stdout);
        }
    }

    inline void tm_dump(const char* name, int layer, const float* p, std::size_t n) {
        if (!tm_dump_on() || !p) return;
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += (double)p[i];
        std::printf("TMDUMP %-28s l=%-3d n=%-8zu sum=%.6f  head=%.6f  tail=%.6f\n",
                    name, layer, n, sum, n ? (double)p[0] : 0.0,
                    n ? (double)p[n - 1] : 0.0);
        std::fflush(stdout);
    }

// --------------------------------------------------------------------------
// Hybrid prefill split: sampled two-loop discrete controller (Metal-free,
// unit-testable).
//
// Discrete control formulation (2026-09-09). One sample per prefill
// (Ts = 1 prefill; the actuator — the CPU row fraction — holds between
// samples), so every loop below is a discrete-time design on that sample
// grid, not a continuous law.
//
// Event-triggered inner structures (plant-based, unchanged from 2026-09-08):
//   * First sample jumps to the per-row rate-ratio prior (a static-map
//     estimate that transfers across prompt lengths; most agent requests
//     are one prefill in a fresh process).
//   * Straggler retreat: balance error below -0.15 shrinks the split
//     fast (slide-mode exit; parking at zero measured IS the loaded
//     optimum: 351 vs 747 t/s at 4 burner cores, T=2000).
//   * Park with a load-scaled re-probe interval (8 prefills quiet, up to
//     64 loaded); a fraction is persisted ONLY when the machine was
//     quiet at save time (L < 0.15) — kStateVersion in tm_tune.h bumped
//     when the persisted quantity changes meaning.
//
// Outer loop — why the balance PI is gone. Its regulated variable
// e = (gpu_ms - cpu_ms)/(gpu_ms + cpu_ms) is DEGENERATE at short
// prompts: the identified static map at T=512 reads e* ≈ 0 across
// frac [0.13, 0.19] (6-round pinned means +0.042, -0.003, +0.002,
// -0.008; per-sample sd 0.06) while wall time varies 869 -> 931 t/s
// over the same band. A reference inside the measurement's null space
// cannot be tracked into the optimum, and the shipped controller sat
// at 869 t/s for twelve straight prefills (equilibrium frac 0.178).
// At T=2000 the map is non-degenerate (e*: +0.249, +0.186, +0.136,
// +0.062, -0.027 over frac 0.15..0.25; sd 0.02) and balance coincides
// with the optimum — which is why the PI worked there.
//
// The outer loop is therefore discrete extremum seeking on the DIRECT
// performance variable (Kiefer-Wolfowitz stochastic approximation with
// dwell averaging):
//   * dither: the base fraction is perturbed by ±δ (δ = 0.02); each arm
//     is held for m = 6 samples and the arm ORDER alternates per cycle
//     so linear drift (thermals, cache state) shifts both means, not
//     their difference.
//   * noise model: relative wall noise sigma_hat is estimated online by
//     an EWMA of |w/w̄ - 1| (init 0.03, consistent with the identified
//     e-sd halved); the decision variable is the two-arm mean difference
//     whose standard error is sigma_hat*sqrt(2/m).
//   * move rule (hypothesis test): |w̄₊ - w̄₋| > 1.5·sigma_hat·sqrt(2/m)
//     → base steps δ AWAY from the slower arm; otherwise no move. With
//     m = 6 this filters 3% wall noise and passes the measured 5-8%
//     contrast (T=512). Near the optimum the gate blocks most moves,
//     bounding limit-cycle amplitude at ±δ.
//   * persistence saves the BASE (the estimated optimum), never the
//     dithered actuator value, under the quiet rule above.
struct HybridSplit {
    // Setpoint of the balance error, kept ONLY as the straggler-retreat
    // threshold's frame (the retreat branch slides it with load); the
    // regulation itself no longer targets balance — see the outer loop above.
    double setpoint = 0.0;         // quiet-machine balance reference
    double setpoint_load = 0.05;   // setpoint slide per unit of contention
    double frac_max = 0.35, frac_min = 0.04;
    double save_max_load = 0.15;   // persist only quiet-regime results
    double retreat_error = -0.15;  // CPU straggler: retreat fast
    // Ambient busy-core probe; overridable for tests and bench harnesses.
    std::function<double()> probe = tm_ambient_busy_cores;

    mutable double frac = -1.0;    // < 0: not initialised
    mutable double load = -1.0;    // last contention reading [0,1]; < 0 unknown
    mutable int idle = 0, updates = 0;
    mutable bool probed = false, seen_first = false;
    mutable double saved = -1.0;
    // --- discrete extremum-seeking outer loop state (all in frac units or
    // relative wall units; noise-modelled, see the header above) ---
    static int esc_m() {
        const char* e = getenv("TM_HYBRID_ESC_M");
        return e ? std::max(2, std::atoi(e)) : 6;   // dwell per dither arm
    }
    static double esc_delta() {
        return std::max(0.005, env_var_double("TM_HYBRID_ESC_DELTA").value_or(0.02));
    }
    mutable double esc_base = -1.0;        // < 0: not initialised (the optimum estimate)
    mutable int esc_phase = 0;             // current dither arm: 0 = +δ, 1 = -δ
    mutable double esc_sum[2] = {0.0, 0.0};// relative-wall accumulators per arm
    mutable int esc_n[2] = {0, 0};
    mutable double esc_cycle_sign = 1.0;   // arm-0 offset sign; alternates per cycle
    mutable double esc_sw = 0.03;          // sigma_hat: EWMA of relative wall noise
    mutable double esc_w = -1.0;           // wall-time EWMA (normaliser)
    mutable int esc_stable = 0;            // consecutive cycles without a move
    mutable int esc_last_dir = 0;          // last applied move: -1/+1
    mutable double esc_damp = 1.0;         // cumulative direction-flip damping (bisection)
    // --- chip-pressure self-regulation (2026-09-09) ---
    // Per-prefill process-pressure sample: page-fault and context-switch
    // deltas over the hybrid prefill (TASK_EVENTS_INFO), normalized by the
    // quiet references measured on this machine (AMX prefill quiet
    // ~110-260 csw, thrashing ~430-5000; faults 2-300). p > ~0.5 means the
    // OS is preempting our CPU rows: wall inflation is then CONTAMINATION,
    // not plant gradient — the ESC must not learn from it, and sustained
    // pressure must make the split YIELD the rows (they are the resource
    // being stolen).
    static double pressure_csw()  { return env_var_double("TM_HYBRID_P_CSW").value_or(400.0); }
    static double pressure_fault() { return env_var_double("TM_HYBRID_P_FAULT").value_or(500.0); }
    // Soft yield (2026-09-11): 0.5 < esc_p <= 0.75 blocks learning but used
    // to keep publishing the UNDEGRADED actuator — a sustained moderate-
    // pressure regime (the common case between clean and thrash) paid the
    // contamination on every prefill while waiting for the hard-park
    // counter. Graded response: shrink the CPU share toward the GPU's
    // resource share as pressure builds (the CPU rows are the stolen
    // resource); a clean sample republishes from esc_base and self-heals.
    static int soft_yield() {
        const char* e = getenv("TM_HYBRID_SOFTYIELD");
        return e ? std::atoi(e) : 0;
    }
    mutable double esc_p = 0.0;            // EWMA of the pressure sample [0, 1+]
    mutable int esc_yield = 0;             // consecutive contaminated samples
    mutable int probe_fails = 0;           // consecutive probe→park cycles
    mutable int probe_open = 0;            // set while a re-probe is being judged
    mutable double freeze_load = -1.0;     // load reading at settle-freeze entry
    // Full reset: the wall-time scale and accumulators belong to the wall
    // regime the ESC was learning; a retreat or park can land the next
    // hybrid prefill in a different regime (loaded vs quiet), so carrying
    // them over corrupts the relative-noise model.
    void esc_reset_keep_base() const {
        esc_w = -1.0;
        esc_sum[0] = esc_sum[1] = 0.0;
        esc_n[0] = esc_n[1] = 0;
        esc_phase = 0;
        esc_cycle_sign = 1.0;
        esc_sw = 0.03;
        esc_stable = 0;
        esc_last_dir = 0;
        esc_damp = 1.0;
        frac = std::clamp(esc_base, 0.0, frac_max);   // undithered while yielding
    }
    void esc_reset() const {
        esc_base = -1.0;
        esc_w = -1.0;
        esc_sum[0] = esc_sum[1] = 0.0;
        esc_n[0] = esc_n[1] = 0;
        esc_phase = 0;
        esc_cycle_sign = 1.0;
        esc_sw = 0.03;
        esc_stable = 0;
    }

    static double pinned_frac() {
        static const double v = [] {
            const char* e = std::getenv("TM_LLAMA_HYBRID");
            if (!e || !*e) return -1.0;                 // auto
            if (std::string(e) == "auto") return -1.0;
            const double f = env_var_double("TM_LLAMA_HYBRID").value_or(-1.0);
            return (f >= 0.0 && f < 0.5) ? f : 0.0;     // pinned (0 = off)
        }();
        return v;
    }
    void refresh_load(bool force) const {
        if (!force && probed && updates % 8 != 0) return;
        const double decayed = load < 0 ? 0.0 : 0.7 * load;
        if (!probe) { if (load < 0) load = 0; return; }
        const double busy = probe();
        if (busy < 0) { if (load < 0) load = 0; return; }
        load = std::clamp((busy - 0.5) / 3.5, 0.0, 1.0);
        load = std::max(load, decayed);
        probed = true;
    }
    double setpoint_now() const {
        return setpoint + setpoint_load * std::max(0.0, load);
    }
    double seed() const {
        if (frac < 0.0) frac = pinned_frac() >= 0.0 ? pinned_frac() : tmtune::hybrid_seed();
        return pinned_frac() >= 0.0 ? pinned_frac() : frac;
    }
    void update(double gpu_ms, double cpu_ms, int gpu_rows, int cpu_rows,
                double wall_ms = 0.0, double pressure = 0.0) const {
        // Pressure EWMA decays toward 0 on clean samples (alpha 0.3), so a
        // single spike neither trips nor lingers.
        esc_p += 0.3 * (pressure - esc_p);
        if (gpu_ms > 0.0 && cpu_ms > 0.0 && getenv("TM_HYBRID_ID"))   // plant identification
            fprintf(stderr, "[hybrid-id] frac %.3f gpu_ms %.1f cpu_ms %.1f e %+.4f wall %.1f\n",
                    frac, gpu_ms, cpu_ms, (gpu_ms - cpu_ms) / (gpu_ms + cpu_ms), wall_ms);
        if (pinned_frac() >= 0.0 || gpu_ms <= 0.0 || cpu_ms <= 0.0) return;
        refresh_load(false);
        ++updates;
        const double e_bal = (gpu_ms - cpu_ms) / (gpu_ms + cpu_ms) - setpoint_now();
        if (!seen_first) {
            // First hybrid prefill: jump to the rate-ratio prior (per-row
            // rates transfer across prompt lengths), pulled toward zero by
            // the measured load. The ESC outer loop starts from there, on
            // the +δ arm.
            const double rc = cpu_rows / std::max(cpu_ms, 1e-6), rg = gpu_rows / std::max(gpu_ms, 1e-6);
            if (rc > 0 && rg > 0)
                frac = std::clamp(0.8 * rc / (rc + rg) * (1.0 - 0.6 * std::max(0.0, load)),
                                  0.0, frac_max);
            seen_first = true;
            esc_base = frac;
            frac = std::clamp(esc_base + esc_delta(), 0.0, frac_max);   // open on the +δ arm
        } else {
            if (e_bal < retreat_error) {              // CPU is the straggler: retreat fast
                frac = std::max(0.0, frac * (1.0 + 0.9 * e_bal));
                esc_reset();                          // ESC state is stale after a retreat
                // Re-open the ESC on the -δ arm: the straggler signal says
                // lower is better, so the next cycle explores DOWNWARD
                // first (a +δ opening would immediately cancel the retreat —
                // regression found by the heavy-load synthetic test).
                esc_cycle_sign = -1.0;
            }
            if (frac < frac_min) {
                frac = 0.0; idle = 0; esc_reset();
                // A probe killed by retreat/gate (never reaching a pressure
                // park) must still count as a failed probe — otherwise the
                // backoff never engages for straggler-thrash episodes
                // (found by the ESC simulation, 2026-09-11).
                if (probe_open) { ++probe_fails; probe_open = 0; }
            }
        }
        if (esc_base < 0.0 && frac > 0.0) {              // re-probe / fresh start
            // Re-seed from the persisted quiet optimum when one exists: a
            // pressure-park recovery then starts AT the known good point
            // and the ESC only trims (a fresh 0.08 probe would pay the
            // whole climb again on every intermittent-thrash episode).
            esc_base = saved > 0.0 ? saved : frac;
            esc_w = -1.0;                                // new wall regime
        }
        if (frac <= 0.0 || esc_base < 0.0) return;

        // ---- outer loop: discrete extremum seeking on relative wall time ----
        const double delta = esc_delta();
        const int m = esc_m();
        if (wall_ms <= 0.0) return;      // no wall sample (legacy caller): hold
        // Pressure handling precedes the feasibility projection (moved
        // 2026-09-11): the projection returns every sample once the base is
        // pinned at the 64-row gate, which made esc_yield unreachable — a
        // thrashing short-prompt workload could NEVER park (found by the ESC
        // simulation) and paid the gate-floor contamination on every prefill.
        if (esc_p > 0.5) {
            // Contaminated sample: thrash, not plant. Learn nothing; count
            // sustained pressure and PARK after 3 of them — the CPU rows
            // are exactly the resource being preempted (measured: csw 65 ->
            // 5000/prefill, wall x2.5), so the correct self-regulation is
            // to stop taking them entirely: GPU-only rows are immune. The
            // load-scaled re-probe owns recovery when it clears, re-seeded
            // from the persisted quiet optimum.
            if (esc_p > 0.75 && ++esc_yield >= 3) {
                frac = 0.0;
                idle = 0;
                if (probe_open) { ++probe_fails; probe_open = 0; }   // probe→park: the probe failed
                esc_reset();
                esc_yield = 0;
            }
            else if (soft_yield() && frac > 0.0) {
                // Graded degradation between "clean" and "thrash": the CPU
                // share is the stolen resource, so shrink the ACTUATOR
                // (not the base) proportionally to pressure — at esc_p 0.6
                // the split still runs at ~70% of its learned share and
                // keeps a gate-viable footprint, instead of paying 100% of
                // the contamination while blocked from learning. esc_base
                // is untouched; the first clean sample republishes
                // esc_base + shift and the split self-heals with no probe.
                // (Probe-failure judgement: a soft-yielded probe that never
                // reaches a clean sample also counts as failed, but only
                // via the park counter — the shrink alone is recoverable
                // in-place and must not inflate probe_fails.)
                frac *= (1.0 - 0.5 * esc_p);
            }
            return;
        }
        esc_yield = 0;
        if (probe_open) { probe_fails = 0; probe_open = 0; }   // probe succeeded: split is live

        if (gpu_rows + cpu_rows > 0 && esc_base < 64.0 / (gpu_rows + cpu_rows)) {
            esc_base = std::min(64.0 / (gpu_rows + cpu_rows) + delta, frac_max);
            esc_reset_keep_base();
            return;
        }
        if (wall_ms <= 0.0) return;      // no wall sample (legacy caller): hold
        if (esc_w < 0.0) esc_w = wall_ms;
        esc_w += 0.25 * (wall_ms - esc_w);
        const double rel = wall_ms / esc_w;
        esc_sw += 0.25 * (std::fabs(rel - 1.0) - esc_sw);   // sigma_hat, E|noise| scale
        esc_sum[esc_phase] += rel;
        ++esc_n[esc_phase];
        const int other = 1 - esc_phase;
        if (esc_n[esc_phase] >= m && esc_n[other] < m) {
            esc_phase = other;               // mid-cycle arm switch
        } else if (esc_n[0] >= m && esc_n[1] >= m) {
            // Cycle complete: the two-mean test decides.
            const double d = esc_sum[0] / esc_n[0] - esc_sum[1] / esc_n[1];
            // Arm 0 carries +cycle_sign·δ. The Kiefer-Wolfowitz gradient
            // estimate of wall vs frac is g = cycle_sign * d / (2δ); the
            // base descends the estimated gradient by one dither step,
            // gated by the noise model (two-mean standard error).
            const double g = esc_cycle_sign * d;
            const double tau = 1.5 * esc_sw * std::sqrt(2.0 / m);
            double moved = 0.0;
            if (std::fabs(d) > tau) {
                // Fixed δ step with direction-flip damping. When the move
                // direction FLIPS relative to the last move, the base just
                // bracketed the optimum: halve the step (discrete bisection
                // on the gradient sign — geometric convergence, and the
                // limit cycle the dither otherwise sustains at a kinked
                // optimum decays instead of persisting). An adaptive
                // |d|/tau-scaled step was tried first and rejected: in a
                // low-noise regime sigma_hat decays until tau under-reads
                // the dither's own induced contrast, the ratio saturates the
                // cap, and every move is 3δ — a sustained ±3δ limit cycle.
                // Noise can flip a sign spuriously; the cost is one
                // shrunken step, not a wrong direction.
                const int dir = g < 0 ? 1 : -1;
                if (esc_last_dir != 0 && dir != esc_last_dir) esc_damp *= 0.5;
                else if (esc_last_dir != 0) esc_damp = std::min(1.0, esc_damp * 2.0);
                const double esc_step = delta * esc_damp;
                esc_base = std::clamp(esc_base + dir * esc_step, frac_min, frac_max);
                esc_last_dir = dir;
                moved = esc_step;
                }
            // Settled means "no further exploration is warranted": the gate
            // blocked the move (contrast within the noise model) or the move
            // was damped under 3/4 δ (the direction-flip bisection has
            // bracketed the optimum). A full-δ-or-more move reopens it.
            esc_stable = (moved > 0.75 * delta) ? 0 : esc_stable + 1;
            if (getenv("TM_DEBUG_POOL"))
                fprintf(stderr, "[llama] hybrid esc: d %+.4f tau %.4f sigma %.3f -> base %.3f\n",
                        d, tau, esc_sw, esc_base);
            esc_sum[0] = esc_sum[1] = 0.0;
            esc_n[0] = esc_n[1] = 0;
            esc_cycle_sign = -esc_cycle_sign;    // alternate arm order: drift cancels
            esc_phase = 0;
        }
        // Publish the actuator for the next prefill: current cycle's arm.
        // In the cycle's second arm the offset flips sign.
        // Arm bookkeeping: the accumulator still filling is the arm being
        // measured; when both are empty (fresh cycle) we open on arm 0.
        const double t_rows = (double)(gpu_rows + cpu_rows);
        double offset = (esc_n[0] < m) ? esc_cycle_sign : -esc_cycle_sign;
        double shift = offset * delta;               // the actuator shift, in frac units
        // Settle-freeze (2026-09-11, default OFF — TM_HYBRID_SETTLEFREEZE=1):
        // a settled ESC keeps dithering ±δ around the optimum forever —
        // bounded, but a real cost on every prefill. After 6 quiet cycles
        // publish the undithered base; reopen on any of the events that reset
        // esc_stable (retreat, park, feasibility projection) or on a material
        // ambient-load change (>0.15 since entry), which reopens exploration.
        if (esc_stable < 6) freeze_load = load;
        const int settlefreeze = getenv("TM_HYBRID_SETTLEFREEZE")
                                     ? std::atoi(getenv("TM_HYBRID_SETTLEFREEZE")) : 0;
        const bool load_drift = freeze_load < 0.0 ||
                                std::fabs(std::max(load, 0.0) - freeze_load) > 0.15;
        if (settlefreeze && esc_stable >= 6 && !load_drift) shift = 0.0;
        // The actuator must stay RUNNABLE: a published frac whose row
            // count falls under the 64-row gate measures nothing and parks
            // the split (frozen-controller bug 2026-09-09). Enforce the
            // viability band [64/T, T/2] on the published value; near a band
            // edge an arm degenerates to the base — safe, just blind on that
        // side.
        double act = esc_base + shift;
        if (t_rows > 0) {
            act = std::max(act, 64.0 / t_rows + 1e-4);   // smallest frac the gate runs
            act = std::min(act, 0.5 - 1e-4);             // GPU keeps at least half
        }
        frac = std::clamp(act, 0.0, frac_max);
        // Persist the BASE (the estimated optimum), never a dithered value.
        if (esc_stable >= 3 && std::fabs(esc_base - saved) > 0.02 &&
            load >= 0.0 && load < save_max_load) {
            tmtune::state_put("hybrid_frac", esc_base);
            saved = esc_base;
            if (getenv("TM_DEBUG_POOL"))
                fprintf(stderr, "[llama] hybrid: esc settled at %.3f (load %.2f), saved\n",
                        esc_base, load);
        }
        if (getenv("TM_DEBUG_POOL"))
            fprintf(stderr, "[llama] hybrid: gpu %.0f ms cpu %.0f ms load %.2f wall %.0f -> frac %.5f (base %.5f)\n",
                    gpu_ms, cpu_ms, std::max(load, 0.0), wall_ms, frac, esc_base);
    }
    int cpu_rows(int T) const {
        const double pinned = pinned_frac();
        const double f = seed();
        // Parked re-probe: the interval scales with contention (8 prefills
        // quiet, up to 64 loaded) — probing the split on a busy machine
        // costs the probe prefill itself (measured 351 vs 747 t/s at 4
        // burner cores), so a loaded host must not pay it often.
        // Probe-failure backoff (2026-09-11): a probe that immediately
        // re-parks (pressure never cleared) evidences sustained contention,
        // not load level — the next attempt should be exponentially more
        // reluctant (x2, x4, capped at 4x). A clean probe resets the counter.
        // Intermittent-thrash machines otherwise pay the probe cost at the
        // full load-scaled rate on every episode.
        const int pback = getenv("TM_HYBRID_PROBE_BACKOFF")
                              ? std::atoi(getenv("TM_HYBRID_PROBE_BACKOFF")) : 0;
        const int fail_mult = pback ? (1 << std::min(probe_fails, 2)) : 1;
        if (f <= 0.0 && pinned < 0.0 &&
            ++idle >= (8 + (int)(56.0 * std::max(0.0, load))) * fail_mult) {
            // The probe prefill is about to pay its cost anyway; refresh the
            // ambient reading with it, or a machine that went quiet while
            // parked would wait for updates%8 to align (updates do not
            // advance while parked) before the setpoint could relax.
            refresh_load(true);
            probe_open = 1;   // judge the probe by the next sample's pressure
            // The probe must be gate-viable for THIS prompt length: a probe
            // below the 64-row gate parks again immediately and the split
            // never recovers (T=512 needs >= 0.125).
            frac = std::max(0.08, 80.0 / T);
            idle = 0;
        }
        const double cur = pinned >= 0.0 ? pinned : frac;
        // Below ~512 rows the GPU alone is faster: the CPU's share of a short
        // prompt costs more than the GPU time it saves (T=256 measured 563
        // t/s hybrid vs 835 GPU-only; T=512 861 vs 790).
        if (cur <= 0.0 || T < 512) return 0;
        int rows = (int)(T * cur) / 8 * 8;
        if (rows < 64) {
            // Gated off while claiming a nonzero fraction: park properly so
            // the re-probe machinery owns recovery (otherwise the split
            // freezes with a live frac that can never run — frozen-controller
            // bug 2026-09-09).
            if (pinned < 0.0) {
                    frac = 0.0; idle = 0;
                    if (probe_open) { ++probe_fails; probe_open = 0; }   // probe died at the gate
                }
            return 0;
        }
        if (rows > T / 2) rows = T / 2 / 8 * 8;
        return rows;
    }
};

// --- per-token GEMV dispatch switches ---------------------------------------
//
// These are read inside per-token, per-projection code: a decode token
// dispatches ~89 projections, and getenv() walks the whole environment array on
// every call. They are resolved ONCE per process, which is the lifetime the A/B
// workflows they exist for already assume (one arm per binary run, or one arm
// per ctest invocation via ENVIRONMENT) and what the `static const` bindings
// beside them already did. No new knobs: this consolidates the existing reads.
//
// grouped_gemv_enabled() stays where it is rather than moving in here: it
// already resolves once through its own `static const`, and test_gemv_group
// --config pins exactly that (it flips TM_GEMV_GROUP in-process and asserts the
// already-resolved value is RETAINED, i.e. cached per process).
struct GemvSwitches {
    bool sdot = true;          // TM_SDOT: int8-quantized activations + sdot kernels
    bool serial = false;       // TM_SERIAL_GEMV: run the kernel inline, no pool
    bool pool_deq = false;     // TM_POOL_DEQ: row-parallel K-quant dequant on the pool
    bool gemm_f32 = true;      // TM_GEMM_F32=0: per-token gemv loop for fp32 weights
    bool debug_roww = false;   // TM_DEBUG_ROWW: present at all (even as "0")
    int kq_chunk = 16;         // TM_KQ_CHUNK: 8 forces the narrow multi-position width
    int kq_amx_min = 64;       // TM_KQ_AMX_MIN: rows below this skip AMX dequant
};

inline const GemvSwitches& gemv_switches() {
    static const GemvSwitches v = [] {
        const auto is_on = [](const char* name) {
            const char* e = std::getenv(name);
            return e && e[0] == '1';
        };
        const auto not_off = [](const char* name) {   // unset, or anything but "0"
            const char* e = std::getenv(name);
            return !e || e[0] != '0';
        };
        GemvSwitches s;
        s.sdot = not_off("TM_SDOT");
        s.serial = is_on("TM_SERIAL_GEMV");
        s.pool_deq = is_on("TM_POOL_DEQ");
        s.gemm_f32 = not_off("TM_GEMM_F32");
        s.debug_roww = std::getenv("TM_DEBUG_ROWW") != nullptr;
        // Strict parse (unlike the atoi this replaces, which read "8junk" as 8):
        // anything that is not exactly 8 keeps the 16-wide dispatch.
        s.kq_chunk = env_var_long("TM_KQ_CHUNK").value_or(16) == 8 ? 8 : 16;
        s.kq_amx_min = static_cast<int>(env_var_long("TM_KQ_AMX_MIN").value_or(64));
        return s;
    }();
    return v;
}

// A GPU failure the host cannot recover from: say what failed and abort. The
// GPU KV cache is authoritative in the modes that reach here, and command
// buffers may be in flight, so there is no unwinding to do and no CPU fallback
// that would produce a correct answer. Four sites used to spell this out as
// `fprintf(stderr, …); std::abort();`; the message is now the only thing that
// varies, and [[noreturn]] lets the compiler see that a guarded `begin` cannot
// fall through.
[[noreturn]] inline void gpu_fatal(const char* what) {
    std::fprintf(stderr, "[llama] %s\n", what);
    std::fflush(stderr);
    std::abort();
}

struct Llama {
    Llama() = default;
    Llama(const Llama&) = delete;
    Llama& operator=(const Llama&) = delete;
    int L = 22, D = 2048, H = 32, KVH = 4, dh = 64, V = 32000, ctx = 2048;
    int max_ctx_ = 2048;   // model max_position_embeddings; ctx auto-sized up to this
    int F = 5632;   // MLP intermediate (TinyLlama default; config.json overrides)
    float theta = 10000.f, eps = 1e-5f;
    // qwen35 (qwen3_5_text) hybrid linear-attention fields. All default to the
    // dense-Llama values so llama/qwen2 configs parse byte-identical. NOTE: for
    // qwen35 `head_dim` != D/H (e.g. 0.8B: D/H = 128 but head_dim = 256), so the
    // attention kernels must switch from `dh = D/H` to these explicit dims.
    std::vector<int> layer_types_;       // per layer: 0 = full-attn, 1 = linear
    int full_attention_interval_ = 0;    // (i+1) % interval == 0 -> full-attn
    int head_dim_ = 0;                   // full-attn head dim (config head_dim)
    // Whether full-attention layers carry per-head QK RMS norms (Qwen3/qwen35)
    // or not (dense llama/qwen2). Detected from the checkpoint in
    // validate_weights, NOT from head_dim_: ordinary llama configs declare
    // `head_dim` too (TinyLlama declares 64), so keying off it demanded q_norm
    // weights from models that have none and made them unloadable.
    bool qk_norm_ = false;
    int lk_heads_ = 0, lv_heads_ = 0;    // linear_num_key/value_heads
    int lk_dim_ = 0, lv_dim_ = 0;        // linear_key/value_head_dim
    int conv_kernel_ = 0;                // linear_conv_kernel_dim
    int ssm_state_ = 0, ssm_rank_ = 0;   // state_size, time_step_rank
    int ssm_groups_ = 0;                 // num_groups (delta groups)
    float ts_min_ = 0.f, ts_max_ = 0.f, ts_floor_ = 0.f;  // time_step_*
    float partial_rotary_ = 1.f;         // partial_rotary_factor (1 = full RoPE)
    int n_rot_ = 0;                      // rotated dims per head (partial RoPE);
                                         // 0 -> dh (full RoPE)
    int mtp_layers_ = 0;                 // mtp_num_hidden_layers
        int mtp_layer_ = -1;                 // layer slot the MTP head is aliased to
        std::vector<float> mtp_h_, mtp_e_, mtp_c_, mtp_x_;   // MTP step scratch
        // Speculative decoding hook: overrides the drafter. -1 = "draft nothing
        // this round" (the loop then emits one token and re-drafts). Production
        // runs leave it null and draft with the MTP head; the test uses it to
        // exercise the accept path deterministically (see spec_generate_greedy).
        using SpecDraftFn = int (*)(void*, const std::vector<int>&,
                                    const std::vector<float>&);
        SpecDraftFn spec_draft_ = nullptr;
        void* spec_draft_ctx_ = nullptr;
        // Trunk hidden state of the last position a forward() produced, and the
        // last-position normalised row it fed the LM head. The MTP head consumes
        // h; keeping it costs a D-float copy per forward.
        std::vector<float> hlast_, xfa_;
        // Pooled-embedding collection (embed_begin / embed_end): the running sum
        // of the FINAL hidden state over every position the pass covered, and how
        // many positions that is. Off by default; costs nothing when off.
        std::vector<float> emb_sum_;
        std::size_t emb_count_ = 0;
        bool emb_collect_ = false;
        // Snapshots of the Gated-DeltaNet recurrent state, taken around a
        // speculative verification pass so a rejected draft can be undone.
        std::vector<std::vector<float>> ssm_snap_, conv_snap_;
    bool attn_output_gate_ = false;
    // The pack ties lm_head to the embedding table (no `lm_head.weight`
    // tensor); alias_tied_lm_head() binds the two names to one block.
    bool tie_word_embeddings_ = false;
    bool mrope_interleaved_ = false;
    // qwen35 linear-attention (Gated DeltaNet) per-layer recurrent state. Both
    // are O(1) in context length and replace the KV cache for those layers:
    //   ssm_st_[l]  [lv_heads_ * lk_dim_ * lv_dim_]  delta-rule state S
    //   conv_st_[l] [(conv_kernel_-1) * conv_channels]  causal conv1d window
    // Allocated for every layer index but only touched by linear layers.
    std::vector<std::vector<float>> ssm_st_, conv_st_;
    int linear_layers_ = 0;              // count of Gated-DeltaNet layers (cache sizing)
    std::map<std::string, Tensor> w;
    // M5 L2: optional quantized weights (load_quant). 2-D tensors stay
    // mmap'd blocks keyed by their safetensors name; 1-D (norms) are
    // dequantized into w at attach. Llama nn.Linear is (out, in) —
    // exactly M4's q8_gemv layout (dot along the in axis).
    std::map<std::string, const tmq::BlockQ8_0*> q8_;
    std::map<std::string, const tmq::BlockQ4_0*> q4_;
    // dtype 4/5/6: K-quant blocks (Q4_1 / Q5_K / Q6_K) kept at their GGUF
    // source byte width. The dtype travels with the pointer because the three
    // share one registry; kqgemv_rows() dispatches on it.
    struct KQuant { const void* blocks; std::uint32_t dtype; };
    std::map<std::string, KQuant> qk_;
    // dtype 3: 2 bytes per element, row-major. Holds tensors whose gguf
    // source was a K-quant / F16 / BF16 — stored at f16 precision instead
    // of being requantized to Q8_0 (see convert_gguf.cpp).
    std::map<std::string, const std::uint16_t*> f16_;
    std::map<std::string, std::pair<int, int>> qshape_;   // (out, in)
    std::unique_ptr<tmmq::Mapped> held_map_;
    std::unique_ptr<tmmq::Demand> held_demand_;   // TM_DEMAND=1 per-segment reader
    std::vector<PageVec> kc, vc;   // per KV head, (ctx, dh); page-aligned for zero-copy GPU views
    // Packed Q8_0 KV rows (TM_KV_KTYPE / TM_KV_VTYPE; spec
    // specs/TENSORMARK_KV_Q8_SPEC.md). Row layout matches the fp32 one: per KV
    // head, `ctx` rows of dh/32 blocks (34 B each) — a Q8_0 KV row is
    // byte-layout-identical to a Q8_0 weight row, so the unpack discipline is
    // the one kq*_kern already uses. This is the ADDITIVE step: the fp32 cache
    // above stays authoritative and is still the only thing read, while every
    // append also writes the packed copy and (TM_KV_SELFCHECK=1) proves it
    // round-trips. The read cutover is a separate, gated commit.
    std::vector<std::vector<std::uint8_t>> kq_, vq_;
    std::vector<float> kv_chk_;    // self-check scratch (one row)
    static int kv_type_env(const char* name) {
        const char* e = std::getenv(name);
        if (!e) e = std::getenv("TM_KV_Q8");   // shorthand: sets both K and V
        if (!e) return 0;                      // 0 = fp32 (default), 1 = q8_0
        if (std::strcmp(e, "q8_0") == 0 || std::strcmp(e, "q8") == 0) return 1;
        return e[0] == '1' ? 1 : 0;
    }
    int kv_kt_ = kv_type_env("TM_KV_KTYPE");
    int kv_vt_ = kv_type_env("TM_KV_VTYPE");
    bool kv_selfcheck_ =
        [](const char* e) { return e && e[0] == '1'; }(std::getenv("TM_KV_SELFCHECK"));
    // Effective types: a request is only usable when dh tiles into complete Q8_0
    // blocks. Derived from dh rather than stored, on purpose — the flags and the
    // arrays they describe are swapped between the staging and live objects, and
    // a flag that can disagree with its allocation writes past the end of an
    // empty vector (that is not hypothetical: both failure modes happened).
    int kv_ktype() const { return (kv_kt_ && dh % tmq::kBlock == 0) ? 1 : 0; }
    int kv_vtype() const { return (kv_vt_ && dh % tmq::kBlock == 0) ? 1 : 0; }
    std::vector<float> h_, q_, k_, v_, att_, ao_, h2_, g_, u_, dn_, xf_, logits_, xbuf_;
    std::vector<float> qkvv_, gu_;
    // qwen35 Gated-DeltaNet scratch (see Llama::block_linear) plus the
    // full-attn interleaved q|gate projection and its gate half.
    std::vector<float> lqkv_, lq_, lk_, lv_, lz_, lo_, lb_, la_, ldec_,
        lconv_, lker_, lkflat_, qfull_, agate_;
    tmgdn::StepScratch gdn_scratch_;   // reused by every delta-rule step()
    int seen = 0;
    // GPU batched prefill copies fused weight blobs into GPU memory.
    // When decode_gpu_ is on, the decode path registers the SAME weights
    // zero-copy from the .tmq mmap — keeping both alive doubles the 7B's
    // GPU footprint past the 8 GB host budget (watchdog panic 2026-09-05).
    // So with decode_gpu_ the prefill GPU path is FORCED off (CPU/AMX
    // prefill), regardless of TM_PREFILL_GPU; explicit TM_PREFILL_GPU=1
    // with TM_DECODE_GPU=1 fails fast instead of silently blowing up.
    // TM_PREFILL_GPU: "1" forces the Metal prefill, "0" forces CPU, unset or
    // "auto" (Metal builds) resolves at load with the decode policy below:
    // GPU when the host is busy, AMX otherwise. Measured 2026-09-06, TinyLlama
    // Q4 512@512 (build/prefill_amx_vs_gpu.json): quiet AMX 320.9 vs GPU
    // 246.3 t/s (0/4 for GPU); four burner cores AMX 111.7 vs GPU 204.8
    // (4/4). AMX keeps 35% under load, the GPU 83%.
    bool gpu_prefill_auto_ =
        [](const char* e) { return !e || !e[0] || std::string(e) == "auto"; }(
            getenv("TM_PREFILL_GPU"));
    bool gpu_prefill_ =
        [](const char* e) { return e && e[0] == '1'; }(
            getenv("TM_PREFILL_GPU"));
    // Set when the auto policy chose BOTH GPU prefill and GPU decode: only
    // for models under the auto weight cap, where the copied prefill blobs
    // plus the NoCopy decode mapping fit comfortably (TinyLlama: ~1.5 GB).
    bool gpu_both_auto_ = false;
    // Context rule, kept as a safety net only: the auto policy now starts
    // decode on the GPU (see resolve_decode_gpu_auto), so this fires only
    // when something forced the CPU path and the context has grown past
    // TM_DECODE_GPU_AUTO_CTX rows, where the GPU's margin is largest.
    bool decode_gpu_auto_fit_ = false;
    bool decode_gpu_ctx_switched_ = false;
    // Experimental Q4 GPU decode: all per-token ops in one command buffer.
    // Prefill stays CPU/AMX to avoid the legacy Metal copied-weight cache.
    // Single-model session; default OFF until large-model safety/perf gates.
    // TM_DECODE_GPU: "1" forces GPU decode, "0" forces CPU, unset or "auto"
    // (Metal builds) resolves at load: GPU when the host is already busy
    // (ambient >= TM_DECODE_GPU_AUTO_BUSY cores, default 1.0) and the Q4
    // weights fit TM_DECODE_GPU_AUTO_MAX_GB (default 2.0 — 7B stays on the
    // CPU until its guarded GPU gate closes). Measured 2026-09-06 on
    // TinyLlama Q4 under controlled burner load (build/gpu_load_ab*.json):
    // quiet CPU 66.7 vs GPU 58.2 t/s (0.87x, 0/4), one burner core 51.6
    // vs 59.3 (1.18x, 4/4), four cores 24.5 vs 53.4 (2.18x, 4/4). The CPU
    // path loses 63% of its speed under load, the GPU path 8%.
    bool decode_gpu_auto_ =
        [](const char* e) { return !e || !e[0] || std::string(e) == "auto"; }(
            getenv("TM_DECODE_GPU"));
    bool decode_gpu_ =
        [](const char* e) { return e && e[0] == '1'; }(
            getenv("TM_DECODE_GPU"));
    // TM_DEMAND=1: CPU-only per-segment demand loader (docs/ON_DEMAND_WEIGHTS.md).
    // GPU decode/prefill register zero-copy Metal views over the single whole-file
    // mapping, which per-segment munmap eviction would invalidate, so demand mode
    // forces the CPU path regardless of the auto policy or an explicit TM_DECODE_GPU.
    bool demand_mode_ =
        [](const char* e) { return e && e[0] == '1'; }(
            getenv("TM_DEMAND"));
    // Eviction lag (layers): demand_layer(l) drops layer (l - W) behind the
    // scan head. The resident set peaks near 2W layers around the token
    // wrap; W = 6 keeps ~12 layers (~580 MB Q4) resident for a 3B model.
    int demand_window_ = static_cast<int>(env_var_long("TM_DEMAND_WIN").value_or(6));
    // Prefetch lookahead (layers): demand_layer(l) F_RDADVISE-reads layers
    // l+1..l+P ahead into the page cache so their later fault hits cache
    // instead of SSD (pipelining). Measured on Qwen 3B (8 GB, model fits in
    // page cache): P=10 is a NET LOSS vs P=0 — 12.47 vs 11.30 s for 60 tok,
    // +4.1 s sys from the read-advise syscalls, no latency win. Default 0
    // (off); kept as a knob for truly oversize models (cold reads) where it
    // might pay, but unproven there.
    int demand_lookahead_ = static_cast<int>(env_var_long("TM_DEMAND_LOOKAHEAD").value_or(0));
    // The GPU memory budget both GPU registration paths fail fast against:
    // TM_GPU_BUDGET_GB when set, else 0.625 of physical RAM. One definition so
    // the fp16 and Q4 lanes cannot drift (they were two copies of the same
    // expression, both fed the variable's VALUE as an accessor argument — see
    // the accessor notes above).
    static double gpu_budget_gb() {
        return env_var_double_checked("TM_GPU_BUDGET_GB", 0.625 * physical_ram_gb());
    }
    size_t q4_weight_bytes() const {
        size_t bytes = 0;
        for (const auto& [n, blk] : q4_) {
            auto it = qshape_.find(n);
            if (it != qshape_.end())
                bytes += (size_t)it->second.first * it->second.second / 32 * 18;
        }
        return bytes;
    }
    // Resolve the auto policy once the weights are known (called from
    // commit_loaded, noexcept: the probe sleeps 150 ms and cannot throw).
    void resolve_decode_gpu_auto() noexcept {
        if (demand_mode_) {
            decode_gpu_ = false; gpu_prefill_ = false; gpu_both_auto_ = false;
            return;
        }
        // A PACKED (Q8_0) KV cache is CPU-only, for the same structural reason
        // the hybrid lane is: the Metal kernels wrap the fp32 cache as zero-copy
        // views and write fp32 rows, so a packed cache has nothing for them to
        // address. Refusing HERE is the single decision point that turns the
        // token prologue and the whole-model stack off together.
        if (kv_ktype() || kv_vtype()) {
            decode_gpu_ = false; gpu_prefill_ = false; gpu_both_auto_ = false;
            gpu_prefill_short_ok_ = false; gpu_ok_ = false;
            return;
        }
        // qwen35 hybrid models have no Metal implementation: `gpu_stack_setup`
        // bails on the first linear layer (no self_attn q/k/v) and the token-decode
        // path reserves a KV slot per layer, which linear layers do not have.
        // The Gated-DeltaNet GPU decode lane that once served them was RETIRED
        // (2026-09-17): it was wrong on K-quant checkpoints and 0.6-0.9x the CPU
        // even when correct, and every shipped qwen35 checkpoint is K-quant. The
        // recurrent state is CPU-only; the v1 GPU prefill stack has no GDN path.
        if (is_hybrid()) {
            decode_gpu_ = false; gpu_prefill_ = false; gpu_both_auto_ = false;
            gpu_prefill_short_ok_ = false; gpu_ok_ = false;
            return;
        }
        if (!decode_gpu_auto_ && !gpu_prefill_auto_) return;
        if (decode_gpu_auto_) decode_gpu_ = false;
        if (gpu_prefill_auto_) gpu_prefill_ = false;
        gpu_both_auto_ = false;
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        try {
            const bool usable = gpu_decode_usable();
            const double busy = tm_ambient_busy_cores();
            const double min_busy = env_var_double_checked("TM_DECODE_GPU_AUTO_BUSY", 1.0);
            // Auto device bounds scale with physical RAM (see physical_ram_gb):
            // on the 8 GB host these land at 2.0 / 4.0 GiB (7B prefill on GPU,
            // decode on CPU); on a larger chip the same fractions let bigger
            // models take the GPU automatically. Explicit env vars still win.
            const double max_gb =
                env_var_double_checked("TM_DECODE_GPU_AUTO_MAX_GB", 0.25 * physical_ram_gb());
            const double prefill_max_gb =
                env_var_double_checked("TM_PREFILL_GPU_AUTO_MAX_GB", 0.5 * physical_ram_gb());
            const double gb = (double)q4_weight_bytes() / (1u << 30);
            const bool want = usable && busy >= min_busy && gb <= prefill_max_gb;
            decode_gpu_auto_fit_ = usable && gb <= max_gb;
            // Decode device (2026-09-07, re-measured after the GPU decode work
            // — half K/V cache, 2-row GEMV, split-K attention, resident greedy
            // chain — which together made the GPU path ~45% faster than when
            // the busy-core and context gates were written):
            //   quiet  ctx 64   CPU 80.4  GPU 85.6 t/s
            //   quiet  ctx 2000 CPU 60.6  GPU 74.7
            //   4 burn ctx 64   CPU 32.9  GPU 72.2
            //   4 burn ctx 2000 CPU 24.5  GPU 62.5
            // The GPU wins every condition measured, so the ambient-load gate
            // is gone and only the memory bound remains (a 7B model still
            // decodes on the CPU until its GPU footprint gate closes).
            // TM_DECODE_GPU=0 forces the CPU path.
            const bool decode_want = decode_gpu_auto_fit_;
            if (gpu_prefill_auto_) gpu_prefill_ = want;
            gpu_prefill_short_ok_ = gpu_prefill_auto_ && usable && gb <= prefill_max_gb && gpu_prefill_short_t() > 0;
            // Explicit TM_PREFILL_GPU=1 keeps the legacy exclusion: it wins
            // and decode stays on the CPU.
            if (decode_gpu_auto_) decode_gpu_ = decode_want && !(!gpu_prefill_auto_ && gpu_prefill_);
            gpu_both_auto_ = gpu_prefill_ && decode_gpu_ && gpu_prefill_auto_ && decode_gpu_auto_;
            if (getenv("TM_DEBUG_POOL"))
                fprintf(stderr, "[llama] gpu auto: ambient %.2f cores (>= %.2f), q4 %.2f GiB (prefill<=%.2f, decode<=%.2f) -> prefill %s, decode %s\n",
                        busy, min_busy, gb, prefill_max_gb, max_gb, gpu_prefill_ ? "GPU" : "CPU", decode_gpu_ ? "GPU" : "CPU");
            (void)want;
        } catch (...) {
            if (decode_gpu_auto_) decode_gpu_ = false;
            if (gpu_prefill_auto_) gpu_prefill_ = false;
            gpu_both_auto_ = false;
        }
#endif
    }
    // One command buffer per LAYER instead of per token (L commit/wait
    // round trips per token — 22 on TinyLlama, 32 on the 32-layer models).
    // Default OFF: per-token single buffer measured faster (spec M6 L3);
    // kept as an A/B knob for larger models where per-layer pipelining
    // can overlap CPU encode with GPU execution.
    bool decode_layer_cb_ =
        [](const char* e) { return e && e[0] == '1'; }(
            getenv("TM_DECODE_LAYER_CB"));
    // AMX prefill (TM_PREFILL_AMX=1): T>1 projections run cblas_sgemm over
    // weights dequantized on the fly into a reused fp32 scratch. The decode
    // qgemv path does NOT ride AMX; this gives quantized prefill an
    // AMX-class CPU GEMM without keeping ~4.4 GB of dequantized weights
    // resident (scratch is only the largest single tensor, 46 MB).
    // Default ON since 2026-09-06: 320-333 t/s at 512@512 vs 84 for the
    // qgemv loop and 246 for the Metal prefill on TinyLlama Q4 (quiet);
    // TM_PREFILL_AMX=0 restores the qgemv loop.
    bool amx_prefill_ =
        [](const char* e) { return !(e && e[0] == '0'); }(getenv("TM_PREFILL_AMX"));
    std::vector<float> deq_;    // AMX prefill fp32 weight scratch (reused)
    // Batched (sgemm) attention vs the per-head sdot/axpy loop (A/B, gates).
    bool attn_naive_ =
        [](const char* e) { return e && e[0] == '1'; }(getenv("TM_ATTN_NAIVE"));
    // Blocked prefill attention: query tokens per block (TM_ATTN_PREFILL_BLOCK,
    // default 128; 0 = per-token loop).
    int attn_prefill_block_ = static_cast<int>(
        env_var_long("TM_ATTN_PREFILL_BLOCK").value_or(128));
    std::vector<float> silu_tmp_;
    bool attn_prefill_pool_ =
        [](const char* e) { return !(e && e[0] == '0'); }(getenv("TM_ATTN_PREFILL_POOL"));
    // Decode (T == 1) attention through the batched sgemm path (default) instead
    // of the per-position sdot/axpy loop. TM_ATTN_DECODE_BATCHED=0 /
    // set_attn_decode_batched(false) restores the loop for A/B; TM_ATTN_NAIVE=1
    // still forces the loop for every T.
    bool attn_decode_batched_ =
        [](const char* e) { return !(e && e[0] == '0'); }(getenv("TM_ATTN_DECODE_BATCHED"));
    // Softmax rows through vDSP/vForce (maxv, vsadd, vvexpf, sve, vsmul) instead
    // of a scalar std::exp loop; TM_ATTN_VSOFTMAX=0 restores the scalar loop.
    bool attn_vsoftmax_ =
        [](const char* e) { return !(e && e[0] == '0'); }(getenv("TM_ATTN_VSOFTMAX"));
    // Experimental, default OFF: decode attention KV groups spread over the
    // GEMV pool (one group per slice, private score rows). Same outputs as
    // the serial path (identical greedy hash); paired A/B at tg1024/ctx2048
    // measured +0.3% median, 4/6 wins — pool wake-up per layer eats the
    // gain at TinyLlama's KVH=4. TM_ATTN_DECODE_POOL=1 enables it for A/B
    // (7B, KVH=8, unmeasured).
    bool attn_decode_pool_ =
        [](const char* e) { return e && e[0] == '1'; }(getenv("TM_ATTN_DECODE_POOL"));
    // gpu2: per-token single-commit GPU decode state. Slot ids name
    // persistent GPU buffers inside metal.h's token encoder.
    enum {
        SLOT_X = 0, SLOT_H = 1, SLOT_QKV = 2, SLOT_AO = 3, SLOT_H2 = 4,
        SLOT_H2S = 5, SLOT_GU = 6, SLOT_DN = 7, SLOT_XF = 8,
        SLOT_LOGITS = 9, SLOT_PROBS = 10, SLOT_TOK = 11, SLOT_BIAS = 12,
        SLOT_KC = 100, SLOT_VC = 200
    };
    // Measured 2026-09-07 (TinyLlama Q4 GPU decode, ctx 64, GPU time per
    // token, interleaved pairs): q|k|v and gate|up as one multi-part GEMV
    // dispatch each was neutral (13.22 vs 13.18 ms); folding the residual adds into the
    // o_proj/down GEMV epilogues cost +0.4 ms and one rope_q+rope_k+v_cache
    // kernel +0.5 ms — the GPU overlaps the separate small dispatches better
    // than the fused kernels ran. Kept one dispatch per tensor/op.
    bool gpu_kv_uploaded_ = false;
    int gpu_from_ = 0;
    // Rows [0, gpu_kv_valid_) of the GPU cache equal the CPU cache (uploaded
    // or pulled back); a later upload sends only [gpu_kv_valid_, seen).
    int gpu_kv_valid_ = 0;
    bool gpu_ok_ = false;
    bool gpu_checked_ = false;
    // The GPU GEMV is Q4-only. A Q8 lm_head therefore used to forfeit the whole
    // decode lane even when every layer weight qualified; when this is set, that
    // one projection is left to the CPU and the rest of the model still runs on
    // the GPU. (`gpu_head_cpu_` also means lm_head is in no GPU weight buffer.)
    bool gpu_head_cpu_ = false;
    int REP_ = H / KVH;   // GQA query heads per KV head (contiguous)

    // The GPU decode path needs every projection as Q4 (the gemv kernel
    // is Q4-only). Checked once; a q8 tensor anywhere means sticky CPU.
    // A K-quant tensor this lane has a kernel for (Q4_1 / Q5_K / Q6_K). The
    // admission gate and the registration loop must agree on exactly this set: a
    // dtype one of them accepts and the other cannot deliver is the shape of the
    // old Q8 bug (the gate passed, registration threw).
    bool kq_registerable(const std::string& n) const {
        const auto it = qk_.find(n);
        return it != qk_.end() &&
               (it->second.dtype == 4u || it->second.dtype == 5u || it->second.dtype == 6u);
    }
    // Register one K-quant tensor into the token weight lane, zero-copy over the
    // .tmq mmap, and record its dtype for the dispatch. Used by the plain decode
    // lane: to the metal layer these are the same wbuf contract,
    // only the block layout (and so the byte length) differs. Returns false when the
    // tensor is absent or its dtype has no kernel, so callers can fall through to
    // f16/Q4_0 registration.
    bool reg_kq(const std::string& n) {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        const auto kq = qk_.find(n);
        if (kq == qk_.end()) return false;
        const unsigned d = kq->second.dtype;
        if (d != 4u && d != 5u && d != 6u) return false;
        const auto qs = qshape_.at(n);
        const std::size_t elems = (std::size_t)qs.first * qs.second;
        const std::size_t bytes =
            elems / tmmq::block_elems(d) * tmmq::block_bytes(d);
        const bool file_backed = held_map_ && held_map_->owned.empty() &&
                                 held_map_->base != nullptr;
        uint64_t w = 0;
        const int id = tm_metal_tok_wbuf(kq->second.blocks, bytes,
                                        held_map_->base, held_map_->size,
                                        file_backed ? 1 : 0, w);
        if (id < 0)
            throw std::runtime_error("metal: K-quant registration failed: " + n);
        gpu_k_[n] = {id, w, d};
        return true;
#else
        // CPU-only builds include this header without the Metal shim, so the one
        // entry point that touches the token lane has to be compiled out — the
        // function used to be a lambda inside a guarded body, which is why moving
        // it here initially broke test_gemv_pool's build.
        (void)n;
        return false;
#endif
    }
    bool gpu_decode_usable() {
        if (gpu_checked_) return gpu_ok_;
        gpu_checked_ = true;
        // K-quant weights in this lane are UNVERIFIED, so they are opt-in. The kernels
        // pass a per-row gate (test_kquant_gemv.cpp), but the only end-to-end K-quant
        // evidence in this repo came from the now-RETIRED Gated-DeltaNet lane, and that
        // lane was wrong AND slower on both checkpoints tried: qwen35_0.8b_q40 (a two-token attractor) and
        // qwen38_4b_q6k (99.8% Q6_K by bytes, ~2.5x the per-step cost, greedy
        // trajectory diverging from the CPU's in the same window). Refusing costs a
        // bandwidth-bound model little: qwen38_4b_q6k decodes at 10.35 t/s on the CPU,
        // i.e. ~40 GB/s of a ~61 GB/s machine ceiling. So the default stays fail-closed
        // and the path is reachable only for measurement, until a numerics gate covers
        // it on a checkpoint whose projection weights are K-quant.
        const bool kq_lane = [] {
            const char* e = getenv("TM_DECODE_KQ");
            return e && e[0] == '1';
        }();
        const bool head_q4 = q4_.count("lm_head.weight") != 0;
        const bool head_kq = kq_lane && kq_registerable("lm_head.weight");
        gpu_head_cpu_ = !head_q4 && !head_kq && q8_.count("lm_head.weight") != 0;
        gpu_ok_ = head_q4 || head_kq || gpu_head_cpu_;
        for (int l = 0; l < L && gpu_ok_; ++l) {
            const std::string p =
                "model.layers." + std::to_string(l) + ".";
            // The admissible set must be exactly what ensure_gpu_weights() can
            // register for this lane: a Q4_0 tensor, or — under TM_DECODE_KQ — a
            // K-quant tensor with a Metal kernel. A dtype that passes here and cannot
            // be registered is the old Q8 bug: refused cleanly is fine, half-registered
            // is not.
            for (const char* t :
                 {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight",
                  "mlp.down_proj.weight"})
                gpu_ok_ = gpu_ok_ &&
                          (q4_.count(p + t) != 0 ||
                           (kq_lane && kq_registerable(p + t)));
        }
        return gpu_ok_;
    }
    // weight handle table: name -> (tok_wbuf id, byte offset in buffer)
    std::map<std::string, std::pair<int, uint64_t>> gpu_w_;
    bool gpu_w_ready_ = false;
    // K-quant weights (Q4_1 / Q5_K / Q6_K) registered into the same token lane. The
    // dtype travels with the handle because one dispatch entry point serves all
    // three; woff is a BYTE offset into the shared weight buffer, exactly as gpu_w_.
    struct GpuKq { int id; uint64_t woff; unsigned dtype; };
    std::map<std::string, GpuKq> gpu_k_;
    // Q4 weights registered zero-copy over one shared .tmq mmap buffer (true)
    // vs per-tensor copy buffers (TM_RESIDENT=1). The fused q|k|v / gate|up
    // decode dispatch needs one shared buffer, so it is gated on this.
    bool gpu_shared_wbuf_ = false;
    mutable bool gpu_weights_used_ = false; // reload cannot invalidate Metal handles
    // fp16 decode path (TM_DECODE_F16=1): half-precision weights loaded
    // from <dir>/model.f16.safetensors, registered as their own tok_wbuf
    // buffers; gpu_h_ mirrors gpu_w_'s contract. The .tmq stays the
    // CPU/fallback source of truth — Q4 GPU registration is skipped so
    // the two never double the footprint.
    std::map<std::string, std::pair<int, uint64_t>> gpu_h_;
    // Owned staging for f16 tensors the .tmq packed at a 2-mod-4 offset; the
    // fp16 GEMV streams half4. See reg_h().
    bool gpu_f16_ = false;
    bool gpu_bias_uploaded_ = false;   // SLOT_BIAS holds the q/k/v bias blob
    std::string model_dir_;   // set by load_quant; f16 safetensors live here
    // RAII mmap for the fp16 safetensors: zero-copy GPU registration and
    // CPU row reads (embed lookup in forward_gpu_one) share the mapping.
    struct F16Map {
        int fd = -1;
        std::uint8_t* base = nullptr;
        std::size_t size = 0;
        std::uint64_t data_begin = 0;   // 8 + header length
        std::map<std::string, tmsf::Entry> entries;
        F16Map() = default;
        F16Map(const F16Map&) = delete;
        F16Map& operator=(const F16Map&) = delete;
        ~F16Map() {
            if (base) ::munmap(base, size);
            if (fd >= 0) ::close(fd);
        }
        [[nodiscard]] bool open(const std::string& path) {
            fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) return false;
            struct ::stat st{};
            if (::fstat(fd, &st) || st.st_size <= 0) { ::close(fd); fd = -1; return false; }
            size = (std::size_t)st.st_size;
            void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (p == MAP_FAILED) { ::close(fd); fd = -1; return false; }
            base = (std::uint8_t*)p;
            std::string meta;
            entries = tmsf::read_header(path, data_begin, meta);
            return true;
        }
    };
    std::unique_ptr<F16Map> held_f16_;
    // CPU-side embed row from the fp16 mapping (forward_gpu_one entry).
    void row_h(const std::string& name, int idx, float* out) const {
        const auto& e = held_f16_->entries.at(name);
        const std::size_t n = (std::size_t)(e.end - e.begin) / 2;
        const std::size_t row_elems = n / (std::size_t)qshape_.at(name).first;
        const std::uint16_t* src =
            (const std::uint16_t*)(held_f16_->base + held_f16_->data_begin + e.begin)
            + (std::size_t)idx * row_elems;
        tmq::fp16_to_fp32_row(src, out, row_elems);
    }
    // Reconcile the prefill/decode GPU-memory conflict once the flags
    // are known (constructor-end / first forward). Explicit
    // TM_PREFILL_GPU=1 together with TM_DECODE_GPU=1 is a fail-fast
    // config error; the silent default resolves to CPU/AMX prefill.
    void reconcile_gpu_memory() {
        if (!decode_gpu_ || gpu_both_auto_) return;
        if (gpu_prefill_) {
            const char* e = getenv("TM_PREFILL_GPU");
            if (e && e[0] == '1')
                throw std::runtime_error(
                    "llama: TM_PREFILL_GPU=1 with TM_DECODE_GPU=1 "
                    "doubles the GPU weight footprint (copied prefill "
                    "blobs + NoCopy decode mapping) — over the 8 GB "
                    "host budget; pick one");
            gpu_prefill_ = false;
        }
    }
    // Estimated GPU resident bytes for the decode path: NoCopy shares
    // the file mapping (no extra RAM beyond mapped pages), copy mode
    // duplicates all Q4 weights in shared memory; KV caches and
    // activation slots are added on top. Fail fast over the budget.
    size_t gpu_decode_bytes_estimate() const {
        const bool file_backed = held_map_ && held_map_->owned.empty() &&
                                 held_map_->base != nullptr;
        size_t bytes = 0;
        if (!file_backed) {
            for (const auto& [n, sh] : qshape_)
                bytes += (size_t)sh.first * sh.second / 32 * 18;
        }
        bytes += kv_cache_bytes_all_layers();   // KV caches (fp32 or packed)
        bytes += (size_t)V * 4 * 2;                    // logits + probs
        bytes += (size_t)(H * dh + 2 * F + 4 * 2048) * 4;  // activations
        return bytes;
    }
    void ensure_gpu_weights() {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (gpu_w_ready_) return;
        reconcile_gpu_memory();
        // fp16 decode path (TM_DECODE_F16=1): half weights from
        // <dir>/model.f16.safetensors — zero-copy NoCopy over the file
        // mmap when maxBufferLength allows, per-tensor copy otherwise.
        // Replaces the Q4 GPU registration entirely (never both).
        if (const char* e = getenv("TM_DECODE_F16"); e && e[0] == '1') {
            const std::string f16p = model_dir_ + "/model.f16.safetensors";
            held_f16_ = std::make_unique<F16Map>();
            if (!held_f16_->open(f16p))
                throw std::runtime_error(
                    "llama: TM_DECODE_F16=1 but " + f16p +
                    " missing (build it with tools/convert_safetensors_f16.py)");
            std::size_t f16_bytes = 0;
            for (int l = 0; l < L; ++l) {
                const std::string pre =
                    "model.layers." + std::to_string(l) + ".";
                for (const char* t :
                     {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                      "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                      "mlp.gate_proj.weight", "mlp.up_proj.weight",
                      "mlp.down_proj.weight"}) {
                    const std::string n = pre + t;
                    const auto& te = held_f16_->entries.at(n);
                    if (te.dtype != "F16")
                        throw std::runtime_error(
                            "llama: " + n + " in " + f16p + " is " + te.dtype + ", F16 required");
                    const auto [out, in] = qshape_.at(n);
                    if ((int)te.shape[0] != out || (int)te.shape[1] != in)
                        throw std::runtime_error("llama: f16 shape mismatch: " + n);
                    f16_bytes += (size_t)out * in * 2;
                }
            }
            f16_bytes += (size_t)qshape_.at("lm_head.weight").first
                       * qshape_.at("lm_head.weight").second * 2;
            f16_bytes += (size_t)qshape_.at("model.embed_tokens.weight").first
                       * qshape_.at("model.embed_tokens.weight").second * 2;
            // Same fail-fast budget guard as the Q4 path; fp16 weights are
            // ~3.6x larger than Q4, so the estimate uses the fp16 bytes.
            {
                size_t bytes = f16_bytes;
                bytes += kv_cache_bytes_all_layers();
                bytes += (size_t)V * 4 * 2;
                bytes += (size_t)(H * dh + 2 * F + 4 * 2048) * 4;
                const double budget_gb = gpu_budget_gb();
                const double need_gb = (double)bytes / (1u << 30);
                if (need_gb > budget_gb)
                    throw std::runtime_error(
                        "llama: GPU fp16 decode estimate " + std::to_string(need_gb) +
                        " GiB exceeds TM_GPU_BUDGET_GB=" + std::to_string(budget_gb) +
                        " (raise the budget for the fp16 lane)");
            }
            struct NameBytes { std::string n; std::size_t bytes; };
            std::vector<NameBytes> wanted;
            for (int l = 0; l < L; ++l) {
                const std::string pre =
                    "model.layers." + std::to_string(l) + ".";
                for (const char* t :
                     {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                      "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                      "mlp.gate_proj.weight", "mlp.up_proj.weight",
                      "mlp.down_proj.weight"})
                    wanted.push_back({pre + t, 0});
            }
            wanted.push_back({"lm_head.weight", 0});
            wanted.push_back({"model.embed_tokens.weight", 0});
            for (auto& wb : wanted) {
                const auto& te = held_f16_->entries.at(wb.n);
                wb.bytes = (std::size_t)(te.end - te.begin);
            }
            // Zero-copy if the whole file fits one NoCopy buffer...
            bool zero_copy_ok = true;
            {
                const long page = sysconf(_SC_PAGESIZE);
                if ((uintptr_t)held_f16_->base % (std::size_t)page)
                    zero_copy_ok = false;
            }
            if (zero_copy_ok) {
                bool all = true;
                for (const auto& wb : wanted) {
                    const void* ptr = held_f16_->base + held_f16_->data_begin
                                    + held_f16_->entries.at(wb.n).begin;
                    uint64_t w = 0;
                    gpu_weights_used_ = true;
                    const int id = tm_metal_tok_wbuf(
                        ptr, wb.bytes, held_f16_->base, held_f16_->size, 1, w);
                    if (id < 0) { all = false; break; }
                    gpu_h_[wb.n] = {id, w};
                }
                if (!all) {
                    gpu_h_.clear();
                    throw std::runtime_error(
                        "llama: fp16 zero-copy registration failed (file may "
                        "exceed maxBufferLength); no copy fallback to avoid "
                        "handle-cache pollution — see F16_METAL_PLAN.md");
                } else {
                    gpu_f16_ = true;
                    gpu_w_ready_ = true;
                    return;
                }
            }
            // ...otherwise per-tensor shared copies (convert once, upload,
            // free the host buffer; peak RAM = 2.2 GiB weights + one tensor).
            for (const auto& wb : wanted) {
                const auto& te = held_f16_->entries.at(wb.n);
                if (te.dtype != "F16")
                    throw std::runtime_error(
                        "llama: " + wb.n + " in " + f16p + " is " + te.dtype + ", F16 required");
                gpu_weights_used_ = true;
                uint64_t w = 0;
                const int id = tm_metal_tok_wbuf(
                    held_f16_->base + held_f16_->data_begin + te.begin,
                    wb.bytes, nullptr, 0, 0, w);
                if (id < 0)
                    throw std::runtime_error("metal: fp16 weight registration failed: " + wb.n);
                gpu_h_[wb.n] = {id, 0};
            }
            gpu_f16_ = true;
            gpu_w_ready_ = true;
            return;
        }
        // Memory budget: the watchdog-panic session had no guard; fail
        // fast instead of finding out via swap death on an 8 GB host.
        // TM_GPU_BUDGET_GB overrides; the default scales with physical RAM.
        {
            const double budget_gb = gpu_budget_gb();
            const double need_gb =
                (double)gpu_decode_bytes_estimate() / (1u << 30);
            if (need_gb > budget_gb)
                throw std::runtime_error(
                    "llama: GPU decode estimate " + std::to_string(need_gb) +
                    " GiB exceeds TM_GPU_BUDGET_GB=" +
                    std::to_string(budget_gb));
        }
        // zero-copy (NoCopy over the page-aligned .tmq mmap) when the
        // model is file-backed; copy mode for the resident heap copy.
        const bool file_backed = held_map_ && held_map_->owned.empty() &&
                                 held_map_->base != nullptr;
        gpu_shared_wbuf_ = file_backed;   // fused decode dispatch needs one buffer
        for (int l = 0; l < L; ++l) {
            const std::string pre =
                "model.layers." + std::to_string(l) + ".";
            for (const char* t :
                 {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight",
                  "mlp.down_proj.weight"}) {
                const std::string n = pre + t;
                gpu_weights_used_ = true; // also latch partial/failed registration
                // Per NAME, not per model: one checkpoint mixes dtypes inside a
                // single layer, so each tensor is registered by what it is. A
                // K-quant tensor goes to gpu_k_ (its own GEMV kernel); Q4_0 goes
                // to gpu_w_ as it always has. lm_head is handled below.
                if (reg_kq(n)) continue;
                uint64_t w = 0;
                const int id = tm_metal_tok_wbuf(
                    q4_.at(n),
                    (size_t)qshape_.at(n).first * qshape_.at(n).second
                        / 32 * 18,
                    held_map_->base, held_map_->size,
                    file_backed ? 1 : 0, w);
                gpu_w_[n] = {id, w};
                if (id < 0)
                    throw std::runtime_error("metal: weight registration failed: " + n);
            }
        }
        uint64_t w = 0;
        gpu_weights_used_ = true;
        const bool head_kq = reg_kq("lm_head.weight");
        gpu_head_cpu_ = !q4_.count("lm_head.weight") && !head_kq &&
                        q8_.count("lm_head.weight") != 0;
        // A Q8 (or otherwise non-Q4) lm_head has no GPU kernel: it stays on the
        // CPU, so it is not registered here (registering would dereference a tensor
        // this lane never holds). See gpu_head_cpu_. A K-quant head took the reg_kq
        // branch instead and is a GPU head, so gpu_head_cpu_ stays false for it.
        const int id = (gpu_head_cpu_ || head_kq)
                           ? -1
                           : tm_metal_tok_wbuf(
                                 q4_.at("lm_head.weight"),
                                 (size_t)qshape_.at("lm_head.weight").first *
                                     qshape_.at("lm_head.weight").second / 32 * 18,
                                 held_map_->base, held_map_->size, file_backed ? 1 : 0, w);
        if (!gpu_head_cpu_ && !head_kq) {
            gpu_w_["lm_head.weight"] = {id, w};
            if (id < 0) throw std::runtime_error("metal: lm_head registration failed");
        }
        gpu_w_ready_ = true;
        if (getenv("TM_DEBUG_GPUW") && !gpu_head_cpu_ && !head_kq) {
            // Checked: a failed probe download leaves the buffer uninitialized,
            // and this block's entire output is memcmp against the host bytes,
            // so a dropped result would print a comparison of garbage.
            const auto probe_dl = [](int slot, size_t off, unsigned char* dst) {
                if (!tm_metal_tok_download(slot, off, dst, 32))
                    throw std::runtime_error("metal: probed weight download failed");
            };
            unsigned char gpub[32], gpub2[32], host[32];
            probe_dl(id, w, gpub);
            memcpy(host, reinterpret_cast<const uint8_t*>(q4_.at("lm_head.weight")), 32);
            const int probe_slot = 500;
            tm_metal_tok_probe_copy(id, w, probe_slot, 32);
            probe_dl(probe_slot, 0, gpub2);
            const auto& l0q = gpu_w_.at(
                "model.layers.0.self_attn.q_proj.weight");
            unsigned char l0g[32], l0h[32], l0gpu[32];
            probe_dl(l0q.first, l0q.second, l0g);
            memcpy(l0h, reinterpret_cast<const uint8_t*>(q4_.at(
                "model.layers.0.self_attn.q_proj.weight")), 32);
            tm_metal_tok_probe_copy(l0q.first, l0q.second, 501, 32);
            probe_dl(501, 0, l0gpu);
            fprintf(stderr,
                    "[gpuw] lm_head id=%d woff=%llu cpu-cmp=%d gpu-cmp=%d | l0q id=%d woff=%llu cpu-cmp=%d gpu-cmp=%d\n",
                    id, (unsigned long long)w, memcmp(gpub, host, 32),
                    memcmp(gpub2, host, 32), l0q.first,
                    (unsigned long long)l0q.second, memcmp(l0g, l0h, 32),
                    memcmp(l0gpu, l0h, 32));
        }
    #else
        throw std::runtime_error("llama: Metal backend not built");
    #endif
    }

    bool quantized() const { return held_map_ != nullptr || held_demand_ != nullptr; }

    // Minimal config.json reader (Llama-family HF configs are flat scalars
    // for everything needed). Missing keys keep current values; a missing
    // file changes nothing (TinyLlama defaults stay). Derives dh = D/H and
    // REP_ = H/KVH — the batched-attention head grouping depends on REP_,
    // so it MUST be recomputed here, not left at the construction value.
    void apply_config(const std::string& path) {
        std::ifstream f(path);
        if (!f) return;
        std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
        auto num = [&](const char* k) -> const char* {
            const std::string pat = std::string("\"") + k + "\"";
            const std::size_t p = s.find(pat);
            if (p == std::string::npos) return nullptr;
            std::size_t q = s.find(':', p + pat.size());
            if (q == std::string::npos) return nullptr;
            ++q;
            while (q < s.size() && (s[q] == ' ' || s[q] == '\t')) ++q;
            return s.c_str() + q;
        };
        auto number_end = [](const char* end) {
            while (std::isspace((unsigned char)*end)) ++end;
            return *end == ',' || *end == '}';
        };
        auto iget = [&](const char* k, int def) {
            const char* v = num(k);
            if (!v) return def;
            char* end = nullptr;
            errno = 0;
            const long value = std::strtol(v, &end, 10);
            if (end == v || errno == ERANGE || value <= 0 ||
                value > std::numeric_limits<int>::max() ||
                !number_end(end))
                throw std::runtime_error("llama: invalid config integer " + std::string(k));
            return (int)value;
        };
        auto fget = [&](const char* k, float def) {
            const char* v = num(k);
            if (!v) return def;
            char* end = nullptr;
            errno = 0;
            const float value = std::strtof(v, &end);
            if (end == v || errno == ERANGE || !std::isfinite(value) || value <= 0.f ||
                !number_end(end))
                throw std::runtime_error("llama: invalid config number " + std::string(k));
            return value;
        };
        const int next_d = iget("hidden_size", D);
        const int next_l = iget("num_hidden_layers", L);
        const int next_h = iget("num_attention_heads", H);
        // absent num_key_value_heads => no GQA (open_llama_7b class)
        const int next_kvh = iget("num_key_value_heads", next_h);
        const int next_f = iget("intermediate_size", F);
        const int next_v = iget("vocab_size", V);
        const float next_eps = fget("rms_norm_eps", eps);
        const float next_theta = fget("rope_theta", theta);
        const int next_max_ctx = iget("max_position_embeddings", max_ctx_);
        // ---- qwen35 (qwen3_5_text) hybrid fields; absent -> defaults ----
        auto bget = [&](const char* k, bool def) {
            const char* v = num(k);
            if (!v) return def;
            if (std::strncmp(v, "true", 4) == 0) return true;
            if (std::strncmp(v, "false", 5) == 0) return false;
            throw std::runtime_error("llama: invalid config bool " + std::string(k));
        };
        full_attention_interval_ = iget("full_attention_interval", 0);
        head_dim_ = iget("head_dim", 0);
        lk_heads_ = iget("linear_num_key_heads", 0);
        lv_heads_ = iget("linear_num_value_heads", 0);
        lk_dim_ = iget("linear_key_head_dim", 0);
        lv_dim_ = iget("linear_value_head_dim", 0);
        conv_kernel_ = iget("linear_conv_kernel_dim", 0);
        // The reference derives the DeltaNet dimensions from the linear_* config
        // keys (qwen35.cpp: head_k_dim = ssm_d_state = linear_key_head_dim,
        // n_k_heads = ssm_n_group = linear_num_key_heads, n_v_heads =
        // ssm_dt_rank = linear_num_value_heads). The `ssm.*` GGUF keys the
        // earlier revision of this parser looked for are never emitted by the
        // converter, so derive them here and let an explicit key override.
        ssm_state_ = iget("state_size", lk_dim_);
        ssm_rank_ = iget("time_step_rank", lv_heads_);
        ssm_groups_ = iget("num_groups", lk_heads_);
        ts_min_ = fget("time_step_min", 0.f);
        ts_max_ = fget("time_step_max", 0.f);
        ts_floor_ = fget("time_step_floor", 0.f);
        partial_rotary_ = fget("partial_rotary_factor", 1.f);
        mtp_layers_ = iget("mtp_num_hidden_layers", 0);
        attn_output_gate_ = bget("attn_output_gate", false);
        tie_word_embeddings_ = bget("tie_word_embeddings", false);
        mrope_interleaved_ = bget("mrope_interleaved", false);
        // partial_rotary_factor arrives nested under `rope_parameters` in the
        // HF config, so fget() above only sees it when a flattened config is
        // used; scan the nested block too (the key name is unique in the file).
        if (partial_rotary_ == 1.f) {
            const std::string pat = "\"partial_rotary_factor\"";
            const std::size_t p = s.find(pat);
            if (p != std::string::npos) {
                std::size_t q = s.find(':', p + pat.size());
                if (q != std::string::npos) {
                    char* end = nullptr;
                    const float v = std::strtof(s.c_str() + q + 1, &end);
                    if (end != s.c_str() + q + 1 && std::isfinite(v) && v > 0.f && v <= 1.f)
                        partial_rotary_ = v;
                }
            }
        }
        {
            const std::string pat = "\"layer_types\"";
            const std::size_t p = s.find(pat);
            if (p != std::string::npos) {
                const std::size_t b = s.find('[', p), e = s.find(']', b);
                if (b != std::string::npos && e != std::string::npos && e > b) {
                    layer_types_.clear();
                    const std::string inner = s.substr(b + 1, e - b - 1);
                    for (std::size_t i = 0; i < inner.size();) {
                        const std::size_t q1 = inner.find('"', i);
                        if (q1 == std::string::npos) break;
                        const std::size_t q2 = inner.find('"', q1 + 1);
                        if (q2 == std::string::npos) break;
                        layer_types_.push_back(
                            inner.substr(q1 + 1, q2 - q1 - 1) == "linear_attention" ? 1 : 0);
                        i = q2 + 1;
                    }
                }
            }
        }
        if (next_d <= 0 || next_l <= 0 || next_h <= 0 || next_kvh <= 0 ||
            next_f <= 0 || next_v <= 0 || next_d % next_h ||
            next_h % next_kvh || next_f % tmq::kBlock)
            throw std::runtime_error("llama: incompatible config " + path);
        D = next_d; L = next_l; H = next_h; KVH = next_kvh;
        F = next_f; V = next_v; eps = next_eps; theta = next_theta;
        max_ctx_ = next_max_ctx;
        // qwen35 sets an explicit head_dim that differs from D/H (0.8B: 128 vs
        // 256). Everything downstream (KV cache rows, attention scale, o_proj
        // width, QK-norm) is sized by the true head dim, so `dh` carries it.
        dh = head_dim_ > 0 ? head_dim_ : D / H;
        REP_ = H / KVH;
        n_rot_ = (int)(partial_rotary_ * (float)dh + 0.5f);
        n_rot_ = std::min(n_rot_, dh);
    }

    // Size the KV cache (ctx) from physical RAM so a larger unified-memory
    // chip holds a longer context automatically. Budget: KV cache <= 12.5%
    // of RAM, capped at the model's max_position_embeddings; TM_CTX overrides
    // to a fixed value. The KV cache is L*KVH*dh floats per token (K+V), so
    // ctx scales inversely with model width. Leaves ctx untouched (2048) when
    // the RAM is unknown or the budget is below the 256-slot floor.
    void auto_size_ctx() {
        if (const long v = env_var_long("TM_CTX").value_or(-1); v > 0) {
            ctx = (int)v;
            return;
        }
        const std::size_t kv_per_token =
            kv_bytes_per_token();
        if (kv_per_token == 0) return;
        const double ram = physical_ram_gb();
        if (ram <= 0) return;
        const std::size_t budget = (std::size_t)(0.125 * ram * (double)(1u << 30));
        const std::size_t cap = max_ctx_ > 0 ? (std::size_t)max_ctx_ : (std::size_t)ctx;
        const std::size_t slots = budget / kv_per_token < cap ? budget / kv_per_token : cap;
        if (slots >= 256) ctx = (int)slots;
    }

    // Auto-size the demand resident window W from physical RAM: keep the
    // resident set (embed + lm_head + 2W·layer + scratch) at ~50% of RAM so a
    // bigger chip keeps more layers resident and re-reads less per token.
    // Only meaningful when eviction is active (demand_mode_); TM_DEMAND_WIN
    // overrides. Uses the Q4 block layout (rows*cols/32*18 bytes) so it is
    // exact, not a heuristic.
    void auto_size_demand_window() {
        if (!demand_mode_ || std::getenv("TM_DEMAND_WIN")) return;
        const double ram = physical_ram_gb();
        if (ram <= 0 || L <= 0) return;
    auto q4b = [&](const std::string& n) -> std::size_t {
        auto it = qshape_.find(n);
        return it == qshape_.end() ? 0
            : (std::size_t)it->second.first * (std::size_t)it->second.second / 32 * 18;
    };
        const std::size_t base = q4b("model.embed_tokens.weight") + q4b("lm_head.weight");
        static const char* tensors[7] = {
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight"};
        const std::string p0 = "model.layers.0.";
        std::size_t layer = 0;
        for (const char* t : tensors) layer += q4b(p0 + t);
        if (layer == 0) return;
        const std::size_t budget = (std::size_t)(0.5 * ram * (double)(1u << 30));
        if (budget <= base) return;
        const std::size_t W = (budget - base) / (2 * layer);
        if (W >= 1 && W <= (std::size_t)L) demand_window_ = (int)W;
    }

    // y = W x over n_out rows of the (out, in) quantized layout.
    template <class Blk>
    static void qgemv(const Blk* Wq, const float* x, int n_in, int n_out, float* y) {
        constexpr bool q4 = std::is_same_v<Blk, tmq::BlockQ4_0>;
        const std::uint64_t nb = (std::uint64_t)(n_in / tmq::kBlock);
        for (int o = 0; o < n_out; ++o) {
            const Blk* r = Wq + (std::uint64_t)o * nb;
            float acc = 0.f;
            for (std::uint64_t b = 0; b < nb; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                float s = 0.f;
                if constexpr (!q4) {
                    for (int i = 0; i < tmq::kBlock; ++i)
                        s += (float)r[b].qs[i] * x[b * tmq::kBlock + i];
                } else {
                    // Q4_0: value = (nibble - 8) * d  (zero-point +8)
                    for (int j = 0; j < tmq::kBlock / 2; ++j) {
                        s += (float)((r[b].qs[j] & 0xF) - 8) * x[b * tmq::kBlock + 2 * j]
                           + (float)((r[b].qs[j] >> 4) - 8) * x[b * tmq::kBlock + 2 * j + 1];
                    }
                }
                acc += s * d;
            }
            y[o] = acc;
        }
    }

#ifdef __ARM_NEON
    // L3: NEON dequant-dot kernels. Same math and per-row accumulation
    // order as qgemv (block partial sum scaled by d, added to acc), so
    // the oracle gates stay valid; rows are independent, so gemv rows
    // can also be split across pool workers without reordering anything.
template <class Blk>
static void dequant_block(const Blk& blk, float32x4_t* qf) {
    if constexpr (std::is_same_v<Blk, tmq::BlockQ8_0>) {
        const int8x16_t q0 = vld1q_s8(blk.qs);
        const int8x16_t q1 = vld1q_s8(blk.qs + 16);
        const int8x16_t v[2] = {q0, q1};
        for (int h = 0; h < 2; ++h) {
            const int16x8_t w0 = vmovl_s8(vget_low_s8(v[h]));
            const int16x8_t w1 = vmovl_s8(vget_high_s8(v[h]));
            qf[4 * h + 0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0)));
            qf[4 * h + 1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0)));
            qf[4 * h + 2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1)));
            qf[4 * h + 3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)));
        }
    } else {
        const uint8x16_t raw = vld1q_u8(blk.qs);
        // byte j = (nibble 2j) | (nibble 2j+1) << 4; zip the nibble lanes
        // back into element order, then apply the Q4_0 zero-point (-8)
        const int8x16_t lo = vsubq_s8(
            vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0xF))), vdupq_n_s8(8));
        const int8x16_t hi = vsubq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), vdupq_n_s8(8));
        const int8x16x2_t z = vzipq_s8(lo, hi);
        const int8x16_t v[2] = {z.val[0], z.val[1]};
        for (int h = 0; h < 2; ++h) {
            const int16x8_t w0 = vmovl_s8(vget_low_s8(v[h]));
            const int16x8_t w1 = vmovl_s8(vget_high_s8(v[h]));
            qf[4 * h + 0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0)));
            qf[4 * h + 1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0)));
            qf[4 * h + 2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1)));
            qf[4 * h + 3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)));
        }
    }
}

template <class Blk>
static void qgemv_rows(const Blk* Wq, const float* x, int n_in,
                       int o0, int o1, float* y) {
#ifdef TM_FORCE_SCALAR
    const std::uint64_t nb0 = (std::uint64_t)(n_in / tmq::kBlock);
    qgemv(Wq + (std::uint64_t)o0 * nb0, x, n_in, o1 - o0, y + o0);
    return;
#endif
    constexpr bool q4 = std::is_same_v<Blk, tmq::BlockQ4_0>;
    (void)q4;
    const std::uint64_t nb = (std::uint64_t)(n_in / tmq::kBlock);
    const float32x4_t vzero = vdupq_n_f32(0.f);
    for (int o = o0; o < o1; ++o) {
        const Blk* r = Wq + (std::uint64_t)o * nb;
        // 4 independent accumulators break the serial FMA dependency chain
        // (a single chain per block throttled this to ~1 MAC/cycle)
        float32x4_t accs[4] = {vzero, vzero, vzero, vzero};
        std::uint64_t b = 0;
        for (; b + 4 <= nb; b += 4) {
            for (int u = 0; u < 4; ++u) {
                float32x4_t qf[8];
                dequant_block(r[b + u], qf);
                const float32x4_t dv =
                    vdupq_n_f32(tmq::fp16_to_fp32(r[b + u].d_fp16));
                float32x4_t part = vzero;
                const float* xb = x + (b + u) * tmq::kBlock;
                for (int k = 0; k < 8; ++k)
                    part = vfmaq_f32(part, qf[k], vld1q_f32(xb + 4 * k));
                accs[u] = vfmaq_f32(accs[u], part, dv);
            }
        }
        for (; b < nb; ++b) {
            float32x4_t qf[8];
            dequant_block(r[b], qf);
            const float32x4_t dv = vdupq_n_f32(tmq::fp16_to_fp32(r[b].d_fp16));
            float32x4_t part = vzero;
            const float* xb = x + b * tmq::kBlock;
            for (int k = 0; k < 8; ++k)
                part = vfmaq_f32(part, qf[k], vld1q_f32(xb + 4 * k));
            accs[0] = vfmaq_f32(accs[0], part, dv);
        }
        const float32x4_t acc = vaddq_f32(vaddq_f32(accs[0], accs[1]),
                                          vaddq_f32(accs[2], accs[3]));
        y[o] = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1)
             + vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
    }
}
#else
    template <class Blk>
    static void qgemv_rows(const Blk* Wq, const float* x, int n_in,
                           int o0, int o1, float* y) {
        const std::uint64_t nb = (std::uint64_t)(n_in / tmq::kBlock);
        qgemv(Wq + (std::uint64_t)o0 * nb, x, n_in, o1 - o0, y + o0);
    }
#endif
    // Dispatch on the stored dtype for a 2-D tensor: y = W x. Rows are
    // independent full dot products, so splitting rows across pool
    // workers changes NOTHING numerically (per-row order preserved).
    // TM_SERIAL_GEMV=1 forces the single-thread path for A/B benches.
    //
    // Parallelism uses a dedicated mini-pool (NOT traink::Pool): the job
    // is a plain struct of raw pointers dispatched under a generation
    // counter — no std::function, no per-dispatch allocation, no type
    // erasure on the hot path. (The shared traink pool showed flaky
    // corruption with heavy bodies; root cause not yet isolated.)
#ifdef __ARM_NEON
    // --- sdot decode path (TM_SDOT, default on; '0' disables) ------------
    // int8 per-block activation quantization (llama.cpp Q8_1 scheme): x is
    // quantized ONCE per projection (all rows share it), each 32-block with
    // its own scale; the dot runs on vdotq_s32 (16 int8 MACs/insn, dotprod
    // on M1) and each block's int32 sum is rescaled by ws[b]*xs[b]. The
    // dequant+fp32-FMA path is issue-bound at ~0.9 instructions/value —
    // the measured qgemv ceiling is ~19 GB/s effective IDENTICALLY on the
    // 0.65 GB and 3.8 GB models (perfect 4x pool/serial scaling), i.e.
    // compute-bound, not bandwidth-bound. ~0.4 instructions/value here.
    struct XQ {
        std::vector<int8_t> q;    // element order (Q8 weights)
        std::vector<int8_t> d;    // per block: 16 even | 16 odd (Q4 weights)
        std::vector<float> s;     // per-block scales
    };
    static bool quant_x_neon_enabled() {
        // Same-binary grouped-decode A/B won all six pairs on M1. Keep a scalar
        // opt-out for diagnosis and workloads not represented by that gate.
        static const bool enabled = [] {
            const char* value = std::getenv("TM_QUANT_X_NEON");
            return !value || value[0] != '0';
        }();
        return enabled;
    }
    static int quant_x_blocks(int n_in, const float* x) {
        if (n_in < 0 || n_in % tmq::kBlock != 0 || (n_in != 0 && x == nullptr))
            throw std::invalid_argument("quant_x requires complete activation blocks");
        return n_in / tmq::kBlock;
    }
    static void quant_x(const float* x, int n_in, XQ& out) {
        const int nb = quant_x_blocks(n_in, x);
        out.q.resize((std::size_t)n_in);
        out.d.resize((std::size_t)n_in);
        out.s.resize((std::size_t)nb);
        quant_x_into(x, n_in, out.q.data(), out.d.data(), out.s.data());
    }

    // The block loop, writing into CALLER-owned slots. Split out of quant_x so the
    // multi-position path can pack T activations into one flat buffer without a
    // second copy: the caller sizes the storage, this fills a slot.
    static void quant_x_into(const float* x, int n_in, int8_t* qbase, int8_t* dbase,
                             float* sbase) {
        const int nb = quant_x_blocks(n_in, x);
        const bool neon = quant_x_neon_enabled();
        for (int b = 0; b < nb; ++b) {
            const float* xb = x + (std::size_t)b * tmq::kBlock;
            float32x4_t am = vdupq_n_f32(0.f);
            for (int k = 0; k < 8; ++k)
                am = vmaxq_f32(am, vabsq_f32(vld1q_f32(xb + 4 * k)));
            const float amax = vmaxvq_f32(am);
            if (!std::isfinite(amax))
                throw std::invalid_argument("quant_x requires finite activations");
            float d = amax / 127.f;
            const float inv = d > 0.f ? 1.f / d : 0.f;
            int8_t* qe = qbase + (std::size_t)b * tmq::kBlock;
            int8_t* qd = dbase + (std::size_t)b * tmq::kBlock;
            if (amax > 0.f && (d == 0.f || !std::isfinite(inv))) {
                // Tiny finite blocks can underflow the scale or overflow its
                // reciprocal. Round the stored scale upward to cover amax,
                // then divide in double; never convert infinity to an integer.
                if ((double)d * 127. < (double)amax)
                    d = std::nextafter(d, std::numeric_limits<float>::infinity());
                sbase[b] = d;
                for (int i = 0; i < tmq::kBlock; ++i) {
                    qe[i] = (int8_t)std::lrint((double)xb[i] / (double)d);
                    qd[i / 2 + (i % 2) * 16] = qe[i];
                }
                continue;
            }
            sbase[b] = d;
            if (!neon) {
                for (int i = 0; i < 16; ++i) {
                    qe[2 * i] = (int8_t)lrintf(xb[2 * i] * inv);
                    qe[2 * i + 1] = (int8_t)lrintf(xb[2 * i + 1] * inv);
                    qd[i] = qe[2 * i];
                    qd[16 + i] = qe[2 * i + 1];
                }
                continue;
            }
            // FRINTX honors the current rounding mode like lrintf; a fixed
            // nearest-even conversion would silently change directed rounding.
            const auto quant16 = [inv](const float* p) {
                const auto quant4 = [inv](const float* v) {
                    return vcvtq_s32_f32(vrndxq_f32(vmulq_n_f32(vld1q_f32(v), inv)));
                };
                const int16x8_t lo = vcombine_s16(vmovn_s32(quant4(p)),
                                                  vmovn_s32(quant4(p + 4)));
                const int16x8_t hi = vcombine_s16(vmovn_s32(quant4(p + 8)),
                                                  vmovn_s32(quant4(p + 12)));
                return vcombine_s8(vmovn_s16(lo), vmovn_s16(hi));
            };
            const int8x16_t lo = quant16(xb), hi = quant16(xb + 16);
            vst1q_s8(qe, lo);
            vst1q_s8(qe + 16, hi);
            vst1q_s8(qd, vuzp1q_s8(lo, hi));       // even elements first
            vst1q_s8(qd + 16, vuzp2q_s8(lo, hi));  // then odd elements
        }
    }
    template <class Blk>
    static void qgemv_sdot_rows(const Blk* Wq, const int8_t* xq,
                                const int8_t* xd, const float* xs,
                                int n_in, int o0, int o1, float* y) {
        constexpr bool q4 = std::is_same_v<Blk, tmq::BlockQ4_0>;
        const int8x16_t m8 = vdupq_n_s8(8);
        const std::uint64_t nb = (std::uint64_t)(n_in / tmq::kBlock);
        // Row-quad: 4 output rows share each x-vector load pair (halves the
        // x-side instructions), and the per-block scale is a SCALAR — the
        // int32x4 block sum converts once and folds via vfmaq_n_f32, so the
        // epilogue is one instruction per block per row (no vaddv/lane-pack
        // per block). NR must be a compile-time constant: with a runtime
        // trip count the compiler spilled accv[] to the stack and the kernel
        // LOST to the simple per-row loop (TinyLlama 45.2 vs 55.4 t/s).
        auto quad_rows = [&](auto nr_c, int o) {
            constexpr int NR = decltype(nr_c)::value;
            float32x4_t accv[NR];
            for (int rr = 0; rr < NR; ++rr) accv[rr] = vdupq_n_f32(0.f);
            for (std::uint64_t b = 0; b < nb; ++b) {
                const int8x16_t xe = vld1q_s8(xd + b * 32);      // even elems
                const int8x16_t xo = vld1q_s8(xd + b * 32 + 16); // odd elems
                const int8x16_t xqe = vld1q_s8(xq + b * 32);
                const int8x16_t xqo = vld1q_s8(xq + b * 32 + 16);
                const float sc = xs[b];
                // Share the Q4 zero-point term across four output rows.
                // Keep four independent SDOT lane sums: reducing to one
                // block sum here would change subsequent fp32 rounding.
                int32x4_t correction = vdupq_n_s32(0);
                if constexpr (q4 && NR == 4) {
                    const int8x16_t minus8 = vdupq_n_s8(-8);
                    correction = vdotq_s32(correction, minus8, xe);
                    correction = vdotq_s32(correction, minus8, xo);
                }
                // Per-block scales for the four rows in ONE hardware fp16->fp32
                // conversion (2026-09-07): the scalar tmq::fp16_to_fp32 with its
                // exponent branch cost ~20 instructions per row-quad-block, about
                // half the dot-product work (CPU decode 58 vs llama.cpp 78 t/s
                // at 4 threads). Block scales are normal halves; subnormal
                // scales would flush only under FZ, which user mode leaves off.
                float s4[4];
                if constexpr (NR == 4) {
                    uint16x4_t dh = vdup_n_u16(0);
                    dh = vld1_lane_u16(&Wq[(std::uint64_t)(o + 0) * nb + b].d_fp16, dh, 0);
                    dh = vld1_lane_u16(&Wq[(std::uint64_t)(o + 1) * nb + b].d_fp16, dh, 1);
                    dh = vld1_lane_u16(&Wq[(std::uint64_t)(o + 2) * nb + b].d_fp16, dh, 2);
                    dh = vld1_lane_u16(&Wq[(std::uint64_t)(o + 3) * nb + b].d_fp16, dh, 3);
                    vst1q_f32(s4, vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(dh)), sc));
                }
                for (int rr = 0; rr < NR; ++rr) {
                    const Blk* r = Wq + (std::uint64_t)(o + rr) * nb;
                    int32x4_t a = vdupq_n_s32(0);
                    if constexpr (q4) {
                        // byte j = (w[2j] & 0xF) | (w[2j+1] << 4).
                        // Quad rows use raw nibbles plus the shared -8 term;
                        // single-row tails subtract 8 directly.
                        const uint8x16_t raw = vld1q_u8(r[b].qs);
                        int8x16_t lo = vreinterpretq_s8_u8(
                            vandq_u8(raw, vdupq_n_u8(0xF)));
                        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
                        if constexpr (NR == 4) {
                            a = correction;
                        } else {
                            lo = vsubq_s8(lo, m8);
                            hi = vsubq_s8(hi, m8);
                        }
                        a = vdotq_s32(a, lo, xe);
                        a = vdotq_s32(a, hi, xo);
                    } else {
                        a = vdotq_s32(a, vld1q_s8(r[b].qs), xqe);
                        a = vdotq_s32(a, vld1q_s8(r[b].qs + 16), xqo);
                    }
                    if constexpr (NR == 4)
                        accv[rr] = vfmaq_n_f32(accv[rr], vcvtq_f32_s32(a), s4[rr]);
                    else
                        accv[rr] = vfmaq_n_f32(accv[rr], vcvtq_f32_s32(a),
                                               tmq::fp16_to_fp32(r[b].d_fp16) * sc);
                }
            }
            for (int rr = 0; rr < NR; ++rr) y[o + rr] = vaddvq_f32(accv[rr]);
        };
        int o = o0;
        for (; o + 4 <= o1; o += 4)
            quad_rows(std::integral_constant<int, 4>{}, o);
        for (; o < o1; ++o) quad_rows(std::integral_constant<int, 1>{}, o);
    }

    // --- K-quant decode GEMV (Q4_1 / Q5_K / Q6_K) ------------------------
    // These tensors are stored at their GGUF source byte width (see tmq.h),
    // so a row is unpacked on the fly rather than dequantized. Doing the
    // unpack in int8 is what makes it viable: a Q6_K code is (code - 32),
    // already an int8 weight, and each 16-value sub-block carries its own
    // scale — exactly the shape vdotq_s32 wants. A fp32 dequantize-then-dot
    // costs ~5 ops/value against ~0.75 here, which is 25 ms vs 4 ms for
    // Qwen3.5-0.8B's 248320x1024 lm_head. Measured on M1: 8.91 ms as F16
    // (508.6 MB) -> 3.6 ms as Q6_K (206.6 MB).
    //
    // Q4_1 and Q5_K are affine (value = a*code + b) rather than purely
    // multiplicative, so each 32-value group also needs sum(x) — two
    // vaddlvq_s8 on activation bytes that are already in registers.
    static std::int32_t kq_sum32(const std::int8_t* x) {
        return (std::int32_t)vaddlvq_s8(vld1q_s8(x)) +
               (std::int32_t)vaddlvq_s8(vld1q_s8(x + 16));
    }

    // Q6_K: value = d * sc[is] * (code - 32), 16 sub-blocks of 16 per
    // 256-value super-block. The codes are permuted within each 128-value
    // half: element (l + 32*j) takes its low nibble from ql[l + 32*(j&1)]
    // (the high nibble for j >= 2), its top 2 bits from qh[l] >> 2j, and its
    // scale from sc[l/16 + 2j].
    // Rebuild one 16-value Q6_K segment as signed codes. J selects which quarter
    // of the 128-value half the 16 values come from (element l + 32*J), so the
    // nibble half, the qh bit position and the activation offset are all
    // compile-time constants, and the high-bit extraction stays an immediate
    // shift (the earlier loop form needed a runtime vector shift per 16 values).
    //
    // This is the expensive half of a Q6_K dot: ~8 instructions per 16 values
    // against one vdotq_s32. It is its own function because the multi-position
    // kernel uses the same unpack function against T activation vectors, and the
    // per-row kernel uses the same unpack function against one x vector. The
    // code still never lands in memory; only the int8 pair does.
    template <int J>
    static int8x16_t kq6k_unpack16(const std::uint8_t* ql, const uint8x16_t qh) {
        const uint8x16_t raw = vld1q_u8(ql + 32 * (J & 1));
        const uint8x16_t nib = (J < 2) ? vandq_u8(raw, vdupq_n_u8(0x0F))
                                       : vshrq_n_u8(raw, 4);
        // vshrq_n_u8 rejects a 0 shift, and J == 0 needs none.
        uint8x16_t shifted = qh;
        if constexpr (2 * J > 0) shifted = vshrq_n_u8(qh, 2 * J);
        const uint8x16_t hi = vandq_u8(shifted, vdupq_n_u8(3));
        // vsliq_n_u8(a, b, 4) is a | ((b & 0x0F) << 4) in ONE instruction where
        // the explicit shift-then-or took two, and it is bit-identical here
        // because `hi` is already masked to two bits. The unpack is this
        // kernel's dominant instruction cost, so an instruction removed from it
        // is worth more than it looks — and this one cannot change a value.
        return vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(nib, hi, 4)),
                        vdupq_n_s8(32));
    }

    // One 16-value dot: unpack, vdotq_s32, horizontal add — the same tree the
    // kernel used before the unpack was factored out.
    template <int J>
    static int32x4_t kq6k_dot16(const std::uint8_t* ql, const uint8x16_t qh,
                                const std::int8_t* x) {
        return vdotq_s32(vdupq_n_s32(0), kq6k_unpack16<J>(ql, qh), vld1q_s8(x));
    }

    template <bool PF>
    static void kqgemv_q6k_rows_impl(const tmq::BlockQ6_K* W, const int8_t* xq,
                                     const float* xs, int n_in, int o0, int o1,
                                     float* y) {
        const std::uint64_t nbs = (std::uint64_t)(n_in / tmq::kKBlock);
        for (int o = o0; o < o1; ++o) {
            const tmq::BlockQ6_K* r = W + (std::uint64_t)o * nbs;
            // Four independent vector accumulators. The dot is only ~9
            // instructions per 16 values, so this kernel is not issue-bound;
            // what stalls it is the dependency chain through the horizontal
            // reduce (a vaddvq per dot is ~5-7 cycles with nothing to overlap).
            // Four chains, plus folding the sub-block factor with an FMA, keep
            // the vdot and FMA pipes busy across super-blocks.
            float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
            float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
            for (std::uint64_t b = 0; b < nbs; ++b) {
                // Software prefetch arm (TM_KQ_PREFETCH). The Q6_K super-block is
                // 210 bytes — a non-power-of-two stride the hardware prefetcher
                // does not stream — so a core's weight rate may be L2/DRAM latency
                // limited rather than issue limited. Templated so the arm costs no
                // branch in the loop.
                if constexpr (PF) {
                    const std::uint64_t ahead = b + 2;
                    if (ahead < nbs) __builtin_prefetch(r[ahead].ql, 0, 3);
                }
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                const std::uint8_t* ql = r[b].ql;
                const std::uint8_t* qh = r[b].qh;
                const std::int8_t* sc = r[b].scales;
                const std::int8_t* xb = xq + b * tmq::kKBlock;
                const float* xsb = xs + (b * tmq::kKBlock) / tmq::kBlock;
                // The 16 sub-block factors, broadcast once per super-block
                // rather than re-derived inside every dot — that fold was ~40%
                // of the kernel's instructions. Sub-block `is` spans values
                // [16*is, 16*is+16) and therefore activation block is>>1.
                float32x4_t vf[16];
                for (int is = 0; is < 16; ++is)
                    vf[is] = vdupq_n_f32(d * (float)sc[is] * xsb[is >> 1]);
                for (int half = 0; half < 2; ++half) {
                    const std::uint8_t* qlh = ql + half * 64;
                    const std::uint8_t* qhh = qh + half * 32;
                    const std::int8_t* xh = xb + half * 128;
                    const float32x4_t* fh = vf + half * 8;
                    // Both qh vectors are reused by all eight dots of this
                    // half; loading them per dot was the other quarter of the
                    // instruction count.
                    const uint8x16_t qha = vld1q_u8(qhh);
                    const uint8x16_t qhb = vld1q_u8(qhh + 16);
                    // l = 0..15 -> sub-block scale index 2J
                    a0 = vfmaq_f32(a0, vcvtq_f32_s32(kq6k_dot16<0>(qlh, qha, xh)), fh[0]);
                    a1 = vfmaq_f32(a1, vcvtq_f32_s32(kq6k_dot16<1>(qlh, qha, xh + 32)), fh[2]);
                    a2 = vfmaq_f32(a2, vcvtq_f32_s32(kq6k_dot16<2>(qlh, qha, xh + 64)), fh[4]);
                    a3 = vfmaq_f32(a3, vcvtq_f32_s32(kq6k_dot16<3>(qlh, qha, xh + 96)), fh[6]);
                    // l = 16..31 -> sub-block scale index 2J+1
                    a0 = vfmaq_f32(a0, vcvtq_f32_s32(kq6k_dot16<0>(qlh + 16, qhb, xh + 16)), fh[1]);
                    a1 = vfmaq_f32(a1, vcvtq_f32_s32(kq6k_dot16<1>(qlh + 16, qhb, xh + 48)), fh[3]);
                    a2 = vfmaq_f32(a2, vcvtq_f32_s32(kq6k_dot16<2>(qlh + 16, qhb, xh + 80)), fh[5]);
                    a3 = vfmaq_f32(a3, vcvtq_f32_s32(kq6k_dot16<3>(qlh + 16, qhb, xh + 112)), fh[7]);
                }
            }
            y[o] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        }
    }

    // The env is read once (the same rule as the flags in gemv_w) and dispatch
    // picks the arm. Both arms are the same arithmetic, so the choice is a
    // measurement (`kq_multi_ab`) and not a guess.
    static void kqgemv_q6k_rows(const tmq::BlockQ6_K* W, const int8_t* xq,
                                const float* xs, int n_in, int o0, int o1,
                                float* y) {
        static const bool pf = [] {
            const char* e = getenv("TM_KQ_PREFETCH");
            return e && e[0] == '1';
        }();
        if (pf)
            kqgemv_q6k_rows_impl<true>(W, xq, xs, n_in, o0, o1, y);
        else
            kqgemv_q6k_rows_impl<false>(W, xq, xs, n_in, o0, o1, y);
    }

    // TQ2 (ternary, 2 bits over a group of 128): w = (code - 1) * d, code in
    // {0,1,2}. The stored block is 32 packed bytes plus one fp16 scale —
    // byte-identical to what an MLX `bits=2, group_size=128` tensor stores for
    // one group, with element j at word j/16, bits 2*(j%16) (verified against
    // mx.dequantize, max error 0). Our block just puts the scale first.
    //
    // `codes` are 0..2, so no high-bit plane exists and no -32 offset is needed:
    // the trit is recovered by the `- sum(x)` term alone (the (code - 1) shift),
    // folded once per 32-value activation group. The unpack expands a 16-byte
    // load (4 words = 64 elements) into four int8x16 planes in ELEMENT order:
    // the four 2-bit planes sit at stride 4, so zip pair-wise and then at 16-bit
    // granularity, which is exactly the stride-4 -> consecutive shuffle. Eight
    // zips and four vdots per 64 values, against Q6_K's ~9 ops per 16 values for
    // the same result with one fewer plane to fold.
    static void tq2_unpack64(const std::uint8_t* q, int8x16_t out[4]) {
        const uint8x16_t v = vld1q_u8(q);
        const uint8x16_t m = vdupq_n_u8(3);
        const uint8x16_t c0 = vandq_u8(v, m);
        const uint8x16_t c1 = vandq_u8(vshrq_n_u8(v, 2), m);
        const uint8x16_t c2 = vandq_u8(vshrq_n_u8(v, 4), m);
        const uint8x16_t c3 = vshrq_n_u8(v, 6);      // top plane, already 2 bits
        const uint8x16_t p01a = vzip1q_u8(c0, c1), p01b = vzip2q_u8(c0, c1);
        const uint8x16_t p23a = vzip1q_u8(c2, c3), p23b = vzip2q_u8(c2, c3);
        out[0] = vreinterpretq_s8_u8(
            vzip1q_u16(vreinterpretq_u16_u8(p01a), vreinterpretq_u16_u8(p23a)));
        out[1] = vreinterpretq_s8_u8(
            vzip2q_u16(vreinterpretq_u16_u8(p01a), vreinterpretq_u16_u8(p23a)));
        out[2] = vreinterpretq_s8_u8(
            vzip1q_u16(vreinterpretq_u16_u8(p01b), vreinterpretq_u16_u8(p23b)));
        out[3] = vreinterpretq_s8_u8(
            vzip2q_u16(vreinterpretq_u16_u8(p01b), vreinterpretq_u16_u8(p23b)));
    }

    static void kqgemv_tq2_rows(const tmq::BlockTq2_34x128* W, const int8_t* xq,
                                const float* xs, int n_in, int o0, int o1,
                                float* y) {
        const std::uint64_t nbs =
            (std::uint64_t)(n_in / tmq::kBlockTq2_34x128);
        const int32x4_t z = vdupq_n_s32(0);
        for (int o = o0; o < o1; ++o) {
            const tmq::BlockTq2_34x128* r = W + (std::uint64_t)o * nbs;
            // Four independent float accumulators, reduced ONCE per row. The
            // first cut reduced inside the loop (a vaddvq per 16 lanes), which
            // is the ~6-cycle dependency the Q6_K kernel's own note warns
            // about and what nothing overlaps — it measured 16.7 GB/s
            // effective against Q6_K's 43. Here the int dot feeds a lane-wise
            // FMA and `sh` carries the (code - 1) shift as one fma per
            // 32-value group, so the whole row is one horizontal reduce.
            float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
            float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
            const int8x16_t one = vdupq_n_s8(1);
            for (std::uint64_t b = 0; b < nbs; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                const std::int8_t* xb = xq + b * tmq::kBlockTq2_34x128;
                const float* sb = xs + b * 4;   // one activation scale per 32 values
                for (int h = 0; h < 2; ++h) {
                    int8x16_t c[4];
                    tq2_unpack64(r[b].qs + 16 * h, c);
                    const std::int8_t* xh = xb + 64 * h;
                    // Each 16-lane plane lies inside one 32-value activation
                    // group, so its whole contribution scales by one number.
                    const float32x4_t sA = vdupq_n_f32(d * sb[2 * h]);
                    const float32x4_t sB = vdupq_n_f32(d * sb[2 * h + 1]);
                    // The (code - 1) shift is a lane-wise subtract BEFORE the
                    // dot, not a `- sum(x)` term after it. Same arithmetic
                    // (sum (code-1)*x == sum code*x - sum x) but it replaces a
                    // horizontal reduction plus a GPR->vector broadcast per
                    // 32-value group with one vsubq_s8 on already-live
                    // registers.
                    a0 = vfmaq_f32(a0, vcvtq_f32_s32(vdotq_s32(z, vsubq_s8(c[0], one), vld1q_s8(xh))), sA);
                    a1 = vfmaq_f32(a1, vcvtq_f32_s32(vdotq_s32(z, vsubq_s8(c[1], one), vld1q_s8(xh + 16))), sA);
                    a2 = vfmaq_f32(a2, vcvtq_f32_s32(vdotq_s32(z, vsubq_s8(c[2], one), vld1q_s8(xh + 32))), sB);
                    a3 = vfmaq_f32(a3, vcvtq_f32_s32(vdotq_s32(z, vsubq_s8(c[3], one), vld1q_s8(xh + 48))), sB);
                }
            }
            y[o] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        }
    }

    // The TQ2 kernel's real cost is the 2-bit -> int8 expansion, and the
    // expensive half of that is putting the four stride-4 planes back into
    // element order (eight vzip per 64 values). A dot only needs its two
    // operands in the SAME order, though — so the reshuffle can move off the
    // weight side entirely and be paid ONCE per call on the activation, which
    // all n_out rows then share. Same arithmetic; the permutation stops being
    // per-row.
    //
    // tq2_prepare lays the activation out so plane p of each 64-value chunk sits
    // contiguously: pq[chunk + p*16 + i] = x[4i + p]. The weight bytes need no
    // reordering at all, because the raw `v & 3`, `(v>>2)&3`, `(v>>4)&3`,
    // `v>>6` planes ARE that layout — which is what lets the kernel drop the
    // zips while still feeding vdotq_s32 two matching operands.
    static void tq2_prepare(const std::int8_t* xq, const float* xs, int n_in,
                            std::vector<std::int8_t>& pq, std::vector<float>& psv) {
        pq.resize((std::size_t)n_in);
        psv.resize((std::size_t)(n_in / 16) * 4);
        for (int b = 0; b < n_in; b += tmq::kBlockTq2_34x128) {
            for (int h = 0; h < 2; ++h) {
                const std::int8_t* src = xq + b + h * 64;
                std::int8_t* dst = pq.data() + b + h * 64;
                for (int p = 0; p < 4; ++p)
                    for (int i = 0; i < 16; ++i) dst[p * 16 + i] = src[4 * i + p];
                // Lane j of the vdot sums permuted positions 4j..4j+3, i.e. this
                // chunk's weights 16j..16j+3 — so lanes 0,1 live in the first
                // 32-value activation group and lanes 2,3 in the second. Two
                // scales in one vector, which is also why the kernel can fold
                // each pair of 16-lane dots before the int -> float convert.
                const int sub = b / tmq::kBlock + 2 * h;
                float* sv = psv.data() + (std::size_t)(b / 64 + h) * 4;
                sv[0] = sv[1] = xs[sub];
                sv[2] = sv[3] = xs[sub + 1];
            }
        }
    }

    // The zip-free TQ2 row kernel. Consumes tq2_prepare's layout: four planes
    // per 16-byte load dotted directly against the permuted activation, the two
    // dots of a 32-value group summed before the convert, and one fma per
    // 128-value block to apply the block's fp16 d. The (code - 1) shift stays a
    // lane-wise vsubq_s8 — with the codes already in registers it is four ops
    // per 64 values, cheaper than the `- sum(x)` alternative it replaced.
    static void kqgemv_tq2p_rows(const tmq::BlockTq2_34x128* W,
                                 const std::int8_t* pq, const float* psv, int n_in,
                                 int o0, int o1, float* y) {
        const std::uint64_t nbs = (std::uint64_t)(n_in / tmq::kBlockTq2_34x128);
        const int32x4_t z = vdupq_n_s32(0);
        const int8x16_t one = vdupq_n_s8(1);
        const uint8x16_t m3 = vdupq_n_u8(3);
        for (int o = o0; o < o1; ++o) {
            const tmq::BlockTq2_34x128* r = W + (std::uint64_t)o * nbs;
            float32x4_t acc = vdupq_n_f32(0.f);
            for (std::uint64_t b = 0; b < nbs; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                const std::int8_t* pb = pq + b * tmq::kBlockTq2_34x128;
                const float* psvb = psv + (std::size_t)b * 8;   // two 64-value chunks
                float32x4_t bacc = vdupq_n_f32(0.f);
                for (int h = 0; h < 2; ++h) {
                    const uint8x16_t v = vld1q_u8(r[b].qs + 16 * h);
                    const int8x16_t c0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m3)), one);
                    const int8x16_t c1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 2), m3)), one);
                    const int8x16_t c2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), m3)), one);
                    const int8x16_t c3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 6)), one);
                    const std::int8_t* p = pb + h * 64;
                    const int32x4_t s01 = vaddq_s32(vdotq_s32(z, c0, vld1q_s8(p)),
                                                    vdotq_s32(z, c1, vld1q_s8(p + 16)));
                    const int32x4_t s23 = vaddq_s32(vdotq_s32(z, c2, vld1q_s8(p + 32)),
                                                    vdotq_s32(z, c3, vld1q_s8(p + 48)));
                    const float32x4_t sv = vld1q_f32(psvb + 4 * h);
                    bacc = vfmaq_f32(bacc, vcvtq_f32_s32(s01), sv);
                    bacc = vfmaq_f32(bacc, vcvtq_f32_s32(s23), sv);
                }
                acc = vfmaq_f32(acc, bacc, vdupq_n_f32(d));
            }
            y[o] = vaddvq_f32(acc);
        }
    }

    // The block-scaled TQ2 path. The kernel above still carries the activation's
    // per-32-value scale through to the arithmetic, which costs one int->float
    // convert (and one fma) per 32 values — four per weight block. Quantizing the
    // activation with ONE scale per 128 values instead, matching the weight
    // block, lets every one of the block's eight int dots accumulate into a
    // single int32x4 that is converted ONCE. The price is a coarser activation
    // quantizer (amax over 128 values rather than 32), which is a numerics
    // change and is gated as one (check_tq2_block_gemv drives the same fp64
    // oracle, and the end-to-end hash is compared against the 32-value path).
    //
    // The (code - 1) shift also gets cheaper here: with one scale for the whole
    // block the correction is a scalar, sum(q) over the block, precomputed per
    // call. That replaces the eight lane-wise vsubq_s8 with one vector subtract,
    // and it is row-independent — every output row subtracts the same amount.
    static void tq2_prepare128(const float* x, int n_in, std::vector<std::int8_t>& pq,
                               std::vector<float>& pscale, std::vector<float>& psum) {
        const int nblocks = n_in / tmq::kBlockTq2_34x128;
        pq.resize((std::size_t)n_in);
        pscale.resize((std::size_t)nblocks);
        psum.resize((std::size_t)nblocks);
        for (int b = 0; b < nblocks; ++b) {
            const float* xb = x + (std::size_t)b * tmq::kBlockTq2_34x128;
            float amax = 0.f;
#if defined(__ARM_NEON)
            // vabsq_f32 and vmaxq_f32 are IEEE-exact (no rounding, no reordering
            // of a max), so this is BIT-IDENTICAL to the scalar reduction below
            // and only removes instructions: 8 vectors instead of 128 scalar
            // fabs+fmax pairs per block. The scalar form is kept for other targets.
            {
                float32x4_t vm = vdupq_n_f32(0.f);
                for (int i = 0; i < tmq::kBlockTq2_34x128; i += 4)
                    vm = vmaxq_f32(vm, vabsq_f32(vld1q_f32(xb + i)));
                amax = vmaxvq_f32(vm);
            }
#else
            for (int i = 0; i < tmq::kBlockTq2_34x128; ++i)
                amax = std::max(amax, std::fabs(xb[i]));
#endif
            if (!(amax == amax) || amax == std::numeric_limits<float>::infinity())
                throw std::invalid_argument("tq2: activation must be finite");
            // Same contract as quant_x: round the stored scale UP when a tiny
            // finite block underflows its reciprocal, so no value ever converts
            // through an infinity.
            float d = amax / 127.f;
            if (amax > 0.f && (d == 0.f || !std::isfinite(1.f / d)))
                d = std::nextafter(d, std::numeric_limits<float>::infinity());
            pscale[b] = d;
            const double inv = d > 0.f ? 1.0 / (double)d : 0.0;
            std::int8_t* qb = pq.data() + (std::size_t)b * tmq::kBlockTq2_34x128;
            std::int64_t sum = 0;
            #if defined(__ARM_NEON)
            // The scalar form below is `(double)xb[i] * inv` rounded with lrint.
            // f64 NEON reproduces that EXACTLY: vcvt_f64_f32 widens without
            // rounding, vmulq_f64 is the same double multiply, and vcvtnq_s64_f64
            // rounds to nearest with ties to EVEN — which is what lrint does under
            // the default FE_TONEAREST. The tempting f32 form is NOT safe: it
            // rounds the product to single precision first and can land on the
            // other side of a .5 boundary, so it would need the fp64 oracle and the
            // trajectory gate. The sum rides int32 lanes and is reduced with exact
            // integer adds, so it is order-independent and equals the scalar sum.
            {
                const float64x2_t vinv = vdupq_n_f64(inv);
                // The double -> int32 step is TWO instructions, not one. The direct
                // FCVTNS Vd.2S, Vn.2D intrinsic is spelled differently by each clang
                // generation — vcvtnq_s32_f64 under Apple's 17, the scalar-named
                // vcvtnd_s32_f64 under clang 22 — and neither spelling exists in BOTH
                // headers this engine is built against, so the portable pair is
                // vcvtnq_s64_f64 (FCVTNS to s64, ties to even) then vqmovn_s64. Exact
                // and identically rounded for every value in [-(2^31-1), 2^31-1];
                // ours are bounded by 127 by construction.
                const auto rnd = [&](float64x2_t v) {
                    return vqmovn_s64(vcvtnq_s64_f64(vmulq_f64(v, vinv)));
                };
                int32x4_t acc = vdupq_n_s32(0);
                for (int i = 0; i < tmq::kBlockTq2_34x128; i += 8) {
                    const float32x4_t lo = vld1q_f32(xb + i);
                    const float32x4_t hi = vld1q_f32(xb + i + 4);
                    const int32x4_t r0 = vcombine_s32(rnd(vcvt_f64_f32(vget_low_f32(lo))),
                                                      rnd(vcvt_f64_f32(vget_high_f32(lo))));
                    const int32x4_t r1 = vcombine_s32(rnd(vcvt_f64_f32(vget_low_f32(hi))),
                                                      rnd(vcvt_f64_f32(vget_high_f32(hi))));
                    acc = vaddq_s32(acc, vaddq_s32(r0, r1));
                    vst1_s8(qb + i, vqmovn_s16(vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1))));
                }
                sum = vaddlvq_s32(acc);
            }
            #else
            for (int i = 0; i < tmq::kBlockTq2_34x128; ++i) {
                const std::int8_t v = (std::int8_t)std::lrint((double)xb[i] * inv);
                qb[i] = v;
                sum += v;
            }
            #endif
            psum[b] = (float)sum;
            // Then permute into the kernel's extraction order (the sum is over
            // the whole block, so it is order-independent and taken first).
            #if defined(__ARM_NEON)
            // dest[h*64 + p*16 + i] = src[h*64 + 4*i + p] is exactly the 4-way byte
            // deinterleave vld4q_s8 performs on the way in: val[p][i] == src[4*i+p].
            // Ten instructions replace what the tmp[] loop cost — 128 loads + 128
            // stores + a 128-byte memcpy per block, times 253 blocks a token.
            // In place is safe: vld4q reads all 64 bytes before any store lands.
            for (int h = 0; h < 2; ++h) {
                std::int8_t* qh = qb + h * 64;
                const int8x16x4_t q = vld4q_s8(qh);
                vst1q_s8(qh, q.val[0]);
                vst1q_s8(qh + 16, q.val[1]);
                vst1q_s8(qh + 32, q.val[2]);
                vst1q_s8(qh + 48, q.val[3]);
            }
            #else
            std::int8_t tmp[128];
            for (int h = 0; h < 2; ++h)
                for (int p = 0; p < 4; ++p)
                    for (int i = 0; i < 16; ++i)
                        tmp[h * 64 + p * 16 + i] = qb[h * 64 + 4 * i + p];
            std::memcpy(qb, tmp, sizeof tmp);
        #endif
        }
    }

    static void kqgemv_tq2b_rows(const tmq::BlockTq2_34x128* W,
                                 const std::int8_t* pq, const float* pscale,
                                 const float* psum, int n_in, int o0, int o1,
                                 float* y) {
        const std::uint64_t nbs = (std::uint64_t)(n_in / tmq::kBlockTq2_34x128);
        const int32x4_t z = vdupq_n_s32(0);
        const uint8x16_t m3 = vdupq_n_u8(3);
        for (int o = o0; o < o1; ++o) {
            const tmq::BlockTq2_34x128* r = W + (std::uint64_t)o * nbs;
            float32x4_t acc = vdupq_n_f32(0.f);
            // The (code - 1) shift is a SCALAR here, not a lane-wise subtract:
            // subtracting psum from all four lanes and then reducing with
            // vaddvq would count the correction four times. The gate caught
            // exactly that (it came out 4x too large). One scalar fma per block
            // is also cheaper than a vector subtract plus a broadcast.
            float corr = 0.f;
            for (std::uint64_t b = 0; b < nbs; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                const std::int8_t* pb = pq + b * tmq::kBlockTq2_34x128;
                // One accumulator per block. Splitting it into four independent
                // chains was tried and measured NEUTRAL (0.568 -> 0.572 ms on
                // gate_proj, 8.640 -> 8.777 on embed — noise), so the vdot
                // dependency chain is not what bounds this loop; keeping the
                // single accumulator keeps the block cheaper to read.
                int32x4_t s = z;
                for (int h = 0; h < 2; ++h) {
                    const uint8x16_t v = vld1q_u8(r[b].qs + 16 * h);
                    const std::int8_t* p = pb + h * 64;
                    s = vaddq_s32(s, vdotq_s32(z, vreinterpretq_s8_u8(vandq_u8(v, m3)), vld1q_s8(p)));
                    s = vaddq_s32(s, vdotq_s32(z, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 2), m3)), vld1q_s8(p + 16)));
                    s = vaddq_s32(s, vdotq_s32(z, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), m3)), vld1q_s8(p + 32)));
                    s = vaddq_s32(s, vdotq_s32(z, vreinterpretq_s8_u8(vshrq_n_u8(v, 6)), vld1q_s8(p + 48)));
                }
                const float dxs = d * pscale[b];
                corr += dxs * psum[b];
                acc = vfmaq_f32(acc, vcvtq_f32_s32(s), vdupq_n_f32(dxs));
            }
            y[o] = vaddvq_f32(acc) - corr;
        }
    }

    // Q4_1: value = code*d + m, 32 values per block, low nibble = element j
    // and high nibble = element j+16 (ggml layout, transcribed in quant.h).
    static void kqgemv_q41_rows(const tmq::BlockQ4_1* W, const int8_t* xq,
                                const float* xs, int n_in, int o0, int o1,
                                float* y) {
        const std::uint64_t nb = (std::uint64_t)(n_in / tmq::kBlock);
        const uint8x16_t m4 = vdupq_n_u8(0x0F);
        const int32x4_t z = vdupq_n_s32(0);
        for (int o = o0; o < o1; ++o) {
            const tmq::BlockQ4_1* r = W + (std::uint64_t)o * nb;
            float acc = 0.f;
            for (std::uint64_t b = 0; b < nb; ++b) {
                const std::int8_t* xb = xq + b * tmq::kBlock;
                const uint8x16_t raw = vld1q_u8(r[b].qs);
                const int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, m4));
                const int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
                const std::int32_t dot =
                    vaddvq_s32(vdotq_s32(z, lo, vld1q_s8(xb))) +
                    vaddvq_s32(vdotq_s32(z, hi, vld1q_s8(xb + 16)));
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                const float m = tmq::fp16_to_fp32(r[b].m_fp16);
                acc += xs[b] * (d * (float)dot + m * (float)kq_sum32(xb));
            }
            y[o] = acc;
        }
    }

    // Q5_K: value = d1*code - m1 with d1 = d*sc, m1 = dmin*mn, 8 sub-blocks
    // of 32 per 256-value super-block. Codes are 5 bits: the low 4 in qs (low
    // nibble for the first group of each 64, high for the second) and the
    // 5th in qh, one bit per 64-value step.
    //
    // SH is the qh bit (2*g + h) and HI picks the qs nibble; both are template
    // parameters so the extraction stays in immediate-shift form, the same
    // reason kq6k_dot16 is templated on J.
    template <int SH, bool HI>
    static std::int32_t kq5k_dot16(const std::uint8_t* ql, const uint8x16_t qh,
                                   const std::int8_t* x) {
        const uint8x16_t raw = vld1q_u8(ql);
        const uint8x16_t nib = HI ? vshrq_n_u8(raw, 4)
                                  : vandq_u8(raw, vdupq_n_u8(0x0F));
        uint8x16_t sh = qh;
        if constexpr (SH > 0) sh = vshrq_n_u8(qh, SH);
        const uint8x16_t hb = vandq_u8(sh, vdupq_n_u8(1));
        const int8x16_t code =
            vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(hb, 4)));
        return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), code, vld1q_s8(x)));
    }

    // One 32-value Q5_K group: the 5-bit dot plus the affine bias, folded with
    // the sub-block scale/min and the activation scale.
    template <int SH, bool HI>
    static float kq5k_group32(const tmq::BlockQ5_K& blk, const uint8x16_t qha,
                              const uint8x16_t qhb, const std::int8_t* x,
                              const float* xsb, int sub, float d, float dmin) {
        std::uint8_t sc, mn;
        tmq::k_scale_min(sub, blk.scales, sc, mn);
        const std::uint8_t* ql = blk.qs + 32 * (SH / 2);
        const std::int32_t dot = kq5k_dot16<SH, HI>(ql, qha, x) +
                                 kq5k_dot16<SH, HI>(ql + 16, qhb, x + 16);
        return xsb[sub] * (d * (float)sc * (float)dot -
                           dmin * (float)mn * (float)kq_sum32(x));
    }

    // One 64-value step: group A (low nibble, bit 2g) then group B (high
    // nibble, bit 2g+1). Both qh vectors are loaded once and reused.
    template <int G>
    static float kq5k_step64(const tmq::BlockQ5_K& blk, const std::int8_t* xg,
                             const float* xsb, float d, float dmin) {
        const uint8x16_t qha = vld1q_u8(blk.qh);
        const uint8x16_t qhb = vld1q_u8(blk.qh + 16);
        return kq5k_group32<2 * G + 0, false>(blk, qha, qhb, xg, xsb,
                                              2 * G + 0, d, dmin) +
               kq5k_group32<2 * G + 1, true>(blk, qha, qhb, xg + 32, xsb,
                                             2 * G + 1, d, dmin);
    }

    static void kqgemv_q5k_rows(const tmq::BlockQ5_K* W, const int8_t* xq,
                                const float* xs, int n_in, int o0, int o1,
                                float* y) {
        const std::uint64_t nbs = (std::uint64_t)(n_in / tmq::kKBlock);
        for (int o = o0; o < o1; ++o) {
            const tmq::BlockQ5_K* r = W + (std::uint64_t)o * nbs;
            float acc = 0.f;
            for (std::uint64_t b = 0; b < nbs; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                const float dmin = tmq::fp16_to_fp32(r[b].dmin_fp16);
                const std::int8_t* xb = xq + b * tmq::kKBlock;
                const float* xsb = xs + (b * tmq::kKBlock) / tmq::kBlock;
                acc += kq5k_step64<0>(r[b], xb + 0, xsb, d, dmin);
                acc += kq5k_step64<1>(r[b], xb + 64, xsb, d, dmin);
                acc += kq5k_step64<2>(r[b], xb + 128, xsb, d, dmin);
                acc += kq5k_step64<3>(r[b], xb + 192, xsb, d, dmin);
            }
            y[o] = acc;
        }
    }

    // Dispatch a K-quant row range on its stored dtype.
    static void kqgemv_rows(const void* W, std::uint32_t dtype, const int8_t* xq,
                            const float* xs, int n_in, int o0, int o1, float* y) {
        switch (dtype) {
            case 4u: kqgemv_q41_rows(static_cast<const tmq::BlockQ4_1*>(W), xq, xs, n_in, o0, o1, y); return;
            case 5u: kqgemv_q5k_rows(static_cast<const tmq::BlockQ5_K*>(W), xq, xs, n_in, o0, o1, y); return;
            case 6u: kqgemv_q6k_rows(static_cast<const tmq::BlockQ6_K*>(W), xq, xs, n_in, o0, o1, y); return;
            case 7u: kqgemv_tq2_rows(static_cast<const tmq::BlockTq2_34x128*>(W), xq, xs, n_in, o0, o1, y); return;
            default: throw std::invalid_argument("llama: unsupported k-quant dtype");
        }
    }

    // --- Multi-position K-quant rows (2 <= T < prefill_loop_t) ----------------
    // The same arithmetic as kqgemv_q6k_rows with the two loops EXCHANGED, so a
    // weight block is unpacked once and then dotted against every position.
    // What that buys: the unpack is ~8 instructions per 16 values against one
    // vdotq_s32, so at T=1 the kernel spends most of its issue slots rebuilding
    // codes the activation does not enter — and the measured ceiling (~19 GB/s
    // effective, identically on 0.65 GB and 3.8 GB models, with a 4x pool/serial
    // scaling) says that instruction cost, not bandwidth, is what the row loop is
    // up against. Here it is paid once per T positions.
    // What it does NOT buy: the 16 sub-block factors of a super-block can no
    // longer sit in registers once T > 1 (that is 16 vectors per position), so
    // they are re-derived per position. The win is the unpack, not a free 1/T.
    //
    // xq/xs hold the T quantized activations: slot t at + t*n_in (scales at
    // + t*n_in/kBlock). Slot-major, i.e. y[t*n_out + o] on the way out.
    template <int T>
    static void kq6k_multi_rows(const tmq::BlockQ6_K* W, const int8_t* xq,
                                const float* xs, int n_in, int n_out, int o0,
                                int o1, float* y) {
        const std::uint64_t nbs = (std::uint64_t)(n_in / tmq::kKBlock);
        const std::int64_t nsb = (std::int64_t)(n_in / tmq::kBlock);
        for (int o = o0; o < o1; ++o) {
            const tmq::BlockQ6_K* r = W + (std::uint64_t)o * nbs;
            float32x4_t acc[T][2];
            for (int t = 0; t < T; ++t) {
                acc[t][0] = vdupq_n_f32(0.f);
                acc[t][1] = vdupq_n_f32(0.f);
            }
            for (std::uint64_t b = 0; b < nbs; ++b) {
                const std::uint8_t* ql = r[b].ql;
                const std::uint8_t* qh = r[b].qh;
                const std::int8_t* sc = r[b].scales;
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                const std::int64_t xbase = (std::int64_t)b * tmq::kKBlock;
                const std::int64_t sbase = xbase / tmq::kBlock;
                for (int half = 0; half < 2; ++half) {
                    const std::uint8_t* qlh = ql + half * 64;
                    const std::uint8_t* qhh = qh + half * 32;
                    const uint8x16_t qha = vld1q_u8(qhh);
                    const uint8x16_t qhb = vld1q_u8(qhh + 16);
                    // The eight 16-value segments of this half, unpacked ONCE for
                    // all T positions. This is the loop exchange.
                    int8x16_t Q[8];
                    Q[0] = kq6k_unpack16<0>(qlh, qha);
                    Q[1] = kq6k_unpack16<1>(qlh, qha);
                    Q[2] = kq6k_unpack16<2>(qlh, qha);
                    Q[3] = kq6k_unpack16<3>(qlh, qha);
                    Q[4] = kq6k_unpack16<0>(qlh + 16, qhb);
                    Q[5] = kq6k_unpack16<1>(qlh + 16, qhb);
                    Q[6] = kq6k_unpack16<2>(qlh + 16, qhb);
                    Q[7] = kq6k_unpack16<3>(qlh + 16, qhb);
                    for (int t = 0; t < T; ++t) {
                        const std::int8_t* x = xq + (std::size_t)t * n_in + xbase +
                                               half * 128;
                        // Sub-block is spans values [16*is, 16*is+16) and an
                        // activation block is 32 values, so BOTH sub-blocks of a
                        // segment pair share one activation scale: sub-block
                        // half*8 + {2s, 2s+1} reads activation block half*4 + s.
                        const float* fa = xs + (std::size_t)t * nsb + sbase + half * 4;
                        // Two accumulator chains per position (the segments with
                        // sub-block scale 2s and 2s+1), the same split the per-row
                        // kernel makes — four chains, one per 16-value row-group,
                        // is what keeps the vdot/FMA pipes busy rather than the
                        // horizontal-reduce dependency.
                        float32x4_t a0 = acc[t][0], a1 = acc[t][1];
                        for (int s = 0; s < 4; ++s) {
                            const float as = fa[s];
                            const float f0 = d * (float)sc[half * 8 + 2 * s] * as;
                            const float f1 = d * (float)sc[half * 8 + 2 * s + 1] * as;
                            a0 = vfmaq_n_f32(
                                a0, vcvtq_f32_s32(
                                        vdotq_s32(vdupq_n_s32(0), Q[s], vld1q_s8(x))),
                                f0);
                            a1 = vfmaq_n_f32(
                                a1,
                                vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), Q[4 + s],
                                                        vld1q_s8(x + 16))),
                                f1);
                            x += 32;
                        }
                        acc[t][0] = a0;
                        acc[t][1] = a1;
                    }
                }
            }
            for (int t = 0; t < T; ++t)
                y[(std::size_t)t * n_out + o] =
                    vaddvq_f32(vaddq_f32(acc[t][0], acc[t][1]));
        }
    }

    // Adapter for the pool: its kernel signature carries ONE activation vector, so
    // the T-slot buffer and the position count travel in the job's weight pointer
    // (the same place the per-row wrapper puts the dtype). dispatch() completes
    // before the publishing frame returns, so the descriptor outlives the job.
    struct KQMultiW {
        const void* blocks;
        const int8_t* xq;
        const float* xs;
        int n_in;
        int n_out;
        int T;
    };
    static void kq_multi_kern(const GemvArgs& a, int o0, int o1) {
        const auto* k = static_cast<const KQMultiW*>(a.w);
        const auto* B = static_cast<const tmq::BlockQ6_K*>(k->blocks);
        float* y = a.y.data();
        switch (k->T) {
            case 2: kq6k_multi_rows<2>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            case 3: kq6k_multi_rows<3>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            case 4: kq6k_multi_rows<4>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            case 5: kq6k_multi_rows<5>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            case 6: kq6k_multi_rows<6>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            case 7: kq6k_multi_rows<7>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            case 8: kq6k_multi_rows<8>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            case 16: kq6k_multi_rows<16>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            case 1: kq6k_multi_rows<1>(B, k->xq, k->xs, k->n_in, k->n_out, o0, o1, y); return;
            default: return;   // T is admitted by gemm_kq_multi; nothing else arrives
        }
    }
#endif

    mutable XQ xq_;             // sdot decode: quantized x (amortized/pass)
    mutable std::vector<float> kq_deq_;   // non-NEON K-quant GEMV scratch
    // TQ2 (dtype 7) decode scratch, rebuilt once per GEMV call and shared by
    // every output row: `tq2_pq_` is the activation in the layout the kernel
    // extracts codes in (see tq2_prepare), and `tq2_psv_` holds the per-64-value
    // float scale vectors (four lanes each). Both are O(n_in); the weight stream
    // is O(n_out * n_in), which is why permuting the ACTIVATION is the whole
    // trick: it deletes the per-row code reshuffle.
    mutable std::vector<std::int8_t> tq2_pq_;
    mutable std::vector<float> tq2_psv_;
    // Block-scaled TQ2 path (tq2_prepare128): one activation scale per 128
    // values, matching the weight block, plus that block's integer sum so the
    // (code - 1) shift is a scalar correction rather than a lane-wise subtract.
    mutable std::vector<float> tq2_pscale_;
    mutable std::vector<float> tq2_psum_;
    // T-slot quantized activations for the multi-position K-quant path: slot t
    // lives at q + t*n_in and its scales at s + t*(n_in/kBlock). Same reason as
    // xq_ (one buffer per instance: a decode never nests).
    mutable XQ xq_multi_;

    // Multi-position K-quant projection: y[t] = W x[t] for every t, with each
    // weight block unpacked once and applied to every position
    // (kq6k_multi_rows). Returns false when the path does not apply.
    //
    // Why: `linear_proj`'s tail for 2 <= T < prefill_loop_t() calls gemv_w once
    // per position, and every one of those calls re-reads and re-unpacks the
    // whole weight row. The measured position-scaling curve (docs/BENCHMARKS.md)
    // shows a k-position pass costing more than a plain step per position — and
    // identically on a dense Q4_0 model as on this hybrid K-quant one, so the cost
    // is the loop order, not the weights' dtype. Exchanging the two loops is the
    // fix, and it reuses the fast path's arithmetic rather than a substitute for
    // it: the FIRST attempt here dequantized a band of rows to fp32 and handed it
    // to cblas_sgemm, which measured 0.17x the plain decode (spec 0.17x against
    // 0.73x for the per-row loop) — it traded int8 sdot for a fp32 dequant plus a
    // tiny-M gemm that Accelerate packs internally, twice the useful bytes moved
    // per position. Reuse the fast path; do not replace it.
    //
    // Numerics: the same int8 dots and the same FMA tree as the per-row kernel,
    // so logits agree to fp32 reassociation. The speculative loop asserts an
    // identical token stream, which is the check that decides if that matters.
    bool gemm_kq_multi(const std::string& name, const float* x, int T, int in,
                       int out, float* y) const {
        if (T < 2 || out <= 0 || in <= 0 || in % tmq::kKBlock != 0)
            return false;
        const auto it = qk_.find(name);
        if (it == qk_.end()) return false;
        if (it->second.dtype != 6u) return false;   // Q6_K only (see kq_multi_kern)
#ifdef __ARM_NEON
        const bool serial = gemv_switches().serial;
        const std::size_t nsb = (std::size_t)(in / tmq::kBlock);   // scales per slot
        const std::size_t nq = (std::size_t)T * (std::size_t)in;
        const std::size_t nst = nsb * (std::size_t)T;
        if (xq_multi_.q.size() < nq) {
            xq_multi_.q.resize(nq);
            // quant_x_into writes both layouts in one pass; the K-quant kernels
            // read q and s only, so `d` is just the price of sharing that code.
            xq_multi_.d.resize(nq);
        }
        if (xq_multi_.s.size() < nst) xq_multi_.s.resize(nst);
        for (int t = 0; t < T; ++t)
            quant_x_into(x + (std::size_t)t * in, in,
                         xq_multi_.q.data() + (std::size_t)t * in,
                         xq_multi_.d.data() + (std::size_t)t * in,
                         xq_multi_.s.data() + (std::size_t)t * nsb);
        // Chunked at the template bound: a wider pass needs the per-position
        // accumulator array AND the unpacked segment registers live at once.
        // ceil(T/chunk) chunks is NOT one arbitrarily wide pass — the weight stream
        // is re-read per chunk, so the cost per position is (unpack/chunk + dot)
        // rather than (unpack/T + dot). It still carries the per-chunk kernel factor
        // (2.38x at 8, measured) over the row loop, and this is what serves 9..63
        // positions, which otherwise fall to the row loop at ~90 ms/position — the
        // ordinary chat prompt range.
        //
        // Width 16 by default: measured against two 8-wide chunks on the same
        // weights and activations, 1.529 vs 1.709 ms per position at T=16 with
        // BIT-IDENTICAL output (the accumulators are per position, so widening only
        // halves how often the weight stream is read). Strictly less work and equal
        // arithmetic, so it cannot regress; TM_KQ_CHUNK=8 forces the narrow width
        // for an A/B. The tail is always finished in <= 8, so a remainder of 9..15
        // can never land on a width the dispatch does not implement — that would
        // silently write nothing for those positions instead of failing.
        const int chunk = gemv_switches().kq_chunk;
        for (int c = 0; c < T;) {
            const int rem = T - c;
            const int tc = (chunk == 16 && rem >= 16) ? 16 : std::min(8, rem);
            KQMultiW kw;
            kw.blocks = it->second.blocks;
            kw.xq = xq_multi_.q.data() + (std::size_t)c * in;
            kw.xs = xq_multi_.s.data() + (std::size_t)c * nsb;
            kw.n_in = in;
            kw.n_out = out;
            kw.T = tc;
            GemvJob j;
            // One chunk of tc token rows: y's extent is THIS chunk's tc*out
            // (the caller advanced the base by c*out), x/xq/xs extents tc*in.
            j.a = GemvArgs{.w = &kw,
                           .x = std::span(x + (std::size_t)c * in, (std::size_t)tc * in),
                           .xq = std::span(kw.xq, (std::size_t)tc * in),
                           .xs = std::span(kw.xs, (std::size_t)tc * nsb),
                           .y = std::span(y + (std::size_t)c * out, (std::size_t)tc * out),
                           .n_in = in};
            j.n_out = out;
            j.kern = &kq_multi_kern;
            if (serial)
                j.kern(j.a, 0, j.n_out);
            else
                gemv_pool().dispatch(j);
            c += tc;
        }
        return true;
#else
        (void)x; (void)y;
        return false;   // the multi kernels are NEON; the per-row loop covers this
#endif
    }

    // Timed activation encoding. Every sdot GEMV needs x as int8 blocks; the
    // timing is a subset slot so the profile can separate the encoding from the
    // weight stream it feeds.
    void quant_x_timed(const float* x, int n_in) const {
        ProfGuard _qx(LlamaProf::Sec::QUANT_X);
        quant_x(x, n_in, xq_);
    }
    void gemv_w(const std::string& name, const float* x, int n_in,
                int n_out, float* y) const {
        // Subset timer (see LlamaProf::Sec::GEMV_CORE): everything from here to
        // return, including the activation quantization and the pool wait.
        ProfGuard _gemv_guard(LlamaProf::Sec::GEMV_CORE);
        // Hoisted: this is the innermost weight dispatch — once per projection, per layer, per
        // token — and getenv() walks the environment array on every call, so reading these per
        // call puts the switches on the path they are meant to be observing. Correctness of
        // shape, not a claimed speed-up: at this scale the cost is below a paired run's noise.
        const bool sdot_on = gemv_switches().sdot;
        const bool serial_gemv = gemv_switches().serial;
        // f16 weights never reach the sdot/q4 kernels below; the fused
        // dequant+GEMV covers every direct caller (embedding, lm_head, conv).
        if (f16_.count(name)) { gemv_f16(name, x, n_in, n_out, y); return; }
        // TQ2 (dtype 7) gets its own dispatch because it needs the activation in
        // tq2_prepare's plane layout, not the element-order one the other sdot
        // kernels read. The permuted buffer is rebuilt here, once per call:
        // O(n_in) against the O(n_out * n_in) weight stream. kqgemv_rows' case 7
        // (the zip form) stays as the reference the regression gate compares
        // against, and as the non-sdot fallback below.
        if (const auto tq = qk_.find(name);
            tq != qk_.end() && tq->second.dtype == 7u) {
#ifdef __ARM_NEON
            // TM_TQ2_ZIP=1 forces the older zip kernel. It exists so the two can
            // be compared PAIRED (A B B A) in one binary: the per-tensor spread on
            // this box is large enough that two unpaired runs decided nothing.
            static const int tq2_mode = [] {
                const char* z = std::getenv("TM_TQ2_ZIP");
                if (z && z[0] == '1') return 2;
                const char* p = std::getenv("TM_TQ2_PER32");
                return (p && p[0] == '1') ? 1 : 0;
            }();
            if (sdot_on && tq2_mode != 2) {
                struct TQ2W {
                    const void* blocks;
                    const std::int8_t* pq;
                    const float* sa;
                    const float* sb;   // block sums; null on the per-32 path
                    int mode;
                };
                if (tq2_mode == 0) {
                    // One activation scale per 128 values, so the block's eight
                    // int dots share a single convert. Skips quant_x entirely —
                    // but it is NOT free, and unlike every other dtype's encoding
                    // it was invisible in the profile: it runs per TENSOR (253 of
                    // them a token) and consecutive projections read the SAME
                    // activation. Charged to the QUANT_X slot so the cost is
                    // measured rather than assumed; the guard is free when the
                    // profile is off.
                    ProfGuard _tq2_prep(LlamaProf::Sec::QUANT_X);
                    tq2_prepare128(x, n_in, tq2_pq_, tq2_pscale_, tq2_psum_);
                } else {
                    quant_x_timed(x, n_in);
                    tq2_prepare(xq_.q.data(), xq_.s.data(), n_in, tq2_pq_, tq2_psv_);
                }
                const TQ2W tw{tq->second.blocks, tq2_pq_.data(),
                              tq2_mode == 0 ? tq2_pscale_.data() : tq2_psv_.data(),
                              tq2_mode == 0 ? tq2_psum_.data() : nullptr,
                              tq2_mode};
                const std::size_t nsc = tq2_mode == 0 ? (std::size_t)(n_in / 128)
                                                      : (std::size_t)(n_in / 16) * 4;
                GemvJob j;
                j.a = GemvArgs{.w = &tw,
                               .x = std::span(x, (std::size_t)n_in),
                               .xq = std::span(tq2_pq_.data(), (std::size_t)n_in),
                               .xs = std::span(tw.sa, nsc),
                               .y = std::span(y, (std::size_t)n_out),
                               .n_in = n_in};
                j.n_out = n_out;
                j.kern = [](const GemvArgs& a, int o0, int o1) {
                    const auto* p = static_cast<const TQ2W*>(a.w);
                    const auto* W =
                        static_cast<const tmq::BlockTq2_34x128*>(p->blocks);
                    if (p->mode == 0)
                        kqgemv_tq2b_rows(W, p->pq, p->sa, p->sb, a.n_in, o0, o1, a.y.data());
                    else
                        kqgemv_tq2p_rows(W, p->pq, p->sa, a.n_in, o0, o1, a.y.data());
                };
                if (serial_gemv)
                    j.kern(j.a, 0, j.n_out);
                else
                    gemv_pool().dispatch(j);
                return;
            }
#endif
        }
        // K-quant weights (Q4_1/Q5_K/Q6_K) ride the same int8 activation
        // quantization as the Q4_Q8 sdot path; only the block unpack differs.
        // The pool's kernel is a plain function pointer, so the dtype travels
        // in a wrapper the pointer refers to — `dispatch()` completes before
        // this frame returns, so the local outlives the job.
        if (const auto kq = qk_.find(name); kq != qk_.end()) {
            struct KQW { const void* blocks; std::uint32_t dtype; };
            const KQW kqw{kq->second.blocks, kq->second.dtype};
#ifdef __ARM_NEON
            if (sdot_on) {
                quant_x_timed(x, n_in);
                GemvJob j;
                j.a = GemvArgs{.w = &kqw,
                               .x = std::span(x, (std::size_t)n_in),
                               .xq = std::span(xq_.q.data(), (std::size_t)n_in),
                               .xs = std::span(xq_.s.data(), (std::size_t)n_in / tmq::kBlock),
                               .xd = std::span(xq_.d.data(), (std::size_t)n_in),
                               .y = std::span(y, (std::size_t)n_out),
                               .n_in = n_in};
                j.n_out = n_out;
                j.kern = [](const GemvArgs& a, int o0, int o1) {
                    const auto* p = static_cast<const KQW*>(a.w);
                    kqgemv_rows(p->blocks, p->dtype, a.xq.data(), a.xs.data(),
                                a.n_in, o0, o1, a.y.data());
                };
                if (gemv_switches().serial)
                    j.kern(j.a, 0, j.n_out);
                else
                    gemv_pool().dispatch(j);
                return;
            }
#endif
            // No NEON (or TM_SDOT=0): dequantize the whole tensor into scratch
            // once and run a dense GEMV. This is the correctness path, not the
            // fast one — it costs a full fp32 copy of the tensor.
            const std::size_t nk = (std::size_t)n_out * (std::size_t)n_in;
            if (kq_deq_.size() < nk) kq_deq_.resize(nk);
            for (int o = 0; o < n_out; ++o)
                dequant_row_kq(kqw.blocks, kqw.dtype, n_in, o, kq_deq_.data() + (std::size_t)o * n_in);
            cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.f,
                        kq_deq_.data(), n_in, x, 1, 0.f, y, 1);
            return;
        }
        const bool q8 = q8_.count(name) != 0;
#ifdef __ARM_NEON
        const bool sdot = gemv_switches().sdot;
        if (sdot) {
            quant_x_timed(x, n_in);
            GemvJob j;
            j.a = GemvArgs{.x = std::span(x, (std::size_t)n_in),
                           .xq = std::span(xq_.q.data(), (std::size_t)n_in),
                           .xs = std::span(xq_.s.data(), (std::size_t)n_in / tmq::kBlock),
                           .xd = std::span(xq_.d.data(), (std::size_t)n_in),
                           .y = std::span(y, (std::size_t)n_out),
                           .n_in = n_in};
            if (q8) {
                j.a.w = q8_.at(name);
                j.kern = [](const GemvArgs& a, int o0, int o1) {
                    qgemv_sdot_rows(static_cast<const tmq::BlockQ8_0*>(a.w),
                                    a.xq.data(), a.xd.data(), a.xs.data(),
                                    a.n_in, o0, o1, a.y.data());
                };
            } else {
                j.a.w = q4_.at(name);
                j.kern = [](const GemvArgs& a, int o0, int o1) {
                    qgemv_sdot_rows(static_cast<const tmq::BlockQ4_0*>(a.w),
                                    a.xq.data(), a.xd.data(), a.xs.data(),
                                    a.n_in, o0, o1, a.y.data());
                };
            }
            j.n_out = n_out;
            if (serial_gemv)
                j.kern(j.a, 0, j.n_out);
            else
                gemv_pool().dispatch(j);
            return;
        }
#endif
        if (serial_gemv) {
            if (q8) qgemv_rows(q8_.at(name), x, n_in, 0, n_out, y);
            else    qgemv_rows(q4_.at(name), x, n_in, 0, n_out, y);
            return;
        }
        GemvJob j;
        j.a = GemvArgs{.x = std::span(x, (std::size_t)n_in),
                       .y = std::span(y, (std::size_t)n_out),
                       .n_in = n_in};
        j.n_out = n_out;
        if (q8) {
            j.a.w = q8_.at(name);
            j.kern = [](const GemvArgs& a, int o0, int o1) {
                qgemv_rows(static_cast<const tmq::BlockQ8_0*>(a.w), a.x.data(),
                           a.n_in, o0, o1, a.y.data());
            };
        } else {
            j.a.w = q4_.at(name);
            j.kern = [](const GemvArgs& a, int o0, int o1) {
                qgemv_rows(static_cast<const tmq::BlockQ4_0*>(a.w), a.x.data(),
                           a.n_in, o0, o1, a.y.data());
            };
        }
        gemv_pool().dispatch(j);
    }
    static bool grouped_gemv_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("TM_GEMV_GROUP");
            return !value || value[0] != '0';
        }();
        return enabled;
    }

    // Whether the grouped path can actually serve THIS model. It resolves its
    // weights through q4_/q8_ only — there is no K-quant kernel behind it — and
    // a K-quant tensor is registered in qk_ and in neither of those two. A model
    // carrying one therefore throws "missing grouped GEMV weight" on its first
    // grouped call: Qwen3.8-4B converted at Q6_K decoded exactly one token and
    // then lost the turn. The ungrouped path does have the K-quant kernels, so
    // route there instead of failing. Conservative by design — a single K-quant
    // tensor anywhere turns grouping off for the whole model, which costs some
    // throughput and never a wrong answer.
    bool grouped_gemv_ok() const { return qk_.empty() && grouped_gemv_enabled(); }

    // Shared-input projections: quantize once and publish one pool generation.
    // Each output retains its own row coordinates; no packed output allocation.
    // Weight maps must have matching qshape_ metadata (out, in), as load_quant
    // installs. Callers still own input/output capacity and weight lifetime.
    template <std::size_t N>
    void gemv_w_group(const std::string (&names)[N], const int (&rows)[N],
                      float* const (&outputs)[N], const float* x, int n_in) const {
        static_assert(N > 0);
        if (n_in <= 0 || n_in % tmq::kBlock || !x)
            throw std::invalid_argument("invalid grouped GEMV input");
        struct Projection { const void* weights; float* output; int rows; bool q8; };
        std::array<Projection, N> projections;
        int total = 0;
        for (std::size_t i = 0; i < N; ++i) {
            if (rows[i] < 0 || rows[i] > std::numeric_limits<int>::max() - total ||
                (rows[i] > 0 && !outputs[i]))
                throw std::invalid_argument("invalid grouped GEMV output");
            const auto shape = qshape_.find(names[i]);
            if (shape == qshape_.end() || shape->second.first < rows[i] ||
                shape->second.second != n_in)
                throw std::invalid_argument("grouped GEMV weight shape mismatch");
            const auto q8_it = q8_.find(names[i]);
            const bool q8 = q8_it != q8_.end();
            const void* weights = nullptr;
            if (q8) {
                weights = q8_it->second;
            } else {
                const auto q4_it = q4_.find(names[i]);
                if (q4_it == q4_.end())
                    throw std::invalid_argument("missing grouped GEMV weight");
                weights = q4_it->second;
            }
            if (rows[i] > 0 && !weights)
                throw std::invalid_argument("null grouped GEMV weight");
            projections[i] = {weights, outputs[i], rows[i], q8};
            total += rows[i];
        }
        if (total == 0) return;
        GemvJob job;
        job.a = GemvArgs{.w = projections.data(),
                         .x = std::span(x, (std::size_t)n_in),
                         .n_in = n_in};
        job.n_out = total;
#ifdef __ARM_NEON
        const bool sdot = gemv_switches().sdot;
        if (sdot) {
            quant_x_timed(x, n_in);
            job.a.xq = std::span(xq_.q.data(), (std::size_t)n_in);
            job.a.xd = std::span(xq_.d.data(), (std::size_t)n_in);
            job.a.xs = std::span(xq_.s.data(), (std::size_t)n_in / tmq::kBlock);
        }
#endif
        // y is NOT one contiguous buffer (each projection owns its output
        // pointer), so the args' y span stays empty and this kernel writes
        // p.output directly.
        job.kern = [](const GemvArgs& a, int first, int last) {
            const auto* parts = static_cast<const Projection*>(a.w);
            const float* input = a.x.data();
            const std::int8_t* xq = a.xq.data();
            const float* xs = a.xs.data();
            const std::int8_t* xd = a.xd.data();
            int width = a.n_in;
            int offset = 0;
            for (std::size_t i = 0; i < N && offset < last; ++i) {
                const auto& p = parts[i];
                const int begin = std::max(first - offset, 0);
                const int end = std::min(last - offset, p.rows);
                if (begin < end) {
                    const auto run = [&](const auto* blocks) {
#ifdef __ARM_NEON
                        if (xq) {
                            qgemv_sdot_rows(blocks, xq, xd, xs, width, begin, end, p.output);
                            return;
                        }
#endif
                        qgemv_rows(blocks, input, width, begin, end, p.output);
                    };
                    if (p.q8) run(static_cast<const tmq::BlockQ8_0*>(p.weights));
                    else run(static_cast<const tmq::BlockQ4_0*>(p.weights));
                }
                offset += p.rows;
            }
        };
        if (gemv_switches().serial)
            job.kern(job.a, 0, total);
        else
            gemv_pool().dispatch(job);  // completes before projections leaves scope
    }

    // M6: batched prefill projection on the GPU (fused Q4-dequant GEMM).
    // Returns false (caller falls back to the per-token CPU gemv loop) when
    // Metal is absent/disabled, the tensor is not Q4, or K is not a
    // multiple of the 32-value block.
    bool batched_q4(const float* A, const std::string& name, float* C,
                    int M, int N, int K) const {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if ((!gpu_prefill_ && !decode_gpu_) || K % 32 || !q4_.count(name))
            return false;
        if (M > 1 && M < prefill_loop_t()) return false;   // tiny prompts: per-token loop (see prefill_loop_t)
        // M=1 through this kernel is the GPU-DECODE experiment: measured
        // 5x SLOWER than the CPU path (v1 prefill-kernel reuse — 16/256
        // threads active per threadgroup, per-call commit/wait, CPU
        // round trips between ops), so it must not engage for prefill
        // sessions; only TM_DECODE_GPU=1 opts in.
        if (M < 2 && !decode_gpu_) return false;
        // M=1: the native GEMV kernel (1 thread/output, full occupancy) —
        // the prefill tile kernel wastes 15/16 of its threads at M=1.
        if (M == 1) {
            if (tm_metal_q4_gemv) {
                gpu_weights_used_ = true;
                return tm_metal_q4_gemv(A, q4_.at(name), C,
                                        (unsigned)N, (unsigned)K);
            }
            return false;
        }
        if (tm_metal_sgemm_q4) {
            gpu_weights_used_ = true;
            return tm_metal_sgemm_q4(A, q4_.at(name), C, (unsigned)M,
                                     (unsigned)N, (unsigned)K);
        }
#endif
        (void)A; (void)name; (void)C; (void)M; (void)N; (void)K;
        return false;
    }
    void set_prefill_gpu(bool on) { gpu_prefill_ = on; }
    void set_prefill_amx(bool on) { amx_prefill_ = on; }
    // GPU layer stack for prefill (metal_llama.h). TM_LLAMA_GPU_STACK=0 keeps
    // the per-layer block() loop with GPU GEMMs.
    void* gpu_stack_ = nullptr;
    unsigned long gpu_stack_prefills_ = 0;
    void gpu_stack_setup() {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        gpu_stack_teardown();
        if (const char* e = std::getenv("TM_LLAMA_GPU_STACK"); e && e[0] == '0') return;
        if (!quantized() || !held_map_ || !held_map_->base) return;
        if (kv_ktype() || kv_vtype()) return;   // packed KV is CPU-only (resolve_decode_gpu_auto)
        std::vector<tm_llama_gpu_layer> ls((std::size_t)L);
        for (int l = 0; l < L; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            const auto q4 = [&](const char* k) -> const void* { auto it = q4_.find(p + k); return it == q4_.end() ? nullptr : static_cast<const void*>(it->second); };
            const auto wf = [&](const char* k) -> const float* { auto it = w.find(p + k); return it == w.end() ? nullptr : it->second.data(); };
            tm_llama_gpu_layer& g = ls[(std::size_t)l];
            g.q = q4("self_attn.q_proj.weight"); g.k = q4("self_attn.k_proj.weight"); g.v = q4("self_attn.v_proj.weight");
            g.o = q4("self_attn.o_proj.weight"); g.gate = q4("mlp.gate_proj.weight"); g.up = q4("mlp.up_proj.weight");
            g.down = q4("mlp.down_proj.weight");
            g.rms1 = wf("input_layernorm.weight"); g.rms2 = wf("post_attention_layernorm.weight");
            g.q_bias = wf("self_attn.q_proj.bias"); g.k_bias = wf("self_attn.k_proj.bias"); g.v_bias = wf("self_attn.v_proj.bias");
            g.kc = kc[l].data(); g.vc = vc[l].data();
            if (!(g.q && g.k && g.v && g.o && g.gate && g.up && g.down && g.rms1 && g.rms2)) return;
        }
        tm_llama_gpu_desc d{};
        d.L = L; d.D = D; d.H = H; d.KVH = KVH; d.dh = dh; d.F = F; d.ctx = ctx; d.theta = theta; d.eps = eps;
        d.map_base = held_map_->base; d.map_size = held_map_->size; d.layers = ls.data();
        gpu_stack_ = tm_metal_llama_create(&d);
#endif
    }
    void gpu_stack_teardown() {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (gpu_stack_) tm_metal_llama_destroy(gpu_stack_);
#endif
        gpu_stack_ = nullptr;
    }
    // True when a T>1 projection of this shape will go to the Metal
    // prefill, so the AMX path must step aside (GPU wins under load, the
    // reason auto selected it); false lets AMX catch shapes the GPU declines
    // instead of falling to the per-token qgemv loop.
    // Short prompts (2026-09-07): the auto policy resolves the prefill
    // device from ambient load at load time, but for M <= 32 the GPU wins
    // even on a quiet host — AMX dequantizes the whole model per call (387
    // ms for a 21-token chat turn) where the Metal Q4 GEMM takes 226 ms — so
    // short prompts go to the GPU whenever it is usable and the model fits
    // the auto budget. TM_PREFILL_GPU_SHORT_T (default 32; 0 disables).
    bool gpu_prefill_short_ok_ = false;   // resolved with the auto policy
    static int gpu_prefill_short_t() {
        static const int t = [] { const char* e = std::getenv("TM_PREFILL_GPU_SHORT_T"); return e && *e ? std::atoi(e) : 32; }();
        return t;
    }
    // Tiny prompts (2026-09-07): below TM_PREFILL_LOOP_T (default 5) tokens
    // the per-token GEMV loop (~12 ms/token) beats both the AMX path (a whole-
    // model dequant per call) and the GPU stack (M=2: 478 ms under MPS kernel
    // selection; the stack pads 5 <= M < 20 to 20 rows itself). A chat turn
    // close feeds 2-3 tokens: 470 -> 24 ms.
    static int prefill_loop_t() {
        static const int t = [] { const char* e = std::getenv("TM_PREFILL_LOOP_T"); return e && *e ? std::atoi(e) : 5; }();
        return t;
    }
    bool amx_ok(int T) const { return amx_prefill_ && T >= prefill_loop_t(); }
    // Rows of a prefill handed to the CPU in hybrid mode. The controller
    // (HybridSplit, defined above Llama) is load-aware: its setpoint slides
    // with measured contention, it persists only quiet-regime results, and
    // its parked re-probe interval scales with the load. TM_LLAMA_HYBRID
    // pins the fraction (e.g. 0.25) or "auto"; rows are rounded to a
    // multiple of 8 and clamped so the GPU keeps at least half the rows and
    // both sides stay above 64 rows.
    mutable HybridSplit hybrid_;
    void hybrid_update(double gpu_ms, double cpu_ms, int gpu_rows, int cpu_rows,
                       double wall_ms = 0.0, double pressure = 0.0) const {
        hybrid_.update(gpu_ms, cpu_ms, gpu_rows, cpu_rows, wall_ms, pressure);
    }
    int hybrid_cpu_rows(int T) const { return hybrid_.cpu_rows(T); }
    // Set while the hybrid split runs this thread's rows: block() must use
    // AMX, not queue its GEMMs behind the GPU stack's command buffers.
    bool force_cpu_block_ = false;
    bool gpu_prefill_ready(int M, int K) const {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (force_cpu_block_) return false;
        if (M < prefill_loop_t()) return false;
        // With the GPU layer stack present (metal_llama.h) the GPU wins at
        // every prompt length on a quiet host too (2026-09-07: 128 tok 573 vs
        // ~300 AMX, 512 tok 757 vs 299), so the busy-host rule only matters
        // for the per-layer GEMM path; the short-prompt rule stays for it.
        const bool short_prompt = gpu_prefill_short_ok_ && (M <= gpu_prefill_short_t() || gpu_stack_ != nullptr);
        // The decode-GPU exclusion guards the legacy copied-blob prefill
        // (double footprint); the layer stack shares the NoCopy mapping, so
        // forced or context-switched GPU decode keeps GPU prefill with it
        // (2026-09-07: TM_DECODE_GPU=1 used to fall to AMX at 113-343 t/s).
        const bool legacy_conflict = decode_gpu_ && !gpu_both_auto_ && gpu_stack_ == nullptr;
        return (gpu_prefill_ || short_prompt) && !legacy_conflict && K % 32 == 0 &&
               M >= 2 && tm_metal_q4_prefill && !q4_.empty();
#else
        (void)M; (void)K;
        return false;
#endif
    }
    void set_attn_naive(bool on) { attn_naive_ = on; }
    void set_attn_decode_batched(bool on) { attn_decode_batched_ = on; }
    void set_attn_vsoftmax(bool on) { attn_vsoftmax_ = on; }
    void set_attn_decode_pool(bool on) { attn_decode_pool_ = on; }

    // One GQA group of batched attention for one token: scores, softmax,
    // V-weighted sum. S is this group's private (REP x allow) scratch.
    static void attn_group(const float* qg, const float* Kh, const float* Vh, float* S,
                           float* out, int REP, int dh, int allow, float scale,
                           bool vsoftmax) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    REP, allow, dh, scale, qg, dh, Kh, dh, 0.f, S, allow);
        for (int r = 0; r < REP; ++r) {
            float* row = S + (std::size_t)r * allow;
            if (vsoftmax) {
                float m = 0.f, z = 0.f;
                vDSP_maxv(row, 1, &m, (vDSP_Length)allow);
                const float nm = -m;
                vDSP_vsadd(row, 1, &nm, row, 1, (vDSP_Length)allow);
                vvexpf(row, row, &allow);
                vDSP_sve(row, 1, &z, (vDSP_Length)allow);
                const float inv = 1.f / z;
                vDSP_vsmul(row, 1, &inv, row, 1, (vDSP_Length)allow);
            } else {
                float m = -1e30f;
                for (int s = 0; s < allow; ++s) m = std::max(m, row[s]);
                float z = 0.f;
                for (int s = 0; s < allow; ++s) {
                    row[s] = std::exp(row[s] - m);
                    z += row[s];
                }
                const float inv = 1.f / z;
                for (int s = 0; s < allow; ++s) row[s] *= inv;
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    REP, dh, allow, 1.f, S, allow, Vh, dh, 0.f, out, dh);
    }
    // GemvPool job adapter: slice [o0, o1) = KV groups; w carries the context.
    struct AttnJob {
        const float* q; const float* K; const float* V; float* S; float* out;
        int REP, dh, allow, ctx; float scale; bool vsoftmax;
    };
    static void attn_group_kern(const GemvArgs& g, int o0, int o1) {
        const auto& a = *static_cast<const AttnJob*>(g.w);
        for (int kv = o0; kv < o1; ++kv)
            attn_group(a.q + (std::size_t)kv * a.REP * a.dh,
                       a.K + (std::size_t)kv * a.ctx * a.dh,
                       a.V + (std::size_t)kv * a.ctx * a.dh,
                       a.S + (std::size_t)kv * a.REP * a.allow,
                       a.out + (std::size_t)kv * a.REP * a.dh,
                       a.REP, a.dh, a.allow, a.scale, a.vsoftmax);
    }
    // Blocked prefill attention unit: (KV group kv, query block blk) — one
    // (B*REP x allow_max) score GEMM, masked softmax rows, one PV GEMM, with
    // thread-local scratch so units run on the GemvPool concurrently (the
    // softmax exp is the part that scales with threads; the sgemms share
    // the AMX units). Units are enumerated largest-allow first so the
    // dynamic slicing balances the causal triangle.
    struct AttnBlockJob {
        const float* q; const float* K; const float* V; float* ao;
        // Row strides, named rather than positional: the query row is 2*H*dh
        // wide when attn_q carries the interleaved output gate, while the
        // attention output row is always H*dh. Both were passed as the same
        // value, so a gated model read its queries at the wrong stride.
        int T, pos0, qstride, aostride, REP, dh, ctx, KVH, B, nblocks; float scale;
    };
    static void attn_block_unit(const AttnBlockJob& a, int kv, int blk) {
        thread_local std::vector<float> S, qb, ob;
        const int t0 = blk * a.B;
        const int bt = std::min(a.B, a.T - t0);
        const int rows = bt * a.REP;
        const int allow_max = a.pos0 + t0 + bt;
        if (qb.size() < (std::size_t)rows * a.dh) qb.resize((std::size_t)rows * a.dh);
        if (ob.size() < (std::size_t)rows * a.dh) ob.resize((std::size_t)rows * a.dh);
        if (S.size() < (std::size_t)rows * allow_max) S.resize((std::size_t)rows * allow_max);
        const float* Kh = a.K + (std::size_t)kv * a.ctx * a.dh;
        const float* Vh = a.V + (std::size_t)kv * a.ctx * a.dh;
        for (int t = 0; t < bt; ++t)
            std::copy_n(a.q + (std::size_t)(t0 + t) * a.qstride + (std::size_t)kv * a.REP * a.dh,
                        (std::size_t)a.REP * a.dh, qb.data() + (std::size_t)t * a.REP * a.dh);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, allow_max, a.dh, a.scale,
                    qb.data(), a.dh, Kh, a.dh, 0.f, S.data(), allow_max);
        for (int r = 0; r < rows; ++r) {
            float* row = S.data() + (std::size_t)r * allow_max;
            int allow = a.pos0 + t0 + r / a.REP + 1;
            float m = 0.f, z = 0.f;
            vDSP_maxv(row, 1, &m, (vDSP_Length)allow);
            const float nm = -m;
            vDSP_vsadd(row, 1, &nm, row, 1, (vDSP_Length)allow);
            vvexpf(row, row, &allow);
            vDSP_sve(row, 1, &z, (vDSP_Length)allow);
            const float inv = 1.f / z;
            vDSP_vsmul(row, 1, &inv, row, 1, (vDSP_Length)allow);
            if (allow < allow_max) std::fill(row + allow, row + allow_max, 0.f);
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, a.dh, allow_max, 1.f,
                    S.data(), allow_max, Vh, a.dh, 0.f, ob.data(), a.dh);
        for (int t = 0; t < bt; ++t)
            std::copy_n(ob.data() + (std::size_t)t * a.REP * a.dh, (std::size_t)a.REP * a.dh,
                        a.ao + (std::size_t)(t0 + t) * a.aostride + (std::size_t)kv * a.REP * a.dh);
    }
    static void attn_block_kern(const GemvArgs& g, int o0, int o1) {
        const auto& a = *static_cast<const AttnBlockJob*>(g.w);
        for (int u = o0; u < o1; ++u) {
            const int rev = a.KVH * a.nblocks - 1 - u;       // largest blocks first
            attn_block_unit(a, rev / a.nblocks, rev % a.nblocks);
        }
    }
    // Blocked prefill for a PACKED KV cache. The fp32 unit above hands two fp32
    // sgemms a `const float*` K/V, which a packed row is not — so before this
    // kernel existed a packed cache fell back to the naive per-token loop and
    // paid a measured 1.58x on prefill. Both routes do the same arithmetic; the
    // difference is READS. The naive route re-reads every KV row once per token
    // (~T^2/2 rows); this one reads each packed K row once per (KV group, query
    // block) unit (~T^2/B) and dots it against the whole block's quantized query
    // rows, and dequantizes each V row once and fans it across them — which is
    // also the spec's prescription ("dequantize each (position, KV head) row
    // once"). No fp32 row is ever materialized for K.
    struct AttnBlockJobPacked {
        const float* q; const std::uint8_t* K; const std::uint8_t* V; float* ao;
        int T, pos0, qstride, aostride, REP, dh, ctx, KVH, B, nblocks; float scale;
    };
    static void attn_block_unit_packed(const AttnBlockJobPacked& a, int kv, int blk) {
        thread_local std::vector<std::int8_t> qq;
        thread_local std::vector<std::uint16_t> qd;
        thread_local std::vector<float> S, vb, ob;
        const int nb = a.dh / tmq::kBlock;
        const std::size_t kvrow = (std::size_t)nb * sizeof(tmq::BlockQ8_0);
        const int t0 = blk * a.B;
        const int bt = std::min(a.B, a.T - t0);
        const int rows = bt * a.REP;
        const int allow_max = a.pos0 + t0 + bt;
        if (qq.size() < (std::size_t)rows * a.dh) qq.resize((std::size_t)rows * a.dh);
        if (qd.size() < (std::size_t)rows * nb) qd.resize((std::size_t)rows * nb);
        if (S.size() < (std::size_t)rows * allow_max) S.resize((std::size_t)rows * allow_max);
        if (ob.size() < (std::size_t)rows * a.dh) ob.resize((std::size_t)rows * a.dh);
        if (vb.size() < (std::size_t)a.dh) vb.resize((std::size_t)a.dh);
        // Quantize every query row in the block once: REP heads x bt tokens.
        for (int t = 0; t < bt; ++t) {
            const float* qg =
                a.q + (std::size_t)(t0 + t) * a.qstride + (std::size_t)kv * a.REP * a.dh;
            for (int r = 0; r < a.REP; ++r) {
                const float* qrow = qg + (std::size_t)r * a.dh;
                std::int8_t* dst = qq.data() + (std::size_t)(t * a.REP + r) * a.dh;
                std::uint16_t* sdst = qd.data() + (std::size_t)(t * a.REP + r) * nb;
                for (int b = 0; b < nb; ++b) {
                    const float* x = qrow + (std::size_t)b * tmq::kBlock;
                    float amax = 0.f;
                    for (int i = 0; i < tmq::kBlock; ++i)
                        amax = std::max(amax, std::fabs(x[i]));
                    const std::uint16_t d16 = tmq::fp32_to_fp16(amax / 127.f);
                    const float d = tmq::fp16_to_fp32(d16);
                    sdst[b] = d16;
                    const float inv = d == 0.f ? 0.f : 1.f / d;
                    for (int i = 0; i < tmq::kBlock; ++i) {
                        const float v = std::min(std::max(x[i] * inv, -127.f), 127.f);
                        dst[(std::size_t)b * tmq::kBlock + i] = (std::int8_t)std::lrintf(v);
                    }
                }
            }
        }
        const auto* Kq = reinterpret_cast<const tmq::BlockQ8_0*>(
            a.K + (std::size_t)kv * (std::size_t)a.ctx * kvrow);
        // Scores: one packed K row read, `rows` int8 dots against it.
        for (int s = 0; s < allow_max; ++s) {
            const tmq::BlockQ8_0* krow = Kq + (std::size_t)s * nb;
            for (int r = 0; r < rows; ++r) {
                const std::int8_t* qrow = qq.data() + (std::size_t)r * a.dh;
                const std::uint16_t* qrowd = qd.data() + (std::size_t)r * nb;
                float acc = 0.f;
#if defined(__aarch64__)
                // One horizontal reduce per dot, not one per block: the block's
                // four int32 lanes are scaled into an fp32 accumulator and only
                // the final `vaddvq` folds them. The block scale is a per-block
                // quantity, so it cannot ride the int accumulator — this is the
                // cheapest shape that still applies it exactly.
                float32x4_t facc = vdupq_n_f32(0.f);
                for (int b = 0; b < nb; ++b) {
                    const std::int8_t* kr = krow[b].qs;
                    const std::int8_t* qb = qrow + (std::size_t)b * tmq::kBlock;
                    const int8x16_t k0 = vld1q_s8(kr), k1 = vld1q_s8(kr + 16);
                    const int8x16_t c0 = vld1q_s8(qb), c1 = vld1q_s8(qb + 16);
                    const int32x4_t d = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, c0), k1, c1);
                    const float sc = tmq::fp16_to_fp32(krow[b].d_fp16) *
                                     tmq::fp16_to_fp32(qrowd[b]);
                    facc = vfmaq_n_f32(facc, vcvtq_f32_s32(d), sc);
                }
                acc = vaddvq_f32(facc);
#else
                for (int b = 0; b < nb; ++b) {
                    const std::int8_t* kr = krow[b].qs;
                    const std::int8_t* qb = qrow + (std::size_t)b * tmq::kBlock;
                    std::int32_t dot = 0;
                    for (int i = 0; i < tmq::kBlock; ++i) dot += (int)kr[i] * (int)qb[i];
                    acc += (float)dot * (tmq::fp16_to_fp32(krow[b].d_fp16) *
                                         tmq::fp16_to_fp32(qrowd[b]));
                }
#endif
                S[(std::size_t)r * allow_max + s] = acc * a.scale;
            }
        }
        for (int r = 0; r < rows; ++r) {
            float* row = S.data() + (std::size_t)r * allow_max;
            const int allow = a.pos0 + t0 + r / a.REP + 1;
            float m = 0.f, z = 0.f;
            vDSP_maxv(row, 1, &m, (vDSP_Length)allow);
            const float nm = -m;
            vDSP_vsadd(row, 1, &nm, row, 1, (vDSP_Length)allow);
            vvexpf(row, row, &allow);
            vDSP_sve(row, 1, &z, (vDSP_Length)allow);
            const float inv = 1.f / z;
            vDSP_vsmul(row, 1, &inv, row, 1, (vDSP_Length)allow);
            if (allow < allow_max) std::fill(row + allow, row + allow_max, 0.f);
        }
        for (int r = 0; r < rows; ++r)
            std::fill_n(ob.data() + (std::size_t)r * a.dh, (std::size_t)a.dh, 0.f);
        const auto* Vq = reinterpret_cast<const tmq::BlockQ8_0*>(
            a.V + (std::size_t)kv * (std::size_t)a.ctx * kvrow);
        for (int s = 0; s < allow_max; ++s) {
            // Dequantize this V row ONCE and fan it across every row that still
            // carries weight — softmax zeroed the masked tail, and the `w == 0`
            // skip is what makes the causal triangle cheap.
            tmq::dequantize_row_q8_0(Vq + (std::size_t)s * nb, vb.data(), (std::size_t)a.dh);
            for (int r = 0; r < rows; ++r) {
                const float w = S[(std::size_t)r * allow_max + s];
                if (w == 0.f) continue;
                float* acc = ob.data() + (std::size_t)r * a.dh;
                for (int i = 0; i < a.dh; ++i) acc[i] += w * vb[(std::size_t)i];
            }
        }
        for (int t = 0; t < bt; ++t)
            std::copy_n(ob.data() + (std::size_t)t * a.REP * a.dh, (std::size_t)a.REP * a.dh,
                        a.ao + (std::size_t)(t0 + t) * a.aostride +
                            (std::size_t)kv * a.REP * a.dh);
    }
    static void attn_block_kern_packed(const GemvArgs& g, int o0, int o1) {
        const auto& a = *static_cast<const AttnBlockJobPacked*>(g.w);
        for (int u = o0; u < o1; ++u) {
            const int rev = a.KVH * a.nblocks - 1 - u;       // largest blocks first
            attn_block_unit_packed(a, rev / a.nblocks, rev % a.nblocks);
        }
    }
    // Dequantize rows [o0, o1) of a quantized (out, in) tensor into a
    // row-major fp32 buffer — same math as dequant_row; the NEON variant
    // reuses dequant_block. out is the BASE row pointer (global row
    // indexing — same convention as qgemv_rows, see GemvPool::run_slices).
    static void dequant_rows(const void* blocks, bool q8, int n_in,
                             int o0, int o1, float* out) {
        const std::uint64_t nb = (std::uint64_t)(n_in / tmq::kBlock);
#ifdef __ARM_NEON
        for (int o = o0; o < o1; ++o) {
            float* dst = out + (std::uint64_t)o * n_in;
            if (q8) {
                const auto* r = static_cast<const tmq::BlockQ8_0*>(static_cast<const void*>(blocks)) +
                               (std::uint64_t)o * nb;
                for (std::uint64_t b = 0; b < nb; ++b) {
                    float32x4_t qf[8];
                    dequant_block(r[b], qf);
                    const float32x4_t dv =
                        vdupq_n_f32(tmq::fp16_to_fp32(r[b].d_fp16));
                    for (int k = 0; k < 8; ++k)
                        vst1q_f32(dst + b * tmq::kBlock + 4 * k,
                                  vmulq_f32(qf[k], dv));
                }
            } else {
                const auto* r = static_cast<const tmq::BlockQ4_0*>(static_cast<const void*>(blocks)) +
                               (std::uint64_t)o * nb;
                for (std::uint64_t b = 0; b < nb; ++b) {
                    float32x4_t qf[8];
                    dequant_block(r[b], qf);
                    const float32x4_t dv =
                        vdupq_n_f32(tmq::fp16_to_fp32(r[b].d_fp16));
                    for (int k = 0; k < 8; ++k)
                        vst1q_f32(dst + b * tmq::kBlock + 4 * k,
                                  vmulq_f32(qf[k], dv));
                }
            }
        }
#else
        for (int o = o0; o < o1; ++o)
            dequant_row(blocks, q8, n_in, o, out + (std::uint64_t)o * n_in);
#endif
    }
    // AMX prefill projection: C = A x W^T for a quantized (out=N, in=K)
    // weight, A row-major (M, K). W is dequantized on the calling thread
    // into the reused fp32 scratch, then one threaded cblas_sgemm
    // (CblasTrans: nn.Linear (out, in) layout) does the math — Accelerate
    // dispatches large sgemm to AMX and parallelizes it internally, which
    // dwarfs the O(N*K) dequant, so the dequant stays serial: a GemvPool
    // variant (row-parallel dequant) measured NO faster and corrupts the
    // heap when interleaved with the batched-attention sgemms (open issue,
    // root cause not isolated — serial dequant + AMX + batched attention
    // passed all gates and benched FASTER: 271 vs 257 t/s). Returns false
    // when the AMX path is off, the tensor is not a quantized 2-D weight,
    // or K is not a multiple of the quant block — caller falls back.
    [[nodiscard]] bool gemm_w(const std::string& name, const float* A, int M, int K,
                int N, float* C) {
        if (!amx_prefill_ || K % tmq::kBlock) return false;
        const bool q8 = q8_.count(name) != 0;
        // K-quant prefill: dequantize the same reused fp32 scratch and let the
        // one threaded sgemm do the math, exactly as the q4/q8 lanes below.
        if (const auto kq = qk_.find(name); kq != qk_.end()) {
            // That dequantize is per CALL, not per position: the whole tensor is
            // unpacked to fp32 (210 bytes of Q6_K -> 1024 bytes, scalar, ~7 GB/s
            // measured) before the sgemm can start, so this lane carries a large
            // FIXED cost that only amortizes at large M. Both lanes fitted from
            // in-engine passes, paired and interleaved in one window on
            // Qwen3.8-4B Q6_K:
            //   this lane   ~3.18 s + 7.5 ms per position  (3.24 s at M=8, 3.30 at
            //               M=16: flat to within 2 % => the sgemm is not the cost)
            //   row path    ~90 ms per position   (measured 88-95 under load)
            //   chunked multi  ~61 ms per position, FLAT from M=8 to M=32 (58.5 /
            //               59.3 / 61.9 quiet) — see gemm_kq_multi
            // Crossover is therefore 3.18 / (0.061 - 0.0075) ~ 59, and this lane
            // waits for it. Admitting it from prefill_loop_t() == 5 instead made
            // every prompt shorter than that 2.5-6x slower than the path it
            // displaced: measured per position at M=8, 405 vs 58.5, and at M=32,
            // 106 vs 62. The lane is still the fastest thing here for a long
            // prompt — at M=256 it is 12 + 7 = 19 ms per position against 61 — so
            // it is kept, and TM_KQ_AMX_MIN overrides the threshold.
            //
            // The real disease is the serial per-call unpack, and the obvious fix —
            // row-parallelising it, as this lane's q4/q8 dequant already offers via
            // TM_POOL_DEQ — is now MEASURED AND REFUTED (2026-09-16, deq_scale:
            // largest K-quant tensor of Qwen3.8-4B, 2542.8 MB of fp32 per pass,
            // three arms interleaved in one window, best of 3, load per arm):
            //   TM_THREADS=4, load 3.99-4.87  serial 2.951 s | pool 9.333 s (0.32x)
            //                                 | plain threads 10.466 s (0.28x)
            //   TM_THREADS=2, load 6.2-7.2    serial 5.123 s | pool 13.989 s (0.37x)
            //                                 | plain threads 11.882 s (0.43x)
            // Plain std::thread workers at default QoS are as slow as the pool, so
            // this is not the launcher, the chunk size or the QoS band: writing one
            // 2.5 GB destination as N interleaved streams is simply the slower shape
            // here. The fixed term stands and the threshold stays where it is.
            // (A whole-model pass showed the pool WINNING 1.22x — 38.5 -> 31.7 s
            // over 217 tensors — which is why this needed paired single-tensor arms
            // and a printed load rather than one favourable window.)
            const int kq_amx_min = gemv_switches().kq_amx_min;
            if (M < kq_amx_min) return false;
            if (deq_.size() < (std::size_t)N * K) deq_.resize((std::size_t)N * K);
            for (int o = 0; o < N; ++o)
                dequant_row_kq(kq->second.blocks, kq->second.dtype, K, o,
                               deq_.data() + (std::size_t)o * K);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                        1.f, A, K, deq_.data(), K, 0.f, C, N);
            return true;
        }
        if (!q8 && !q4_.count(name)) {
            // fp32 weights (equal-precision control lane): direct sgemm,
            // no dequant scratch, no batched_q4_parts split copy
            // (TENSORMARK_FP32_PREFILL_SPEC P2.1, 2026-09-12).
            // TM_GEMM_F32=0 restores the per-token gemv loop (A/B gate).
            if (!w.count(name)) return false;
            if (!gemv_switches().gemm_f32) return false;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                        1.f, A, K, w.at(name).data(), K, 0.f, C, N);
            return true;
        }
        const void* blocks = q8 ? static_cast<const void*>(q8_.at(name))
                                : static_cast<const void*>(q4_.at(name));
        if (deq_.size() < (std::size_t)N * K) deq_.resize((std::size_t)N * K);
        // TM_POOL_DEQ=1 dispatches the dequant on the GemvPool (row-parallel).
        // Post ld-fix it passes the previously-fatal shapes repeatedly (the
        // old corruption was the batched-attention lda/ldc OOB, not the
        // pool — see repro_poolbug.cpp and the M5 spec); serial stays the
        // default because it measured faster (271 vs 257 t/s A/B).
        if (gemv_switches().pool_deq) {
            GemvJob j;
            j.a = GemvArgs{.w = blocks,
                           .y = std::span(deq_.data(), (std::size_t)N * K),
                           .n_in = K};
            j.n_out = N;
            j.kern = q8
                ? [](const GemvArgs& a, int o0, int o1) {
                      dequant_rows(a.w, true, a.n_in, o0, o1, a.y.data());
                  }
                : [](const GemvArgs& a, int o0, int o1) {
                      dequant_rows(a.w, false, a.n_in, o0, o1, a.y.data());
                  };
            gemv_pool().dispatch(j);
        } else {
            dequant_rows(blocks, q8, K, 0, N, deq_.data());
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.f,
                    A, K, deq_.data(), K, 0.f, C, N);
        return true;
    }
    // M6: fused multi-matrix prefill (one dispatch for q/k/v or gate/up).
    // C[M x Ntot] where Ntot = sum(Ns); weights concatenated on the GPU once.
    bool batched_q4_parts(const float* A, const std::string* names,
                          const int* Ns, int nn, float* C,
                          int M, int K) const {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        // The legacy prefill backend retains copied weight blobs. Do not mix
        // it with NoCopy token decode: that duplicates most of the model even
        // in a SINGLE process. Use CPU/AMX prefill until buffers are shared.
        if (!gpu_prefill_ || (decode_gpu_ && !gpu_both_auto_) || K % 32 ||
            M < 2 || M < prefill_loop_t() || !tm_metal_q4_prefill)
            return false;
        const void* wqs[8];
        unsigned ns[8];
        if (nn > 8) return false;
        unsigned ntot = 0;
        for (int i = 0; i < nn; ++i) {
            if (!q4_.count(names[i]) || K % 32) return false;
            wqs[i] = q4_.at(names[i]);
            ns[i] = (unsigned)Ns[i];
            ntot += ns[i];
        }
        (void)ntot;
        gpu_weights_used_ = true;
        return tm_metal_q4_prefill(A, wqs, ns, (unsigned)nn, C, (unsigned)M,
                                   (unsigned)K);
#endif
        (void)A; (void)names; (void)Ns; (void)nn; (void)C; (void)M; (void)K;
        return false;
    }
    // Debug/bisect helper: run rows [o0, o1) on the calling thread only.
    void gemv_serial_rows(const std::string& name, int o0, int o1,
                          const float* x, float* y) const {
        const int ni = qshape_.at(name).second;
        if (q8_.count(name))
            qgemv_rows(q8_.at(name), x, ni, o0, o1, y);
        else
            qgemv_rows(q4_.at(name), x, ni, o0, o1, y);
    }
    // Dequantize one (out,) row of a quantized tensor (2-D via maps, or
    // any row/block pointer via the static core).
    static void dequant_row(const void* blocks, bool q8, int n_in, int row,
                            float* dst) {
        const std::uint64_t nb = (std::uint64_t)(n_in / tmq::kBlock);
        if (q8) {
            const auto* r = static_cast<const tmq::BlockQ8_0*>(static_cast<const void*>(blocks)) + (std::uint64_t)row * nb;
            for (std::uint64_t b = 0; b < nb; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                for (int i = 0; i < tmq::kBlock; ++i)
                    dst[b * tmq::kBlock + i] = (float)r[b].qs[i] * d;
            }
        } else {
            const auto* r = static_cast<const tmq::BlockQ4_0*>(static_cast<const void*>(blocks)) + (std::uint64_t)row * nb;
            for (std::uint64_t b = 0; b < nb; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                for (int j = 0; j < tmq::kBlock / 2; ++j) {
                    dst[b * tmq::kBlock + 2 * j] = (float)((r[b].qs[j] & 0xF) - 8) * d;
                    dst[b * tmq::kBlock + 2 * j + 1] = (float)((r[b].qs[j] >> 4) - 8) * d;
                }
            }
        }
    }
    // Dequantize one row of a K-quant tensor. Used by the 1-D loader path and
    // by row_w for embed/gate lookups; the 2-D GEMV path never calls it (it
    // unpacks in int8, see kqgemv_rows).
    //
    // Q6_K under __ARM_NEON reuses kq6k_unpack16 (the GEMV kernel's unpacker)
    // instead of the scalar bit loop in quant.h. Layout, from the scalar's
    // contract: within a 128-value half, element (l + 32j), j = 0..3, takes its
    // low nibble from ql[l + 32*(j&1)] (high nibble for j >= 2), its top 2 bits
    // from qh[l] >> 2j, and its scale from sc[l/16 + 2j]. kq6k_unpack16<J>
    // already produces exactly the 16 lanes of that shape for l = 0..15 with
    // ql's (J&1)-th 32-byte chunk — so each half is eight unpacks (l = 0..15
    // and l = 16..31), each widened, converted and scaled by one
    // broadcast d * sc[is + 2j]. Bit-exactness: the scalar evaluates
    // d * sc * q in that order with |q| <= 48 (exact in fp32), and the unpack
    // is the same identity the GEMV kernel is gated against.
    static void dequant_row_q6k_neon(const tmq::BlockQ6_K* r, float* dst,
                                     std::size_t n_in) {
        for (std::size_t b = 0; b < n_in / tmq::kKBlock; ++b, dst += tmq::kKBlock) {
            const float d = tmq::fp16_to_fp32(r[b].d_fp16);
            const std::uint8_t* ql = r[b].ql;
            const std::uint8_t* qh = r[b].qh;
            const std::int8_t* sc = r[b].scales;
            for (int half = 0; half < tmq::kKBlock; half += 128, ql += 64, qh += 32, sc += 8) {
                const uint8x16_t qha = vld1q_u8(qh);
                const uint8x16_t qhb = vld1q_u8(qh + 16);
                for (int is = 0; is < 2; ++is) {
                    const std::uint8_t* qll = ql + 16 * is;
                    const uint8x16_t qhx = is == 0 ? qha : qhb;
                    const int8x16_t q[4] = {
                        kq6k_unpack16<0>(qll, qhx), kq6k_unpack16<1>(qll, qhx),
                        kq6k_unpack16<2>(qll, qhx), kq6k_unpack16<3>(qll, qhx),
                    };
                    for (int j = 0; j < 4; ++j) {
                        const float32x4_t f = vdupq_n_f32(d * (float)sc[is + 2 * j]);
                        const int16x8_t w = vmovl_s8(vget_low_s8(q[j]));
                        const int16x8_t w2 = vmovl_s8(vget_high_s8(q[j]));
                        float* y = dst + half + 32 * j + 16 * is;
                        vst1q_f32(y,      vmulq_f32(f, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)))));
                        vst1q_f32(y + 4,  vmulq_f32(f, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)))));
                        vst1q_f32(y + 8,  vmulq_f32(f, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2)))));
                        vst1q_f32(y + 12, vmulq_f32(f, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2)))));
                    }
                }
            }
        }
    }
    static void dequant_row_kq(const void* blocks, std::uint32_t dtype, int n_in,
                               int row, float* dst) {
        const std::uint32_t be = tmmq::block_elems(dtype);
        const std::uint64_t nb = (std::uint64_t)(n_in / be);
        const std::size_t bs = tmmq::block_bytes(dtype);
        const std::uint8_t* r = (const std::uint8_t*)blocks + (std::size_t)row * nb * bs;
        switch (dtype) {
            case 4u: tmq::dequantize_row_q4_1(static_cast<const tmq::BlockQ4_1*>(static_cast<const void*>(r)), dst, (std::size_t)n_in); return;
            case 5u: tmq::dequantize_row_q5_K(static_cast<const tmq::BlockQ5_K*>(static_cast<const void*>(r)), dst, (std::size_t)n_in); return;
            case 6u:
#ifdef __ARM_NEON
                dequant_row_q6k_neon(static_cast<const tmq::BlockQ6_K*>(static_cast<const void*>(r)), dst, (std::size_t)n_in);
#else
                tmq::dequantize_row_q6_K(static_cast<const tmq::BlockQ6_K*>(static_cast<const void*>(r)), dst, (std::size_t)n_in);
#endif
                return;
            case 7u: {
                // Ternary 2-bit rows have no NEON dequantize (the fused GEMV is
                // the fast path); this scalar body serves row_w's embed/gate
                // lookups and the non-NEON correctness path alike.
                const auto* t = static_cast<const tmq::BlockTq2_34x128*>(
                    static_cast<const void*>(r));
                for (std::size_t b = 0;
                     b < (std::size_t)n_in / tmq::kBlockTq2_34x128; ++b) {
                    const float d = tmq::fp16_to_fp32(t[b].d_fp16);
                    for (int wrd = 0; wrd < 8; ++wrd) {
                        std::uint32_t word;
                        std::memcpy(&word, t[b].qs + 4 * wrd, sizeof word);
                        for (int k = 0; k < 16; ++k)
                            dst[b * tmq::kBlockTq2_34x128 + (std::size_t)wrd * 16 + k] =
                                (float)((int)((word >> (2 * k)) & 3u) - 1) * d;
                    }
                }
                return;
            }
            default: throw std::invalid_argument("llama: unsupported k-quant dtype");
        }
    }
    // dtype-dispatched row dequantize for any quantized dtype (0..6).
    static void dequant_row_any(const void* blocks, std::uint32_t dtype, int n_in,
                                float* dst) {
        if (dtype == 0u || dtype == 1u) { dequant_row(blocks, dtype == 0u, n_in, 0, dst); return; }
        dequant_row_kq(blocks, dtype, n_in, 0, dst);
    }
    void row_w(const std::string& name, int row, float* dst) const {
        const bool roww_debug = gemv_switches().debug_roww;
        if (roww_debug)
            fprintf(stderr, "[row_w] %s (in qshape: %d)\n", name.c_str(),
                    (int)qshape_.count(name));
        const int n_in = qshape_.at(name).second;
        if (f16_.count(name)) {
            const std::uint16_t* W = f16_.at(name);
            if (!W) throw std::runtime_error("llama: f16 tensor not resident: " + name);
            const std::uint16_t* r = W + (std::size_t)row * n_in;
            tmq::fp16_to_fp32_row(r, dst, (std::size_t)n_in);
            return;
        }
        if (const auto kq = qk_.find(name); kq != qk_.end()) {
            if (!kq->second.blocks)
                throw std::runtime_error("llama: k-quant tensor not resident: " + name);
            dequant_row_kq(kq->second.blocks, kq->second.dtype, n_in, row, dst);
            return;
        }
        dequant_row(q8_.count(name)
                        ? static_cast<const void*>(q8_.at(name))
                        : static_cast<const void*>(q4_.at(name)),
                    q8_.count(name), n_in, row, dst);
    }

    // Attach quantized weights from a .tmq file (M4 format). 1-D tensors
    // (norms) are dequantized into w; 2-D stay mapped. Replaces load().
    void load_quant(const std::string& tmq_path) {
        require_reload_safe();
        Llama next;
        next.copy_config_from(*this);
        // config.json sits next to the .tmq (same snapshot dir); overrides
        // the TinyLlama defaults for other families BEFORE kc/vc sizing.
        {
            const std::size_t slash = tmq_path.find_last_of('/');
            next.model_dir_ = slash == std::string::npos
                ? std::string(".") : tmq_path.substr(0, slash);
            next.apply_config(next.model_dir_ + "/config.json");
        }
        // TM_DEMAND=1: per-segment reader (docs/ON_DEMAND_WEIGHTS.md). Every
        // tensor is faulted in here and held; eviction is the hot-loop
        // follow-up. Demand mode is CPU-only — the GPU decode/prefill paths
        // register zero-copy Metal views over a stable whole-file mapping,
        // which per-segment munmap eviction would invalidate.
        static const bool demand = [] {
            const char* e = std::getenv("TM_DEMAND");
            return e && e[0] == '1';
        }();
        if (demand) {
            // CPU-only enforcement lives in demand_mode_/resolve_decode_gpu_auto().
            next.held_demand_ = std::make_unique<tmmq::Demand>();
            if (!next.held_demand_->open(tmq_path))
                throw std::runtime_error("tmq open failed: " + tmq_path);
            for (const auto& s : next.held_demand_->segs) {
                const bool q8 = s.dtype == 0;
                const auto& shape = s.shape;
                if (shape.empty() || shape.size() > 2)
                    throw std::runtime_error("llama: unsupported tensor rank: " + s.name);
                for (auto d : shape)
                    if (d > (std::uint32_t)std::numeric_limits<int>::max())
                        throw std::runtime_error("llama: tensor dimension exceeds int: " + s.name);
                if (shape.size() == 2) {
                    const bool is_f16 = s.dtype == 3;
                    if (!is_f16 && shape[1] % tmmq::block_elems(s.dtype))
                        throw std::runtime_error("llama: partial quantized row: " + s.name);
                    next.qshape_[s.name] = {(int)shape[0], (int)shape[1]};
                    // embed/lm_head are hot every token and stay pinned; the
                    // per-layer weights fault lazily in demand_layer() (null
                    // pointer here = metadata only, not yet resident).
                    const bool pinned = s.name == "model.embed_tokens.weight" ||
                                        s.name == "lm_head.weight";
                    const void* blocks = nullptr;
                    if (pinned) {
                        blocks = next.held_demand_->data(s.name);
                        if (!blocks)
                            throw std::runtime_error("llama: demand fault failed: " + s.name);
                    }
                    if (is_f16) next.f16_[s.name] = (const std::uint16_t*)blocks;
                    else if (q8) next.q8_[s.name] = static_cast<const tmq::BlockQ8_0*>(static_cast<const void*>(blocks));
                    else if (s.dtype == 1) next.q4_[s.name] = static_cast<const tmq::BlockQ4_0*>(static_cast<const void*>(blocks));
                    else next.qk_[s.name] = {blocks, s.dtype};
                } else {
                    // 1-D (norms, biases, ssm params): dequantize (or raw-F32
                    // copy for dtype 2) into w, then release the segment — they
                    // are never re-faulted.
                    const void* blocks = next.held_demand_->data(s.name);
                    if (!blocks)
                        throw std::runtime_error("llama: demand fault failed: " + s.name);
                    const int n = (int)shape[0];
                    next.w.emplace(s.name, Tensor(std::vector<int>{n}));
                    if (s.dtype == 2)
                        std::memcpy(next.w.at(s.name).data(), blocks, (std::size_t)n * sizeof(float));
                    else
                        dequant_row_any(blocks, s.dtype, n, next.w.at(s.name).data());
                    // Best-effort reclaim: a refused munmap leaves this one
                    // segment resident, so the window is larger than intended —
                    // never wrong data, which is why it is not an error here.
                    (void)next.held_demand_->evict(s.name);
                }
            }
        } else {
            next.held_map_ = std::make_unique<tmmq::Mapped>();
            if (!next.held_map_->open(tmq_path))
                throw std::runtime_error("tmq open failed: " + tmq_path);
            for (const auto& [name, ti] : next.held_map_->tensors) {
                const bool q8 = ti.dtype == 0;
                if (ti.shape.empty() || ti.shape.size() > 2)
                    throw std::runtime_error("llama: unsupported tensor rank: " + name);
                for (auto d : ti.shape)
                    if (d > (std::uint32_t)std::numeric_limits<int>::max())
                        throw std::runtime_error("llama: tensor dimension exceeds int: " + name);
                if (ti.shape.size() == 2) {
                    const bool is_f16 = ti.dtype == 3;
                    const std::uint32_t be = tmmq::block_elems(ti.dtype);
                    if (!is_f16 && ti.shape[1] % be)
                        throw std::runtime_error("llama: partial quantized row: " + name);
                    if (is_f16) next.f16_[name] = (const std::uint16_t*)ti.blocks;
                    else if (q8) next.q8_[name] = static_cast<const tmq::BlockQ8_0*>(static_cast<const void*>(ti.blocks));
                    else if (ti.dtype == 1) next.q4_[name] = static_cast<const tmq::BlockQ4_0*>(static_cast<const void*>(ti.blocks));
                    else next.qk_[name] = {ti.blocks, ti.dtype};
                    next.qshape_[name] = {(int)ti.shape[0], (int)ti.shape[1]};
                } else if (ti.shape.size() == 1) {
                    // 1-D (norms, biases, ssm params): dtype 2 = raw F32 in
                    // `blocks`; otherwise dequantize the block into w.
                    const int n = (int)ti.shape[0];
                    next.w.emplace(name, Tensor(std::vector<int>{n}));
                    if (ti.dtype == 2) {
                        std::memcpy(next.w.at(name).data(), ti.blocks,
                                    (std::size_t)n * sizeof(float));
                    } else {
                        dequant_row_any(ti.blocks, ti.dtype, n, next.w.at(name).data());
                    }
                }
            }
        }
        next.alias_tied_lm_head();
        next.alias_mtp_weights();
        next.validate_weights();
        next.auto_size_ctx();
        next.auto_size_demand_window();
        next.allocate_kv();
        commit_loaded(next);
    }

    ~Llama() { gpu_stack_teardown(); }

    void load(const std::string& dir) {
        require_reload_safe();
        Llama next;
        next.copy_config_from(*this);
        next.apply_config(dir + "/config.json");
        std::uint64_t db = 0; std::string meta;
        auto entries = tmsf::read_header(dir + "/model.safetensors", db, meta);
        std::ifstream f(dir + "/model.safetensors", std::ios::binary);
        for (auto& [name, e] : entries) {
            if (e.dtype != "F32") throw std::runtime_error("llama: F32 only");
            next.w.emplace(name, tmsf::load_tensor(f, e, db));
        }
        next.alias_tied_lm_head();
        next.alias_mtp_weights();
        next.validate_weights();
        next.auto_size_ctx();
        next.allocate_kv();
        commit_loaded(next);
    }

private:
    void require_reload_safe() const {
        // Metal handles may refer to the old mmap or pointer-keyed copied weights.
        // Do not reset process-global GPU state belonging to another live model.
        if (gpu_weights_used_ || gpu_w_ready_)
            throw std::runtime_error("llama: reload after Metal weight use is unsupported; start a new inference process");
    }
    void copy_config_from(const Llama& from) noexcept {
        L = from.L; D = from.D; H = from.H; KVH = from.KVH; dh = from.dh;
        V = from.V; ctx = from.ctx; F = from.F; theta = from.theta;
        eps = from.eps; REP_ = from.REP_;
        layer_types_ = from.layer_types_;
        full_attention_interval_ = from.full_attention_interval_;
        head_dim_ = from.head_dim_; n_rot_ = from.n_rot_;
        lk_heads_ = from.lk_heads_; lv_heads_ = from.lv_heads_;
        lk_dim_ = from.lk_dim_; lv_dim_ = from.lv_dim_;
        conv_kernel_ = from.conv_kernel_;
        ssm_state_ = from.ssm_state_; ssm_rank_ = from.ssm_rank_;
        ssm_groups_ = from.ssm_groups_;
        ts_min_ = from.ts_min_; ts_max_ = from.ts_max_; ts_floor_ = from.ts_floor_;
        partial_rotary_ = from.partial_rotary_;
        mtp_layers_ = from.mtp_layers_;
        mtp_layer_ = from.mtp_layer_;
        attn_output_gate_ = from.attn_output_gate_;
        tie_word_embeddings_ = from.tie_word_embeddings_;
        mrope_interleaved_ = from.mrope_interleaved_;
        linear_layers_ = from.linear_layers_;
        // Load-derived, but it must survive the swap: validate_weights() sets it
        // on the staging object, and the forward path reads it from the live one.
        qk_norm_ = from.qk_norm_;
    }
    void validate_config() const {
        if (L <= 0 || D <= 0 || H <= 0 || KVH <= 0 || ctx <= 0 || dh <= 0 ||
            V <= 0 || F <= 0 || D % H || H % KVH || REP_ != H / KVH ||
            (head_dim_ <= 0 && dh != D / H) || dh % 2 ||
            F % tmq::kBlock || !std::isfinite(eps) || eps <= 0.f ||
            !std::isfinite(theta) || theta <= 0.f)
            throw std::runtime_error("llama: invalid model dimensions");
        if (is_hybrid()) {
            // qwen35: the Gated-DeltaNet layer geometry, per the reference
            // (qwen35.cpp load_arch_tensors). head_k_dim == head_v_dim ==
            // linear_*_head_dim, and the conv spans q|k|v of the fused qkv proj.
            // Required only when the model declares linear layers: an interval of
            // 1 means every layer is full attention, and a fixture may omit them.
            const bool has_linear =
                lk_heads_ > 0 || lv_heads_ > 0 ||
                std::any_of(layer_types_.begin(), layer_types_.end(),
                            [](int v) { return v == 1; });
            if (has_linear && (head_dim_ <= 0 || lk_heads_ <= 0 || lv_heads_ <= 0 ||
                lk_dim_ <= 0 || lv_dim_ <= 0 || conv_kernel_ < 2 ||
                lv_heads_ % lk_heads_ || lk_dim_ % 2 || lv_dim_ % 2 ||
                (head_dim_ % 2) || n_rot_ <= 0 || n_rot_ > dh || n_rot_ % 2))
                throw std::runtime_error("llama: invalid qwen35 linear-attention dimensions");
        }
    }
public:
    // qwen35 keeps its MTP (multi-token-prediction) head as an extra layer whose
    // tensors the converter namespaces `mtp.*`. Aliasing them onto
    // model.layers.<L>.* lets the ordinary dense layer path run the head
    // unchanged: block() already implements exactly the attention, QK norms,
    // partial RoPE, sigmoid output gate and SwiGLU FFN the reference builds in
    // build_nextn_layer.
    void alias_mtp_weights() {
        static const char* probe = "mtp.eh_proj.weight";
        if (!qshape_.count(probe) && !f16_.count(probe) && !q4_.count(probe) &&
            !q8_.count(probe))
            return;
        mtp_layer_ = L;
        mtp_layers_ = 1;
        const std::string dst = "model.layers." + std::to_string(L) + ".";
        auto alias = [&](const char* suffix) {
            const std::string src = std::string("mtp.") + suffix;
            if (auto it = qshape_.find(src); it != qshape_.end()) qshape_[dst + suffix] = it->second;
            if (auto it = q4_.find(src); it != q4_.end()) q4_[dst + suffix] = it->second;
            if (auto it = q8_.find(src); it != q8_.end()) q8_[dst + suffix] = it->second;
            if (auto it = f16_.find(src); it != f16_.end()) f16_[dst + suffix] = it->second;
            // qk_ is NOT optional here. A checkpoint that quantizes its MTP head as a K-quant
            // (Qwen3.8-4B-Distill Q6_K does) keeps those rows in qk_, so omitting this alias
            // leaves the head's layer slot empty and block() throws `map::at` on the first
            // draft — which is how this was found. The plumbing had only ever run against
            // gen_mtp_fixture.py, whose head is F16/Q8_0, so the gap could not show up there.
            if (auto it = qk_.find(src); it != qk_.end()) qk_[dst + suffix] = it->second;
            if (auto it = w.find(src); it != w.end()) w.emplace(dst + suffix, it->second);
        };
        for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight",
                              "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                              "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                              "mlp.gate_proj.weight", "mlp.up_proj.weight",
                              "mlp.down_proj.weight"})
            alias(n);
    }

    bool mtp_ready() const { return mtp_layer_ >= 0; }

    // Tied embeddings: a pack with `tie_word_embeddings: true` ships no
    // `lm_head.weight` — the head IS the embedding table. The forward path
    // resolves the head by name, so bind the same block pointer under both
    // names. Zero copy: one tensor, two keys; nothing is duplicated, which
    // matters because duplicating a 400 MB embedding table to satisfy a lookup
    // would undo the point of the ternary pack.
    void alias_tied_lm_head() {
        if (!tie_word_embeddings_) return;
        const char* dst = "lm_head.weight";
        const char* src = "model.embed_tokens.weight";
        const bool head_present =
            qshape_.count(dst) || f16_.count(dst) || q4_.count(dst) ||
            q8_.count(dst) || qk_.count(dst) || w.count(dst);
        const bool src_present =
            qshape_.count(src) || f16_.count(src) || q4_.count(src) ||
            q8_.count(src) || qk_.count(src) || w.count(src);
        if (head_present || !src_present) return;
        if (auto it = qshape_.find(src); it != qshape_.end()) qshape_[dst] = it->second;
        if (auto it = q4_.find(src); it != q4_.end()) q4_[dst] = it->second;
        if (auto it = q8_.find(src); it != q8_.end()) q8_[dst] = it->second;
        if (auto it = qk_.find(src); it != qk_.end()) qk_[dst] = it->second;
        if (auto it = f16_.find(src); it != f16_.end()) f16_[dst] = it->second;
        if (auto it = w.find(src); it != w.end()) w.emplace(dst, it->second);
    }

    // One MTP step: given the trunk's hidden state h for the token it just
    // emitted (`next_token`), predict the token AFTER it. That is what a
    // speculative decoder drafts with and then verifies against the trunk.
    //
    // Reference (llama.cpp qwen35.cpp build_nextn_layer):
    //   e = embed(next_token);  concat = [rms(e, enorm) | rms(h, hnorm)]
    //   x = eh_proj @ concat;   x = <one full-attention layer>(x)
    //   logits = shared_head @ rms(x, shared_head_norm)
    std::vector<float> mtp_step(const std::vector<float>& h, int next_token, int pos) {
        if (!mtp_ready())
            throw std::runtime_error("llama: this model has no MTP head");
        if ((int)h.size() < D) throw std::invalid_argument("llama: mtp h too short");
        const auto Wm = [&](const char* k) -> const float* {
            return w.at(std::string("mtp.") + k).data();
        };
        if (mtp_h_.size() < (std::size_t)D) mtp_h_.resize(D);
        if (mtp_e_.size() < (std::size_t)D) mtp_e_.resize(D);
        if (mtp_c_.size() < (std::size_t)2 * D) mtp_c_.resize((std::size_t)2 * D);
        if (mtp_x_.size() < (std::size_t)D) mtp_x_.resize(D);
        // e = the NEXT token's embedding (the head's own table when it has one)
        const std::string emb = qshape_.count("mtp.nextn_embed_tokens.weight")
                                    ? "mtp.nextn_embed_tokens.weight"
                                    : "model.embed_tokens.weight";
        row_w(emb, next_token, mtp_e_.data());
        rms(h.data(), mtp_h_.data(), Wm("hnorm.weight"), D, eps);
        rms(mtp_e_.data(), mtp_e_.data(), Wm("enorm.weight"), D, eps);
        std::copy_n(mtp_e_.data(), D, mtp_c_.data());
        std::copy_n(mtp_h_.data(), D, mtp_c_.data() + D);
        linear_proj("mtp.eh_proj.weight", mtp_c_.data(), 1, 2 * D, D, mtp_x_.data());
        // the head layer proper, writing its own KV slot at `pos`
        block(mtp_layer_, mtp_x_, 1, pos);
        // shared output norm + LM head (both overridable by the head's own)
        if (xf_.size() < (std::size_t)D) xf_.resize(D);
        const float* nw = w.count("mtp.shared_head_norm.weight")
                              ? Wm("shared_head_norm.weight")
                              : w.at("model.norm.weight").data();
        rms(mtp_x_.data(), xf_.data(), nw, D, eps);
        const std::string hd = (f16_.count("mtp.shared_head_head.weight") ||
                                q4_.count("mtp.shared_head_head.weight") ||
                                q8_.count("mtp.shared_head_head.weight"))
                                   ? "mtp.shared_head_head.weight" : "lm_head.weight";
        if (logits_.size() != (std::size_t)V) logits_.resize(V);
        gemv_w(hd, xf_.data(), D, V, logits_.data());
        return logits_;
    }

    // ---- MTP speculative decoding -------------------------------------------
    //
    // The MTP head predicts the token AFTER the one the trunk just emitted, so a
    // decoder can draft that token, let the trunk verify it inside one batched
    // forward, and keep it when the trunk agrees. Greedy verification is exact:
    // the emitted sequence is bit-identical to a plain greedy loop, because every
    // accepted token is one the trunk itself argmaxed at that position (see
    // test_mtp.cpp, which asserts the equivalence rather than assuming it).
    //
    // Two things make verification more than a bookkeeping exercise here:
    //   * logits for EVERY position of the verify pass are needed, not just the
    //     last (forward() gained an optional out-parameter for that);
    //   * a rejected draft has already mutated the Gated-DeltaNet recurrent state
    //     of the linear layers, which — unlike the position-addressed KV cache —
    //     cannot be rewound by moving `seen` back. It is snapshotted and
    //     restored instead.

    int argmax_of(const std::vector<float>& lg) const {
        return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    }

    void set_spec_draft(SpecDraftFn fn, void* ctx) {
        spec_draft_ = fn;
        spec_draft_ctx_ = ctx;
    }

    // Is a labelled draft available for the NEXT round? Both the trunk's KV and
    // the recurrent state must be rewindable, which rules out a GPU-resident KV.
    bool spec_ready() const {
        if (gpu_kv_uploaded_) return false;
        return mtp_ready() || spec_draft_ != nullptr;
    }

    void snap_states() {
        if ((int)ssm_snap_.size() != L) ssm_snap_.resize(L);
        if ((int)conv_snap_.size() != L) conv_snap_.resize(L);
        for (int l = 0; l < L; ++l) {
            ssm_snap_[l] = ssm_st_[l];
            conv_snap_[l] = conv_st_[l];
        }
    }

    void restore_states() {
        for (int l = 0; l < L; ++l) {
            if (ssm_snap_[l].size() == ssm_st_[l].size()) ssm_st_[l] = ssm_snap_[l];
            if (conv_snap_[l].size() == conv_st_[l].size()) conv_st_[l] = conv_snap_[l];
        }
    }

    // Greedy decoding with one MTP lookahead per round. Identical output to the
    // plain loop `x = first; repeat { y = argmax(forward(x)); x = y }`; the
    // difference is that a round spends one batched forward over two positions
    // and can therefore advance two tokens.
    //
    //   *drafted / *accepted / *rejected, when non-null, receive the round counts
    //   (acceptance rate = accepted / drafted; 0/0 when the head drafts nothing).
    //
    // Without an MTP head and without a set_spec_draft() hook this degenerates to
    // the plain loop — which is what makes it usable as its own reference.
    std::vector<int> spec_generate_greedy(int first, int n, int* drafted = nullptr,
                                          int* accepted = nullptr,
                                          int* rejected = nullptr) {
        // Read once, not per round. This switch guards a stderr write inside the verify loop, and
        // getenv() walks the environment array on every call — a debug flag that costs work on
        // the path it is meant to observe is the same defect as a printf left in a hot loop.
        static const bool spec_debug_ = getenv("TM_SPEC_DEBUG") != nullptr;
        // Where a speculative round actually spends its time. The opt-in note on
        // spec_bench blamed snap_states() for the 0.82x, but the snapshot is
        // linear_layers_ * (lv_heads_*lk_dim_*lv_dim_ + (conv_kernel_-1)*conv_channels)
        // floats -- ~50 MB on Qwen3.8-4B -- which is ~1.5 ms of memcpy per round
        // against rounds that cost hundreds of ms. That is an estimate, not a
        // measurement, which is exactly what kept the path opt-in. TM_SPEC_TIMING=1
        // prints the four terms separately so the fix targets the real one.
        static const bool spec_time_ = getenv("TM_SPEC_TIMING") != nullptr;
        using spec_clk = std::chrono::steady_clock;
        // A tick that costs nothing when the flag is off. The note on spec_debug_
        // above states the rule for this file: a debug switch must not do work on
        // the path it observes, and an unconditional steady_clock::now() per round
        // (or per verify) would be exactly that.
        auto spec_tick = [&]() {
            return spec_time_ ? spec_clk::now() : spec_clk::time_point{};
        };
        auto spec_ms = [](spec_clk::time_point a, spec_clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const auto spec_t0 = spec_tick();
        double tm_draft = 0, tm_snap = 0, tm_verify = 0, tm_replay = 0;
        int rounds = 0;
        int nd = 0, na = 0, nr = 0;
        std::vector<int> out;
        if (n > 0) out.push_back(first);
        if (n <= 1) {
            if (drafted) *drafted = nd;
            if (accepted) *accepted = na;
            if (rejected) *rejected = nr;
            return out;
        }
        std::vector<float> lg = forward({first});   // copy: forward returns an alias
        std::vector<float> rows;                    // verify-pass logits, T x V
        while ((int)out.size() < n) {
            const int t = argmax_of(lg);
            int d = -1;
            if (spec_ready() && (int)out.size() + 1 < n) {
                ++rounds;
                const auto c0 = spec_tick();
                if (spec_draft_) d = spec_draft_(spec_draft_ctx_, out, lg);
                else d = argmax_of(mtp_step(hlast_, t, seen));
                if (spec_time_) tm_draft += spec_ms(c0, spec_clk::now());
            }
            if (d < 0) {               // nothing to verify: one plain step
                lg = forward({t});
                out.push_back(t);
                continue;
            }
            ++nd;
            const int p0 = seen;
            const bool snap = linear_layers_ > 0;
            if (snap) {
                const auto c0 = spec_tick();
                snap_states();
                if (spec_time_) tm_snap += spec_ms(c0, spec_clk::now());
            }
            const auto cv = spec_tick();
            forward({t, d}, -1, &rows);     // one batched pass over both positions
            if (spec_time_) tm_verify += spec_ms(cv, spec_clk::now());
            const float* r0 = rows.data();
            const int y1 = (int)(std::max_element(r0, r0 + V) - r0);
            if (spec_debug_)
                fprintf(stderr, "[spec] p0=%d t=%d d=%d row0argmax=%d %s\n", p0, t, d,
                        y1, d == y1 ? "accept" : "reject");
            if (d == y1) {
                ++na;
                out.push_back(t);
                out.push_back(d);
                // rows[1] predicts the token after `d`: it is the next round's
                // starting logits, which is where the second token comes from.
                lg.assign(rows.begin() + V, rows.end());
            } else {
                ++nr;
                const auto c0 = spec_tick();
                if (snap) {
                    // The Gated-DeltaNet state is not position-addressed: rolling
                    // the snapshot back undoes d, but it also undoes t, which is
                    // accepted. `t` is replayed at its own position so the state
                    // advances by it exactly once. (Correctness first; a linear
                    // model pays one token of work per rejection.)
                    restore_states();
                    seen = p0;
                    lg = forward({t});
                } else {
                    // Position-addressed KV: t's row is already written and row 0
                    // of the verify pass IS the standalone logits for position p0,
                    // so the rejection costs nothing beyond dropping d.
                    seen = p0 + 1;
                    lg.assign(rows.begin(), rows.begin() + V);
                }
                if (spec_time_) tm_replay += spec_ms(c0, spec_clk::now());
                // Rows past the rewind point are stale relative to the GPU mirror,
                // if one exists: keep the upload window honest (the GPU upload
                // clamps by `seen`, and this keeps the invariant it relies on).
                gpu_kv_valid_ = std::min(gpu_kv_valid_, seen);
                out.push_back(t);
            }
        }
        if (spec_time_) {
            const double tot = spec_ms(spec_t0, spec_clk::now());
            const double other = tot - tm_draft - tm_snap - tm_verify - tm_replay;
            fprintf(stderr,
                    "[spec-time] rounds=%d drafted=%d accepted=%d rejected=%d | "
                    "draft %.1f ms  snap %.1f ms  verify %.1f ms  replay %.1f ms  "
                    "other %.1f ms  | total %.1f ms\n",
                    rounds, nd, na, nr, tm_draft, tm_snap, tm_verify, tm_replay, other, tot);
        }
        if (drafted) *drafted = nd;
        if (accepted) *accepted = na;
        if (rejected) *rejected = nr;
        return out;
    }

    // Not const: it records `qk_norm_`, the one derived property the forward
    // path needs and the only way to detect QK norms (see the member's note).
    void validate_weights() {
        validate_config();
        // Format-level bounds only establish that declared tensors fit the
        // file. Inference kernels use architectural dimensions, so validate
        // every required shape before allocating KV or publishing any views.
        const auto require_shape = [&](const std::string& name,
                                       const std::vector<int>& expected) {
            // Quantized 2-D weights live in qshape_ (their blocks are keyed
            // separately); F32 tensors — norms, the DeltaNet decay/bias vectors
            // and the depthwise conv kernel — are dense in w. Accept either.
            if (expected.size() == 2) {
                const auto found = qshape_.find(name);
                if (found != qshape_.end() &&
                    found->second == std::pair{expected[0], expected[1]})
                    return;
            }
            const auto found = w.find(name);
            if (found != w.end() && found->second.shape() == expected) return;
            throw std::runtime_error("llama: missing or incompatible required tensor: " + name);
        };
        // validate_config ensures KVH <= H and dh == D/H, so this product fits D.
        const int kv_width = KVH * dh;
        require_shape("model.embed_tokens.weight", {V, D});
        require_shape("lm_head.weight", {V, D});
        require_shape("model.norm.weight", {D});
        const int q_width = H * dh * (attn_output_gate_ ? 2 : 1);
        const int conv_ch = is_hybrid() ? linear_conv_channels() : 0;
        const int value_width = lv_heads_ * lv_dim_;
        qk_norm_ = false;
        for (int layer = 0; layer < L; ++layer) {
            const std::string prefix = "model.layers." + std::to_string(layer) + ".";
            require_shape(prefix + "input_layernorm.weight", {D});
            require_shape(prefix + "post_attention_layernorm.weight", {D});
            require_shape(prefix + "mlp.gate_proj.weight", {F, D});
            require_shape(prefix + "mlp.up_proj.weight", {F, D});
            require_shape(prefix + "mlp.down_proj.weight", {D, F});
            if (is_linear_layer(layer)) {
                // Gated DeltaNet layer (llama.cpp qwen35.cpp load_block_trunk,
                // is_recr branch).
                require_shape(prefix + "linear_attn.qkv_proj.weight", {conv_ch, D});
                require_shape(prefix + "linear_attn.gate_proj.weight", {value_width, D});
                require_shape(prefix + "linear_attn.conv1d.weight", {conv_kernel_, conv_ch});
                require_shape(prefix + "linear_attn.dt_proj.bias", {lv_heads_});
                require_shape(prefix + "linear_attn.a_log", {lv_heads_});
                require_shape(prefix + "linear_attn.alpha.weight", {lv_heads_, D});
                require_shape(prefix + "linear_attn.beta.weight", {lv_heads_, D});
                require_shape(prefix + "linear_attn.norm.weight", {lv_dim_});
                require_shape(prefix + "linear_attn.out_proj.weight", {D, value_width});
                continue;
            }
            require_shape(prefix + "self_attn.q_proj.weight", {q_width, D});
            require_shape(prefix + "self_attn.k_proj.weight", {kv_width, D});
            require_shape(prefix + "self_attn.v_proj.weight", {kv_width, D});
            require_shape(prefix + "self_attn.o_proj.weight", {D, H * dh});
            // Require the QK norms only where the checkpoint actually has them.
            if (qshape_.count(prefix + "self_attn.q_norm.weight") != 0 ||
                w.count(prefix + "self_attn.q_norm.weight") != 0) {
                require_shape(prefix + "self_attn.q_norm.weight", {dh});
                require_shape(prefix + "self_attn.k_norm.weight", {dh});
                qk_norm_ = true;
            }
        }
    }
    // ---- packed KV helpers (TM_KV_*; specs/TENSORMARK_KV_Q8_SPEC.md) --------
    // Row width in bytes: dh/32 Q8_0 blocks (34 B each) or dh floats.
    std::size_t kv_row_bytes(int type) const {
        return type ? (std::size_t)(dh / tmq::kBlock) * sizeof(tmq::BlockQ8_0)
                    : (std::size_t)dh * sizeof(float);
    }
    // How many KV slots EXIST: the trunk plus the MTP head's own slot when the
    // checkpoint has one. This is allocate_kv()'s rule, and it lives here so the
    // allocation and the accounting cannot drift apart — they had, in both
    // directions (see kv_cache_slots below).
    int kv_slot_total() const { return L + (mtp_layers_ > 0 ? 1 : 0); }
    // How many of those slots own a real K/V cache. The Gated-DeltaNet layers
    // hold O(1) recurrent state instead, so they are not a cache cost. NOTE this
    // is deliberately NOT kv_layer_count(): that one counts the *stack's*
    // attention layers and excludes the MTP slot, which is right for per-layer
    // work and wrong for a memory budget — the MTP slot is allocated like any
    // other and is part of the footprint even when speculation is off. Using the
    // layer count here under-measured the cache by one slot (12.5 % on an 8+1
    // hybrid model).
    int kv_cache_slots() const {
        int n = 0;
        for (int l = 0; l < kv_slot_total(); ++l)
            if (!is_linear_layer(l)) ++n;
        return n;
    }
    // Per-token cost over the slots that actually cache — the convention
    // auto_size_ctx sizes the context with.
    std::size_t kv_bytes_per_token() const {
        return (std::size_t)kv_cache_slots() * (std::size_t)KVH *
               (kv_row_bytes(kv_ktype()) + kv_row_bytes(kv_vtype()));
    }
    // The same count at the full context — what the GPU budget estimates add on
    // top of the weights. It used to be `L * KVH * ctx * row`, i.e. the TRUNK
    // length for every layer: 32 slots for a model whose hybrid geometry has 9,
    // a 3.56x over-estimate, and it excluded the MTP slot for a dense model at
    // the same time.
    std::size_t kv_cache_bytes_all_layers() const {
        return (std::size_t)kv_cache_slots() * (std::size_t)KVH * (std::size_t)ctx *
               (kv_row_bytes(kv_ktype()) + kv_row_bytes(kv_vtype()));
    }
    // TM_KV_SELFCHECK=1: the stored blocks must be a fixed point of
    // dequantize -> quantize. Q8_0 is idempotent by construction (the
    // dequantized row's amax is the d it was built from), so a mismatch is a
    // row-addressing or layout bug — the exact failure class a packed cache
    // introduces, and one that is otherwise silent.
    void kv_selfcheck_rows(int l, int first, int count) {
        const std::size_t nb = (std::size_t)(dh / tmq::kBlock);
        const std::size_t row_bytes = nb * sizeof(tmq::BlockQ8_0);
        if (kv_chk_.size() < (std::size_t)dh) kv_chk_.resize((std::size_t)dh);
        std::vector<std::uint8_t> back(row_bytes);
        const auto check = [&](const std::vector<std::uint8_t>& buf, const char* which) {
            for (int kv = 0; kv < KVH; ++kv)
                for (int t = 0; t < count; ++t) {
                    const std::size_t row =
                        (std::size_t)kv * (std::size_t)ctx + (std::size_t)(first + t);
                    const std::uint8_t* src = buf.data() + row * row_bytes;
                    tmq::dequantize_row_q8_0(reinterpret_cast<const tmq::BlockQ8_0*>(src),
                                             kv_chk_.data(), (std::size_t)dh);
                    tmq::quantize_row_q8_0(kv_chk_.data(),
                                           reinterpret_cast<tmq::BlockQ8_0*>(back.data()),
                                           (std::size_t)dh);
                    if (std::memcmp(src, back.data(), row_bytes) != 0)
                        throw std::runtime_error(
                            std::string("llama: TM_KV_SELFCHECK failed: ") + which +
                            " layer " + std::to_string(l) + " kv " + std::to_string(kv) +
                            " row " + std::to_string(first + t));
                }
        };
        if (kv_ktype()) check(kq_[(std::size_t)l], "K");
        if (kv_vtype()) check(vq_[(std::size_t)l], "V");
    }
    // ---- packed-KV read primitives (step 2 of the spec) ---------------------
    // The query is quantized once per (token, head) and then dotted against each
    // K row with int8 vdot: no dequantized row exists anywhere, which is the
    // constraint the "no storage-only phase" arithmetic imposes. V has no such
    // trick — its weight is a per-position float — so that row is dequantized
    // straight into the accumulator, one pass, every value used once.
    std::vector<std::int8_t> kv_qs_;      // quantized query values
    std::vector<std::uint16_t> kv_qd_;    // per-block query scales (fp16 bits)
    void kv_quant_query(const float* q, int dh) {
        const int nb = dh / tmq::kBlock;
        if (kv_qs_.size() < (std::size_t)dh) kv_qs_.resize((std::size_t)dh);
        if (kv_qd_.size() < (std::size_t)nb) kv_qd_.resize((std::size_t)nb);
        for (int b = 0; b < nb; ++b) {
            const float* x = q + (std::size_t)b * tmq::kBlock;
            float amax = 0.f;
            for (int i = 0; i < tmq::kBlock; ++i) amax = std::max(amax, std::fabs(x[i]));
            const std::uint16_t d16 = tmq::fp32_to_fp16(amax / 127.f);
            const float d = tmq::fp16_to_fp32(d16);
            kv_qd_[(std::size_t)b] = d16;
            const float inv = d == 0.f ? 0.f : 1.f / d;
            for (int i = 0; i < tmq::kBlock; ++i) {
                const float v = std::min(std::max(x[i] * inv, -127.f), 127.f);
                kv_qs_[(std::size_t)b * tmq::kBlock + i] = (std::int8_t)std::lrintf(v);
            }
        }
    }
    float kv_row_dot(const tmq::BlockQ8_0* row, int nb) const {
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            const std::int8_t* r = row[b].qs;
            const std::int8_t* qb = kv_qs_.data() + (std::size_t)b * tmq::kBlock;
            std::int32_t dot = 0;
#if defined(__aarch64__)
            const int8x16_t r0 = vld1q_s8(r), r1 = vld1q_s8(r + 16);
            const int8x16_t q0 = vld1q_s8(qb), q1 = vld1q_s8(qb + 16);
            dot = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), r0, q0), r1, q1));
#else
            for (int i = 0; i < tmq::kBlock; ++i) dot += (int)r[i] * (int)qb[i];
#endif
            acc += (float)dot * (tmq::fp16_to_fp32(row[b].d_fp16) *
                                 tmq::fp16_to_fp32(kv_qd_[(std::size_t)b]));
        }
        return acc;
    }
    static void kv_row_axpy(float w, const tmq::BlockQ8_0* row, float* acc, int nb) {
        for (int b = 0; b < nb; ++b) {
            const float d = w * tmq::fp16_to_fp32(row[b].d_fp16);
            const std::int8_t* r = row[b].qs;
            float* a = acc + (std::size_t)b * tmq::kBlock;
#if defined(__aarch64__)
            for (int i = 0; i < tmq::kBlock; i += 8) {
                const int16x8_t v = vmovl_s8(vld1_s8(r + i));
                const float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v)));
                const float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v)));
                vst1q_f32(a + i, vfmaq_n_f32(vld1q_f32(a + i), lo, d));
                vst1q_f32(a + i + 4, vfmaq_n_f32(vld1q_f32(a + i + 4), hi, d));
            }
#else
            for (int i = 0; i < tmq::kBlock; ++i) a[i] += d * (float)r[i];
#endif
        }
    }
    void allocate_kv() {
        validate_config();
        std::size_t slots = (std::size_t)ctx;
        for (int factor : {KVH, dh}) {
            if (slots > std::vector<float>().max_size() / (std::size_t)factor)
                throw std::runtime_error("llama: KV dimensions overflow");
            slots *= (std::size_t)factor;
        }
        // Only full-attention layers own a KV cache; the Gated-DeltaNet layers
        // carry O(1) recurrent state instead (allocated below). Leaving the
        // linear slots empty is what makes a hybrid model fit: for the 27B it
        // removes 48/64 of the KV footprint.
        // The MTP head owns a KV cache of its own, in a slot one past the trunk.
        const int nslots = kv_slot_total();
        kc.clear(); vc.clear();
        kc.resize((std::size_t)nslots); vc.resize((std::size_t)nslots);
        for (int l = 0; l < nslots; ++l) {
            if (is_linear_layer(l)) continue;
            // Only for tensors that still use fp32: a packed tensor's rows live
            // in kq_/vq_, and leaving this empty is what makes the memory win
            // real. Every reader must therefore be type-aware — the fp32 readers
            // all live behind a `kbuf`/`vbuf` test, and the append below is
            // guarded on emptiness.
            if (!kv_ktype()) kc[(std::size_t)l] = PageVec(slots);
            if (!kv_vtype()) vc[(std::size_t)l] = PageVec(slots);
        }
        // Packed copy of the same rows (TM_KV_*). A model whose head dim does not
        // tile into complete Q8_0 blocks stays fp32 and says so, once per process
        // (kv_ktype() carries the fallback; nothing is mutated).
        if ((kv_kt_ || kv_vt_) && dh % tmq::kBlock != 0) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "llama: TM_KV_* ignored — dh %d is not a multiple of %d\n",
                             dh, tmq::kBlock);
            }
        }
        kq_.clear(); vq_.clear();
        kq_.resize((std::size_t)nslots); vq_.resize((std::size_t)nslots);
        {
            const std::size_t pack_row =
                (std::size_t)(dh / tmq::kBlock) * sizeof(tmq::BlockQ8_0);
            for (int l = 0; l < nslots; ++l) {
                if (is_linear_layer(l)) continue;
                if (kv_ktype())
                    kq_[(std::size_t)l].assign((std::size_t)KVH * (std::size_t)ctx * pack_row, 0);
                if (kv_vtype())
                    vq_[(std::size_t)l].assign((std::size_t)KVH * (std::size_t)ctx * pack_row, 0);
            }
        }
        if (is_hybrid()) {
            const std::size_t ssm_sz = (std::size_t)lv_heads_ *
                                       (std::size_t)lk_dim_ * (std::size_t)lv_dim_;
            const std::size_t conv_sz = (std::size_t)(conv_kernel_ - 1) *
                                        (std::size_t)linear_conv_channels();
            ssm_st_.assign((std::size_t)L, std::vector<float>(ssm_sz, 0.f));
            conv_st_.assign((std::size_t)L, std::vector<float>(conv_sz, 0.f));
            linear_layers_ = 0;
            for (int l = 0; l < L; ++l)
                if (is_linear_layer(l)) ++linear_layers_;
        }
    }

    // Number of layers holding a KV cache (all layers for dense models, the
    // full-attention subset for hybrids). The context budget is spent only on
    // the layers that actually cache K/V.
    int kv_layer_count() const {
        if (!is_hybrid()) return L;
        int n = 0;
        for (int l = 0; l < L; ++l)
            if (!is_linear_layer(l)) ++n;
        return n > 0 ? n : L;
    }
    void commit_loaded(Llama& next) noexcept {
        copy_config_from(next);
        w.swap(next.w); q8_.swap(next.q8_); q4_.swap(next.q4_);
        f16_.swap(next.f16_); qk_.swap(next.qk_);
        qshape_.swap(next.qshape_); std::swap(held_map_, next.held_map_);
        std::swap(held_demand_, next.held_demand_);
        model_dir_.swap(next.model_dir_);   // fp16 safetensors live next to the .tmq
        kc.swap(next.kc); vc.swap(next.vc);
        // Same rule as the recurrent state below: the packed KV is allocate_kv()'d
        // on `next`, so it must move over with the fp32 cache or the live object
        // keeps an empty vector and the append writes past the end (TM_KV_Q8=1
        // segfaulted on exactly that before this line existed).
        kq_.swap(next.kq_); vq_.swap(next.vq_);
        // The Gated-DeltaNet recurrent state is allocate_kv()'d on `next` like
        // the KV cache; it must move over with it, or the live object keeps an
        // empty vector and block_linear indexes out of bounds.
        ssm_st_.swap(next.ssm_st_); conv_st_.swap(next.conv_st_);
        gpu_kv_uploaded_ = false; gpu_kv_valid_ = 0; gpu_from_ = 0;   // GPU slots hold the old model's rows
        reset_cache();
        gpu_checked_ = false; gpu_ok_ = false;
        gpu_w_.clear(); gpu_w_ready_ = false; gpu_weights_used_ = false;
        gpu_shared_wbuf_ = false;
        gpu_bias_uploaded_ = false;   // SLOT_BIAS holds the old model's bias blob
        resolve_decode_gpu_auto();
        gpu_stack_setup();
    }

public:

    void reset_cache() {
        seen = 0;
        gpu_kv_uploaded_ = false;
        gpu_from_ = 0;
        gpu_kv_valid_ = 0;
        // Recurrent state is per-sequence conversation state, not weights:
        // a new sequence starts from zero.
        for (auto& s : ssm_st_) std::fill(s.begin(), s.end(), 0.f);
        for (auto& s : conv_st_) std::fill(s.begin(), s.end(), 0.f);
    }

    // Drop everything after token `n`, so the next forward appends from there.
    //
    // This is what makes a cached conversation reusable: a caller that already holds
    // a prefix can continue it instead of re-loading it, and a caller that wants a
    // prefix REBUILT (because the tokens that followed were sampled rather than
    // tokenized) can rewind and re-prefill from text — one token of difference is
    // enough to change the continuation, which is how the shim got a wrong answer
    // from a faster cache (docs/BENCHMARKS.md).
    //
    // Sound only where the state at a position is fully determined by the tokens
    // before it. The recurrent (Gated-DeltaNet) layers are not, so they refuse
    // rather than silently return a sequence that never happened.
    [[nodiscard]] bool rewind_to(int n) {
        if (!ssm_st_.empty() || !conv_st_.empty()) return false;
        if (n < 0 || n > seen) return false;
        seen = n;
        // Rows past `n` are unreachable, but the GPU copy has to be re-uploaded
        // rather than assumed: it was built for a longer sequence.
        gpu_kv_uploaded_ = false;
        gpu_from_ = 0;
        gpu_kv_valid_ = 0;
        return true;
    }

     // qwen35 / qwen3_5_text hybrid: interleaved Gated-DeltaNet (linear) and
     // full-attention layers. Detected from either the explicit `layer_types`
     // array or `full_attention_interval` ((l+1) % interval == 0 -> full attn,
     // per llama.cpp qwen35.cpp load_arch_hparams).
     bool is_hybrid() const {
         return !layer_types_.empty() || full_attention_interval_ > 0 ||
                lk_heads_ > 0;
     }
     bool is_linear_layer(int l) const {
         // The MTP head is a full-attention layer; its slot sits one past the
         // trunk, where (l+1) % interval would otherwise misfile it.
         if (l == mtp_layer_) return false;
         if (!layer_types_.empty())
             return l < (int)layer_types_.size() && layer_types_[(std::size_t)l] == 1;
         if (full_attention_interval_ > 0)
             return (l + 1) % full_attention_interval_ != 0;
         return false;
     }
     // Fused qkv width of a Gated-DeltaNet layer: q(D_k*H_k) | k(D_k*H_k) |
     // v(D_v*H_v), the reference's conv_dim = key_dim*2 + value_dim.
     int linear_conv_channels() const { return 2 * lk_heads_ * lk_dim_ + lv_heads_ * lv_dim_; }

    static void rms(const float* x, float* out, const float* wt, int n, float eps) {
#ifdef __ARM_NEON
        // 4-lane sum of squares + one fused scale pass (the scalar serial
        // chain was 285 ms of a 2000-token prefill across the two norms).
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            const float32x4_t v0 = vld1q_f32(x + i), v1 = vld1q_f32(x + i + 4);
            a0 = vfmaq_f32(a0, v0, v0); a1 = vfmaq_f32(a1, v1, v1);
        }
        float ss = vaddvq_f32(vaddq_f32(a0, a1));
        for (; i < n; ++i) ss += x[i] * x[i];
        const float r = 1.f / std::sqrt(ss / n + eps);
        const float32x4_t rv = vdupq_n_f32(r);
        i = 0;
        for (; i + 4 <= n; i += 4)
            vst1q_f32(out + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), rv), vld1q_f32(wt + i)));
        for (; i < n; ++i) out[i] = x[i] * r * wt[i];
#else
        float ss = 0.f;
        for (int i = 0; i < n; ++i) ss += x[i] * x[i];
        const float r = 1.f / std::sqrt(ss / n + eps);
        for (int i = 0; i < n; ++i) out[i] = x[i] * r * wt[i];
#endif
    }

    // RoPE inverse frequencies for the full head width and for the partial
    // rotation, plus the per-position cos/sin scratch rope_qk refills. All three
    // are keyed on (theta, width), so a manual theta/dh change on a loaded model
    // rebuilds them instead of leaving a stale table — see tmgdn::RoPeFreq.
    mutable tmgdn::RoPeFreq rope_freq_full_;
    mutable tmgdn::RoPeFreq rope_freq_partial_;
    mutable std::vector<float> rope_sincos_;

    // RoPE on one head (dh floats), absolute position p.
    void rope_head(float* r, int p) const {
        const std::span<const float> inv = rope_freq_full_.get(dh, theta);
        for (int j = 0; j < dh / 2; ++j) {
            // The phase is materialized and the paired libm operation is
            // spelled EXPLICITLY, matching rope_qk below. Leaving either path to
            // the compiler's own cos/sin spelling made the two disagree by 1 ULP
            // under sanitizer instrumentation (where Clang does not combine
            // separate sin/cos calls), and rope_qk's shared-table path must stay
            // bit-identical to this one — the reload gate memcmps them.
            const float phase = (float)p * inv[(std::size_t)j];
            float c, s;
            ::__sincosf(phase, &s, &c);
            const float a = r[j], b = r[j + dh / 2];
            r[j] = a * c - b * s;
            r[j + dh / 2] = b * c + a * s;
        }
    }

    // Each Q/K head at a position uses the same rotation. Share the expensive
    // transcendental evaluations without changing each pair's arithmetic: the
    // inverse frequencies come from the cached table (same pow, same argument)
    // and the cos/sin pair is evaluated once per position instead of once per
    // head, then shared by the rotate loop below.
    void rope_qk(float* q, float* k, int tokens, int pos0) const {
        // The query row stride is 2*H*dh when attn_q carries the interleaved
        // output gate: the split compacts the query into the front H*dh of each
        // row and parks the gate in the half beside it, so the rows are twice
        // the attention width. Assuming H*dh rotated the wrong memory for every
        // token past the first in a gated model whose RoPE is not partial —
        // qwen35/3.8 take the partial branch below and were unaffected, which is
        // why this survived. Equal to H*dh when ungated, so dense models are
        // bit-identical.
        const int qstride = attn_output_gate_ ? 2 * H * dh : H * dh;
        // Scalar and vector negated products can round differently in directed
        // modes. Preserve the established per-head implementation in that case.
        if (std::fegetround() != FE_TONEAREST) {
            for (int t = 0; t < tokens; ++t) {
                for (int hd = 0; hd < H; ++hd)
                    rope_head(q + (std::size_t)t * qstride + (std::size_t)hd * dh,
                              pos0 + t);
                for (int kv = 0; kv < KVH; ++kv)
                    rope_head(k + ((std::size_t)t * KVH + kv) * dh, pos0 + t);
            }
            return;
        }
        const int half = dh / 2;
        const std::span<const float> inv = rope_freq_full_.get(dh, theta);
        if ((int)rope_sincos_.size() < half * 2) rope_sincos_.resize((std::size_t)half * 2);
        float* cosines = rope_sincos_.data();
        float* sines = cosines + half;
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < half; ++j) {
                const float phase = (float)(pos0 + t) * inv[(std::size_t)j];
                ::__sincosf(phase, &sines[j], &cosines[j]);
            }
            const auto rotate = [&](float* r) {
                for (int j = 0; j < half; ++j) {
                    const float c = cosines[j], s = sines[j];
                    const float a = r[j], b = r[j + half];
                    r[j] = a * c - b * s;
                    r[j + half] = b * c + a * s;
                }
            };
            float* qt = q + (std::size_t)t * qstride;
            float* kt = k + (std::size_t)t * KVH * dh;
            for (int hd = 0; hd < H; ++hd) rotate(qt + (std::size_t)hd * dh);
            for (int kv = 0; kv < KVH; ++kv) rotate(kt + (std::size_t)kv * dh);
        }
    }

    // gpu2: one token, ONE command buffer. The CPU prefill fills kc/vc
    // (uploaded once on the first GPU token); afterwards the GPU KV cache
    // is authoritative for positions >= gpu_from_. A later multi-token
    // forward downloads the GPU rows back and returns to the CPU path.
    // Shared by forward_gpu_one and generate_greedy_gpu: weights, slots,
    // one-time KV upload.
    void gpu_token_prologue() {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        const int QDg = H * dh, KVDg = KVH * dh;
        ensure_gpu_weights();
        if (!tm_metal_tok_reserve(SLOT_QKV, (size_t)(QDg + 2 * KVDg) * 4) ||
            !tm_metal_tok_reserve(SLOT_GU, (size_t)2 * F * 4) ||
            !tm_metal_tok_reserve(SLOT_PROBS, (size_t)H * ctx * 4))
            throw std::runtime_error("metal: packed slot reservation failed");
        if (!gpu_kv_uploaded_) {
            // Only rows [0, seen) are ever read (allow = pos + 1), and rows
            // [0, gpu_kv_valid_) already match: upload [gpu_kv_valid_, seen)
            // per kv head instead of the whole ctx-sized cache.
            const int r0 = std::min(gpu_kv_valid_, seen);
            for (int l = 0; l < L; ++l) {
                // A hybrid's linear layers have no KV cache at all — kc[l] is
                // empty — and the slot allocator rejects a 0-byte request, so
                // this loop aborted the whole GPU decode lane on a model whose
                // attention path never touches those slots anyway.
                if (kc[l].empty()) continue;
                if (!tm_metal_tok_kv_reserve(SLOT_KC + l, kc[l].size()) ||
                    !tm_metal_tok_kv_reserve(SLOT_VC + l, vc[l].size()))
                    throw std::runtime_error("metal: KV cache slot reservation failed");
                for (int kv = 0; kv < KVH; ++kv) {
                    const size_t off = (size_t)kv * ctx * dh + (size_t)r0 * dh, n = (size_t)(seen - r0) * dh;
                    if (n && (!tm_metal_tok_kv_upload(SLOT_KC + l, off, kc[l].data() + off, n) ||
                              !tm_metal_tok_kv_upload(SLOT_VC + l, off, vc[l].data() + off, n)))
                        throw std::runtime_error("metal: KV cache upload failed");
                }
            }
            gpu_kv_valid_ = seen;
            gpu_kv_uploaded_ = true;
            gpu_from_ = seen;
        }
        // Qwen2 attention bias: upload the contiguous [q | k | v] bias blob for
        // every layer into SLOT_BIAS once (absent for Llama-2 -> a no-op).
        if (!gpu_bias_uploaded_ && w.count("model.layers.0.self_attn.q_proj.bias")) {
            const size_t per = (size_t)(QDg + 2 * KVDg);
            std::vector<float> blob((size_t)L * per);
            for (int l = 0; l < L; ++l) {
                const std::string pre = "model.layers." + std::to_string(l) + ".self_attn.";
                const float* bq = w.at(pre + "q_proj.bias").data();
                const float* bk = w.at(pre + "k_proj.bias").data();
                const float* bv = w.at(pre + "v_proj.bias").data();
                float* dst = blob.data() + (size_t)l * per;
                std::copy_n(bq, QDg, dst);
                std::copy_n(bk, KVDg, dst + QDg);
                std::copy_n(bv, KVDg, dst + QDg + KVDg);
            }
            if (!tm_metal_tok_reserve(SLOT_BIAS, blob.size() * 4) ||
                !tm_metal_tok_upload(SLOT_BIAS, 0, blob.data(), blob.size() * 4))
                throw std::runtime_error("metal: attention bias upload failed");
        }
        gpu_bias_uploaded_ = true;
#endif
    }
    // One token's kernels from the first layer norm through lm_head; x is
    // expected in SLOT_X, logits land in SLOT_LOGITS. Encoded into the open
    // token command buffer.
    void encode_gpu_token(int pos) {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        const int QDg = H * dh, KVDg = KVH * dh;
        const float scale = 1.f / std::sqrt((float)dh);
        // Weight-handle + GEMV dispatch: fp16 (TM_DECODE_F16) or Q4.
        // Per NAME, not per model: a qwen35 hybrid legitimately mixes Q4_0
        // dense weights (gpu_w_) with f16 Gated-DeltaNet weights (gpu_h_), and
        // the dispatch follows the tensor's own registration.
        const auto gw = [&](const std::string& n)
            -> const std::pair<int, uint64_t>& {
            // The fused q|k|v / gate|up seg kernel is Q4_0-only, so a K-quant tensor
            // reaching here is a bug in the caller's seg-vs-separate decision, not a
            // case to paper over with a Q4 offset.
            if (gpu_k_.count(n))
                throw std::runtime_error("metal: K-quant weight in the fused seg path: " + n);
            auto h = gpu_h_.find(n);
            if (h != gpu_h_.end()) return h->second;
            return gpu_w_.at(n);
        };
        const auto gmv = [&](int xslot, const std::string& n, int yslot,
                             size_t yoff, unsigned N, unsigned K) {
            // Per NAME, not per model: one shipped qwen35 checkpoint mixes dtypes
            // inside a single layer — Q4_0 qkv_proj/gate_proj, Q5_K out_proj, Q4_1
            // mlp.down_proj on the first layers, Q6_K lm_head.
            if (const auto k = gpu_k_.find(n); k != gpu_k_.end()) {
                tm_metal_tok_gemv_kq(xslot, k->second.id, k->second.woff, yslot, yoff,
                                     N, K, k->second.dtype);
                return;
            }
            auto h = gpu_h_.find(n);
            if (h != gpu_h_.end())
                tm_metal_tok_gemv_h(xslot, h->second.first, h->second.second,
                                    yslot, yoff, N, K);
            else {
                const auto& g = gpu_w_.at(n);
                tm_metal_tok_gemv(xslot, g.first, g.second, yslot, yoff, N, K);
            }
        };
        // Fused q|k|v / gate|up dispatch gate (A/B harness): default on,
        // TM_METAL_GEMV_FUSE=0 restores the separate per-tensor dispatches.
        static const bool gemv_fuse = [] {
            const char* e = std::getenv("TM_METAL_GEMV_FUSE");
            return !(e && e[0] == '0');
        }();
        for (int l = 0; l < L; ++l) {
            const std::string pre =
                "model.layers." + std::to_string(l) + ".";
            if (is_linear_layer(l))
                // Gated-DeltaNet layers never take the dense GPU decode lane. The
                // lane that served them (TM_METAL_GDN) was RETIRED (2026-09-17):
                // it was wrong on K-quant checkpoints and 0.6-0.9x the CPU when
                // correct. resolve_decode_gpu_auto() forces every hybrid onto the
                // CPU, so reaching here means lane admission and this loop disagree.
                throw std::runtime_error(
                    "metal: Gated-DeltaNet layer reached the dense GPU decode lane");
            tm_metal_tok_rmsnorm(SLOT_X,
                w.at(pre + "input_layernorm.weight").data(), SLOT_H,
                D, eps);
            // The fused q|k|v seg kernel is Q4_0-only, and gw() REFUSES a K-quant
            // tensor — so this decision has to be made before any handle is fetched.
            // Fetching them eagerly (as this did) threw mid-encode on a K-quant
            // layer, which leaves a half-encoded command buffer: fluent garbage
            // from token 1, invariant to every GEMV knob. Same species as the
            // gate|up site below, and the reason both now test gpu_k_ first.
            const bool qkv_seg_ok =
                gemv_fuse && !gpu_f16_ && gpu_shared_wbuf_ &&
                !gpu_k_.count(pre + "self_attn.q_proj.weight") &&
                !gpu_k_.count(pre + "self_attn.k_proj.weight") &&
                !gpu_k_.count(pre + "self_attn.v_proj.weight");
            if (!qkv_seg_ok) {
                gmv(SLOT_H, pre + "self_attn.q_proj.weight", SLOT_QKV, 0, QDg, D);
                gmv(SLOT_H, pre + "self_attn.k_proj.weight", SLOT_QKV, (size_t)QDg * 4, KVDg, D);
                gmv(SLOT_H, pre + "self_attn.v_proj.weight", SLOT_QKV, (size_t)(QDg + KVDg) * 4, KVDg, D);
            } else {
                // Fused q|k|v: one dispatch over QD+2*KVD rows, three weight
                // segments at their file offsets. Segment boundaries are 2-aligned
                // (all dims are 32-multiples); identical per-row order.
                const auto& gq = gw(pre + "self_attn.q_proj.weight");
                const auto& gk = gw(pre + "self_attn.k_proj.weight");
                const auto& gv = gw(pre + "self_attn.v_proj.weight");
                const uint64_t qkv_woff[3] = {gq.second, gk.second, gv.second};
                const unsigned qkv_rows[3] = {(unsigned)QDg, (unsigned)KVDg, (unsigned)KVDg};
                tm_metal_tok_gemv_seg(SLOT_H, gq.first, SLOT_QKV, 0,
                                      (unsigned)(QDg + 2 * KVDg), D,
                                      qkv_woff, qkv_rows, 3);
            }
            if (w.count(pre + "self_attn.q_proj.bias"))
                tm_metal_tok_bias_qkv(SLOT_QKV, SLOT_BIAS,
                                      (size_t)l * (QDg + 2 * KVDg) * 4,
                                      (unsigned)(QDg + 2 * KVDg));
            tm_metal_tok_rope_q(SLOT_QKV, H, dh, pos, theta);
            tm_metal_tok_rope_k_cache(SLOT_QKV, (size_t)QDg * 4,
                                      SLOT_KC + l, KVH, dh, ctx, pos, theta);
            tm_metal_tok_v_cache(SLOT_QKV, (size_t)(QDg + KVDg) * 4,
                                 SLOT_VC + l, KVH, dh, ctx, pos);
            tm_metal_tok_attn(SLOT_QKV, SLOT_KC + l, SLOT_VC + l,
                              SLOT_PROBS, SLOT_AO, QDg, H, KVH, dh, ctx,
                              pos + 1, scale, REP_);
            gmv(SLOT_AO, pre + "self_attn.o_proj.weight", SLOT_H2, 0, D, D);
            tm_metal_tok_add(SLOT_X, SLOT_H2, D);
            tm_metal_tok_rmsnorm(SLOT_X,
                w.at(pre + "post_attention_layernorm.weight").data(),
                SLOT_H2S, D, eps);
            // A K-quant gate|up cannot take the fused seg kernel (Q4_0-only, see gw()),
            // and mlp.down_proj is Q4_1 on the first layers of a qwen35 byte-diet build.
            // Fetching these handles unconditionally THREW on that checkpoint mid-encode;
            // caught upstream (see the catch on the forward path) the token completed on a
            // half-encoded command buffer — fluent garbage from token 1, invariant to every
            // GEMV knob because the K-quant kernels never ran. Same species as the dead
            // lm_head locals documented below.
            const bool seg_ok =
                gemv_fuse && !gpu_f16_ && gpu_shared_wbuf_ &&
                !gpu_k_.count(pre + "mlp.gate_proj.weight") &&
                !gpu_k_.count(pre + "mlp.up_proj.weight");
            if (!seg_ok) {
                gmv(SLOT_H2S, pre + "mlp.gate_proj.weight", SLOT_GU, 0, F, D);
                gmv(SLOT_H2S, pre + "mlp.up_proj.weight", SLOT_GU, (size_t)F * 4, F, D);
            } else {
                const auto& gg = gw(pre + "mlp.gate_proj.weight");
                const auto& gu = gw(pre + "mlp.up_proj.weight");
                const uint64_t gu_woff[2] = {gg.second, gu.second};
                const unsigned gu_rows[2] = {(unsigned)F, (unsigned)F};
                tm_metal_tok_gemv_seg(SLOT_H2S, gg.first, SLOT_GU, 0,
                                      (unsigned)(2 * F), D, gu_woff, gu_rows, 2);
            }
            tm_metal_tok_silu_mul(SLOT_GU, 0, SLOT_GU, (size_t)F * 4, F);
            // q4_.at() and gw() both throw for a Q4_1 down_proj (this checkpoint has it on
            // layers 0-2) and neither result was used: gmv() below is the dtype-routed
            // dispatch. These three dead locals were the actual crash-to-garbage site.
            gmv(SLOT_GU, pre + "mlp.down_proj.weight", SLOT_DN, 0, D, F);
            tm_metal_tok_add(SLOT_X, SLOT_DN, D);
            if (decode_layer_cb_) {
                // Per-layer commit: end this layer's buffer, start the next.
                if (!tm_metal_tok_flush())
                    throw std::runtime_error("metal: per-layer command failed");
                // The final norm + lm_head still encode after the last
                // layer, so re-begin unconditionally.
                if (!tm_metal_tok_begin()) gpu_fatal("gpu layer begin failed");
            }
        }
        tm_metal_tok_rmsnorm(SLOT_X, w.at("model.norm.weight").data(),
                             SLOT_XF, D, eps);
        if (!gpu_head_cpu_) {
            // Only the gmv() call below is live. Two dead locals used to sit here
            // (`q4_.at("lm_head.weight")` and `gw("lm_head.weight")`) and BOTH throw
            // before it on a checkpoint whose lm_head is Q6_K — which is every shipped
            // qwen35 build, so the hybrid lane died on its first token at the very last
            // dispatch of the stack, after the whole layer loop had run. gmv() is the
            // dtype-aware dispatch; nothing else here needs the weight handle.
            gmv(SLOT_XF, "lm_head.weight", SLOT_LOGITS, 0, V, D);
        }
        // gpu_head_cpu_: nothing here — the projection has no Q4 kernel, so
        // forward_gpu_one takes SLOT_XF back and runs it on the CPU.
#else
        (void)pos;
#endif
    }
    std::vector<float>& forward_gpu_one(int tok) {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        static bool said = false;
        if (!said && getenv("TM_DEBUG_POOL")) {
            said = true;
            fprintf(stderr, "[gpu-one] T=1 decode entering the GPU token lane\n");
        }
        gpu_token_prologue();
        std::vector<float> x0(D);
        // The row comes from the f16 SAFETENSORS mapping only when that mapping is
        // actually open; otherwise it comes from the .tmq. gpu_f16_ answers a
        // different question ("does this lane hold f16 tensors in gpu_h_?") — a
        // Gated-DeltaNet lane sets it true while never opening held_f16_, and
        // testing it here dereferenced a null map.
        if (held_f16_) row_h("model.embed_tokens.weight", tok, x0.data());
        else row_w("model.embed_tokens.weight", tok, x0.data());
        // A dropped result here would run the GPU on whatever the slot held
        // last. Throwing, not aborting: forward()'s auto-GPU catch demotes to
        // the CPU path when the GPU KV is not yet authoritative, and re-throws
        // when it is.
        if (!tm_metal_tok_upload(SLOT_X, 0, x0.data(), (size_t)D * 4))
            throw std::runtime_error("metal: token hidden-state upload failed");
        if (!tm_metal_tok_begin()) {
            // sticky-GPU mode: the GPU KV cache is authoritative, so a
            // failed begin cannot fall back silently — abort loudly.
            gpu_fatal("gpu token begin failed");
        }
        encode_gpu_token(seen);
        if (gpu_head_cpu_) {
            // End the command buffer by pulling the normed hidden row back: the
            // layer stack must complete before the CPU head reads it, and the
            // download is D floats (8 KB at D=2048) against the 65 MB a Q8 lm_head
            // would stream per token.
            if (xf_.size() < (std::size_t)D) xf_.resize(D);
            if (!tm_metal_tok_end(SLOT_XF, xf_.data(), (unsigned)D))
                throw std::runtime_error("metal: token execution failed");
            if (logits_.size() != (std::size_t)V) logits_.resize(V);
            gemv_w("lm_head.weight", xf_.data(), D, V, logits_.data());
            seen += 1;
            return logits_;
        }
        if (!tm_metal_tok_end(SLOT_LOGITS, logits_.data(), V))
            throw std::runtime_error("metal: token execution failed");
        seen += 1;
        return logits_;
#else
        (void)tok;
        return logits_;
#endif
    }
    // GPU-resident greedy decode (2026-09-07): feeds `first`, then n-1 more
    // tokens each chosen by an argmax kernel over the previous logits and
    // embedded on the GPU, all n command buffers committed back to back so
    // the host never sits between tokens. out receives the n-1 argmax
    // tokens; logits() afterwards holds the last token's logits; the cache
    // advances by n. Requires the GPU decode path with a Q4 embedding table
    // (claim_gpu_for_chain()); callers fall back to forward() otherwise.
    const std::vector<float>& logits() const { return logits_; }
    // The old name, kept for callers in files this lane does not own
    // (llama_chat.cpp). It forwards; prefer claim_gpu_for_chain().
    bool gpu_greedy_chain_ok() { return claim_gpu_for_chain(); }
    // Claims the GPU decode path for a chained greedy run and reports whether it
    // got it. Named for the side effect, not as a predicate: it registers the
    // weights, uploads the KV cache and latches the slot state, so the old
    // `_ok` name invited a caller to ignore a false result and leave the session
    // half-claimed. [[nodiscard]] for the same reason: a false result is not a
    // statement about the session, it is a refusal the caller must handle.
    [[nodiscard]] bool claim_gpu_for_chain() {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        // Under the auto policy a greedy chain claims the GPU (paired
        // 2026-09-07: ctx 64 chain 84.1 t/s vs CPU decode 79.6, ctx 2000
        // 71.4 vs 59), and the GPU cache is authoritative from then on.
        if (decode_gpu_auto_ && !decode_gpu_ && decode_gpu_auto_fit_ && quantized()) {
            decode_gpu_ = true;
            if (getenv("TM_DEBUG_POOL")) fprintf(stderr, "[llama] decode auto: greedy chain -> GPU\n");
        }
        // The resident greedy chain argmaxes SLOT_LOGITS on the GPU, which a
        // CPU-side lm_head never fills. The embedding conjunct is REGISTRATION
        // state, not eligibility: the chain registers the table itself in
               // gpu_chain_begin via gpu_token_prologue, so a caller that asks before
                // any GPU token ran (the natural "may I?" probe) must not be refused
                // for a table nothing has had a reason to register yet. Eligibility is
               // the WEIGHT being present in a registrable dtype; TM_DEBUG_CHAIN=1
               // names the conjunct that refuses.
        const bool emb_reg = gpu_h_.count("model.embed_tokens.weight") != 0 ||
                             gpu_w_.count("model.embed_tokens.weight") != 0 ||
                             gpu_k_.count("model.embed_tokens.weight") != 0;
        const bool emb_have = f16_.count("model.embed_tokens.weight") != 0 ||
                              q4_.count("model.embed_tokens.weight") != 0 ||
                              kq_registerable("model.embed_tokens.weight");
        const bool ok = !gpu_head_cpu_ && (emb_reg || emb_have) &&
                        decode_gpu_ && seen > 0 && gpu_decode_usable() &&
                        tm_metal_tok_end_async != nullptr;
        if (!ok && getenv("TM_DEBUG_CHAIN"))
            fprintf(stderr, "[llama] chain refused: head_cpu=%d emb_reg=%d emb_have=%d "
                            "decode_gpu=%d seen=%d usable=%d async=%d\n",
                    (int)gpu_head_cpu_, (int)emb_reg, (int)emb_have,
                    (int)decode_gpu_, seen, (int)gpu_decode_usable(),
                    (int)(tm_metal_tok_end_async != nullptr));
        return ok;
#else
        return false;
#endif
    }
    // Chain API. begin() feeds `first` at the current position and chooses
    // n tokens on the GPU, one per step from that step's logits (argmax when
    // temperature <= 0, else Gumbel-max temperature sampling with `seed`);
    // chosen tokens 0..n-2 are fed by the following steps, chosen token n-1
    // is left for the caller (the natural `first` of the next chain), so
    // chunks link without any host-side sampling. All n command buffers are
    // committed at once; poll(i) returns chosen token i as soon as it has
    // landed (-1 while pending); end() waits, exposes the last logits, and
    // advances the cache by n — or by keep+1 when the caller stopped at
    // chosen token `keep` (EOS: fed tokens stay, the EOS is not fed): rows
    // the chain wrote past that are simply overwritten later.
    int chain_n_ = 0, chain_seen0_ = 0;
    void gpu_chain_begin(int first, int n, float temperature, unsigned seed) {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (n <= 0) throw std::invalid_argument("llama: chain needs n >= 1");
        if (!claim_gpu_for_chain())
            throw std::runtime_error("llama: chain needs the GPU decode path");
        if (seen + n > ctx) throw std::invalid_argument("llama: chain exceeds context");
        if (chain_n_) throw std::runtime_error("llama: chain already running");
        gpu_token_prologue();
        if (!tm_metal_tok_reserve(SLOT_TOK, sizeof(unsigned) * (size_t)n))
            throw std::runtime_error("metal: token slot reservation failed");
        if (gpu_f16_) {
            const std::string nme = "model.embed_tokens.weight";
            if (!gpu_h_.count(nme)) {
                const auto& te = held_f16_->entries.at(nme);
                uint64_t w = 0;
                gpu_weights_used_ = true;
                const int id = tm_metal_tok_wbuf(
                    held_f16_->base + held_f16_->data_begin + te.begin,
                    (std::size_t)(te.end - te.begin),
                    held_f16_->base, held_f16_->size, 1, w);
                if (id < 0)
                    throw std::runtime_error("metal: fp16 embedding registration failed");
                gpu_h_[nme] = {id, w};
            }
        } else if (q4_.count("model.embed_tokens.weight") &&
                   !gpu_w_.count("model.embed_tokens.weight")) {
            // Guarded on q4_ as well: on a checkpoint whose embedding is Q6_K this
            // branch used to run and throw, and the registration is already done by
            // reg_kq in that case.
            const std::string nme = "model.embed_tokens.weight";
            const bool file_backed = held_map_ && held_map_->owned.empty() && held_map_->base != nullptr;
            uint64_t w = 0;
            const int id = tm_metal_tok_wbuf(q4_.at(nme),
                (size_t)qshape_.at(nme).first * qshape_.at(nme).second / 32 * 18,
                held_map_->base, held_map_->size, file_backed ? 1 : 0, w);
            if (id < 0) throw std::runtime_error("metal: embedding registration failed");
            gpu_w_[nme] = {id, w};
        } else if (kq_registerable("model.embed_tokens.weight") &&
                   !gpu_k_.count("model.embed_tokens.weight")) {
            // A K-quant embedding is gathered by the K-quant embed kernel out of the
            // same zero-copy wbuf contract, so the chain registers it the same way
            // the Q4 embedding is registered above.
            reg_kq("model.embed_tokens.weight");
        }
        const std::string en = "model.embed_tokens.weight";
        const auto kge = gpu_k_.find(en);
        // A copy, not a reference: which table holds the embedding depends on the
        // checkpoint, and .at() on the wrong one is the throw that used to kill the
        // lane's first token.
        const std::pair<int, uint64_t> ge =
            kge != gpu_k_.end() ? std::pair<int, uint64_t>{kge->second.id, kge->second.woff}
                                : (gpu_f16_ ? gpu_h_.at(en) : gpu_w_.at(en));
        std::vector<unsigned> sentinel((size_t)n, 0xFFFFFFFFu);
        const auto chain_upload = [](int slot, const void* src, size_t bytes) {
            if (!tm_metal_tok_upload(slot, 0, src, bytes))
                throw std::runtime_error("metal: chained input upload failed");
        };
        chain_upload(SLOT_TOK, sentinel.data(), sizeof(unsigned) * (size_t)n);
        std::vector<float> x0(D);
        row_w("model.embed_tokens.weight", first, x0.data());
        chain_upload(SLOT_X, x0.data(), (size_t)D * 4);
        const float inv_temp = temperature > 0.0f ? 1.0f / temperature : 0.0f;
        for (int i = 0; i < n; ++i) {
            if (!tm_metal_tok_begin()) gpu_fatal("gpu token begin failed");
            if (i > 0) {
                if (kge != gpu_k_.end())
                    tm_metal_tok_embed_kq(SLOT_TOK, (unsigned)(i - 1), ge.first, ge.second,
                                          SLOT_X, (unsigned)D, kge->second.dtype);
                else if (gpu_f16_)
                    tm_metal_tok_embed_h(SLOT_TOK, (unsigned)(i - 1),
                                         ge.first, ge.second, SLOT_X, (unsigned)D);
                else
                    tm_metal_tok_embed_q4(SLOT_TOK, (unsigned)(i - 1),
                                          ge.first, ge.second, SLOT_X, (unsigned)D);
            }
            encode_gpu_token(seen + i);
            if (inv_temp > 0.0f) tm_metal_tok_sample(SLOT_LOGITS, SLOT_TOK, (unsigned)i, (unsigned)V, inv_temp, seed);
            else tm_metal_tok_argmax(SLOT_LOGITS, SLOT_TOK, (unsigned)i, (unsigned)V);
            if (!tm_metal_tok_end_async()) throw std::runtime_error("metal: token commit failed");
        }
        chain_n_ = n; chain_seen0_ = seen;
#else
        (void)first; (void)n; (void)temperature; (void)seed;
        throw std::runtime_error("llama: chain needs Metal");
#endif
    }
    int gpu_chain_poll(int i) const {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (i < 0 || i >= chain_n_) return -1;
        const unsigned t = tm_metal_tok_peek(SLOT_TOK, (unsigned)i);
        return t == 0xFFFFFFFFu ? -1 : (int)t;
#else
        (void)i; return -1;
#endif
    }
    // Returns the n chosen tokens (the last one not yet fed). keep < 0: all
    // consumed (cache += n); keep >= 0: the caller stopped at chosen token
    // `keep` (not fed, e.g. EOS) — cache position becomes seen0 + keep + 1.
    std::vector<int> gpu_chain_end(int keep = -1) {
        std::vector<int> out;
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (!chain_n_) return out;
        if (!tm_metal_tok_wait_pending()) throw std::runtime_error("metal: chained token execution failed");
        std::vector<unsigned> toks((size_t)chain_n_);
        // Checked like the wait_pending() above: a dropped download would hand
        // back uninitialized tokens or logits, i.e. silently wrong output.
        if (!tm_metal_tok_download(SLOT_TOK, 0, toks.data(),
                                  sizeof(unsigned) * (size_t)chain_n_) ||
            !tm_metal_tok_download(SLOT_LOGITS, 0, logits_.data(),
                                  sizeof(float) * (size_t)V))
            throw std::runtime_error("metal: chained result download failed");
        out.assign(toks.begin(), toks.end());
        seen = keep < 0 ? chain_seen0_ + chain_n_ : chain_seen0_ + keep + 1;
        chain_n_ = 0;
#else
        (void)keep;
#endif
        return out;
    }
    // n forwards (first + n-1 chosen tokens); returns the n-1 chosen tokens
    // that were fed, like a host loop of n forwards and n-1 argmaxes.
    std::vector<int> generate_greedy_gpu(int first, int n) {
        gpu_chain_begin(first, n, 0.0f, 0);
        std::vector<int> t = gpu_chain_end();
        t.pop_back();
        return t;
    }
    // pos0 < 0 (default): absolute position comes from the cache counter
    // `seen` — sequential decode/prefill needs no manual tracking. An
    // explicit pos0 must equal seen: KV storage and attention share the same
    // position. Call reset_cache() to start a new sequence at position zero.
    // logits_all, when given, receives the logits of EVERY position of this pass
    // (T x V), not just the last — what speculative verification compares a draft
    // against. It forces the batched path: the per-token GPU branch below cannot
    // produce the earlier rows, so a session in that mode is left to the plain
    // loop (see spec_generate_greedy).
    // `ids` is borrowed for the call only, so it takes a span: the decode paths
    // inside this file call forward({tok}) once per token, and overloads taking
    // const std::vector<int>& made each of those a heap allocation. The
    // initializer_list overload below keeps `forward({tok})` source-compatible
    // and allocation-free (the braced list becomes a temporary array).
    std::vector<float>& forward(std::span<const int> ids, int pos0 = -1,
                                std::vector<float>* logits_all = nullptr) {
        tm_thread_qos_once();   // interactive band: an agent waits on every token
        // Read ONCE. This flag guards three debug prints below, one of which sits on the
        // per-token path — and `getenv()` walks the environment array on every call, so testing
        // it per token makes the observer cost work on the thing it observes. Same defect as a
        // printf left in a hot loop, and it reads harmless at every call site.
        static const bool pool_debug_ = getenv("TM_DEBUG_POOL") != nullptr;
        const int T = (int)ids.size();
        const int base = pos0 < 0 ? seen : pos0;
        if (T <= 0 || base < 0 || base > ctx || T > ctx - base)
            throw std::invalid_argument("llama: sequence exceeds context or is empty");
        for (int token : ids)
            if (token < 0 || token >= V)
                throw std::invalid_argument("llama: token outside vocabulary");
        if (base != seen)
            throw std::invalid_argument("llama: explicit position must equal cache position");
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (decode_gpu_auto_ && !decode_gpu_ && decode_gpu_auto_fit_ && T == 1 && quantized()) {
            static const double ctx_switch =
                env_var_double_checked("TM_DECODE_GPU_AUTO_CTX", 384.0);
            if (ctx_switch > 0 && seen >= ctx_switch) {
                decode_gpu_ = true;
                decode_gpu_ctx_switched_ = true;
                if (pool_debug_)
                    fprintf(stderr, "[llama] decode auto: context %d >= %.0f -> GPU\n", seen, ctx_switch);
            }
        }
#endif
        if (pool_debug_ && T == 1)
            fprintf(stderr, "[dispatch] T=1 quant=%d decode_gpu=%d seen=%d usable=%d\n",
                    (int)quantized(), (int)decode_gpu_, seen, (int)gpu_decode_usable());
        // !emb_collect_: the per-token GPU branch sees only the last position's
        // hidden row, so a collecting pass has to stay on the batched path.
        // `decode_gpu_` records a REQUEST; it is not evidence that the token lane
        // can run, and only the check below makes it one. A refused lane must not
        // keep the bit set: the CPU block() loop reads it to route its Q4_0
        // projections through batched_q4(M=1) — the per-call GPU GEMV, one buffer
        // alloc + upload + commit + wait + download EACH, which that branch's own
        // comment measured at 5x the CPU path (and block_linear's `!decode_gpu_`
        // guard cost the AMX prefill too). Paired on Qwen3.5-0.8B Q4_0 f16lm, 1024
        // tokens, quiet host: the since-retired Gated-DeltaNet GPU lane
        // (TM_DECODE_GPU=1 TM_METAL_GDN=1), which refuses on this checkpoint's
        // mlp.down_proj (Q4_1), ran 20 t/s against the plain CPU's 42. Demote here — at the first token the lane
        // would have served — and NOT at load: eligibility is memoized per weight
        // set, so computing it at load memoises against the PREVIOUS weights, which
        // is the stale-eligibility bug test_llama_reload exists to refuse.
        if (T == 1 && decode_gpu_ && !gpu_decode_usable()) decode_gpu_ = false;
        if (!emb_collect_ && quantized() && T == 1 && decode_gpu_ && seen > 0 &&
            gpu_decode_usable()) {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
            if (!decode_gpu_auto_) return forward_gpu_one(ids[0]);
            // Auto-selected GPU: a setup failure (budget, registration)
            // demotes this session to CPU decode instead of aborting it.
            try {
                return forward_gpu_one(ids[0]);
            } catch (const std::runtime_error& e) {
                if (gpu_kv_uploaded_) throw;   // GPU KV is authoritative: no clean CPU resume
                decode_gpu_ = false;
                if (pool_debug_)
                    fprintf(stderr, "[llama] decode auto: GPU setup failed (%s) -> CPU\n", e.what());
            }
#endif
        }
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (!emb_collect_ && logits_all == nullptr && gpu_kv_uploaded_ && T > 1 &&
            T < prefill_loop_t() && decode_gpu_ && quantized()) {
            // A few tokens (a chat turn close) while the GPU cache is
            // authoritative: run them one by one on the GPU instead of
            // pulling the cache back and re-uploading it on the next token
            // (2026-09-07: turn close 55 -> ~35 ms at ctx 1100).
            for (int t = 0; t + 1 < T; ++t) forward_gpu_one(ids[t]);
            return forward_gpu_one(ids[T - 1]);
        }
        if (gpu_kv_uploaded_ && T > 1) {
            // GPU cache is authoritative for [gpu_from_, seen): pull those
            // rows back so the CPU prefill continues from the right state.
            for (int l = 0; l < L; ++l)
                for (int kv = 0; kv < KVH; ++kv) {
                    const size_t src =
                        ((size_t)kv * ctx + gpu_from_) * dh;
                    float* dk = kc[l].data() + (size_t)kv * ctx * dh
                              + (size_t)gpu_from_ * dh;
                    float* dv = vc[l].data() + (size_t)kv * ctx * dh
                              + (size_t)gpu_from_ * dh;
                    (void)tm_metal_tok_kv_download(SLOT_KC + l, src, dk,
                        (size_t)(seen - gpu_from_) * dh);
                    (void)tm_metal_tok_kv_download(SLOT_VC + l, src, dv,
                        (size_t)(seen - gpu_from_) * dh);
                }
            gpu_kv_uploaded_ = false;
            gpu_kv_valid_ = seen;   // GPU rows [0, seen) still equal the CPU rows
        }
#endif
        // One activation buffer per model, not two. xbuf_ was the embedding
        // destination and `x` was a fresh T*D copy of it, so every forward paid
        // a heap allocation and a T*D memcpy before the first layer ran. The
        // embedding now lands in the buffer the layers mutate. Safe as a shared
        // buffer because forward() is not reentrant (no path inside it calls
        // forward again).
        if (xbuf_.size() < (std::size_t)T * D) xbuf_.resize((std::size_t)T * D);
        if (quantized()) {
            for (int t = 0; t < T; ++t)
                row_w("model.embed_tokens.weight", ids[t],
                      xbuf_.data() + (std::size_t)t * D);
        } else {
            const float* emb = w.at("model.embed_tokens.weight").data();
            for (int t = 0; t < T; ++t)
                std::copy_n(emb + (std::size_t)ids[t] * D, D,
                            xbuf_.begin() + (std::size_t)t * D);
        }
        std::vector<float>& x = xbuf_;   // the residual stream the layers mutate
        bool gpu_stack_done = false;
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        // GPU layer stack (2026-09-07, metal_llama.h): whole prefill on the
        // GPU when the runtime would use GPU GEMMs anyway. Falls back to the
        // per-layer block() loop on any refusal.
        if (T > 1 && gpu_stack_ && gpu_prefill_ready(T, D)) {
            llama_prof().start();
            // Hybrid split (2026-09-07): the GPU takes the leading rows and
            // this thread runs the tail rows through block(), one layer
            // behind — causal attention means the GPU's rows never need the
            // CPU's, and the CPU's rows read the GPU's K/V rows once that
            // layer's command buffer has completed (the caches are shared
            // page-aligned memory). GPU ~780-830 tok/s, CPU/AMX ~270-350, so
            // the tail is sized by TM_LLAMA_HYBRID (fraction of rows given to
            // the CPU; 0 = GPU only, the default until gated).
            const int cpu_rows = hybrid_cpu_rows(T);
            if (cpu_rows > 0) {
                const int gpu_rows = T - cpu_rows;
                const auto wall0 = std::chrono::steady_clock::now();
                if (tm_metal_llama_prefill_begin(gpu_stack_, x.data(), gpu_rows, base) != 0) {
                    std::vector<float> xc(x.begin() + (std::size_t)gpu_rows * D, x.end());
                    bool ok = true;
                    double cpu_ms = 0;
                    force_cpu_block_ = true;
                    for (int l = 0; l < L && ok; ++l) {
                        ok = tm_metal_llama_prefill_wait_layer(l) != 0;
                        if (ok) {
                            const auto t0 = std::chrono::steady_clock::now();
                            block(l, xc, cpu_rows, base + gpu_rows); // hybrid GPU path (dense only)
                            cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        }
                    }
                    force_cpu_block_ = false;
                    ok = (tm_metal_llama_prefill_end() != 0) && ok;
                    if (ok)
                        hybrid_update(tm_metal_llama_prefill_gpu_ms(), cpu_ms, gpu_rows, cpu_rows,
                                      std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - wall0).count());
                    if (ok) {
                        std::copy(xc.begin(), xc.end(), x.begin() + (std::size_t)gpu_rows * D);
                        gpu_stack_done = true;
                        ++gpu_stack_prefills_;
                    } else {
                        throw std::runtime_error("llama: hybrid prefill failed");
                    }
                }
            }
            if (!gpu_stack_done) {
                gpu_stack_done = tm_metal_llama_prefill(gpu_stack_, x.data(), T, base) != 0;
                if (gpu_stack_done) ++gpu_stack_prefills_;
            }
            llama_prof().mark(LlamaProf::Sec::GPU_STACK);
        }
#endif
        if (!gpu_stack_done)
            for (int l = 0; l < L; ++l) dispatch_block(l, x, T, base);
        seen += T;
        if (xf_.size() < (std::size_t)D) xf_.resize(D);
        rms(x.data() + (std::size_t)(T - 1) * D, xf_.data(),
            w.at("model.norm.weight").data(), D, eps);
        tm_dump("result_norm", -1, xf_.data(), (std::size_t)D);
        if (logits_.size() != (std::size_t)V) logits_.resize(V);
        llama_prof().start();
        if (quantized()) {
            // GPU decode: the M=1 lm_head rides the fused kernel too.
            // (Braced: the trailing else below must stay bound to
            // quantized(), not to this inner if — it reads the fp32
            // lm_head tensor, which quantized models don't have.)
            if (!batched_q4(xf_.data(), "lm_head.weight",
                            logits_.data(), 1, V, D))
                gemv_w("lm_head.weight", xf_.data(), D, V, logits_.data());
        } else
            cblas_sgemv(CblasRowMajor, CblasNoTrans, V, D, 1.f,
                        w.at("lm_head.weight").data(), D, xf_.data(), 1, 0.f,
                        logits_.data(), 1);
        llama_prof().mark(LlamaProf::Sec::LMHEAD);
        if (emb_collect_) {
            // x still holds the final hidden state of all T positions here, which
            // is exactly the tensor whose last row feeds the LM head.
            if ((int)emb_sum_.size() != D) emb_sum_.assign(D, 0.f);
            for (int t = 0; t < T; ++t) {
                const float* row = x.data() + (std::size_t)t * D;
                for (int d = 0; d < D; ++d) emb_sum_[(std::size_t)d] += row[d];
            }
            emb_count_ += (std::size_t)T;
        }
        // Speculative verification compares a draft against every position of the
        // pass, so the earlier rows are produced here; the last row is the one the
        // block above just computed and is copied rather than recomputed. The
        // default path (logits_all == nullptr) does no extra work.
        if (mtp_ready()) {
            if ((int)hlast_.size() != D) hlast_.resize(D);
            std::copy_n(x.data() + (std::size_t)(T - 1) * D, D, hlast_.data());
        }
        if (logits_all) {
            logits_all->resize((std::size_t)T * V);
            if (xfa_.size() < (std::size_t)D) xfa_.resize(D);
            for (int t = 0; t + 1 < T; ++t) {
                float* dst = logits_all->data() + (std::size_t)t * V;
                rms(x.data() + (std::size_t)t * D, xfa_.data(),
                    w.at("model.norm.weight").data(), D, eps);
                if (quantized()) {
                    if (!batched_q4(xfa_.data(), "lm_head.weight", dst, 1, V, D))
                        gemv_w("lm_head.weight", xfa_.data(), D, V, dst);
                } else {
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, V, D, 1.f,
                                w.at("lm_head.weight").data(), D, xfa_.data(), 1,
                                0.f, dst, 1);
                }
            }
            std::copy_n(logits_.data(), V,
                        logits_all->data() + (std::size_t)(T - 1) * V);
        }
        if (tm_dump_on()) {
            double sum = 0.0;
            for (int i = 0; i < V; ++i) sum += (double)logits_[(std::size_t)i];
            std::printf("TMDUMP %-28s l=%-3d n=%-8d sum=%.6f\n",
                        "result_output", -1, V, sum);
            std::fflush(stdout);
        }
        return logits_;
    }
    // `forward({tok})` / `forward({a, b})`: the braced list becomes a temporary
    // array, so the one-token decode calls allocate nothing. Kept as an overload
    // rather than a std::vector<int> default so a caller holding a real vector
    // binds the span version directly, with no conversion and no copy.
    std::vector<float>& forward(std::initializer_list<int> ids, int pos0 = -1,
                                std::vector<float>* logits_all = nullptr) {
        return forward(std::span<const int>(ids.begin(), ids.size()), pos0, logits_all);
    }

    // ---- pooled embeddings -------------------------------------------------
    // The embedding this engine exposes is the MEAN of the final-layer hidden
    // state over the positions of the pass, L2-normalised by default. That is the
    // same tensor whose last row the LM head consumes, so the path is covered by
    // the logits numerics gates transitively — but the pooling arithmetic is this
    // code's own, and the returned vector is not independently reference-checked.
    //
    // Collecting forces the batched path (see the emb_collect_ guards in
    // forward()): the per-token GPU branches cannot see the earlier positions'
    // hidden rows. Embed a prompt the normal way — reset_cache(), one forward()
    // over the whole prompt — and the batched path is what runs anyway.
    void embed_begin() {
        emb_sum_.assign(D, 0.f);
        emb_count_ = 0;
        emb_collect_ = true;
    }
    std::vector<float> embed_end(bool normalize = true) {
        emb_collect_ = false;
        if (emb_count_ == 0 || (int)emb_sum_.size() != D) return {};
        std::vector<float> out(emb_sum_);
        const float inv = 1.f / (float)emb_count_;
        for (float& value : out) value *= inv;
        if (normalize) {
            double squared = 0.0;
            for (float value : out) squared += (double)value * value;
            const double norm = std::sqrt(squared);
            if (norm > 0.0)
                for (float& value : out) value = (float)((double)value / norm);
        }
        return out;
    }

    // Demand-mode per-layer fault + evict-behind. Faults layer l's seven
    // weight segments and evicts layer (l - demand_window_) behind the scan
    // head (Belady-optimal MRU for the circular decode scan; see
    // docs/ON_DEMAND_WEIGHTS.md). No-op outside TM_DEMAND=1.
    void demand_layer(int l) {
        if (!demand_mode_ || !held_demand_) return;
        // A Gated-DeltaNet layer has a different tensor set than a dense one
        // (llama.cpp qwen35.cpp load_block_trunk); pick by the target layer's
        // type, because lookahead also faults layers ahead of the scan head.
        const auto for_each = [&](int layer, auto&& fn) {
            static const char* attn[4] = {
                "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight"};
            static const char* lin[4] = {
                "linear_attn.qkv_proj.weight", "linear_attn.gate_proj.weight",
                "linear_attn.out_proj.weight", "linear_attn.alpha.weight"};
            static const char* mlp[3] = {
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"};
            const std::string p = "model.layers." + std::to_string(layer) + ".";
            if (is_linear_layer(layer)) { for (const char* t : lin) fn(p + t); }
            else                        { for (const char* t : attn) fn(p + t); }
            for (const char* t : mlp) fn(p + t);
        };
        const auto fault = [&](int layer) {
            for_each(layer, [&](const std::string& n) {
                const void* blocks = held_demand_->data(n);
                if (!blocks) throw std::runtime_error("llama: demand fault failed: " + n);
                if (q8_.count(n)) q8_[n] = static_cast<const tmq::BlockQ8_0*>(static_cast<const void*>(blocks));
                else if (q4_.count(n)) q4_[n] = static_cast<const tmq::BlockQ4_0*>(static_cast<const void*>(blocks));
                else if (f16_.count(n)) f16_[n] = (const std::uint16_t*)blocks;
            });
        };
        fault(l);
        // Warm layer l's bytes into the page cache so the dequant that
        // follows faults cache-warm (sequential ~2.8 GB/s) instead of
        // serialized 16 KiB SSD faults (~15-460 MB/s). This is what turns
        // oversize decode from fault-bound into bandwidth-bound.
        for_each(l, [&](const std::string& n) { held_demand_->warm(n); });
        // Pipelining: read-ahead the next P layers so their fault hits page
        // cache instead of SSD. No-op when already resident or P=0.
        for (int a = 1; a <= demand_lookahead_ && l + a < L; ++a)
            for_each(l + a, [&](const std::string& n) { held_demand_->prefetch(n); });
        const int behind = l - demand_window_;
        if (behind >= 0)
            // Best-effort, for the same reason as the eviction in load_quant:
            // refusing to unmap costs a larger resident window, not bad data.
            for_each(behind, [&](const std::string& n) { (void)held_demand_->evict(n); });
    }

    void block(int l, std::vector<float>& x, int T, int pos0) {
        demand_layer(l);
        const auto W = [&](const std::string& k) -> const float* {
            return w.at("model.layers." + std::to_string(l) + "." + k).data();
        };
        const int QD = H * dh, KVD = KVH * dh;
        // qwen35 full-attention layers emit the output gate from `attn_q`,
        // head-interleaved as [q(dh) | gate(dh)] per head (llama.cpp
        // build_layer_attn views Qcur_full with a 2*dh head stride), and
        // scale the attention output by sigmoid(gate) before o_proj.
        // The attention-output row stride is H*dh; that equals D for every
        // dense model, so this is a no-op there.
        const int QPROJ = attn_output_gate_ ? 2 * QD : QD;
        LlamaProf& P = llama_prof();
        P.start();
        if (h_.size() < (std::size_t)T * D) h_.resize((std::size_t)T * D);
        for (int t = 0; t < T; ++t)
            rms(x.data() + (std::size_t)t * D, h_.data() + (std::size_t)t * D,
                W("input_layernorm.weight"), D, eps);
        P.mark(LlamaProf::Sec::RMS1);
        tm_dump("attn_norm", l, h_.data(), (std::size_t)T * D);
        tm_dump_row("attn_norm", l, h_.data(), (std::size_t)D);

        // q, k, v projections
        if (q_.size() < (std::size_t)T * QPROJ) q_.resize((std::size_t)T * QPROJ);
        if (k_.size() < (std::size_t)T * KVD) k_.resize((std::size_t)T * KVD);
        if (v_.size() < (std::size_t)T * KVD) v_.resize((std::size_t)T * KVD);
        if (attn_output_gate_) {
            // attn_q carries the interleaved gate, so q is twice the
            // attention width and the fused q|k|v blob does not apply.
            const std::string sq = "model.layers." + std::to_string(l) +
                                   ".self_attn.q_proj.weight";
            const std::string sk = "model.layers." + std::to_string(l) +
                                   ".self_attn.k_proj.weight";
            const std::string sv = "model.layers." + std::to_string(l) +
                                   ".self_attn.v_proj.weight";
            linear_proj(sq, h_.data(), T, D, QPROJ, q_.data());
            linear_proj(sk, h_.data(), T, D, KVD, k_.data());
            linear_proj(sv, h_.data(), T, D, KVD, v_.data());
            tm_dump_row("qfull", l, q_.data(), (std::size_t)T * QPROJ);
            tm_dump_row("kfull", l, k_.data(), (std::size_t)KVD);
        } else if (T == 1) {
            // Llama nn.Linear weights are (out, in): y = W x. That's the
            // NoTrans gemv — GPT-2's Conv1D was (in, out), Trans there.
            if (quantized()) {
                // GPU decode: one fused q|k|v dispatch (same blob cache
                // as prefill — weights upload ONCE per tensor), then
                // split; falls back to 3 CPU gemvs.
                const std::string qp =
                    "model.layers." + std::to_string(l)
                    + ".self_attn.q_proj.weight";
                const std::string kp =
                    "model.layers." + std::to_string(l)
                    + ".self_attn.k_proj.weight";
                const std::string vp =
                    "model.layers." + std::to_string(l)
                    + ".self_attn.v_proj.weight";
                const std::string qkv_names[3] = {qp, kp, vp};
                const int qkv_ns[3] = {QD, KVD, KVD};
                const int qkv_tot = QD + 2 * KVD;
                if (qkvv_.size() < (std::size_t)qkv_tot)
                    qkvv_.resize((std::size_t)qkv_tot);
                if (batched_q4_parts(h_.data(), qkv_names, qkv_ns, 3,
                                     qkvv_.data(), 1, D)) {
                    std::copy_n(qkvv_.data(), QD, q_.data());
                    std::copy_n(qkvv_.data() + QD, KVD, k_.data());
                    std::copy_n(qkvv_.data() + QD + KVD, KVD, v_.data());
                } else if (grouped_gemv_ok()) {
                    float* const outputs[3] = {q_.data(), k_.data(), v_.data()};
                    gemv_w_group(qkv_names, qkv_ns, outputs, h_.data(), D);
                } else {
                    gemv_w(qp, h_.data(), D, QD, q_.data());
                    gemv_w(kp, h_.data(), D, KVD, k_.data());
                    gemv_w(vp, h_.data(), D, KVD, v_.data());
                }
            } else {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, QD, D, 1.f, W("self_attn.q_proj.weight"),
                            D, h_.data(), 1, 0.f, q_.data(), 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, KVD, D, 1.f, W("self_attn.k_proj.weight"),
                            D, h_.data(), 1, 0.f, k_.data(), 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, KVD, D, 1.f, W("self_attn.v_proj.weight"),
                            D, h_.data(), 1, 0.f, v_.data(), 1);
            }
        } else if (quantized()) {
            // M6: fused prefill — ONE Q4-dequant GPU GEMM for q|k|v
            // (falls back to the T row-wise gemv loop when Metal is
            // absent/disabled or the tensors are not Q4).
            const std::string qp =
                "model.layers." + std::to_string(l) + ".self_attn.q_proj.weight";
            const std::string kp =
                "model.layers." + std::to_string(l) + ".self_attn.k_proj.weight";
            const std::string vp =
                "model.layers." + std::to_string(l) + ".self_attn.v_proj.weight";
            // AMX prefill (TM_PREFILL_AMX=1): dequant-to-fp32 +
            // threaded cblas_sgemm per projection. Tried ahead of the
            // GPU fused path so the two can be A/B'd on one binary.
            if (amx_ok(T) && !gpu_prefill_ready(T, D) &&
                gemm_w(qp, h_.data(), T, D, QD, q_.data()) &&
                gemm_w(kp, h_.data(), T, D, KVD, k_.data()) &&
                gemm_w(vp, h_.data(), T, D, KVD, v_.data())) {
                // AMX prefill did q|k|v
            } else {
            const std::string qkv_names[3] = {qp, kp, vp};
            const int qkv_ns[3] = {QD, KVD, KVD};
            const int qkv_tot = QD + 2 * KVD;
            if (qkvv_.size() < (std::size_t)T * qkv_tot)
                qkvv_.resize((std::size_t)T * qkv_tot);
            if (batched_q4_parts(h_.data(), qkv_names, qkv_ns, 3,
                                 qkvv_.data(), T, D)) {
                P.mark(LlamaProf::Sec::QKV);
                for (int t = 0; t < T; ++t) {
                    const float* src = qkvv_.data() + (std::size_t)t * qkv_tot;
                    std::copy_n(src, QD, q_.data() + (std::size_t)t * QPROJ);
                    std::copy_n(src + QD, KVD, k_.data() + (std::size_t)t * KVD);
                    std::copy_n(src + QD + KVD, KVD,
                                v_.data() + (std::size_t)t * KVD);
                }
            } else {
                for (int t = 0; t < T; ++t) {
                    gemv_w(qp, h_.data() + (std::size_t)t * D, D, QD,
                           q_.data() + (std::size_t)t * QPROJ);
                    gemv_w(kp, h_.data() + (std::size_t)t * D, D, KVD,
                           k_.data() + (std::size_t)t * KVD);
                    gemv_w(vp, h_.data() + (std::size_t)t * D, D, KVD,
                           v_.data() + (std::size_t)t * KVD);
                }
            }
            }
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, QD, D,
                        1.f, h_.data(), D, W("self_attn.q_proj.weight"), D, 0.f,
                        q_.data(), QD);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, KVD, D,
                        1.f, h_.data(), D, W("self_attn.k_proj.weight"), D, 0.f,
                        k_.data(), KVD);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, KVD, D,
                        1.f, h_.data(), D, W("self_attn.v_proj.weight"), D, 0.f,
                        v_.data(), KVD);
        }
        tm_dump("Qcur_full", l, q_.data(), (std::size_t)T * QPROJ);
        tm_dump("Kcur", l, k_.data(), (std::size_t)T * KVD);
        tm_dump("Vcur", l, v_.data(), (std::size_t)T * KVD);
        P.mark(LlamaProf::Sec::QKV_SPLIT);
        // Qwen2-family attention bias (q/k/v projections only, absent in
        // Llama-2): add when the converted model carries the bias tensors.
        {
            auto it_q = w.find("model.layers." + std::to_string(l) + ".self_attn.q_proj.bias");
            if (it_q != w.end()) {
                const float* bq = it_q->second.data();
                const float* bk = w.at("model.layers." + std::to_string(l) + ".self_attn.k_proj.bias").data();
                const float* bv = w.at("model.layers." + std::to_string(l) + ".self_attn.v_proj.bias").data();
                for (int t = 0; t < T; ++t) {
                    float* const qt = q_.data() + (std::size_t)t * QPROJ;
                    float* const kt = k_.data() + (std::size_t)t * KVD;
                    float* const vt = v_.data() + (std::size_t)t * KVD;
                    for (int i = 0; i < QD; ++i) qt[i] += bq[i];
                    for (int i = 0; i < KVD; ++i) { kt[i] += bk[i]; vt[i] += bv[i]; }
                }
            }
        }
        if (attn_output_gate_) {
            // attn_q is a Qwidth = 2*H*dh projection; split off the output gate.
            // llama.cpp's graph views it per head as q(dh) then gate(dh)
            // (`ggml_view_3d(Qcur_full, n_embd_head, n_head, ..., nb1 = 2*dh)`),
            // but the GGUF row order is a property of the checkpoint, so the
            // engine takes the layout from TM_QGATE_SPLIT and both are testable
            // in one binary against llama-debug's Qcur_normed sum.
            static const bool halves = [] {
                const char* e = std::getenv("TM_QGATE_SPLIT");
                return e && std::string(e) == "halves";
            }();
            if (agate_.size() < (std::size_t)T * QD) agate_.resize((std::size_t)T * QD);
            for (int t = 0; t < T; ++t) {
                const float* src = q_.data() + (std::size_t)t * QPROJ;
                float* dstq = q_.data() + (std::size_t)t * QPROJ;
                float* g = agate_.data() + (std::size_t)t * QD;
                if (halves) {
                    // [q(0..QD) | gate(QD..2QD)]
                    for (int hd = 0; hd < H; ++hd) {
                        std::copy_n(src + (std::size_t)hd * dh, dh,
                                    dstq + (std::size_t)hd * dh);
                        std::copy_n(src + (std::size_t)QD + (std::size_t)hd * dh, dh,
                                    g + (std::size_t)hd * dh);
                    }
                } else {
                    // head-interleaved: [q_h(dh) | gate_h(dh)] per head
                    for (int hd = 0; hd < H; ++hd) {
                        // No copy into `g` here: the q half is written by the
                        // loop below, and the gate half by the one after it.
                        // A stray `copy_n(src + hd*2*dh, dh, g + (hd+1)*dh)` used
                        // to sit here and landed a full row past the slot for
                        // hd == H-1 — one row off the end of agate_ on the last
                        // token, which SIGSEGVs once T is large enough that the
                        // allocation ends on a page boundary (T=649 died, T=169
                        // survived by luck of placement).
                        for (int d = 0; d < dh; ++d)
                            dstq[(std::size_t)hd * dh + d] = src[(std::size_t)hd * 2 * dh + d];
                        for (int d = 0; d < dh; ++d)
                            g[(std::size_t)hd * dh + d] = src[(std::size_t)hd * 2 * dh + dh + d];
                    }
                }
            }
        }
        if (qk_norm_) {
            // qwen35 applies a per-head RMS norm to Q and K before RoPE
            // (llama.cpp build_layer_attn, attn_q_norm/attn_k_norm).
            const float* qn_w = W("self_attn.q_norm.weight");
            const float* kn_w = W("self_attn.k_norm.weight");
            for (int t = 0; t < T; ++t)
                for (int hd = 0; hd < H; ++hd)
                    rms(q_.data() + (std::size_t)t * QPROJ + (std::size_t)hd * dh,
                        q_.data() + (std::size_t)t * QPROJ + (std::size_t)hd * dh,
                        qn_w, dh, eps);
            for (int t = 0; t < T; ++t)
                for (int kv = 0; kv < KVH; ++kv)
                    rms(k_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh,
                        k_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh,
                        kn_w, dh, eps);
        }
        // Partial RoPE when the config rotates only part of each head
        // (qwen35: n_rot = 64 of 256). Text-only MRoPE reduces to this:
        // all three MRoPE position ids are equal, so the section
        // interleaving is inert and the half-split NORM pairing applies.
        if (n_rot_ > 0 && n_rot_ < dh) {
            // One frequency table for the rotated width and one cos/sin table
            // per position, shared by every Q and K head (was: a pow, a cos and
            // a sin per pair per head).
            const int p_half = n_rot_ / 2;
            const std::span<const float> inv = rope_freq_partial_.get(n_rot_, theta);
            if ((int)rope_sincos_.size() < p_half * 2)
                rope_sincos_.resize((std::size_t)p_half * 2);
            float* cosines = rope_sincos_.data();
            float* sines = cosines + p_half;
            for (int t = 0; t < T; ++t) {
                tmgdn::rope_sincos(cosines, sines, inv, pos0 + t);
                for (int hd = 0; hd < H; ++hd)
                    tmgdn::rope_partial_head(
                        q_.data() + (std::size_t)t * QPROJ + (std::size_t)hd * dh,
                        n_rot_, cosines, sines);
                for (int kv = 0; kv < KVH; ++kv)
                    tmgdn::rope_partial_head(
                        k_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh,
                        n_rot_, cosines, sines);
            }
        } else {
            rope_qk(q_.data(), k_.data(), T, pos0);
        }
        // Only the first QD floats of each row are the query now: the
        // de-interleave compacted q into [head][dh] and parked the gate in
        // agate_, leaving the back half of q_ stale.
        for (int t = 0; t < T; ++t)
            tm_dump("Qcur_normed", l, q_.data() + (std::size_t)t * QPROJ, (std::size_t)QD);
        tm_dump("Kcur_normed", l, k_.data(), (std::size_t)T * KVD);
        P.mark(LlamaProf::Sec::ROPE);
        tm_dump_row("qrope", l, q_.data(), (std::size_t)T * QPROJ);
        // Same checkpoint one step later for K, which is what lands in the cache:
        // the KV-format work (docs/KV_COMPRESSION_NOTES.md, specs/TENSORMARK_KV_Q8_SPEC.md)
        // needs real post-RoPE rows, and RoPE has already been applied above —
        // "Kcur_normed" is a pre-existing name, not a claim about the stage.
        tm_dump_row("krope", l, k_.data(), (std::size_t)T * KVD);

        // append K,V rows to the per-KV-head cache
        auto& K = kc[l]; auto& Vv = vc[l];
        // Rows land at pos0, not `seen`: they agree on the normal path, but a
        // hybrid prefill runs this on the TAIL rows (pos0 = seen + gpu_rows)
        // while `seen` still points at the start of the whole prefill.
        const int cached = pos0;
        for (int t = 0; t < T; ++t)
            for (int kv = 0; kv < KVH; ++kv) {
                // A packed tensor has no fp32 vectors at all (allocate_kv
                // leaves them empty), so each copy is guarded on emptiness
                // rather than on the type flags twice.
                if (!K.empty())
                std::copy_n(k_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh, dh,
                            K.begin() + (std::size_t)kv * ctx * dh
                                      + (std::size_t)(cached + t) * dh);
                if (!Vv.empty())
                std::copy_n(v_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh, dh,
                            Vv.begin() + (std::size_t)kv * ctx * dh
                                       + (std::size_t)(cached + t) * dh);
            }
        // ADDITIVE step of the packed-KV plan: keep an independent Q8_0 copy of
        // every row written above. Nothing reads it yet (the read cutover is
        // the next commit), so this cannot move any existing number — and
        // TM_KV_SELFCHECK=1 is what proves the layout before that cutover.
        if (kv_ktype() || kv_vtype()) {
            const std::size_t nb = (std::size_t)(dh / tmq::kBlock);
            for (int t = 0; t < T; ++t)
                for (int kv = 0; kv < KVH; ++kv) {
                    const std::size_t row =
                        (std::size_t)kv * (std::size_t)ctx + (std::size_t)(cached + t);
                    const std::size_t off = row * nb * sizeof(tmq::BlockQ8_0);
                    if (kv_ktype())
                        tmq::quantize_row_q8_0(
                            k_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh,
                            reinterpret_cast<tmq::BlockQ8_0*>(kq_[(std::size_t)l].data() + off),
                            (std::size_t)dh);
                    if (kv_vtype())
                        tmq::quantize_row_q8_0(
                            v_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh,
                            reinterpret_cast<tmq::BlockQ8_0*>(vq_[(std::size_t)l].data() + off),
                            (std::size_t)dh);
                }
            if (kv_selfcheck_) kv_selfcheck_rows(l, cached, T);
        }

        P.mark(LlamaProf::Sec::KV_APPEND);
        // attention: H query heads against REP-repeated KV heads.
        // Batched (default): the REP = H/KVH query heads sharing a KV head
        // are CONTIGUOUS in each q_ row (hd = kv*REP .. kv*REP+REP-1), so
        // per (t, kv group) the scores are one sgemm (REP x dh) x (dh x
        // allow) and the V-weighted sum another (REP x allow) x (allow x
        // dh) — 2*KVH*T sgemms instead of 2*H*T cblas sdot/axpy calls on
        // dh=64 vectors, which dominated long-context prefill (the O(T^2)
        // loop, not GEMM throughput, dropped 1024-token prompts to ~95
        // t/s). TM_ATTN_NAIVE=1 / set_attn_naive(true) keeps the per-head
        // sdot/axpy loop for A/B benches and oracle gates.
        if (att_.size() < (std::size_t)std::max(T * QD, H * ctx))
            att_.resize((std::size_t)std::max(T * QD, H * ctx));
        if (ao_.size() < (std::size_t)T * QD) ao_.resize((std::size_t)T * QD);
        const float scale = 1.f / std::sqrt((float)dh);
        // T == 1 (decode) also takes the batched path since 2026-09-06: the
        // per-position loop issued 2*H*allow tiny BLAS calls per layer and
        // was 26% of a decode token at ~500 context (sample-profiled), on the
        // main thread while pool workers idled. Batched decode: strict Q4
        // oracle 32/32 / max err 9.3e-5 (loop: 9.7e-5); TinyLlama Q4 paired
        // same-window +8.7% at tg256/ctx384 and +31% at tg1024/ctx2048 (6/6
        // pairs each); it also makes T=1 and T>1 continuations agree, which
        // the loop failed under SDOT at the 0.02 replay bar. The earlier
        // 27/32 result predates the batched-path lda/ldc fix. Greedy hashes
        // differ from the loop's (fp reorder on near-ties), not bit-stable.
        // A packed cache reaches the blocked path (below) when BOTH tensors are
        // packed and the prompt is long enough to block; a mixed-type cache and
        // a short prompt stay on the loop, which handles either type per row.
        const bool packed_kv = kv_ktype() || kv_vtype();
        const bool packed_blocked =
            kv_ktype() && kv_vtype() && T >= 8 && attn_prefill_block_ > 0;
        if (attn_naive_ || (T == 1 && !attn_decode_batched_) ||
            (packed_kv && !packed_blocked)) {
        // Packed-KV read cutover (spec step 2): this per-head loop reads Q8_0
        // rows directly; the sgemm and decode-pool paths still assume fp32 and
        // would read garbage from a packed buffer, so a packed cache the
        // blocked path cannot take lands here. TM_ATTN_NAIVE=1 keeps it as the
        // A/B arm and the oracle for the blocked packed kernel.
        const int kvnb = dh / tmq::kBlock;
        const std::size_t kvrow = (std::size_t)kvnb * sizeof(tmq::BlockQ8_0);
        const std::uint8_t* kbuf = kv_ktype() ? kq_[(std::size_t)l].data() : nullptr;
        const std::uint8_t* vbuf = kv_vtype() ? vq_[(std::size_t)l].data() : nullptr;
        for (int t = 0; t < T; ++t) {
            const int allow = pos0 + t + 1;
            for (int hd = 0; hd < H; ++hd) {
                const int kv = hd / REP_;             // GQA mapping
                const float* qh = q_.data() + (std::size_t)t * QPROJ + (std::size_t)hd * dh;
                float m = -1e30f;
                if (kbuf) {
                    kv_quant_query(qh, dh);
                    const auto* Kq = reinterpret_cast<const tmq::BlockQ8_0*>(
                        kbuf + (std::size_t)kv * (std::size_t)ctx * kvrow);
                    for (int s = 0; s < allow; ++s) {
                        att_[s] = kv_row_dot(Kq + (std::size_t)s * kvnb, kvnb) * scale;
                        m = std::max(m, att_[s]);
                    }
                } else {
                    const float* Kh = &K[(std::size_t)kv * ctx * dh];
                    for (int s = 0; s < allow; ++s) {
                        att_[s] = cblas_sdot(dh, qh, 1, &Kh[(std::size_t)s * dh], 1) * scale;
                        m = std::max(m, att_[s]);
                    }
                }
                float z = 0.f;
                for (int s = 0; s < allow; ++s) { att_[s] = std::exp(att_[s] - m); z += att_[s]; }
                float* acc = ao_.data() + (std::size_t)t * QD + (std::size_t)hd * dh;
                std::fill(acc, acc + dh, 0.f);
                if (vbuf) {
                    const auto* Vq = reinterpret_cast<const tmq::BlockQ8_0*>(
                        vbuf + (std::size_t)kv * (std::size_t)ctx * kvrow);
                    for (int s = 0; s < allow; ++s)
                        kv_row_axpy(att_[s] / z, Vq + (std::size_t)s * kvnb, acc, kvnb);
                } else {
                    const float* Vh = &Vv[(std::size_t)kv * ctx * dh];
                    for (int s = 0; s < allow; ++s)
                        cblas_saxpy(dh, att_[s] / z, &Vh[(std::size_t)s * dh], 1, acc, 1);
                }
            }
        }
        } else if (packed_blocked) {
            // Packed blocked prefill: same tiling and the same largest-block-
            // first ordering as the fp32 path below, so the pool balances the
            // causal triangle identically; only the row representation differs.
            const int B = attn_prefill_block_;
            const int nblocks = (T + B - 1) / B;
            AttnBlockJobPacked a{q_.data(),       kq_[(std::size_t)l].data(),
                                 vq_[(std::size_t)l].data(), ao_.data(),
                                 T,           pos0,   QPROJ, QD,
                                 REP_,        dh,     ctx,   KVH,
                                 B,           nblocks, scale};
            if (attn_prefill_pool_ && gemv_pool().nthreads > 1) {
                GemvJob j;
                j.kern = &attn_block_kern_packed; j.a.w = &a;
                j.n_out = KVH * nblocks; j.chunk = 1;
                gemv_pool().dispatch(j);
            } else {
                for (int u = 0; u < KVH * nblocks; ++u)
                    attn_block_unit_packed(a, u / nblocks, u % nblocks);
            }
        } else if (T == 1 && attn_decode_pool_ && gemv_pool().nthreads > 1 && KVH > 1) {
            // Decode: one KV group per pool slice, each with private score
            // rows (att_ holds KVH * REP * allow floats); the pool's own
            // completion barrier orders the writes before o_proj reads ao_.
            const int allow = pos0 + 1;
            AttnJob a{q_.data(), K.data(), Vv.data(), att_.data(), ao_.data(),
                      REP_, dh, allow, ctx, scale, attn_vsoftmax_};
            GemvJob j;
            j.kern = &attn_group_kern;
            j.a.w = &a;
            j.n_out = KVH; j.chunk = 1;
            gemv_pool().dispatch(j);
        } else if (T >= 8 && attn_prefill_block_ > 0) {
            // Blocked prefill attention (2026-09-07): the per-token loop
            // below issues 2 tiny sgemms + REP softmax rows per (token, KV
            // group) — 352k BLAS calls for a 2000-token prompt, 41% of the
            // prefill. Units of B query tokens x one KV group run on the
            // GemvPool (see attn_block_unit); TM_ATTN_PREFILL_POOL=0 runs
            // them on the calling thread.
            const int B = attn_prefill_block_;
            const int nblocks = (T + B - 1) / B;
            AttnBlockJob a{q_.data(), K.data(), Vv.data(), ao_.data(),
                           T, pos0, QPROJ, QD, REP_, dh, ctx, KVH, B, nblocks, scale};
            if (attn_prefill_pool_ && gemv_pool().nthreads > 1) {
                GemvJob j;
                j.kern = &attn_block_kern; j.a.w = &a;
                j.n_out = KVH * nblocks; j.chunk = 1;
                gemv_pool().dispatch(j);
            } else {
                for (int u = 0; u < KVH * nblocks; ++u) attn_block_unit(a, u / nblocks, u % nblocks);
            }
        } else {
            for (int t = 0; t < T; ++t) {
                const int allow = pos0 + t + 1;
                float* S = att_.data();               // REP_ x allow, ld=allow
                for (int kv = 0; kv < KVH; ++kv) {
                    // S(REP x allow) = qg(REP x dh) x Kh^T(dh x allow). The
                    // REP query heads are contiguous WITHIN token t's row
                    // (stride dh), so A has lda=dh — NOT QD (that would read
                    // token t+r's heads: OOB + garbage). Output rows are the
                    // REP heads inside token t's row: ldc=dh, not D.
                    attn_group(q_.data() + (std::size_t)t * QPROJ + (std::size_t)kv * REP_ * dh,
                               &K[(std::size_t)kv * ctx * dh], &Vv[(std::size_t)kv * ctx * dh],
                               S, ao_.data() + (std::size_t)t * QD + (std::size_t)kv * REP_ * dh,
                               REP_, dh, allow, scale, attn_vsoftmax_);
                }
            }
        }
        P.mark(LlamaProf::Sec::ATTN);
        tm_dump("attn_pregate", l, ao_.data(), (std::size_t)T * QD);
        // full row span: every token, so positions >= 1 (where RoPE stops
        // being the identity) are compared too.
        tm_dump_row("ao_pregate", l, ao_.data(), (std::size_t)T * QD);
        tm_dump_row("agate", l, agate_.data(), agate_.size() ? (std::size_t)QD : 0);
        if (attn_output_gate_) {
            // llama.cpp build_layer_attn: cur = cur * sigmoid(gate), applied
            // head-major, matching the attention output layout.
            for (int t = 0; t < T; ++t)
                tmgdn::sigmoid_gate(ao_.data() + (std::size_t)t * QD,
                                    agate_.data() + (std::size_t)t * QD, QD);
        }
        tm_dump("attn_gated", l, ao_.data(), (std::size_t)T * QD);
        tm_dump_row("ao_gated", l, ao_.data(), (std::size_t)QD);
        if (h2_.size() < (std::size_t)T * D) h2_.resize((std::size_t)T * D);
        const auto oname = "model.layers." + std::to_string(l) + ".self_attn.o_proj.weight";
        if (quantized()) {
            if (T == 1 && decode_gpu_ &&
                batched_q4(ao_.data(), oname, h2_.data(), 1, D, QD)) {
                // GPU decode o_proj
            } else if (T == 1)
                gemv_w(oname, ao_.data(), QD, D, h2_.data());
            else if (amx_ok(T) && !gpu_prefill_ready(T, QD) &&
                     gemm_w(oname, ao_.data(), T, QD, D, h2_.data())) {
                // AMX prefill
            } else if (!batched_q4(ao_.data(), oname, h2_.data(), T, D, QD))
                for (int t = 0; t < T; ++t)
                    gemv_w(oname, ao_.data() + (std::size_t)t * QD, QD, D,
                           h2_.data() + (std::size_t)t * D);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, D, QD, 1.f,
                        ao_.data(), QD, W("self_attn.o_proj.weight"), QD, 0.f,
                        h2_.data(), D);
        }
        tm_dump("attn_output(o_proj)", l, h2_.data(), (std::size_t)T * D);
        tm_dump_row("attn_out", l, h2_.data(), (std::size_t)D);
        P.mark(LlamaProf::Sec::OPROJ);
        // plain loop on purpose: DRAM-bound (T*D >= 2^21 floats) — vDSP_vadd measured slower (see autograd.h tm_elem_stream)
        for (int i = 0; i < T * D; ++i) x[i] += h2_[i];
        tm_dump("attn_residual", l, x.data(), (std::size_t)T * D);
        P.mark(LlamaProf::Sec::RES1);

        ffn(l, x, T);
    }


    // SwiGLU FFN + post-attention residual, shared by the dense and the
    // Gated-DeltaNet layer paths (llama.cpp build_layer_ffn is identical for
    // both layer kinds).
    void ffn(int l, std::vector<float>& x, int T) {
        const auto W = [&](const std::string& k) -> const float* {
            return w.at("model.layers." + std::to_string(l) + "." + k).data();
        };
        LlamaProf& P = llama_prof();
        // MLP: SwiGLU — down( silu(gate(h2)) * up(h2) )
        if (h2s_.size() < (std::size_t)T * D) h2s_.resize((std::size_t)T * D);
        for (int t = 0; t < T; ++t)
            rms(x.data() + (std::size_t)t * D, h2s_.data() + (std::size_t)t * D,
                W("post_attention_layernorm.weight"), D, eps);
        P.mark(LlamaProf::Sec::RMS2);
        if (g_.size() < (std::size_t)T * F) g_.resize((std::size_t)T * F);
        if (u_.size() < (std::size_t)T * F) u_.resize((std::size_t)T * F);
        const auto gname = "model.layers." + std::to_string(l) + ".mlp.gate_proj.weight";
        const auto uname = "model.layers." + std::to_string(l) + ".mlp.up_proj.weight";
        const auto dname = "model.layers." + std::to_string(l) + ".mlp.down_proj.weight";
        if (quantized()) {
            // AMX prefill tried first when enabled (A/B vs GPU fused) — for
            // T > 1 only: this site lacked the guard the other three have,
            // so with AMX prefill on by default (2026-09-06) every DECODE
            // token dequantized gate|up (the largest weights) to fp32 for a
            // one-row GEMM: CPU decode 7.7 t/s instead of ~65 (profiling
            // showed dequant_rows running on the main thread during decode).
            if (T > 1 && amx_ok(T) && !gpu_prefill_ready(T, D) &&
                gemm_w(gname, h2s_.data(), T, D, F, g_.data()) &&
                gemm_w(uname, h2s_.data(), T, D, F, u_.data())) {
                // AMX prefill did gate|up
            } else {
            const std::string gu_names[2] = {gname, uname};
            const int gu_ns[2] = {F, F};
            if (gu_.size() < (std::size_t)T * 2 * F) gu_.resize((std::size_t)T * 2 * F);
            if ((T > 1 || decode_gpu_) &&
                batched_q4_parts(h2s_.data(), gu_names, gu_ns, 2,
                                 gu_.data(), T, D)) {
                for (int t = 0; t < T; ++t) {
                    std::copy_n(gu_.data() + (std::size_t)t * 2 * F, F,
                                g_.data() + (std::size_t)t * F);
                    std::copy_n(gu_.data() + (std::size_t)t * 2 * F + F, F,
                                u_.data() + (std::size_t)t * F);
                }
            } else if (T == 1 && grouped_gemv_ok()) {
                float* const outputs[2] = {g_.data(), u_.data()};
                gemv_w_group(gu_names, gu_ns, outputs, h2s_.data(), D);
            } else {
                for (int t = 0; t < T; ++t) {
                    gemv_w(gname, h2s_.data() + (std::size_t)t * D, D, F,
                           g_.data() + (std::size_t)t * F);
                    gemv_w(uname, h2s_.data() + (std::size_t)t * D, D, F,
                           u_.data() + (std::size_t)t * F);
                }
            }
            }
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, F, D,
                        1.f, h2s_.data(), D, W("mlp.gate_proj.weight"), D, 0.f, g_.data(), F);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, F, D,
                        1.f, h2s_.data(), D, W("mlp.up_proj.weight"), D, 0.f, u_.data(), F);
        }
        P.mark(LlamaProf::Sec::GATE_UP);
        // silu(g) * u through vForce/vDSP in 64K-float strides (a scalar
        // expf per element was 0.4 s of a 2000-token prefill).
        {
            const std::size_t n = (std::size_t)T * F;
            if (silu_tmp_.size() < 65536) silu_tmp_.resize(65536);
            for (std::size_t i0 = 0; i0 < n; i0 += 65536) {
                const int len = (int)std::min<std::size_t>(65536, n - i0);
                float* g = g_.data() + i0; float* tmp = silu_tmp_.data();
                vDSP_vneg(g, 1, tmp, 1, (vDSP_Length)len);           // -g
                vvexpf(tmp, tmp, &len);                              // e^-g
                const float one = 1.f;
                vDSP_vsadd(tmp, 1, &one, tmp, 1, (vDSP_Length)len);  // 1 + e^-g
                vDSP_vdiv(tmp, 1, g, 1, g, 1, (vDSP_Length)len);     // g / (1 + e^-g)
                vDSP_vmul(g, 1, u_.data() + i0, 1, g, 1, (vDSP_Length)len);
            }
        }
        P.mark(LlamaProf::Sec::SILU);
        if (dn_.size() < (std::size_t)T * D) dn_.resize((std::size_t)T * D);
        if (quantized()) {
            if (T == 1 && decode_gpu_ &&
                batched_q4(g_.data(), dname, dn_.data(), 1, D, F)) {
                // GPU decode down_proj
            } else if (T == 1)
                gemv_w(dname, g_.data(), F, D, dn_.data());
            else if (amx_ok(T) && !gpu_prefill_ready(T, F) &&
                     gemm_w(dname, g_.data(), T, F, D, dn_.data())) {
                // AMX prefill
            } else if (!batched_q4(g_.data(), dname, dn_.data(), T, D, F))
                for (int t = 0; t < T; ++t)
                    gemv_w(dname, g_.data() + (std::size_t)t * F, F, D,
                           dn_.data() + (std::size_t)t * D);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, D, F,
                        1.f, g_.data(), F, W("mlp.down_proj.weight"), F, 0.f, dn_.data(), D);
        }
        P.mark(LlamaProf::Sec::DOWN);
        for (int i = 0; i < T * D; ++i) x[i] += dn_[i];
        P.mark(LlamaProf::Sec::RES2);
    }

    // Fused dequant+GEMV for an f16 weight, dispatched through the shared
    // pool (the same mechanism gemm_w uses for its row-parallel dequant; a
    // capture-less lambda is a valid GemvJob::kern). The kernel gets the row
    // base, x, n_in and the [o0,o1) slice, so no payload field is needed.
    void gemv_f16(const std::string& name, const float* x, int n_in, int n_out,
                  float* y) const {
        const std::uint16_t* W = f16_.at(name);
        if (!W) throw std::runtime_error("llama: f16 tensor not resident: " + name);
        GemvJob j;
        j.a = GemvArgs{.w = W,
                       .x = std::span(x, (std::size_t)n_in),
                       .y = std::span(y, (std::size_t)n_out),
                       .n_in = n_in};
        j.n_out = n_out;
        j.kern = [](const GemvArgs& a, int o0, int o1) {
            const float* xp = a.x.data();
            int ni = a.n_in;
            const std::uint16_t* Wp = static_cast<const std::uint16_t*>(a.w);
            float* yp = a.y.data();
            for (int o = o0; o < o1; ++o) {
                const std::uint16_t* row = Wp + (std::size_t)o * ni;
                float acc = 0.f;
                int i = 0;
#ifdef __ARM_NEON
                // The scalar form below spends a software fp16->fp32 conversion
                // (shift, exponent rebias, denormal fixup) on every element, which
                // made the f16 tensors the dominant decode cost: measured 69% of
                // accounted time on Qwen3.5-0.8B (o_proj 53.6% + down_gemm 15.1%),
                // at ~6x the per-weight cost of the Q4_0 sdot path. NEON converts
                // four halves per instruction, so eight weights cost two converts
                // and two FMAs. Tail elements stay scalar.
                float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
                for (; i + 8 <= ni; i += 8) {
                    const float16x4_t lo = vreinterpret_f16_u16(vld1_u16(row + i));
                    const float16x4_t hi = vreinterpret_f16_u16(vld1_u16(row + i + 4));
                    a0 = vfmaq_f32(a0, vcvt_f32_f16(lo), vld1q_f32(xp + i));
                    a1 = vfmaq_f32(a1, vcvt_f32_f16(hi), vld1q_f32(xp + i + 4));
                }
                const float32x4_t s = vaddq_f32(a0, a1);
                const float32x2_t p = vadd_f32(vget_low_f32(s), vget_high_f32(s));
                acc = vget_lane_f32(p, 0) + vget_lane_f32(p, 1);
#endif
                for (; i < ni; ++i) acc += tmq::fp16_to_fp32(row[i]) * xp[i];
                yp[o] = acc;
            }
        };
        gemv_pool().dispatch(j);
    }

    // One projection layer `y = W x` for a quantized tensor, picking the same
    // device ladder the dense path uses: fused GPU dispatch (off for hybrid
    // models), AMX dequant-GEMM for prefill, per-token qgemv otherwise.
    void linear_proj(const std::string& name, const float* x, int T, int in,
                     int out, float* y) {
        if (f16_.count(name)) {
            // Precision-preserved f16 weights. T > 1 loops one GEMV per row: a
            // batched f16 kernel is a later optimisation and this tag is only
            // reachable for the tensors the converter promoted to f16.
            for (int t = 0; t < T; ++t)
                gemv_f16(name, x + (std::size_t)t * in, in, out,
                         y + (std::size_t)t * out);
            return;
        }
        if (!quantized()) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, out, in, 1.f,
                        x, in, w.at(name).data(), in, 0.f, y, out);
            return;
        }
        if (T == 1) {
            if (!batched_q4(x, name, y, 1, out, in)) gemv_w(name, x, in, out, y);
            return;
        }
        if (batched_q4(x, name, y, T, out, in)) return;
        if (amx_ok(T) && gemm_w(name, x, T, in, out, y)) return;
        // 2 <= T < prefill_loop_t(): without this, the loop below calls gemv_w once
        // per position and every one of those calls re-reads and re-unpacks each
        // weight row. kq6k_multi_rows unpacks each block once for all T positions.
        // TM_KQ_MULTI=0 restores the per-row loop, which is also the reference this
        // has to reproduce token-for-token.
        static const bool kq_multi = [] {
            const char* e = getenv("TM_KQ_MULTI");
            return !e || e[0] != '0';
        }();
        if (kq_multi && gemm_kq_multi(name, x, T, in, out, y)) return;
        for (int t = 0; t < T; ++t)
            gemv_w(name, x + (std::size_t)t * in, in, out, y + (std::size_t)t * out);
    }

    // Materialise the [kernel_tap, channel] depthwise conv1d kernel as dense
    // fp32. It is small (4 x 6144 for the 0.8B) and read once per layer call.
    // ggml declares the depthwise kernel as {d_conv, conv_dim}, i.e. ne0 (the
    // FASTEST axis) is the tap — so the file order is channel-major with the
    // tap innermost: element (tap, channel) lives at channel*taps + tap.
    // conv1d_silu wants [tap][channel] rows, so transposing here is mandatory;
    // reading it as [tap][channel] silently scrambles the kernel.
    void conv_kernel_dense(const std::string& name, int taps, int channels,
                           std::vector<float>& dst) {
        const std::size_t n = (std::size_t)taps * channels;
        if (dst.size() < n) dst.resize(n);
        if (lkflat_.size() < n) lkflat_.resize(n);
        auto q8 = q8_.find(name);
        auto q4 = q4_.find(name);
        auto f16 = f16_.find(name);
        auto f32 = w.find(name);
        if (q8 != q8_.end())      dequant_rows(q8->second, true, (int)n, 0, 1, lkflat_.data());
        else if (q4 != q4_.end()) dequant_rows(q4->second, false, (int)n, 0, 1, lkflat_.data());
        else if (f16 != f16_.end() && f16->second) {
            const std::uint16_t* W = f16->second;
            for (std::size_t i = 0; i < n; ++i) lkflat_[i] = tmq::fp16_to_fp32(W[i]);
        } else if (f32 != w.end() && f32->second.data())
            std::memcpy(lkflat_.data(), f32->second.data(), n * sizeof(float));
        else throw std::runtime_error("llama: missing conv1d kernel " + name);
        for (int c = 0; c < channels; ++c)
            for (int j = 0; j < taps; ++j)
                dst[(std::size_t)j * channels + c] = lkflat_[(std::size_t)c * taps + j];
    }

    // Layer-type dispatch: full attention for the interleaved dense layers,
    // Gated DeltaNet for the linear-attention layers (llama.cpp qwen35.cpp
    // `hparams.is_recr(il)`).
    void dispatch_block(int l, std::vector<float>& x, int T, int pos0) {
        // x entering layer l is layer l-1's `post_ffn` (the input embed at l=0).
        if (l == 0 || l == 3 || l == L - 1) tm_dump("post_ffn(prev)", l, x.data(), (std::size_t)T * D);
        if (is_linear_layer(l)) block_linear(l, x, T);
        else block(l, x, T, pos0);
        if (l == 0 || l == 3 || l == L - 1) tm_dump("post_ffn", l, x.data(), (std::size_t)T * D);
    }

    // ---------------------------------------------------------------------
    // Gated DeltaNet layer (qwen35 linear attention). Pipeline pinned from
    // llama.cpp src/models/qwen35.cpp::build_layer_attn_linear:
    //
    //   inp        = input_norm(x)
    //   qkv        = qkv_proj @ inp          # q(D_k*H_k) | k(D_k*H_k) | v(D_v*H_v)
    //   z          = gate_proj @ inp         # [H_v * D_v], a SEPARATE tensor
    //   beta       = sigmoid(beta_proj @ inp)                      [H_v]
    //   decay      = exp(ssm_a * softplus(alpha_proj @ inp + dt))  [H_v]
    //   conv       = silu(depthwise_conv1d(qkv))    # causal, kernel K
    //   q,k,v      = split(conv);  q,k = l2_norm(q), l2_norm(k);  q /= sqrt(D_k)
    //   S_t        = decay*S_{t-1} + k (x) (beta * (v - S_{t-1}^T k))
    //   o          = gated_rms_norm(S_t^T q, ssm_norm, z)   # rms_norm * silu(z)
    //   out        = out_proj @ o
    //
    // S is [D_k, D_v] per value head and lives in ssm_st_[l] — O(1) in context
    // length, which is why the layer has no KV cache. This path runs on the CPU
    // only (resolve_decode_gpu_auto forces the CPU for hybrid models).
    void block_linear(int l, std::vector<float>& x, int T) {
        demand_layer(l);
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const auto Wf = [&](const char* k) -> const float* { return w.at(p + k).data(); };
        LlamaProf& P = llama_prof();
        P.start();
        if (is_hybrid()) {
            // The GGUF layout of V/z/out_proj heads is already the tiled order
            // ggml_repeat consumes (conversion/qwen.py `_LinearAttentionVReorderBase`).
        }
        const int Dk = lk_dim_, Dv = lv_dim_, Hk = lk_heads_, Hv = lv_heads_;
        const int K = conv_kernel_, C = linear_conv_channels();
        const int qn = Hk * Dk, vn = Hv * Dv;
        if (T <= 0) return;

        // input_norm
        if (h_.size() < (std::size_t)T * D) h_.resize((std::size_t)T * D);
        for (int t = 0; t < T; ++t)
            rms(x.data() + (std::size_t)t * D, h_.data() + (std::size_t)t * D,
                Wf("input_layernorm.weight"), D, eps);
        P.mark(LlamaProf::Sec::RMS1);

        // fused q|k|v, the separate z gate, and the per-value-head beta/decay
        if (lqkv_.size() < (std::size_t)T * C) lqkv_.resize((std::size_t)T * C);
        if (lz_.size() < (std::size_t)T * vn) lz_.resize((std::size_t)T * vn);
        tm_dump("attn_norm", l, h_.data(), (std::size_t)T * D);
        tm_dump_row("attn_norm", l, h_.data(), (std::size_t)D);
        linear_proj(p + "linear_attn.qkv_proj.weight", h_.data(), T, D, C, lqkv_.data());
        linear_proj(p + "linear_attn.gate_proj.weight", h_.data(), T, D, vn, lz_.data());
        tm_dump("linear_attn_qkv_mixed", l, lqkv_.data(), (std::size_t)T * C);
        tm_dump("z", l, lz_.data(), (std::size_t)T * vn);
        if (lb_.size() < (std::size_t)T * Hv) lb_.resize((std::size_t)T * Hv);
        if (la_.size() < (std::size_t)T * Hv) la_.resize((std::size_t)T * Hv);
        linear_proj(p + "linear_attn.beta.weight", h_.data(), T, D, Hv, lb_.data());
        linear_proj(p + "linear_attn.alpha.weight", h_.data(), T, D, Hv, la_.data());
        const float* dt_bias = Wf("linear_attn.dt_proj.bias");
        const float* ssm_a = Wf("linear_attn.a_log");
        if (ldec_.size() < (std::size_t)T * Hv) ldec_.resize((std::size_t)T * Hv);
        for (int t = 0; t < T; ++t) {
            float* b = lb_.data() + (std::size_t)t * Hv;
            for (int h = 0; h < Hv; ++h) b[h] = tmgdn::sigmoid(b[h]);
            tmgdn::gdn_decay(ldec_.data() + (std::size_t)t * Hv,
                             la_.data() + (std::size_t)t * Hv, dt_bias, ssm_a, Hv);
        }
        tm_dump("beta_sigmoid", l, lb_.data(), (std::size_t)T * Hv);
        // NOTE: la_ holds the RAW alpha projection and ldec_ the decay
        // exp(g); llama.cpp's `a_softplus` is softplus(alpha+dt) and its
        // `gate` is the log-decay g. Compare la_ to `alpha-<l>` and, for the
        // decay, re-derive exp(g) from the reference's gate elements - a sum
        // cannot be exponentiated after the fact.
        tm_dump("alpha(raw)", l, la_.data(), (std::size_t)T * Hv);
        tm_dump("decay(exp g)", l, ldec_.data(), (std::size_t)T * Hv);
        if (tm_dump_on()) {
            // llama.cpp names the two middle quantities: `a_softplus` is
            // softplus(alpha+dt) and `gate` is that times ssm_a (the log
            // decay). Sums, not exponentials, so they compare directly.
            double s_sp = 0.0, s_g = 0.0;
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < Hv; ++h) {
                    const double sp = (double)tmgdn::softplus(
                        la_[(std::size_t)t * Hv + h] + dt_bias[h]);
                    s_sp += sp;
                    s_g += sp * (double)ssm_a[h];
                }
            std::printf("TMDUMP %-28s l=%-3d n=%-8d sum=%.6f\n", "a_softplus(ref)", l, T * Hv, s_sp);
            std::printf("TMDUMP %-28s l=%-3d n=%-8d sum=%.6f\n", "gate(ref)", l, T * Hv, s_g);
            std::fflush(stdout);
        }
        P.mark(LlamaProf::Sec::GDN_PROJ);

        // causal depthwise conv over the fused stream + SiLU
        if (lconv_.size() < (std::size_t)T * C) lconv_.resize((std::size_t)T * C);
        conv_kernel_dense(p + "linear_attn.conv1d.weight", K, C, lker_);
        tm_dump("conv_kernel", l, lker_.data(), (std::size_t)K * C);
        tm_dump("conv_state_before", l, conv_st_[(std::size_t)l].data(),
                (std::size_t)(K - 1) * C);
        tmgdn::conv1d_silu(lconv_.data(), lqkv_.data(), conv_st_[(std::size_t)l].data(),
                           lker_.data(), T, C, K);
        tm_dump("conv_output_silu", l, lconv_.data(), (std::size_t)T * C);

        // split, per-head L2 norm, and the 1/sqrt(D_k) query scale
        if (lq_.size() < (std::size_t)T * qn) lq_.resize((std::size_t)T * qn);
        if (lk_.size() < (std::size_t)T * qn) lk_.resize((std::size_t)T * qn);
        if (lv_.size() < (std::size_t)T * vn) lv_.resize((std::size_t)T * vn);
        const float qscale = 1.f / std::sqrt((float)Dk);
        for (int t = 0; t < T; ++t) {
            const float* src = lconv_.data() + (std::size_t)t * C;
            float* q = lq_.data() + (std::size_t)t * qn;
            float* k = lk_.data() + (std::size_t)t * qn;
            std::copy_n(src, qn, q);
            std::copy_n(src + qn, qn, k);
            std::copy_n(src + 2 * qn, vn, lv_.data() + (std::size_t)t * vn);
            tmgdn::l2_norm(q, Hk, Dk, eps);
            tmgdn::l2_norm(k, Hk, Dk, eps);
            for (int i = 0; i < qn; ++i) q[i] *= qscale;
        }

        // q_/k_ are post-L2-norm and q is already scaled here, so these
        // correspond to llama.cpp's `q_conv_predelta` / `k_conv_predelta`.
        tm_dump("q_conv_predelta", l, lq_.data(), (std::size_t)T * qn);
        tm_dump("k_conv_predelta", l, lk_.data(), (std::size_t)T * qn);
        tm_dump("v_conv", l, lv_.data(), (std::size_t)T * vn);
        P.mark(LlamaProf::Sec::GDN_CONV);
        // the gated delta rule (state lives in ssm_st_[l])
        if (lo_.size() < (std::size_t)T * vn) lo_.resize((std::size_t)T * vn);
        for (int t = 0; t < T; ++t)
            tmgdn::step(gdn_scratch_, ssm_st_[(std::size_t)l].data(),
                        lq_.data() + (std::size_t)t * qn,
                        lk_.data() + (std::size_t)t * qn,
                        lv_.data() + (std::size_t)t * vn,
                        lb_.data() + (std::size_t)t * Hv,
                        ldec_.data() + (std::size_t)t * Hv,
                        lo_.data() + (std::size_t)t * vn, Hk, Hv, Dk, Dv);
        tm_dump("attn_output(recur)", l, lo_.data(), (std::size_t)T * vn);
        P.mark(LlamaProf::Sec::GDN_RECUR);

        // gated RMS norm (rms_norm * silu(z)) then the output projection
        for (int t = 0; t < T; ++t)
            tmgdn::gated_rms_norm(lo_.data() + (std::size_t)t * vn,
                                  Wf("linear_attn.norm.weight"),
                                  lz_.data() + (std::size_t)t * vn, Hv, Dv, eps);
        if (h2_.size() < (std::size_t)T * D) h2_.resize((std::size_t)T * D);
        tm_dump("final_output", l, lo_.data(), (std::size_t)T * vn);
        linear_proj(p + "linear_attn.out_proj.weight", lo_.data(), T, vn, D, h2_.data());
        tm_dump("linear_attn_out", l, h2_.data(), (std::size_t)T * D);
        P.mark(LlamaProf::Sec::GDN_OUT);
        for (int i = 0; i < T * D; ++i) x[i] += h2_[i];
        tm_dump("attn_residual", l, x.data(), (std::size_t)T * D);
        P.mark(LlamaProf::Sec::RES1);
        ffn(l, x, T);
    }

    std::vector<float> h2s_;
};

}  // namespace tmllama
