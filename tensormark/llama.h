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
extern "C" bool tm_metal_sgemm_q4(const float* A, const void* Wq, float* C,
                                  unsigned M, unsigned N, unsigned K)
    __attribute__((weak_import));
extern "C" bool tm_metal_q4_prefill(const float* A, const void* const* Wqs,
                                    const unsigned* Ns, unsigned nparts,
                                    float* C, unsigned M, unsigned K)
    __attribute__((weak_import));
extern "C" bool tm_metal_q4_gemv(const float* x, const void* Wq, float* y,
                                 unsigned N, unsigned K)
    __attribute__((weak_import));
extern "C" bool tm_metal_tok_begin(void) __attribute__((weak_import));
extern "C" bool tm_metal_tok_flush(void) __attribute__((weak_import));
extern "C" bool tm_metal_tok_reserve(int slot, size_t bytes)
    __attribute__((weak_import));
extern "C" bool tm_metal_tok_end(int outslot, float* dst, unsigned n)
    __attribute__((weak_import));
extern "C" bool tm_metal_tok_upload(int slot, size_t byte_off,
                                    const void* src, size_t bytes)
    __attribute__((weak_import));
extern "C" bool tm_metal_tok_end_async(void) __attribute__((weak_import));
extern "C" bool tm_metal_tok_wait_pending(void) __attribute__((weak_import));
extern "C" void tm_metal_tok_argmax(int xslot, int tokslot, unsigned idx, unsigned n)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_embed_q4(int tokslot, unsigned idx, int wid, uint64_t woff,
                                      int xslot, unsigned D)
    __attribute__((weak_import));
extern "C" void tm_metal_tok_sample(int xslot, int tokslot, unsigned idx, unsigned n, float inv_temp, unsigned seed)
    __attribute__((weak_import));
extern "C" unsigned tm_metal_tok_peek(int tokslot, unsigned idx) __attribute__((weak_import));
extern "C" bool tm_metal_tok_download(int slot, size_t byte_off,
                                      void* dst, size_t bytes)
    __attribute__((weak_import));
// K/V cache rows in the slot's element type (fp32 or half per the Metal
// policy); offsets and counts in elements.
extern "C" bool tm_metal_tok_kv_reserve(int slot, size_t n)
    __attribute__((weak_import));
extern "C" bool tm_metal_tok_kv_upload(int slot, size_t elem_off,
                                       const float* src, size_t n)
    __attribute__((weak_import));
extern "C" bool tm_metal_tok_kv_download(int slot, size_t elem_off,
                                         float* dst, size_t n)
    __attribute__((weak_import));
extern "C" bool tm_metal_tok_probe_copy(int srcslot, size_t src_off,
                                        int dstslot, unsigned n)
    __attribute__((weak_import));
extern "C" int tm_metal_tok_wbuf(const void* tensor_ptr, size_t bytes,
                                 const void* map_base, size_t map_size,
                                 int zero_copy, uint64_t& woff_out)
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
extern "C" void tm_metal_tok_embed_h(int tokslot, unsigned idx, int wid,
                                     uint64_t woff, int xslot, unsigned D)
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
#include <random>
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

// Strict numeric environment-variable parsing. Unlike atoi/atof, it rejects
// trailing garbage: "128abc" parses as nothing instead of silently meaning
// 128. On any failure, `fallback` is returned. Built on strtol/strtod (not
// from_chars): the CI runner's libc++ lacks the floating-point overloads.
inline double env_double(const char* e, double fallback) {
    if (!e || !*e) return fallback;
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    return end == e + std::strlen(e) ? v : fallback;
}
inline long env_long(const char* e, long fallback) {
    if (!e || !*e) return fallback;
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    return end == e + std::strlen(e) ? v : fallback;
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
    enum Sec { RMS1, QKV, QKV_SPLIT, ROPE, KV_APPEND, ATTN, OPROJ, RES1, RMS2, GATE_UP, SILU, DOWN, RES2, GPU_STACK, N };
    static constexpr const char* names[N] = {"rmsnorm1", "qkv_gemm", "qkv_split", "rope", "kv_append", "attention",
                                             "o_proj", "residual1", "rmsnorm2", "gate_up_gemm", "silu_mul", "down_gemm", "residual2",
                                             "gpu_stack"};
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
    void mark(Sec s) { if (!on) return; const auto t = std::chrono::steady_clock::now(); acc[s] += std::chrono::duration<double, std::milli>(t - t0).count(); t0 = t; }
};
inline LlamaProf& llama_prof() { static LlamaProf p; return p; }

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
        const char* e = getenv("TM_HYBRID_ESC_DELTA");
        return e ? std::max(0.005, env_double(e, 0.02)) : 0.02;
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
    static double pressure_csw()  { const char* e = getenv("TM_HYBRID_P_CSW");   return e ? env_double(e, 400.0) : 400.0; }
    static double pressure_fault() { const char* e = getenv("TM_HYBRID_P_FAULT"); return e ? env_double(e, 500.0) : 500.0; }
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
            const double f = env_double(e, -1.0);
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
            const double before = esc_base;
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

struct Llama {
    Llama() = default;
    Llama(const Llama&) = delete;
    Llama& operator=(const Llama&) = delete;
    int L = 22, D = 2048, H = 32, KVH = 4, dh = 64, V = 32000, ctx = 2048;
    int F = 5632;   // MLP intermediate (TinyLlama default; config.json overrides)
    float theta = 10000.f, eps = 1e-5f;
    std::map<std::string, Tensor> w;
    // M5 L2: optional quantized weights (load_quant). 2-D tensors stay
    // mmap'd blocks keyed by their safetensors name; 1-D (norms) are
    // dequantized into w at attach. Llama nn.Linear is (out, in) —
    // exactly M4's q8_gemv layout (dot along the in axis).
    std::map<std::string, const tmq::BlockQ8_0*> q8_;
    std::map<std::string, const tmq::BlockQ4_0*> q4_;
    std::map<std::string, std::pair<int, int>> qshape_;   // (out, in)
    std::unique_ptr<tmmq::Mapped> held_map_;
    std::vector<PageVec> kc, vc;   // per KV head, (ctx, dh); page-aligned for zero-copy GPU views
    std::vector<float> h_, q_, k_, v_, att_, ao_, h2_, g_, u_, dn_, xf_, logits_, xbuf_;
    std::vector<float> qkvv_, gu_;
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
    static double env_double(const char* name, double fallback) {
        const char* e = getenv(name);
        if (!e || !e[0]) return fallback;
        char* end = nullptr;
        const double v = std::strtod(e, &end);
        if (end == e || *end || !(v >= 0.0))
            throw std::invalid_argument(std::string(name) + " must be a non-negative number");
        return v;
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
        if (!decode_gpu_auto_ && !gpu_prefill_auto_) return;
        if (decode_gpu_auto_) decode_gpu_ = false;
        if (gpu_prefill_auto_) gpu_prefill_ = false;
        gpu_both_auto_ = false;
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        try {
            const bool usable = gpu_decode_usable();
            const double busy = tm_ambient_busy_cores();
            const double min_busy = env_double("TM_DECODE_GPU_AUTO_BUSY", 1.0);
            const double max_gb = env_double("TM_DECODE_GPU_AUTO_MAX_GB", 2.0);
            const double gb = (double)q4_weight_bytes() / (1u << 30);
            const bool want = usable && busy >= min_busy && gb <= max_gb;
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
            gpu_prefill_short_ok_ = gpu_prefill_auto_ && usable && gb <= max_gb && gpu_prefill_short_t() > 0;
            // Explicit TM_PREFILL_GPU=1 keeps the legacy exclusion: it wins
            // and decode stays on the CPU.
            if (decode_gpu_auto_) decode_gpu_ = decode_want && !(!gpu_prefill_auto_ && gpu_prefill_);
            gpu_both_auto_ = gpu_prefill_ && decode_gpu_ && gpu_prefill_auto_ && decode_gpu_auto_;
            if (getenv("TM_DEBUG_POOL"))
                fprintf(stderr, "[llama] gpu auto: ambient %.2f cores (>= %.2f), q4 %.2f GiB (<= %.2f) -> prefill %s, decode %s\n",
                        busy, min_busy, gb, max_gb, gpu_prefill_ ? "GPU" : "CPU", decode_gpu_ ? "GPU" : "CPU");
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
        env_long(getenv("TM_ATTN_PREFILL_BLOCK"), 128));
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
        SLOT_LOGITS = 9, SLOT_PROBS = 10, SLOT_TOK = 11, SLOT_KC = 100, SLOT_VC = 200
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
    int REP_ = H / KVH;   // GQA query heads per KV head (contiguous)

    // The GPU decode path needs every projection as Q4 (the gemv kernel
    // is Q4-only). Checked once; a q8 tensor anywhere means sticky CPU.
    bool gpu_decode_usable() {
        if (gpu_checked_) return gpu_ok_;
        gpu_checked_ = true;
        gpu_ok_ = q4_.count("lm_head.weight") != 0;
        for (int l = 0; l < L && gpu_ok_; ++l) {
            const std::string p =
                "model.layers." + std::to_string(l) + ".";
            for (const char* t :
                 {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight",
                  "mlp.down_proj.weight"})
                gpu_ok_ = gpu_ok_ && q4_.count(p + t) != 0;
        }
        return gpu_ok_;
    }
    // weight handle table: name -> (tok_wbuf id, byte offset in buffer)
    std::map<std::string, std::pair<int, uint64_t>> gpu_w_;
    bool gpu_w_ready_ = false;
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
    bool gpu_f16_ = false;
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
        bool open(const std::string& path) {
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
        const std::uint32_t sign = 0;
        for (std::size_t i = 0; i < row_elems; ++i) {
            const std::uint32_t ex = (src[i] >> 10) & 0x1fu, fr = src[i] & 0x03ffu;
            out[i] = ex == 0
                ? std::ldexpf((float)fr, -24) * ((src[i] & 0x8000u) ? -1.f : 1.f)
                : std::bit_cast<float>(sign | (std::uint32_t)(src[i] & 0x8000u) << 16
                    | ((ex == 31 ? 0xffu : ex + 112u) << 23) | (fr << 13));
        }
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
        bytes += (size_t)L * 2 * KVH * ctx * dh * 4;   // KV caches
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
                bytes += (size_t)L * 2 * KVH * ctx * dh * 4;
                bytes += (size_t)V * 4 * 2;
                bytes += (size_t)(H * dh + 2 * F + 4 * 2048) * 4;
                const double budget_gb = env_double(getenv("TM_GPU_BUDGET_GB"), 5.0);
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
        // TM_GPU_BUDGET_GB overrides the conservative default.
        {
            const double budget_gb = env_double(getenv("TM_GPU_BUDGET_GB"), 5.0);
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
                uint64_t w = 0;
                gpu_weights_used_ = true; // also latch partial/failed registration
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
        const int id = tm_metal_tok_wbuf(
            q4_.at("lm_head.weight"),
            (size_t)qshape_.at("lm_head.weight").first
                * qshape_.at("lm_head.weight").second / 32 * 18,
            held_map_->base, held_map_->size, file_backed ? 1 : 0, w);
        gpu_w_["lm_head.weight"] = {id, w};
        if (id < 0) throw std::runtime_error("metal: lm_head registration failed");
        gpu_w_ready_ = true;
        if (getenv("TM_DEBUG_GPUW")) {
            unsigned char gpub[32], gpub2[32], host[32];
            tm_metal_tok_download(id, w, gpub, 32);
            memcpy(host, (const uint8_t*)q4_.at("lm_head.weight"), 32);
            const int probe_slot = 500;
            tm_metal_tok_probe_copy(id, w, probe_slot, 32);
            tm_metal_tok_download(probe_slot, 0, gpub2, 32);
            const auto& l0q = gpu_w_.at(
                "model.layers.0.self_attn.q_proj.weight");
            unsigned char l0g[32], l0h[32], l0gpu[32];
            tm_metal_tok_download(l0q.first, l0q.second, l0g, 32);
            memcpy(l0h, (const uint8_t*)q4_.at(
                "model.layers.0.self_attn.q_proj.weight"), 32);
            tm_metal_tok_probe_copy(l0q.first, l0q.second, 501, 32);
            tm_metal_tok_download(501, 0, l0gpu, 32);
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

    bool quantized() const { return held_map_ != nullptr; }

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
        if (next_d <= 0 || next_l <= 0 || next_h <= 0 || next_kvh <= 0 ||
            next_f <= 0 || next_v <= 0 || next_d % next_h ||
            next_h % next_kvh || next_f % tmq::kBlock)
            throw std::runtime_error("llama: incompatible config " + path);
        D = next_d; L = next_l; H = next_h; KVH = next_kvh;
        F = next_f; V = next_v; eps = next_eps; theta = next_theta;
        dh = D / H;
        REP_ = H / KVH;
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
    static void quant_x(const float* x, int n_in, XQ& out) {
        if (n_in < 0 || n_in % tmq::kBlock != 0 || (n_in != 0 && x == nullptr))
            throw std::invalid_argument("quant_x requires complete activation blocks");
        const bool neon = quant_x_neon_enabled();
        const int nb = n_in / tmq::kBlock;
        out.q.resize((std::size_t)n_in);
        out.d.resize((std::size_t)n_in);
        out.s.resize((std::size_t)nb);
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
            int8_t* qe = out.q.data() + (std::size_t)b * tmq::kBlock;
            int8_t* qd = out.d.data() + (std::size_t)b * tmq::kBlock;
            if (amax > 0.f && (d == 0.f || !std::isfinite(inv))) {
                // Tiny finite blocks can underflow the scale or overflow its
                // reciprocal. Round the stored scale upward to cover amax,
                // then divide in double; never convert infinity to an integer.
                if ((double)d * 127. < (double)amax)
                    d = std::nextafter(d, std::numeric_limits<float>::infinity());
                out.s[b] = d;
                for (int i = 0; i < tmq::kBlock; ++i) {
                    qe[i] = (int8_t)std::lrint((double)xb[i] / (double)d);
                    qd[i / 2 + (i % 2) * 16] = qe[i];
                }
                continue;
            }
            out.s[b] = d;
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
#endif

    mutable XQ xq_;             // sdot decode: quantized x (amortized/pass)

    void gemv_w(const std::string& name, const float* x, int n_in,
                int n_out, float* y) const {
        const bool q8 = q8_.count(name) != 0;
#ifdef __ARM_NEON
        static const bool sdot =
            [](const char* e) { return !e || e[0] != '0'; }(
                getenv("TM_SDOT"));
        if (sdot) {
            quant_x(x, n_in, xq_);
            GemvJob j;
            j.x = x; j.y = y; j.n_in = n_in; j.n_out = n_out;
            j.xq = xq_.q.data(); j.xs = xq_.s.data(); j.xd = xq_.d.data();
            if (q8) {
                j.w = q8_.at(name);
                j.kern = [](const void* w, const float*,
                            const int8_t* xqp, const float* xsp,
                            const int8_t* xdp, int ni, int o0, int o1,
                            float* yv) {
                    qgemv_sdot_rows((const tmq::BlockQ8_0*)w, xqp, xdp, xsp,
                                    ni, o0, o1, yv);
                };
            } else {
                j.w = q4_.at(name);
                j.kern = [](const void* w, const float*,
                            const int8_t* xqp, const float* xsp,
                            const int8_t* xdp, int ni, int o0, int o1,
                            float* yv) {
                    qgemv_sdot_rows((const tmq::BlockQ4_0*)w, xqp, xdp, xsp,
                                    ni, o0, o1, yv);
                };
            }
            if (const char* ser = getenv("TM_SERIAL_GEMV");
                ser && ser[0] == '1')
                j.kern(j.w, j.x, j.xq, j.xs, j.xd, j.n_in, 0, j.n_out, j.y);
            else
                gemv_pool().dispatch(j);
            return;
        }
#endif
        const char* ser = getenv("TM_SERIAL_GEMV");
        if (ser && ser[0] == '1') {
            if (q8) qgemv_rows(q8_.at(name), x, n_in, 0, n_out, y);
            else    qgemv_rows(q4_.at(name), x, n_in, 0, n_out, y);
            return;
        }
        GemvJob j;
        j.x = x; j.y = y; j.n_in = n_in; j.n_out = n_out;
        if (q8) {
            j.w = q8_.at(name);
            j.kern = [](const void* w, const float* xv, const int8_t*,
                        const float*, const int8_t*, int ni,
                        int o0, int o1, float* yv) {
                qgemv_rows((const tmq::BlockQ8_0*)w, xv, ni, o0, o1, yv);
            };
        } else {
            j.w = q4_.at(name);
            j.kern = [](const void* w, const float* xv, const int8_t*,
                        const float*, const int8_t*, int ni,
                        int o0, int o1, float* yv) {
                qgemv_rows((const tmq::BlockQ4_0*)w, xv, ni, o0, o1, yv);
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
        job.w = projections.data(); job.x = x;
        job.n_in = n_in; job.n_out = total;
#ifdef __ARM_NEON
        static const bool sdot = [](const char* value) { return !value || value[0] != '0'; }(
            std::getenv("TM_SDOT"));
        if (sdot) {
            quant_x(x, n_in, xq_);
            job.xq = xq_.q.data(); job.xd = xq_.d.data(); job.xs = xq_.s.data();
        }
#endif
        job.kern = [](const void* weights, const float* input, const int8_t* xq,
                      const float* xs, const int8_t* xd, int width,
                      int first, int last, float*) {
            const auto* parts = static_cast<const Projection*>(weights);
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
        const char* serial = std::getenv("TM_SERIAL_GEMV");
        if (serial && serial[0] == '1')
            job.kern(job.w, job.x, job.xq, job.xs, job.xd, n_in, 0, total, nullptr);
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
        std::vector<tm_llama_gpu_layer> ls((std::size_t)L);
        for (int l = 0; l < L; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            const auto q4 = [&](const char* k) -> const void* { auto it = q4_.find(p + k); return it == q4_.end() ? nullptr : (const void*)it->second; };
            const auto wf = [&](const char* k) -> const float* { auto it = w.find(p + k); return it == w.end() ? nullptr : it->second.data(); };
            tm_llama_gpu_layer& g = ls[(std::size_t)l];
            g.q = q4("self_attn.q_proj.weight"); g.k = q4("self_attn.k_proj.weight"); g.v = q4("self_attn.v_proj.weight");
            g.o = q4("self_attn.o_proj.weight"); g.gate = q4("mlp.gate_proj.weight"); g.up = q4("mlp.up_proj.weight");
            g.down = q4("mlp.down_proj.weight");
            g.rms1 = wf("input_layernorm.weight"); g.rms2 = wf("post_attention_layernorm.weight");
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
    static void attn_group_kern(const void* w, const float*, const int8_t*, const float*,
                                const int8_t*, int, int o0, int o1, float*) {
        const auto& a = *static_cast<const AttnJob*>(w);
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
        int T, pos0, QD, D, REP, dh, ctx, KVH, B, nblocks; float scale;
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
            std::copy_n(a.q + (std::size_t)(t0 + t) * a.QD + (std::size_t)kv * a.REP * a.dh,
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
                        a.ao + (std::size_t)(t0 + t) * a.D + (std::size_t)kv * a.REP * a.dh);
    }
    static void attn_block_kern(const void* w, const float*, const int8_t*, const float*,
                                const int8_t*, int, int o0, int o1, float*) {
        const auto& a = *static_cast<const AttnBlockJob*>(w);
        for (int u = o0; u < o1; ++u) {
            const int rev = a.KVH * a.nblocks - 1 - u;       // largest blocks first
            attn_block_unit(a, rev / a.nblocks, rev % a.nblocks);
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
                const auto* r = (const tmq::BlockQ8_0*)blocks
                              + (std::uint64_t)o * nb;
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
                const auto* r = (const tmq::BlockQ4_0*)blocks
                              + (std::uint64_t)o * nb;
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
    bool gemm_w(const std::string& name, const float* A, int M, int K,
                int N, float* C) {
        if (!amx_prefill_ || K % tmq::kBlock) return false;
        const bool q8 = q8_.count(name) != 0;
        if (!q8 && !q4_.count(name)) {
            // fp32 weights (equal-precision control lane): direct sgemm,
            // no dequant scratch, no batched_q4_parts split copy
            // (TENSORMARK_FP32_PREFILL_SPEC P2.1, 2026-09-12).
            // TM_GEMM_F32=0 restores the per-token gemv loop (A/B gate).
            if (!w.count(name)) return false;
            static const bool on = [] {
                const char* e = std::getenv("TM_GEMM_F32");
                return !(e && e[0] == '0');
            }();
            if (!on) return false;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                        1.f, A, K, w.at(name).data(), K, 0.f, C, N);
            return true;
        }
        const void* blocks = q8 ? (const void*)q8_.at(name)
                                : (const void*)q4_.at(name);
        if (deq_.size() < (std::size_t)N * K) deq_.resize((std::size_t)N * K);
        // TM_POOL_DEQ=1 dispatches the dequant on the GemvPool (row-parallel).
        // Post ld-fix it passes the previously-fatal shapes repeatedly (the
        // old corruption was the batched-attention lda/ldc OOB, not the
        // pool — see repro_poolbug.cpp and the M5 spec); serial stays the
        // default because it measured faster (271 vs 257 t/s A/B).
        if (getenv("TM_POOL_DEQ") && getenv("TM_POOL_DEQ")[0] == '1') {
            GemvJob j;
            j.w = blocks;
            j.x = nullptr;
            j.y = deq_.data();
            j.n_in = K;
            j.n_out = N;
            j.kern = q8
                ? [](const void* w, const float*, const int8_t*, const float*,
                     const int8_t*, int ni, int o0, int o1,
                     float* y) { dequant_rows(w, true, ni, o0, o1, y); }
                : [](const void* w, const float*, const int8_t*, const float*,
                     const int8_t*, int ni, int o0, int o1,
                     float* y) { dequant_rows(w, false, ni, o0, o1, y); };
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
            const auto* r = (const tmq::BlockQ8_0*)blocks + (std::uint64_t)row * nb;
            for (std::uint64_t b = 0; b < nb; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                for (int i = 0; i < tmq::kBlock; ++i)
                    dst[b * tmq::kBlock + i] = (float)r[b].qs[i] * d;
            }
        } else {
            const auto* r = (const tmq::BlockQ4_0*)blocks + (std::uint64_t)row * nb;
            for (std::uint64_t b = 0; b < nb; ++b) {
                const float d = tmq::fp16_to_fp32(r[b].d_fp16);
                for (int j = 0; j < tmq::kBlock / 2; ++j) {
                    dst[b * tmq::kBlock + 2 * j] = (float)((r[b].qs[j] & 0xF) - 8) * d;
                    dst[b * tmq::kBlock + 2 * j + 1] = (float)((r[b].qs[j] >> 4) - 8) * d;
                }
            }
        }
    }
    void row_w(const std::string& name, int row, float* dst) const {
        if (getenv("TM_DEBUG_ROWW"))
            fprintf(stderr, "[row_w] %s (in qshape: %d)\n", name.c_str(),
                    (int)qshape_.count(name));
        const int n_in = qshape_.at(name).second;
        dequant_row(q8_.count(name) ? (const void*)q8_.at(name)
                                    : (const void*)q4_.at(name),
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
                if (ti.shape[1] % tmq::kBlock)
                    throw std::runtime_error("llama: partial quantized row: " + name);
                if (q8) next.q8_[name] = (const tmq::BlockQ8_0*)ti.blocks;
                else    next.q4_[name] = (const tmq::BlockQ4_0*)ti.blocks;
                next.qshape_[name] = {(int)ti.shape[0], (int)ti.shape[1]};
            } else if (ti.shape.size() == 1) {
                // 1-D (norms): dequantize straight into w; the block
                // pointer comes from the mapped tensor, not the maps.
                const int n = (int)ti.shape[0];
                next.w.emplace(name, Tensor(std::vector<int>{n}));
                dequant_row(ti.blocks, q8, n, 0, next.w.at(name).data());
            }
        }
        next.validate_weights();
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
        next.validate_weights();
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
    }
    void validate_config() const {
        if (L <= 0 || D <= 0 || H <= 0 || KVH <= 0 || ctx <= 0 || dh <= 0 ||
            V <= 0 || F <= 0 || D % H || H % KVH || REP_ != H / KVH ||
            dh != D / H || dh % 2 ||
            F % tmq::kBlock || !std::isfinite(eps) || eps <= 0.f ||
            !std::isfinite(theta) || theta <= 0.f)
            throw std::runtime_error("llama: invalid model dimensions");
    }
    void validate_weights() const {
        validate_config();
        // Format-level bounds only establish that declared tensors fit the
        // file. Inference kernels use architectural dimensions, so validate
        // every required shape before allocating KV or publishing any views.
        const auto require_shape = [&](const std::string& name,
                                       const std::vector<int>& expected) {
            if (quantized() && expected.size() == 2) {
                const auto found = qshape_.find(name);
                if (found != qshape_.end() &&
                    found->second == std::pair{expected[0], expected[1]})
                    return;
            } else {
                const auto found = w.find(name);
                if (found != w.end() && found->second.shape() == expected)
                    return;
            }
            throw std::runtime_error("llama: missing or incompatible required tensor: " + name);
        };
        // validate_config ensures KVH <= H and dh == D/H, so this product fits D.
        const int kv_width = KVH * dh;
        require_shape("model.embed_tokens.weight", {V, D});
        require_shape("lm_head.weight", {V, D});
        require_shape("model.norm.weight", {D});
        for (int layer = 0; layer < L; ++layer) {
            const std::string prefix = "model.layers." + std::to_string(layer) + ".";
            require_shape(prefix + "input_layernorm.weight", {D});
            require_shape(prefix + "post_attention_layernorm.weight", {D});
            require_shape(prefix + "self_attn.q_proj.weight", {D, D});
            require_shape(prefix + "self_attn.k_proj.weight", {kv_width, D});
            require_shape(prefix + "self_attn.v_proj.weight", {kv_width, D});
            require_shape(prefix + "self_attn.o_proj.weight", {D, D});
            require_shape(prefix + "mlp.gate_proj.weight", {F, D});
            require_shape(prefix + "mlp.up_proj.weight", {F, D});
            require_shape(prefix + "mlp.down_proj.weight", {D, F});
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
        kc.assign(L, PageVec(slots));
        vc.assign(L, PageVec(slots));
    }
    void commit_loaded(Llama& next) noexcept {
        copy_config_from(next);
        w.swap(next.w); q8_.swap(next.q8_); q4_.swap(next.q4_);
        qshape_.swap(next.qshape_); std::swap(held_map_, next.held_map_);
        model_dir_.swap(next.model_dir_);   // fp16 safetensors live next to the .tmq
        kc.swap(next.kc); vc.swap(next.vc);
        gpu_kv_uploaded_ = false; gpu_kv_valid_ = 0; gpu_from_ = 0;   // GPU slots hold the old model's rows
        reset_cache();
        gpu_checked_ = false; gpu_ok_ = false;
        gpu_w_.clear(); gpu_w_ready_ = false; gpu_weights_used_ = false;
        gpu_shared_wbuf_ = false;
        resolve_decode_gpu_auto();
        gpu_stack_setup();
    }

public:

    void reset_cache() {
        seen = 0;
        gpu_kv_uploaded_ = false;
        gpu_from_ = 0;
        gpu_kv_valid_ = 0;
    }

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

    // RoPE on one head (dh floats), absolute position p.
    void rope_head(float* r, int p) const {
        for (int j = 0; j < dh / 2; ++j) {
            const float inv = std::pow(theta, -2.f * j / (float)dh);
            const float c = std::cos((float)p * inv), s = std::sin((float)p * inv);
            const float a = r[j], b = r[j + dh / 2];
            r[j] = a * c - b * s;
            r[j + dh / 2] = b * c + a * s;
        }
    }

    // Each Q/K head at a position uses the same rotation. Share the expensive
    // transcendental evaluations without changing each pair's arithmetic.
    // No cached coefficients: manual theta/dh changes cannot leave stale state.
    void rope_qk(float* q, float* k, int tokens, int pos0) const {
        // Scalar and vector negated products can round differently in directed
        // modes. Preserve the established per-head implementation in that case.
        if (std::fegetround() != FE_TONEAREST) {
            for (int t = 0; t < tokens; ++t) {
                for (int hd = 0; hd < H; ++hd)
                    rope_head(q + ((std::size_t)t * H + hd) * dh, pos0 + t);
                for (int kv = 0; kv < KVH; ++kv)
                    rope_head(k + ((std::size_t)t * KVH + kv) * dh, pos0 + t);
            }
            return;
        }
        const int half = dh / 2;
        std::vector<float> coefficients((std::size_t)half * 2);
        float* cosines = coefficients.data();
        float* sines = cosines + half;
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < half; ++j) {
                const float inv = std::pow(theta, -2.f * j / (float)dh);
                // Explicitly use the paired libm operation emitted for
                // rope_head. Instrumented stores can otherwise prevent Clang
                // from combining separate sin/cos calls, changing their bits.
                ::__sincosf((float)(pos0 + t) * inv, &sines[j], &cosines[j]);
            }
            const auto rotate = [&](float* r) {
                for (int j = 0; j < half; ++j) {
                    const float c = cosines[j], s = sines[j];
                    const float a = r[j], b = r[j + half];
                    r[j] = a * c - b * s;
                    r[j + half] = b * c + a * s;
                }
            };
            float* qt = q + (std::size_t)t * H * dh;
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
        const auto gw = [&](const std::string& n)
            -> const std::pair<int, uint64_t>& {
            return gpu_f16_ ? gpu_h_.at(n) : gpu_w_.at(n);
        };
        const auto gmv = [&](int xslot, const std::string& n, int yslot,
                             size_t yoff, unsigned N, unsigned K) {
            const auto& g = gw(n);
            if (gpu_f16_) tm_metal_tok_gemv_h(xslot, g.first, g.second,
                                              yslot, yoff, N, K);
            else tm_metal_tok_gemv(xslot, g.first, g.second, yslot, yoff, N, K);
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
            const void* qkv_w[3] = {
                q4_.at(pre + "self_attn.q_proj.weight"),
                q4_.at(pre + "self_attn.k_proj.weight"),
                q4_.at(pre + "self_attn.v_proj.weight")};
            unsigned qkv_n[3] = {(unsigned)QDg, (unsigned)KVDg,
                                 (unsigned)KVDg};
            tm_metal_tok_rmsnorm(SLOT_X,
                w.at(pre + "input_layernorm.weight").data(), SLOT_H,
                D, eps);
            const auto& gq = gw(pre + "self_attn.q_proj.weight");
            const auto& gk = gw(pre + "self_attn.k_proj.weight");
            const auto& gv = gw(pre + "self_attn.v_proj.weight");
            if (!gemv_fuse || gpu_f16_ || !gpu_shared_wbuf_) {
                gmv(SLOT_H, pre + "self_attn.q_proj.weight", SLOT_QKV, 0, QDg, D);
                gmv(SLOT_H, pre + "self_attn.k_proj.weight", SLOT_QKV, (size_t)QDg * 4, KVDg, D);
                gmv(SLOT_H, pre + "self_attn.v_proj.weight", SLOT_QKV, (size_t)(QDg + KVDg) * 4, KVDg, D);
            } else {
                // Fused q|k|v: one dispatch over QD+2*KVD rows, three weight
                // segments at their file offsets. Segment boundaries are 2-aligned
                // (all dims are 32-multiples); identical per-row order.
                const uint64_t qkv_woff[3] = {gq.second, gk.second, gv.second};
                const unsigned qkv_rows[3] = {(unsigned)QDg, (unsigned)KVDg, (unsigned)KVDg};
                tm_metal_tok_gemv_seg(SLOT_H, gq.first, SLOT_QKV, 0,
                                      (unsigned)(QDg + 2 * KVDg), D,
                                      qkv_woff, qkv_rows, 3);
            }
            tm_metal_tok_rope_q(SLOT_QKV, H, dh, pos, theta);
            tm_metal_tok_rope_k_cache(SLOT_QKV, (size_t)QDg * 4,
                                      SLOT_KC + l, KVH, dh, ctx, pos, theta);
            tm_metal_tok_v_cache(SLOT_QKV, (size_t)(QDg + KVDg) * 4,
                                 SLOT_VC + l, KVH, dh, ctx, pos);
            tm_metal_tok_attn(SLOT_QKV, SLOT_KC + l, SLOT_VC + l,
                              SLOT_PROBS, SLOT_AO, QDg, H, KVH, dh, ctx,
                              pos + 1, scale, REP_);
            const void* o_w[1] = {q4_.at(pre + "self_attn.o_proj.weight")};
            unsigned o_n[1] = {(unsigned)D};
            const auto& go = gw(pre + "self_attn.o_proj.weight");
            gmv(SLOT_AO, pre + "self_attn.o_proj.weight", SLOT_H2, 0, D, D);
            tm_metal_tok_add(SLOT_X, SLOT_H2, D);
            tm_metal_tok_rmsnorm(SLOT_X,
                w.at(pre + "post_attention_layernorm.weight").data(),
                SLOT_H2S, D, eps);
            const void* gu_w[2] = {q4_.at(pre + "mlp.gate_proj.weight"),
                                   q4_.at(pre + "mlp.up_proj.weight")};
            unsigned gu_n[2] = {(unsigned)F, (unsigned)F};
            const auto& gg = gw(pre + "mlp.gate_proj.weight");
            const auto& gu = gw(pre + "mlp.up_proj.weight");
            if (!gemv_fuse || gpu_f16_ || !gpu_shared_wbuf_) {
                gmv(SLOT_H2S, pre + "mlp.gate_proj.weight", SLOT_GU, 0, F, D);
                gmv(SLOT_H2S, pre + "mlp.up_proj.weight", SLOT_GU, (size_t)F * 4, F, D);
            } else {
                const uint64_t gu_woff[2] = {gg.second, gu.second};
                const unsigned gu_rows[2] = {(unsigned)F, (unsigned)F};
                tm_metal_tok_gemv_seg(SLOT_H2S, gg.first, SLOT_GU, 0,
                                      (unsigned)(2 * F), D, gu_woff, gu_rows, 2);
            }
            tm_metal_tok_silu_mul(SLOT_GU, 0, SLOT_GU, (size_t)F * 4, F);
            const void* d_w[1] = {q4_.at(pre + "mlp.down_proj.weight")};
            unsigned d_n[1] = {(unsigned)D};
            const auto& gd = gw(pre + "mlp.down_proj.weight");
            gmv(SLOT_GU, pre + "mlp.down_proj.weight", SLOT_DN, 0, D, F);
            tm_metal_tok_add(SLOT_X, SLOT_DN, D);
            if (decode_layer_cb_) {
                // Per-layer commit: end this layer's buffer, start the next.
                if (!tm_metal_tok_flush())
                    throw std::runtime_error("metal: per-layer command failed");
                // The final norm + lm_head still encode after the last
                // layer, so re-begin unconditionally.
                if (!tm_metal_tok_begin()) {
                    std::fprintf(stderr, "[llama] gpu layer begin failed\n");
                    std::abort();
                }
            }
        }
        tm_metal_tok_rmsnorm(SLOT_X, w.at("model.norm.weight").data(),
                             SLOT_XF, D, eps);
        const void* lm_w[1] = {q4_.at("lm_head.weight")};
        unsigned lm_n[1] = {(unsigned)V};
        const auto& glm_ = gw("lm_head.weight");
        gmv(SLOT_XF, "lm_head.weight", SLOT_LOGITS, 0, V, D);
#else
        (void)pos;
#endif
    }
    std::vector<float>& forward_gpu_one(int tok) {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        gpu_token_prologue();
        std::vector<float> x0(D);
        if (gpu_f16_) row_h("model.embed_tokens.weight", tok, x0.data());
        else row_w("model.embed_tokens.weight", tok, x0.data());
        tm_metal_tok_upload(SLOT_X, 0, x0.data(), (size_t)D * 4);
        if (!tm_metal_tok_begin()) {
            // sticky-GPU mode: the GPU KV cache is authoritative, so a
            // failed begin cannot fall back silently — abort loudly.
            std::fprintf(stderr, "[llama] gpu token begin failed\n");
            std::abort();
        }
        encode_gpu_token(seen);
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
    // (gpu_greedy_chain_ok()); callers fall back to forward() otherwise.
    const std::vector<float>& logits() const { return logits_; }
    bool gpu_greedy_chain_ok() {
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        // Under the auto policy a greedy chain claims the GPU (paired
        // 2026-09-07: ctx 64 chain 84.1 t/s vs CPU decode 79.6, ctx 2000
        // 71.4 vs 59), and the GPU cache is authoritative from then on.
        if (decode_gpu_auto_ && !decode_gpu_ && decode_gpu_auto_fit_ && quantized()) {
            decode_gpu_ = true;
            if (getenv("TM_DEBUG_POOL")) fprintf(stderr, "[llama] decode auto: greedy chain -> GPU\n");
        }
        return (gpu_f16_ ? gpu_h_.count("model.embed_tokens.weight") != 0
                         : q4_.count("model.embed_tokens.weight") != 0) &&
               decode_gpu_ && seen > 0 && gpu_decode_usable() &&
               tm_metal_tok_end_async != nullptr;
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
        if (!gpu_greedy_chain_ok()) throw std::runtime_error("llama: chain needs the GPU decode path");
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
        } else if (!gpu_w_.count("model.embed_tokens.weight")) {
            const std::string nme = "model.embed_tokens.weight";
            const bool file_backed = held_map_ && held_map_->owned.empty() && held_map_->base != nullptr;
            uint64_t w = 0;
            const int id = tm_metal_tok_wbuf(q4_.at(nme),
                (size_t)qshape_.at(nme).first * qshape_.at(nme).second / 32 * 18,
                held_map_->base, held_map_->size, file_backed ? 1 : 0, w);
            if (id < 0) throw std::runtime_error("metal: embedding registration failed");
            gpu_w_[nme] = {id, w};
        }
        const auto& ge = gpu_f16_ ? gpu_h_.at("model.embed_tokens.weight")
                                  : gpu_w_.at("model.embed_tokens.weight");
        std::vector<unsigned> sentinel((size_t)n, 0xFFFFFFFFu);
        tm_metal_tok_upload(SLOT_TOK, 0, sentinel.data(), sizeof(unsigned) * (size_t)n);
        std::vector<float> x0(D);
        row_w("model.embed_tokens.weight", first, x0.data());
        tm_metal_tok_upload(SLOT_X, 0, x0.data(), (size_t)D * 4);
        const float inv_temp = temperature > 0.0f ? 1.0f / temperature : 0.0f;
        for (int i = 0; i < n; ++i) {
            if (!tm_metal_tok_begin()) { std::fprintf(stderr, "[llama] gpu token begin failed\n"); std::abort(); }
            if (i > 0) {
                if (gpu_f16_)
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
        tm_metal_tok_download(SLOT_TOK, 0, toks.data(), sizeof(unsigned) * (size_t)chain_n_);
        tm_metal_tok_download(SLOT_LOGITS, 0, logits_.data(), sizeof(float) * (size_t)V);
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
    std::vector<float>& forward(const std::vector<int>& ids, int pos0 = -1) {
        tm_thread_qos_once();   // interactive band: an agent waits on every token
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
            static const double ctx_switch = env_double("TM_DECODE_GPU_AUTO_CTX", 384.0);
            if (ctx_switch > 0 && seen >= ctx_switch) {
                decode_gpu_ = true;
                decode_gpu_ctx_switched_ = true;
                if (getenv("TM_DEBUG_POOL"))
                    fprintf(stderr, "[llama] decode auto: context %d >= %.0f -> GPU\n", seen, ctx_switch);
            }
        }
#endif
        if (quantized() && T == 1 && decode_gpu_ && seen > 0 &&
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
                if (getenv("TM_DEBUG_POOL"))
                    fprintf(stderr, "[llama] decode auto: GPU setup failed (%s) -> CPU\n", e.what());
            }
#endif
        }
#if defined(__APPLE__) && defined(TM_HAVE_METAL)
        if (gpu_kv_uploaded_ && T > 1 && T < prefill_loop_t() && decode_gpu_ && quantized()) {
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
                    tm_metal_tok_kv_download(SLOT_KC + l, src, dk,
                        (size_t)(seen - gpu_from_) * dh);
                    tm_metal_tok_kv_download(SLOT_VC + l, src, dv,
                        (size_t)(seen - gpu_from_) * dh);
                }
            gpu_kv_uploaded_ = false;
            gpu_kv_valid_ = seen;   // GPU rows [0, seen) still equal the CPU rows
        }
#endif
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
        std::vector<float> x(xbuf_.begin(), xbuf_.begin() + (std::size_t)T * D);
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
                            block(l, xc, cpu_rows, base + gpu_rows);
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
            llama_prof().mark(LlamaProf::GPU_STACK);
        }
#endif
        if (!gpu_stack_done)
            for (int l = 0; l < L; ++l) block(l, x, T, base);
        seen += T;
        if (xf_.size() < (std::size_t)D) xf_.resize(D);
        rms(x.data() + (std::size_t)(T - 1) * D, xf_.data(),
            w.at("model.norm.weight").data(), D, eps);
        if (logits_.size() != (std::size_t)V) logits_.resize(V);
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
        return logits_;
    }

    void block(int l, std::vector<float>& x, int T, int pos0) {
        const auto W = [&](const std::string& k) -> const float* {
            return w.at("model.layers." + std::to_string(l) + "." + k).data();
        };
        const int QD = H * dh, KVD = KVH * dh;
        LlamaProf& P = llama_prof();
        P.start();
        if (h_.size() < (std::size_t)T * D) h_.resize((std::size_t)T * D);
        for (int t = 0; t < T; ++t)
            rms(x.data() + (std::size_t)t * D, h_.data() + (std::size_t)t * D,
                W("input_layernorm.weight"), D, eps);
        P.mark(LlamaProf::RMS1);

        // q, k, v projections
        if (q_.size() < (std::size_t)T * QD) q_.resize((std::size_t)T * QD);
        if (k_.size() < (std::size_t)T * KVD) k_.resize((std::size_t)T * KVD);
        if (v_.size() < (std::size_t)T * KVD) v_.resize((std::size_t)T * KVD);
        if (T == 1) {
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
                } else if (grouped_gemv_enabled()) {
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
                P.mark(LlamaProf::QKV);
                for (int t = 0; t < T; ++t) {
                    const float* src = qkvv_.data() + (std::size_t)t * qkv_tot;
                    std::copy_n(src, QD, q_.data() + (std::size_t)t * QD);
                    std::copy_n(src + QD, KVD, k_.data() + (std::size_t)t * KVD);
                    std::copy_n(src + QD + KVD, KVD,
                                v_.data() + (std::size_t)t * KVD);
                }
            } else {
                for (int t = 0; t < T; ++t) {
                    gemv_w(qp, h_.data() + (std::size_t)t * D, D, QD,
                           q_.data() + (std::size_t)t * QD);
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
        P.mark(LlamaProf::QKV_SPLIT);
        rope_qk(q_.data(), k_.data(), T, pos0);
        P.mark(LlamaProf::ROPE);

        // append K,V rows to the per-KV-head cache
        auto& K = kc[l]; auto& Vv = vc[l];
        // Rows land at pos0, not `seen`: they agree on the normal path, but a
        // hybrid prefill runs this on the TAIL rows (pos0 = seen + gpu_rows)
        // while `seen` still points at the start of the whole prefill.
        const int cached = pos0;
        for (int t = 0; t < T; ++t)
            for (int kv = 0; kv < KVH; ++kv) {
                std::copy_n(k_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh, dh,
                            K.begin() + (std::size_t)kv * ctx * dh
                                      + (std::size_t)(cached + t) * dh);
                std::copy_n(v_.data() + (std::size_t)t * KVD + (std::size_t)kv * dh, dh,
                            Vv.begin() + (std::size_t)kv * ctx * dh
                                       + (std::size_t)(cached + t) * dh);
            }

        P.mark(LlamaProf::KV_APPEND);
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
        if (ao_.size() < (std::size_t)T * D) ao_.resize((std::size_t)T * D);
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
        if (attn_naive_ || (T == 1 && !attn_decode_batched_)) {
        for (int t = 0; t < T; ++t) {
            const int allow = pos0 + t + 1;
            for (int hd = 0; hd < H; ++hd) {
                const int kv = hd / REP_;             // GQA mapping
                const float* qh = q_.data() + (std::size_t)t * QD + (std::size_t)hd * dh;
                const float* Kh = &K[(std::size_t)kv * ctx * dh];
                const float* Vh = &Vv[(std::size_t)kv * ctx * dh];
                float m = -1e30f;
                for (int s = 0; s < allow; ++s) {
                    att_[s] = cblas_sdot(dh, qh, 1, &Kh[(std::size_t)s * dh], 1) * scale;
                    m = std::max(m, att_[s]);
                }
                float z = 0.f;
                for (int s = 0; s < allow; ++s) { att_[s] = std::exp(att_[s] - m); z += att_[s]; }
                float* acc = ao_.data() + (std::size_t)t * D + (std::size_t)hd * dh;
                std::fill(acc, acc + dh, 0.f);
                for (int s = 0; s < allow; ++s)
                    cblas_saxpy(dh, att_[s] / z, &Vh[(std::size_t)s * dh], 1, acc, 1);
            }
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
            j.w = &a;
            j.n_in = 0; j.n_out = KVH; j.chunk = 1;
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
                           T, pos0, QD, D, REP_, dh, ctx, KVH, B, nblocks, scale};
            if (attn_prefill_pool_ && gemv_pool().nthreads > 1) {
                GemvJob j;
                j.kern = &attn_block_kern; j.w = &a;
                j.n_in = 0; j.n_out = KVH * nblocks; j.chunk = 1;
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
                    attn_group(q_.data() + (std::size_t)t * QD + (std::size_t)kv * REP_ * dh,
                               &K[(std::size_t)kv * ctx * dh], &Vv[(std::size_t)kv * ctx * dh],
                               S, ao_.data() + (std::size_t)t * D + (std::size_t)kv * REP_ * dh,
                               REP_, dh, allow, scale, attn_vsoftmax_);
                }
            }
        }
        P.mark(LlamaProf::ATTN);
        if (h2_.size() < (std::size_t)T * D) h2_.resize((std::size_t)T * D);
        const auto oname = "model.layers." + std::to_string(l) + ".self_attn.o_proj.weight";
        if (quantized()) {
            if (T == 1 && decode_gpu_ &&
                batched_q4(ao_.data(), oname, h2_.data(), 1, D, D)) {
                // GPU decode o_proj
            } else if (T == 1)
                gemv_w(oname, ao_.data(), D, D, h2_.data());
            else if (amx_ok(T) && !gpu_prefill_ready(T, D) &&
                     gemm_w(oname, ao_.data(), T, D, D, h2_.data())) {
                // AMX prefill
            } else if (!batched_q4(ao_.data(), oname, h2_.data(), T, D, D))
                for (int t = 0; t < T; ++t)
                    gemv_w(oname, ao_.data() + (std::size_t)t * D, D, D,
                           h2_.data() + (std::size_t)t * D);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, D, D, 1.f,
                        ao_.data(), D, W("self_attn.o_proj.weight"), D, 0.f, h2_.data(), D);
        }
        P.mark(LlamaProf::OPROJ);
        // plain loop on purpose: DRAM-bound (T*D >= 2^21 floats) — vDSP_vadd measured slower (see autograd.h tm_elem_stream)
        for (int i = 0; i < T * D; ++i) x[i] += h2_[i];
        P.mark(LlamaProf::RES1);

        // MLP: SwiGLU — down( silu(gate(h2)) * up(h2) )
        if (h2s_.size() < (std::size_t)T * D) h2s_.resize((std::size_t)T * D);
        for (int t = 0; t < T; ++t)
            rms(x.data() + (std::size_t)t * D, h2s_.data() + (std::size_t)t * D,
                W("post_attention_layernorm.weight"), D, eps);
        P.mark(LlamaProf::RMS2);
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
            } else if (T == 1 && grouped_gemv_enabled()) {
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
        P.mark(LlamaProf::GATE_UP);
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
        P.mark(LlamaProf::SILU);
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
        P.mark(LlamaProf::DOWN);
        for (int i = 0; i < T * D; ++i) x[i] += dn_[i];
        P.mark(LlamaProf::RES2);
        (void)l;
    }

    std::vector<float> h2s_;
};

}  // namespace tmllama
