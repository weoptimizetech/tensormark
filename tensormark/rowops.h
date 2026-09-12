// tensormark/rowops.h — NEON row reductions shared by the autograd engine and
// the inference runtimes.
#pragma once
#include <cstddef>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace tmg {
// Row reductions with 8 independent lanes: a plain `s += x[i]` chain is a
// serial dependency the compiler must keep under strict fp (128 adds x ~4
// cycles per (4096,128) row), which made layernorm's mean/var/backward sums
// the op's dominant cost. Summation order differs from the serial loop by
// ~1e-7 relative; gates are tolerance-based.
inline float tm_row_sum(const float* __restrict x, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = vaddq_f32(a0, vld1q_f32(x + i));
        a1 = vaddq_f32(a1, vld1q_f32(x + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; ++i) s += x[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += x[i];
    return s;
#endif
}
inline float tm_row_sumsq_dev(const float* __restrict x, float m, int n) {  // sum (x-m)^2
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float32x4_t mv = vdupq_n_f32(m);
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(x + i), mv);
        const float32x4_t d1 = vsubq_f32(vld1q_f32(x + i + 4), mv);
        a0 = vfmaq_f32(a0, d0, d0);
        a1 = vfmaq_f32(a1, d1, d1);
    }
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; ++i) { const float d = x[i] - m; s += d * d; }
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < n; ++i) { const float d = x[i] - m; s += d * d; }
    return s;
#endif
}
// s1 = sum g*gm, s2 = sum g*gm*(x-m)*rstd (layernorm backward row sums)
inline void tm_row_ln_sums(const float* __restrict g, const float* __restrict gm,
                           const float* __restrict x, float m, float rstd, int n,
                           float& s1, float& s2) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float32x4_t mv = vdupq_n_f32(m), rv = vdupq_n_f32(rstd);
    float32x4_t p0 = vdupq_n_f32(0.f), p1 = vdupq_n_f32(0.f);
    float32x4_t q0 = vdupq_n_f32(0.f), q1 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t dn0 = vmulq_f32(vld1q_f32(g + i), vld1q_f32(gm + i));
        const float32x4_t dn1 = vmulq_f32(vld1q_f32(g + i + 4), vld1q_f32(gm + i + 4));
        const float32x4_t xh0 = vmulq_f32(vsubq_f32(vld1q_f32(x + i), mv), rv);
        const float32x4_t xh1 = vmulq_f32(vsubq_f32(vld1q_f32(x + i + 4), mv), rv);
        p0 = vaddq_f32(p0, dn0); p1 = vaddq_f32(p1, dn1);
        q0 = vfmaq_f32(q0, dn0, xh0); q1 = vfmaq_f32(q1, dn1, xh1);
    }
    float a = vaddvq_f32(vaddq_f32(p0, p1)), b = vaddvq_f32(vaddq_f32(q0, q1));
    for (; i < n; ++i) { const float dn = g[i] * gm[i]; a += dn; b += dn * (x[i] - m) * rstd; }
    s1 = a; s2 = b;
#else
    float a = 0.f, b = 0.f;
    for (int i = 0; i < n; ++i) { const float dn = g[i] * gm[i]; a += dn; b += dn * (x[i] - m) * rstd; }
    s1 = a; s2 = b;
#endif
}
}  // namespace tmg
