// tensormark/softmax_row.h — one row softmax, fused into three sweeps.
//
// The row softmax a decoder runs per (head, query row) was five separate vDSP
// passes over the row (maxv, vsadd, vvexpf, sve, vsmul): 5 reads and 3 writes
// per element. Two of those sweeps only exist because the primitives cannot
// see each other — the max subtraction is folded into exp, and the row sum is
// accumulated by the same loop that writes the exponentials. What stays is
// max | exp+sum | scale: 3 reads and 2 writes. Measured on the causal tile,
// 1.33x at T=512 and 1.23x at T=2048 on the M1 this was written for.
//
// The exp is a polynomial with an exponent assembled in the integer unit, so
// the fused loop needs no libm call per element. It is defined on inputs <= 0
// (a softmax subtracts the row max), where the relative error against
// std::exp is below 3e-7 — one order under the differences the surrounding
// GEMMs already carry. Non-NEON targets (the Windows port) keep the scalar
// libm loop, where none of this is on the hot path.
#pragma once

#include "tm_blas_host.h"

#include <algorithm>
#include <cmath>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace tmrow {

#if defined(__ARM_NEON)
// exp(x) for x <= 0: n = round(x / ln2), r = x - n*ln2 (two-part ln2 so the
// argument reduction does not lose the low bits), exp(r) by its Taylor series
// to r^6 — |r| <= ln2/2 there, so the truncation term is 3.4e-7 relative —
// and 2^n built by writing the biased exponent.
inline float32x4_t exp_nonpos(float32x4_t x) {
    // Below about -87 the result is 0 in fp32; clamping keeps 2^n a normal
    // number and the polynomial inside its range.
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.4426950408889634f)));
    float32x4_t r = vfmaq_f32(x, n, vdupq_n_f32(-0.6931471824645996f));
    r = vfmaq_f32(r, n, vdupq_n_f32(1.904654299957768e-9f));
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    const int32x4_t e = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
    return vmulq_f32(p, vreinterpretq_f32_s32(vshlq_n_s32(e, 23)));
}
#endif

// Row softmax over [0, len) in place, then zero [len, zero_to) — the causal
// mask's dead columns, which later GEMMs read. `row` has at least `zero_to`
// elements. Nothing here allocates or branches per element.
inline void softmax_row(float* row, int len, int zero_to) {
    float m;
    vDSP_maxv(row, 1, &m, vDSP_Length(len));
    const float negm = -m;
    float z;
#if defined(__ARM_NEON)
    const float32x4_t shift = vdupq_n_f32(negm);
    // Four independent accumulators: one chained vaddq would cap the loop at
    // its latency (~4 cycles per 4 elements), which is exactly the rate the
    // exp it is meant to beat already reaches.
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        const float32x4_t e0 = exp_nonpos(vaddq_f32(vld1q_f32(row + i), shift));
        const float32x4_t e1 = exp_nonpos(vaddq_f32(vld1q_f32(row + i + 4), shift));
        const float32x4_t e2 = exp_nonpos(vaddq_f32(vld1q_f32(row + i + 8), shift));
        const float32x4_t e3 = exp_nonpos(vaddq_f32(vld1q_f32(row + i + 12), shift));
        vst1q_f32(row + i, e0);
        vst1q_f32(row + i + 4, e1);
        vst1q_f32(row + i + 8, e2);
        vst1q_f32(row + i + 12, e3);
        a0 = vaddq_f32(a0, e0);
        a1 = vaddq_f32(a1, e1);
        a2 = vaddq_f32(a2, e2);
        a3 = vaddq_f32(a3, e3);
    }
    for (; i + 4 <= len; i += 4) {
        const float32x4_t e = exp_nonpos(vaddq_f32(vld1q_f32(row + i), shift));
        vst1q_f32(row + i, e);
        a0 = vaddq_f32(a0, e);
    }
    z = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < len; ++i) {
        const float e = std::exp(row[i] + negm);
        row[i] = e;
        z += e;
    }
#else
    z = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float e = std::exp(row[i] + negm);
        row[i] = e;
        z += e;
    }
#endif
    const float inv = 1.0f / z;
    vDSP_vsmul(row, 1, &inv, row, 1, vDSP_Length(len));
    if (len < zero_to) std::fill(row + len, row + zero_to, 0.0f);
}

}  // namespace tmrow
