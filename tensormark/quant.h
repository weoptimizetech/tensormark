// tensormark/quant.h — M4: llama.cpp-compatible Q8_0 / Q4_0 block formats.
//
// Block layouts (ggml reference semantics, https://github.com/ggml-org/ggml):
//   Q8_0 (34 B): fp16 scale d + 32 int8 q;  w = d * q,
//                d = amax / 127, q = round(w / d) clamped to [-127, 127].
//   Q4_0 (18 B): fp16 scale d + 16 bytes of 4-bit nibbles (low nibble =
//                element 2j, high = 2j+1); w = (n - 8) * d.
//                ggml's quantize_row_q4_0_ref: d = max_signed / -8 where
//                max_signed is the largest-magnitude element WITH SIGN;
//                nibble = clamp(round(w / d) + 8, 0, 15). The asymmetry
//                ([-8d, +7d]) is inherent to Q4_0; quantize and dequant here
//                share the rule, and the PPL harness is the quality judge.
//
// Dequant is the reference for the gate: Q8_0 must be bit-exact vs an
// independent implementation; Q4_0 dequant is exact by construction.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <cmath>
#if defined(__aarch64__) && defined(__clang__)
#include <arm_neon.h>
#endif
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace tmq {

constexpr int kBlock = 32;

// TMQ/GGUF payloads may begin at any byte offset. These are serialized blocks,
// not naturally aligned C++ records; keep both size AND alignment contractual.
#pragma pack(push, 1)
struct BlockQ8_0 {           // 34 bytes
    std::uint16_t d_fp16;    // IEEE 754 half
    std::int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must be 34 bytes");

struct BlockQ4_0 {           // 18 bytes
    std::uint16_t d_fp16;
    std::uint8_t  qs[16];    // low nibble = element 2j, high = 2j+1
};
static_assert(sizeof(BlockQ4_0) == 18, "Q4_0 block must be 18 bytes");
#pragma pack(pop)
static_assert(alignof(BlockQ8_0) == 1 && alignof(BlockQ4_0) == 1,
              "quantized blocks must support unaligned serialized payloads");

// ---- fp16 <-> fp32 (IEEE half; scalar reference — NEON kernels must match
//      these scales bit-for-bit) ----
inline float bits_to_float(std::uint32_t bits) {
    return std::bit_cast<float>(bits);
}

namespace detail {
inline float fp16_to_fp32_scalar(std::uint16_t h) {
    const std::uint32_t sign = (std::uint32_t)(h & 0x8000u) << 16;
    const std::uint32_t exp  = (h & 0x7c00u) >> 10;
    const std::uint32_t man  = h & 0x03ffu;
    if (exp == 0) {
        // subnormal: value = man * 2^-24, exact as a single float multiply
        if (man == 0) return bits_to_float(sign);
        float f = (float)man * 0x1p-24f;
        return bits_to_float(sign | (std::bit_cast<std::uint32_t>(f) & 0x7fffffffu));
    }
    if (exp == 31) return bits_to_float(sign | 0x7f800000u | (man << 13));
    return bits_to_float(sign | ((exp + 112) << 23) | (man << 13));
}
} // namespace detail

inline float fp16_to_fp32(std::uint16_t h) {
#if defined(__aarch64__) && defined(__clang__)
    // A normal half is represented exactly in fp32, under every rounding mode.
    // Keep subnormals/zeros/NaNs/infinities on the bit-preserving scalar path:
    // native conversion can flush half subnormals or quiet NaN payloads under
    // FPCR controls. This avoids changing those established semantics.
    const unsigned exponent = h & 0x7c00u;
    if (exponent != 0 && exponent != 0x7c00u) {
        __fp16 value;
        std::memcpy(&value, &h, sizeof(value));
        return (float)value;
    }
#endif
    return detail::fp16_to_fp32_scalar(h);
}

// A row of halves, under the same contract as fp16_to_fp32: bit-identical to
// the scalar form element-wise, for every encoding, under every FPCR setting.
// Runs of eight all-normal halves — every weight row in practice — convert on
// the vector unit (the exact hardware conversion fp16_to_fp32 already uses for
// normal halves); any group containing a zero/subnormal/inf/NaN half falls back
// to the scalar per-element form, because FPCR.FZ16/AHP change how the vector
// unit handles those encodings and the established semantics are the scalar
// ones. Replaces two divergent row loops (row_h's inline bit gymnastics with
// std::ldexpf, row_w's element-wise fp16_to_fp32): same results, one body.
inline void fp16_to_fp32_row(const std::uint16_t* src, float* dst,
                             std::size_t n) {
    std::size_t i = 0;
#if defined(__aarch64__) && defined(__clang__)
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t h = vld1q_u16(src + i);
        const uint16x8_t ex = vandq_u16(h, vdupq_n_u16(0x7c00));
        const uint16x8_t special =
            vorrq_u16(vceqzq_u16(ex), vceqq_u16(ex, vdupq_n_u16(0x7c00)));
        if (vmaxvq_u16(special) == 0) {
            vst1q_f32(dst + i,
                      vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h))));
            vst1q_f32(dst + i + 4,
                      vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(h))));
        } else {
            for (int j = 0; j < 8; ++j) dst[i + j] = fp16_to_fp32(src[i + j]);
        }
    }
#endif
    for (; i < n; ++i) dst[i] = fp16_to_fp32(src[i]);
}

inline std::uint16_t fp32_to_fp16(float f) {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t exp32 = (x >> 23) & 0xffu;
    const std::uint32_t man = x & 0x007fffffu;
    if (exp32 == 0xffu)                                    // inf / nan
        return (std::uint16_t)(sign | 0x7c00u | (man ? 0x0200u : 0u));
    std::int32_t exp = (std::int32_t)exp32 - 127 + 15;
    if (exp >= 31) return (std::uint16_t)(sign | 0x7c00u); // overflow -> inf
    if (exp <= 0) {                                        // subnormal / zero
        if (exp < -10) return (std::uint16_t)sign;
        const std::uint32_t m = man | 0x00800000u;         // implicit 1
        const int shift = 14 - exp;                        // 14..24
        const std::uint32_t q = m >> shift;
        const std::uint32_t rem = m & ((1u << shift) - 1u);
        const std::uint32_t tie = 1u << (shift - 1);
        const std::uint32_t r = q + ((rem > tie || (rem == tie && (q & 1u))) ? 1u : 0u);
        return (std::uint16_t)(sign | r);                  // r <= 0x3ff here
    }
    // normal: round the 13 truncated bits, nearest-even
    std::uint32_t r = man >> 13;
    const std::uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    std::int32_t exp2 = exp;
    if (r >> 10) { ++exp2; r = 0; }                        // mantissa overflow
    return (std::uint16_t)(sign | ((std::uint32_t)exp2 << 10) | (r & 0x3ffu));
}

inline std::size_t nblocks(std::size_t n) { return n / kBlock; }

namespace detail {
inline void validate_row_buffers(const void* input, const void* output, std::size_t n) {
    if (n % kBlock != 0 || (n != 0 && (!input || !output)))
        throw std::invalid_argument("quantized rows require complete blocks and non-null buffers");
}

// Preflight the WHOLE row before publishing any blocks. Silently saturating an
// unrepresentable scale can destroy model weights; storing infinity yields NaNs
// on dequantization (even for zero codes). Ordinary encoding stays unchanged.
inline void validate_quantize_row(const float* input, const void* output,
                                  std::size_t n, float divisor) {
    validate_row_buffers(input, output, n);
    float amax = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(input[i]))
            throw std::invalid_argument("quantization requires finite weights");
        amax = std::max(amax, std::fabs(input[i]));
    }
    if (!std::isfinite(fp16_to_fp32(fp32_to_fp16(amax / divisor))))
        throw std::overflow_error("weight scale is not representable as finite fp16");
}
}  // namespace detail

// ---- Q8_0: d = amax/127 (fp16), q = round(w/d) in [-127, 127] ----
inline void quantize_row_q8_0(const float* x, BlockQ8_0* out, std::size_t n) {
    detail::validate_quantize_row(x, out, n, 127.f);
    for (std::size_t b = 0; b < nblocks(n); ++b, x += kBlock, ++out) {
        float amax = 0.f;
        for (int i = 0; i < kBlock; ++i) amax = std::max(amax, std::fabs(x[i]));
        const std::uint16_t dh = fp32_to_fp16(amax / 127.f);
        const float d = fp16_to_fp32(dh);
        out->d_fp16 = dh;
        if (d == 0.f) {
            for (int i = 0; i < kBlock; ++i) out->qs[i] = 0;
            continue;
        }
        const float inv = 1.f / d;
        for (int i = 0; i < kBlock; ++i) {
            const float v = std::min(std::max(x[i] * inv, -127.f), 127.f);
            out->qs[i] = (std::int8_t)std::lrintf(v);
        }
    }
}

inline void dequantize_row_q8_0(const BlockQ8_0* in, float* x, std::size_t n) {
    detail::validate_row_buffers(in, x, n);
    for (std::size_t b = 0; b < nblocks(n); ++b, x += kBlock, ++in) {
        const float d = fp16_to_fp32(in->d_fp16);
        for (int i = 0; i < kBlock; ++i) x[i] = (float)in->qs[i] * d;
    }
}

// ---- K-quant super-blocks (ggml "k-quants"), 256 values each --------------
//
// A K-quant super-block carries its own 2-D lattice: an fp16 super-scale plus
// per-sub-block scales (and, for Q4_K/Q5_K, per-sub-block mins) stored as 6-bit
// codes packed into a 12-byte `scales` array. The layouts below are the wire
// format of ggml-common.h, so a GGUF K-quant tensor is copied into a .tmq
// BIT-EXACTLY — no requantization, no fp32 round trip. That matters: these are
// the tensors llama.cpp deliberately keeps at higher precision (a Q4_0 GGUF
// stores token_embd/output at Q6_K), and they are also the largest per-token
// reads in decode, so their byte width IS the decode speed.
constexpr int kKBlock = 256;

#pragma pack(push, 1)
struct BlockQ4_1 {           // 20 bytes; 32 values: w = q*d + m
    std::uint16_t d_fp16;
    std::uint16_t m_fp16;
    std::uint8_t  qs[16];    // low nibble = element j, high nibble = j + 16
};
static_assert(sizeof(BlockQ4_1) == 20, "Q4_1 block must be 20 bytes");

struct BlockQ5_K {           // 176 bytes; 256 values: w = q*d + m (5-bit codes)
    std::uint16_t d_fp16;    // super-block scale for the quantized scales
    std::uint16_t dmin_fp16; // super-block scale for the quantized mins
    std::uint8_t  scales[12];// 8 sub-blocks, scales and mins, 6 bits each
    std::uint8_t  qh[32];    // high bit of each 5-bit code
    std::uint8_t  qs[128];   // low 4 bits
};
static_assert(sizeof(BlockQ5_K) == 176, "Q5_K block must be 176 bytes");

struct BlockQ6_K {           // 210 bytes; 256 values: w = d * sc * q
    std::uint8_t  ql[128];   // low 4 bits of each 6-bit code
    std::uint8_t  qh[64];    // high 2 bits
    std::int8_t   scales[16];// 16 sub-blocks of 16, signed 8-bit scales
    std::uint16_t d_fp16;    // super-block scale
};
static_assert(sizeof(BlockQ6_K) == 210, "Q6_K block must be 210 bytes");
#pragma pack(pop)
static_assert(alignof(BlockQ4_1) == 1 && alignof(BlockQ5_K) == 1 &&
              alignof(BlockQ6_K) == 1,
              "K-quant blocks must support unaligned serialized payloads");

// Unpack one 6-bit scale/min pair from a Q4_K/Q5_K `scales` array (ggml's
// get_scale_min_k4, transcribed). The first four sub-blocks carry their scale
// in the low 6 bits of scales[j] and their min in scales[j+4]; the last four
// take the low nibble of scales[j+4] (and the high nibbles of scales[j-4] and
// scales[j]) as the top 2 bits.
inline void k_scale_min(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        d = (std::uint8_t)((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        m = (std::uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ---- Q4_1: 32 values, w = q*d + m ----------------
inline void dequantize_row_q4_1(const BlockQ4_1* in, float* x, std::size_t n) {
    detail::validate_row_buffers(in, x, n);
    for (std::size_t b = 0; b < n / kBlock; ++b, x += kBlock, ++in) {
        const float d = fp16_to_fp32(in->d_fp16);
        const float m = fp16_to_fp32(in->m_fp16);
        for (int j = 0; j < 16; ++j) {
            x[j]      = (float)(in->qs[j] & 0x0fu) * d + m;
            x[j + 16] = (float)(in->qs[j] >> 4) * d + m;
        }
    }
}

// ---- Q5_K: 256 values, w = q*d1 - m1 -------------
inline void dequantize_row_q5_K(const BlockQ5_K* in, float* x, std::size_t n) {
    if (n % kKBlock != 0 || (n != 0 && !in) || (n != 0 && !x))
        throw std::invalid_argument("Q5_K rows require complete super-blocks");
    for (std::size_t b = 0; b < n / kKBlock; ++b, x += kKBlock, ++in) {
        const float d = fp16_to_fp32(in->d_fp16);
        const float dm = fp16_to_fp32(in->dmin_fp16);
        const std::uint8_t* ql = in->qs;
        const std::uint8_t* qh = in->qh;
        int is = 0;
        std::uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < kKBlock; j += 64) {
            std::uint8_t sc, m;
            k_scale_min(is + 0, in->scales, sc, m);
            const float d1 = d * (float)sc, m1 = dm * (float)m;
            k_scale_min(is + 1, in->scales, sc, m);
            const float d2 = d * (float)sc, m2 = dm * (float)m;
            for (int l = 0; l < 32; ++l)
                x[j + l] = d1 * (float)((ql[l] & 0x0fu) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                x[j + 32 + l] = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 = (std::uint8_t)(u1 << 2);
            u2 = (std::uint8_t)(u2 << 2);
        }
    }
}

// ---- Q6_K: 256 values, w = d * sc * q, q = code - 32 -------------
// Note the sign convention: `scales` is int8, so a sub-block scale is signed.
// The codes are stored 4+2 bits and land in a permuted order — within each
// 128-value half, element (l + 32*j) for j = 0..3 takes its low nibble from
// ql[l + 32*(j&1)] (high nibble for j >= 2), its top 2 bits from qh[l] shifted
// by 2*j, and its scale from sc[l/16 + 2*j].
inline void dequantize_row_q6_K(const BlockQ6_K* in, float* x, std::size_t n) {
    if (n % kKBlock != 0 || (n != 0 && !in) || (n != 0 && !x))
        throw std::invalid_argument("Q6_K rows require complete super-blocks");
    for (std::size_t b = 0; b < n / kKBlock; ++b, x += kKBlock, ++in) {
        const float d = fp16_to_fp32(in->d_fp16);
        const std::uint8_t* ql = in->ql;
        const std::uint8_t* qh = in->qh;
        const std::int8_t*  sc = in->scales;
        float* y = x;
        for (int half = 0; half < kKBlock; half += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int)((ql[l] & 0x0fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0x0fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                y[l +  0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ---- Q4_0: d = max_signed / -8 (ggml reference rule), nibble offset +8 ----
inline void quantize_row_q4_0(const float* x, BlockQ4_0* out, std::size_t n) {
    detail::validate_quantize_row(x, out, n, 8.f);
    for (std::size_t b = 0; b < nblocks(n); ++b, x += kBlock, ++out) {
        float amax = 0.f;
        float max_signed = 0.f;
        for (int i = 0; i < kBlock; ++i) {
            const float a = std::fabs(x[i]);
            if (a > amax) { amax = a; max_signed = x[i]; }
        }
        const float d = max_signed != 0.f ? -max_signed / 8.f : 0.f;
        const std::uint16_t dh = fp32_to_fp16(d);
        const float df = fp16_to_fp32(dh);
        out->d_fp16 = dh;
        const float inv = df != 0.f ? 1.f / df : 0.f;
        for (int i = 0; i < kBlock; i += 2) {
            int n0, n1;
            if (df == 0.f) {
                n0 = n1 = 8;                       // zero block -> nibble 8 = 0*d
            } else {
                n0 = (int)std::lrintf(x[i] * inv) + 8;
                n1 = (int)std::lrintf(x[i + 1] * inv) + 8;
                n0 = std::min(15, std::max(0, n0));
                n1 = std::min(15, std::max(0, n1));
            }
            out->qs[i / 2] = (std::uint8_t)((n1 << 4) | n0);
        }
    }
}

inline void dequantize_row_q4_0(const BlockQ4_0* in, float* x, std::size_t n) {
    detail::validate_row_buffers(in, x, n);
    for (std::size_t b = 0; b < nblocks(n); ++b, x += kBlock, ++in) {
        const float d = fp16_to_fp32(in->d_fp16);
        for (int j = 0; j < 16; ++j) {
            const int n0 = in->qs[j] & 0x0fu;
            const int n1 = in->qs[j] >> 4;
            x[2 * j]     = (float)(n0 - 8) * d;
            x[2 * j + 1] = (float)(n1 - 8) * d;
        }
    }
}

// ---- Sub-2-bit weights: the PrismML "Bonsai" family -----------------------
//
// THREE layouts are in the wild and they do NOT share a block shape, so the
// names below are layout-explicit on purpose. The GGUF type id is NOT enough
// to identify one: type 42 carried a 34-byte block in the shipped April pack
// and an 18-byte one in the September fork, and reading either at the
// other's stride is silent garbage rather than an error.
//
//   BlockTq2_34x128   34 B / 128 values   ternary   <- the SHIPPED Bonsai pack
//   BlockTq2_18x64    18 B /  64 values   ternary   <- PrismML-Eng/llama.cpp
//   BlockTq1_18x128   18 B / 128 values   binary    <- PrismML-Eng/llama.cpp
//
// MEASURED, not transcribed, for the shipped one:
// `prism-ml/Ternary-Bonsai-4B-gguf / Ternary-Bonsai-4B-Q2_0.gguf` — 16 MB
// range-fetched, header and 3000 weight blocks decoded. Its 253 weight
// tensors sit at exactly 34.000 bytes per 128 values = 2.125 bits/weight,
// the pack's advertised 2.13. The accounting closes exactly (blk.0.attn_k,
// 2560x1024, spans 103841504 - 103145184 = 696320 = 1024 rows x 20 blocks x
// 34), and only that stride yields a smooth positive scale field
// (0.0146 .. 0.0388) — d-last or an 18-byte stride gives negative, chaotic
// scales instead. The code split is 35.1 / 29.9 / 34.9 / 0.0 %, and that
// trailing 0.0 % is a PREDICTION that holds: with d = max|x|, a code meaning
// +2d is unreachable.
//
// The two 18-byte blocks are transcribed from the fork's head
// (`ggml-common.h`, `ggml-quants.c`), where GGML_TYPE_Q1_0 = 41 is binary
// with d = mean|x| and GGML_TYPE_Q2_0 = 42 is ternary with d = max|x| and
// clamp(round(w/d) + 1, 0, 3). No shipped file of either shape has been
    // through this loader, so neither is on the gate; they are here so the
    // shapes are NAMED and testable, not because a file demands them.
//
// ONE dot serves all three, and it is why the format is cheap to read: the
// encoding is affine in its code, so
//   sum_j w_j y_j = d * (sum_lo + 2*sum_hi - sum_j y_j)
    // and the baseline is ONE precomputed block sum of the ACTIVATION, not one
    // multiply per weight (`block_q_n_dot_y` in the fork's Metal kernel).
constexpr int kBlockTq2_34x128 = 128;
constexpr int kBlockTq2_18x64  = 64;
constexpr int kBlockTq1_18x128 = 128;

#pragma pack(push, 1)
              struct BlockTq2_34x128 {     // 34 bytes; 128 values: w = (code - 1) * d
    std::uint16_t d_fp16;    // d = max|x| over the 128
    std::uint8_t  qs[32];    // 2-bit codes, 4 per byte, low bits first
};
static_assert(sizeof(BlockTq2_34x128) == 34, "TQ2 34x128 block must be 34 bytes");

struct BlockTq2_18x64 {      // 18 bytes; 64 values: w = (code - 1) * d
    std::uint16_t d_fp16;
    std::uint8_t  qs[16];
};
static_assert(sizeof(BlockTq2_18x64) == 18, "TQ2 18x64 block must be 18 bytes");

struct BlockTq1_18x128 {     // 18 bytes; 128 values: w = d * (2*bit - 1)
    std::uint16_t d_fp16;    // d = mean|x| over the 128
    std::uint8_t  qs[16];    // one sign bit per value, low bit first
};
static_assert(sizeof(BlockTq1_18x128) == 18, "TQ1 18x128 block must be 18 bytes");
#pragma pack(pop)
static_assert(alignof(BlockTq2_34x128) == 1 && alignof(BlockTq2_18x64) == 1 &&
              alignof(BlockTq1_18x128) == 1,
              "ternary blocks must support unaligned serialized payloads");

namespace detail {
// Same preflight contract as validate_quantize_row, for a block size other
// than kBlock: complete blocks, non-null buffers, finite weights. The fp16
// scale itself is checked per block, where it is computed.
template <int QK>
inline void validate_weight_row(const float* input, const void* output, std::size_t n) {
    if (n % QK != 0 || (n != 0 && (!input || !output)))
        throw std::invalid_argument("quantized rows require complete blocks and non-null buffers");
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(input[i]))
            throw std::invalid_argument("quantization requires finite weights");
}

template <int QK>
inline void validate_weight_row(const void* input, float* output, std::size_t n) {
    if (n % QK != 0 || (n != 0 && (!input || !output)))
        throw std::invalid_argument("quantized rows require complete blocks and non-null buffers");
}
}  // namespace detail

namespace detail {
// The ternary codec: 4 codes per byte, low bits first; code in [0,3] means
// {-1, 0, +1, +2} * d, with d = max|x|. Shared by both TQ2 shapes — only the
// block and byte counts differ, so one body keeps them from drifting apart.
template <class Block, int QK, int NB>
inline void quantize_ternary(const float* x, Block* out, std::size_t n) {
    validate_weight_row<QK>(x, out, n);
    for (std::size_t b = 0; b < n / QK; ++b, x += QK, ++out) {
        float amax = 0.f;
        for (int j = 0; j < QK; ++j) amax = std::max(amax, std::fabs(x[j]));
        const std::uint16_t dh = fp32_to_fp16(amax);
        const float d = fp16_to_fp32(dh);
        out->d_fp16 = dh;
        for (int j = 0; j < NB; ++j) out->qs[j] = 0;
        const float inv = d != 0.f ? 1.f / d : 0.f;
        for (int j = 0; j < QK; ++j) {
            // 00 = -1, 01 = 0, 10 = +1, 11 = +2. With d = max|x| the last is
            // unreachable — the property the shipped pack's code histogram shows
            // — but the clamp stays so a malformed row cannot wrap.
            int q = (int)std::lrintf(x[j] * inv) + 1;
            q = std::min(3, std::max(0, q));
            out->qs[j >> 2] |= (std::uint8_t)((unsigned)q << (2 * (j & 3)));
        }
    }
}

template <class Block, int QK>
inline void dequantize_ternary(const Block* in, float* x, std::size_t n) {
    validate_weight_row<QK>(in, x, n);
    for (std::size_t b = 0; b < n / QK; ++b, x += QK, ++in) {
        const float d = fp16_to_fp32(in->d_fp16);
        for (int j = 0; j < QK; ++j)
            x[j] = (float)((int)((in->qs[j >> 2] >> (2 * (j & 3))) & 0x03u) - 1) * d;
    }
}

// The CODE-COUNTING dot, not the dequantized-value one: acc_lo counts the codes
// whose bit 0 is set, acc_hi those with bit 1, and the -1 baseline collapses
// into the caller's block sum of the activation.
template <class Block, int QK>
inline float dot_ternary(const Block& b, const float* y, float sum_y) {
    const float d = fp16_to_fp32(b.d_fp16);
    float acc_lo = 0.f, acc_hi = 0.f;
    for (int j = 0; j < QK; ++j) {
        const unsigned q = (b.qs[j >> 2] >> (2 * (j & 3))) & 0x03u;
        if (q & 1u) acc_lo += y[j];
        if (q & 2u) acc_hi += y[j];
    }
    return d * (acc_lo + 2.f * acc_hi - sum_y);
}
}  // namespace detail

// ---- TQ2 34x128: the shipped Bonsai pack's block ----
inline void quantize_row_tq2_34x128(const float* x, BlockTq2_34x128* out, std::size_t n) {
    detail::quantize_ternary<BlockTq2_34x128, kBlockTq2_34x128, 32>(x, out, n);
}
inline void dequantize_row_tq2_34x128(const BlockTq2_34x128* in, float* x, std::size_t n) {
    detail::dequantize_ternary<BlockTq2_34x128, kBlockTq2_34x128>(in, x, n);
}
inline float dot_block_tq2_34x128(const BlockTq2_34x128& b, const float* y, float sum_y) {
    return detail::dot_ternary<BlockTq2_34x128, kBlockTq2_34x128>(b, y, sum_y);
}

// ---- TQ2 18x64: the fork head's Q2_0 (same codec, smaller block) ----
inline void quantize_row_tq2_18x64(const float* x, BlockTq2_18x64* out, std::size_t n) {
    detail::quantize_ternary<BlockTq2_18x64, kBlockTq2_18x64, 16>(x, out, n);
}
inline void dequantize_row_tq2_18x64(const BlockTq2_18x64* in, float* x, std::size_t n) {
    detail::dequantize_ternary<BlockTq2_18x64, kBlockTq2_18x64>(in, x, n);
}
inline float dot_block_tq2_18x64(const BlockTq2_18x64& b, const float* y, float sum_y) {
    return detail::dot_ternary<BlockTq2_18x64, kBlockTq2_18x64>(b, y, sum_y);
}


// ---- TQ1 18x128: the fork head's Q1_0 — binary, d = mean|x| ----
inline void quantize_row_tq1_18x128(const float* x, BlockTq1_18x128* out, std::size_t n) {
    detail::validate_weight_row<kBlockTq1_18x128>(x, out, n);
    for (std::size_t b = 0; b < n / kBlockTq1_18x128; ++b, x += kBlockTq1_18x128, ++out) {
        float sum_abs = 0.f;
        for (int j = 0; j < kBlockTq1_18x128; ++j) sum_abs += std::fabs(x[j]);
        const std::uint16_t dh = fp32_to_fp16(sum_abs / (float)kBlockTq1_18x128);
        out->d_fp16 = dh;
        for (int j = 0; j < 16; ++j) out->qs[j] = 0;
        for (int j = 0; j < kBlockTq1_18x128; ++j)
            if (x[j] >= 0.f) out->qs[j >> 3] |= (std::uint8_t)(1u << (j & 7));
    }
}

inline void dequantize_row_tq1_18x128(const BlockTq1_18x128* in, float* x, std::size_t n) {
    detail::validate_weight_row<kBlockTq1_18x128>(in, x, n);
    for (std::size_t b = 0; b < n / kBlockTq1_18x128; ++b, x += kBlockTq1_18x128, ++in) {
        const float d = fp16_to_fp32(in->d_fp16);
        for (int j = 0; j < kBlockTq1_18x128; ++j)
            x[j] = ((in->qs[j >> 3] >> (j & 7)) & 1u) ? d : -d;
    }
}

inline float dot_block_tq1_18x128(const BlockTq1_18x128& b, const float* y, float sum_y) {
    const float d = fp16_to_fp32(b.d_fp16);
    float acc = 0.f;
    for (int j = 0; j < kBlockTq1_18x128; ++j)
        if ((b.qs[j >> 3] >> (j & 7)) & 1u) acc += y[j];
    return d * (2.f * acc - sum_y);
}

// `sum_y` is the activation sum over ONE block, hoisted by the caller because it
// is shared by every weight block that reads the same activation row. The dots
// above count codes; dequantize-then-dot is the ORACLE the tests compare
// against, never the implementation.
inline float block_sum_y(const float* y, int count) {
    float s = 0.f;
    for (int j = 0; j < count; ++j) s += y[j];
    return s;
}

}  // namespace tmq
