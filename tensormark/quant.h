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

}  // namespace tmq
