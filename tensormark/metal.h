// tensormark/metal.h — M6 optional Metal backend.
//
// ObjC++ ONLY: include from .mm translation units. The C++ core must never
// include this. Build with -DTM_HAVE_METAL; without it nothing here
// compiles and the core is Metal-free (spec criterion 1).
//
// Design notes:
//  - Runtime shader compilation via newLibraryWithSource: no offline
//    metallib build step. Source of truth is kMetalSrc below, mirrored in
//    metal_kernels.metal for readability/Xcode tooling.
//  - GEMM dispatch uses MPSMatrixMultiplication (Apple-tuned GPU GEMM).
//    Measured: hand-rolled tiled kernels plateau at ~175-250 GFLOPS on M1
//    regardless of tile shape / lane coalescing / barrier count / re-traffic,
//    while cblas+AMX hits ~900 GFLOPS. MPS is the route the spec allows.
//  - Every entry point returns false on any failure and bumps the fallback
//    counter (spec: silent, measurable fallback — a counter, not a print).
//  - Buffers are cached and reused across calls (per-call newBufferWithBytes
//    measured 3-10x slower than the kernel itself on Llama shapes).
//  - Kernel v2: threadgroup tile 16 rows x 64 cols, each thread computes a
//    1x4 output strip with float4 accumulation.
#pragma once
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <atomic>
#include "tm_tune.h"   // machine-specific static parameters (build/autotune.py)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <utility>
#include <algorithm>
#include <array>
#include <vector>

namespace tmgpu {

inline std::atomic<unsigned long>& fallback_count() {
    static std::atomic<unsigned long> n{0};
    return n;
}
// Successful GPU prefill GEMM dispatches (tests assert the path engaged).
inline std::atomic<unsigned long>& prefill_count() {
    static std::atomic<unsigned long> n{0};
    return n;
}

namespace detail {
inline __strong id<MTLDevice>& dev() { static id<MTLDevice> d = nil; return d; }
inline __strong id<MTLCommandQueue>& q() { static id<MTLCommandQueue> q = nil; return q; }
inline __strong id<MTLLibrary>& lib() { static id<MTLLibrary> l = nil; return l; }
inline __strong id<MTLLibrary>& libq4() { static id<MTLLibrary> l = nil; return l; }
inline bool& tried() { static bool t = false; return t; }

// Cached, grow-only buffers: ba/bb inputs, bc output.
struct Bufs {
    __strong id<MTLBuffer> a, b, c;
};
inline Bufs& bufs() { static Bufs b; return b; }

inline id<MTLBuffer> grow(__strong id<MTLBuffer>& buf, size_t bytes) {
    if (!buf || [buf length] < bytes)
        buf = [dev() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return buf;
}

// Keep in sync with metal_kernels.metal.
inline NSString* kMetalSrc() {
    return @R"MET(
    #include <metal_stdlib>
    using namespace metal;
    kernel void cvt_f32_f16(
    const device float* src [[buffer(0)]],
    device half*        dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
    {
    if (gid < n) dst[gid] = src[gid];
    }
    kernel void cvt_f16_f32(
    const device half*  src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
    {
    if (gid < n) dst[gid] = src[gid];
    }
kernel void sgemm_tiled(
    const device float* A     [[buffer(0)]],
    const device float* B     [[buffer(1)]],
    device float*       C     [[buffer(2)]],
    constant uint&      M     [[buffer(3)]],
    constant uint&      N     [[buffer(4)]],
    constant uint&      K     [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint2 tpt [[thread_position_in_threadgroup]])
{
    // v3: 64x64 threadgroup tile, each thread computes a 4x4 block
    // (4 float4 row accumulators). Global re-traffic drops from
    // 2*M*K*N/16 to 2*M*K*N/64 bytes (10x) -- the naive kernel was
    // bandwidth-bound at ~250 GFLOPS, not compute-bound.
    threadgroup float As[32][64];
    threadgroup float Bs[32][64];
    const uint r0 = tgp.y * 64 + tpt.y * 4;
    const uint c0 = tgp.x * 64 + tpt.x * 4;
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint kt = 0; kt < K; kt += 32) {
        const uint li = tpt.y * 16 + tpt.x;
        for (uint j = 0; j < 8; ++j) {          // As: 2048 floats / 256 thr
            const uint idx = li + 256 * j;      // lanes coalesced (was li*8+j)
            const uint kr = idx >> 6, kc = idx & 63;
            const uint gr = tgp.y * 64 + kc;
            As[kr][kc] = (gr < M && kt + kr < K)
                ? A[gr * K + kt + kr] : 0.0f;
                }
        for (uint j = 0; j < 8; ++j) {          // Bs: 2048 floats / 256 thr
            const uint idx = li + 256 * j;      // lanes coalesced (was li*8+j)
            const uint kr = idx >> 6, kc = idx & 63;
            const uint nc = tgp.x * 64 + kc;
            Bs[kr][kc] = (kt + kr < K && nc < N) ? B[(kt + kr) * N + nc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; ++k) {
            const float4 bv = *(threadgroup float4*)&Bs[k][tpt.x * 4];
            a0 = fma(As[k][tpt.y * 4 + 0], bv, a0);
            a1 = fma(As[k][tpt.y * 4 + 1], bv, a1);
            a2 = fma(As[k][tpt.y * 4 + 2], bv, a2);
            a3 = fma(As[k][tpt.y * 4 + 3], bv, a3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    if (r0 + 3 < M && c0 + 3 < N) {
        *(device float4*)&C[(r0 + 0) * N + c0] = a0;
        *(device float4*)&C[(r0 + 1) * N + c0] = a1;
        *(device float4*)&C[(r0 + 2) * N + c0] = a2;
        *(device float4*)&C[(r0 + 3) * N + c0] = a3;
    } else {
        for (uint r = 0; r < 4 && r0 + r < M; ++r)
            for (uint j = 0; j < 4 && c0 + j < N; ++j) {
                const float4 av = r == 0 ? a0 : r == 1 ? a1 : r == 2 ? a2 : a3;
                C[(r0 + r) * N + c0 + j] = av[j];
            }
    }
}
)MET";
}

inline NSString* kMetalSrcQ4() {
return @R"MET(
#include <metal_stdlib>
using namespace metal;
// M=1-native Q4_0 GEMV: y[n] = sum_k x[k] * W[n,k]. One thread per output;
// each thread streams its own weight row (18B/block, read exactly once)
// and dequantizes nibble pairs into float4 lanes that match the x float4
// loads (values 2j,2j+1 are consecutive -> lane-aligned). Full occupancy:
// every thread computes for its whole life, no threadgroup smem, no tail
// path -- the fix for the prefill-kernel-at-M=1 underutilization (16/256
// threads active) measured at 5x slower than CPU.
kernel void q4_gemv(
const device float* x  [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device float*       y  [[buffer(2)]],
constant uint&      N  [[buffer(3)]],
constant uint&      K  [[buffer(4)]],
constant uint64_t&  woff [[buffer(5)]],
uint gid [[thread_position_in_grid]])
{
if (gid >= N) return;
const uint nb = K >> 5;
const device uchar* row = Wq + woff + (size_t)gid * nb * 18;
float4 acc = 0.0f;
for (uint b = 0; b < nb; ++b, row += 18) {
// fp16 scale assembled from BYTES: with woff from the file mapping the
// half* load would be misaligned (tensor offsets are arbitrary bytes)
// and silently return garbage on the GPU.
const uint16_t d16 = (uint16_t)row[0] | ((uint16_t)row[1] << 8);
const float d = (float)(as_type<half>(d16));
const device uchar* q = row + 2;
const device float* xb = x + b * 32;
for (uint j = 0; j < 16; j += 2) {
float4 wv;
wv[0] = (float)((int)(q[j]   & 0xF) - 8);
wv[1] = (float)((int)(q[j]   >> 4) - 8);
wv[2] = (float)((int)(q[j+1] & 0xF) - 8);
wv[3] = (float)((int)(q[j+1] >> 4) - 8);
acc += wv * (*(device const float4*)(xb + 2 * j)) * d;
}
}
y[gid] = acc[0] + acc[1] + acc[2] + acc[3];
}
// Cooperative row reduction: adjacent lanes stream adjacent Q4 blocks rather
// than unrelated full rows. Specialize the lane count so division/reduction
// become compile-time shifts. Subgroups never cross a 32-lane SIMD group.
constant uint gemv_row_lanes [[function_constant(0)]];
constant bool gemv_block_scale [[function_constant(1)]];
kernel void q4_gemv_coop(
const device float* x [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]],
constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]])
{
    const uint n = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    if (n >= N) return;
    const uint nb = K >> 5;
    float4 acc = 0.0f;
    for (uint b = lane; b < nb; b += gemv_row_lanes) {
        const device uchar* block = Wq + woff + ((size_t)n * nb + b) * 18;
        const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
        const float scale = float(as_type<half>(bits));
        // The scale is constant over all 32 values: apply it once to the
        // block's vector dot, not to every dequantized four-value fragment.
        float4 block_acc = 0.0f;
        for (uint j = 0; j < 16; j += 2) {
            const uchar a = block[2 + j], c = block[3 + j];
            const float4 w = float4(int(a & 15) - 8, int(a >> 4) - 8,
                                    int(c & 15) - 8, int(c >> 4) - 8);
            const float4 xv = *(device const float4*)(x + b * 32 + 2 * j);
            if (gemv_block_scale) block_acc += w * xv;
            else acc += w * xv * scale;
        }
        if (gemv_block_scale) acc += block_acc * scale;
    }
    float sum = acc.x + acc.y + acc.z + acc.w;
    for (uint hop = gemv_row_lanes / 2; hop; hop /= 2)
        sum += simd_shuffle_down(sum, hop);
    if (lane == 0) y[n] = sum;
}
// R rows per thread: a thread loads its x block once and dots it with the
// same block of R consecutive rows, cutting the L1 traffic for x (128 B per
// 18 B weight block) R-fold. Row-group g = gid / lanes covers rows
// R*g .. R*g+R-1; lanes stream blocks as in q4_gemv_coop.
// Measured 2026-09-07 on a cache-cold dependent chain of 2048x2048 GEMVs
// (24 distinct tensors, 56 MB; interleaved in-process sweep, medians of 7):
// rows 1 / 16 lanes / 64 threads (the former default) 45.9 GB/s; rows 2 /
// 16 / 128 or 256: 54.1-54.4; rows 4 / 16: 52.0-52.2; rows 8: <= 49.9;
// 8 or 4 lanes always worse. Also tried and dropped: 16-bit block loads
// (+3%, and .tmq tensor offsets are often odd) and packed_uchar4 nibble
// loads (no gain). In the model (TinyLlama Q4 GPU decode, ctx 64) rows 2
// took GPU time per token 13.24 -> 12.04 ms with identical greedy hashes.
template <uint R>
inline void q4_gemv_rows_body(const device float* x, const device uchar* Wq, device float* y,
                              uint N, uint K, uint64_t woff, uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint nb = K >> 5;
    const uint rows = min(R, N - n0);
    float4 acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nb; b += gemv_row_lanes) {
        const device float4* xb = (const device float4*)(x + b * 32);
        float4 xv[8];
        for (uint j = 0; j < 8; ++j) xv[j] = xb[j];
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* block = Wq + woff + ((size_t)(n0 + r) * nb + b) * 18;
            const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
            const float scale = float(as_type<half>(bits));
            float4 block_acc = 0.0f;
            for (uint j = 0; j < 16; j += 2) {
                const uchar a = block[2 + j], c = block[3 + j];
                const float4 w = float4(int(a & 15) - 8, int(a >> 4) - 8,
                                        int(c & 15) - 8, int(c >> 4) - 8);
                block_acc += w * xv[j / 2];
            }
            acc[r] += block_acc * scale;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r].x + acc[r].y + acc[r].z + acc[r].w;
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2)
            sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}
// Mask-dot variant (idea from llama.cpp's mul_vec_q_n_f32 / block_q_n_dot_y):
// pre-scale the activations by 1, 1/16, 1/256, 1/4096 so a raw 16-bit window
// of the packed nibbles multiplies them directly — no per-value shift, no
// subtract-8 (the -8 offset becomes one term, 8 * sum(x), per block) and no
// per-nibble integer maths. The x pre-scale is paid once per block and shared
// by all R rows. Bytes are read as packed_uchar2 because .tmq blocks sit at
// odd offsets, where a ushort load would fault or return garbage.
template <uint R>
inline void q4_gemv_mask_body(const device float* x, const device uchar* Wq, device float* y,
                              uint N, uint K, uint64_t woff, uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint nb = K >> 5;
    const uint rows = min(R, N - n0);
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nb; b += gemv_row_lanes) {
        const device float4* xb = (const device float4*)(x + b * 32);
        float4 xs[8]; float sumx = 0.0f;
        const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
        for (uint j = 0; j < 8; ++j) {
            const float4 v = xb[j];
            sumx += v.x + v.y + v.z + v.w;
            xs[j] = v * sc;
        }
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* blk = Wq + woff + ((size_t)(n0 + r) * nb + b) * 18;
            const ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
            const float d = float(as_type<half>(dbits));
            const device packed_uchar2* qp = (const device packed_uchar2*)(blk + 2);
            float4 a = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                const uchar2 p = uchar2(qp[j]);
                const uint u = uint(p.x) | (uint(p.y) << 8);
                a += xs[j] * float4(u & 0x000Fu, u & 0x00F0u, u & 0x0F00u, u & 0xF000u);
            }
            acc[r] += d * (a.x + a.y + a.z + a.w - 8.0f * sumx);
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r];
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2) sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}
kernel void q4_gemv_mask_r2(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q4_gemv_mask_body<2>(x, Wq, y, N, K, woff, gid); }
kernel void q4_gemv_mask_r4(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q4_gemv_mask_body<4>(x, Wq, y, N, K, woff, gid); }
kernel void q4_gemv_coop_r2(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q4_gemv_rows_body<2>(x, Wq, y, N, K, woff, gid); }
kernel void q4_gemv_coop_r4(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q4_gemv_rows_body<4>(x, Wq, y, N, K, woff, gid); }
// Segmented cooperative GEMV (decode fusion): one dispatch computes the
// q|k|v (or gate|up) projections from the same activation. Each segment has
// its own (byte offset, row count) into the shared zero-copy weight buffer.
// Rows are R per thread; segment boundaries must be R-aligned (every Llama
// projection dim is a multiple of 32, so this always holds). The per-row
// accumulation order is IDENTICAL to q4_gemv_rows_body, so the result is
// bit-for-bit equal to the separate dispatches it replaces.
template <uint R>
inline void q4_gemv_rows_seg_body(
    const device float* x, const device uchar* Wq, device float* y,
    uint N, uint K, const device uint* seg_rows,
    const device uint64_t* seg_woff, uint nseg, uint gid)
{
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    uint seg = 0, seg_start = 0;
    for (uint s = 0; s < nseg; ++s) {
        if (n0 < seg_start + seg_rows[s]) { seg = s; break; }
        seg_start += seg_rows[s];
    }
    const uint local_n0 = n0 - seg_start;
    const uint64_t woff = seg_woff[seg];
    const uint rows = min(R, seg_rows[seg] - local_n0);
    const uint nb = K >> 5;
    float4 acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nb; b += gemv_row_lanes) {
        const device float4* xb = (const device float4*)(x + b * 32);
        float4 xv[8];
        for (uint j = 0; j < 8; ++j) xv[j] = xb[j];
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* block = Wq + woff + ((size_t)(local_n0 + r) * nb + b) * 18;
            const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
            const float scale = float(as_type<half>(bits));
            float4 block_acc = 0.0f;
            for (uint j = 0; j < 16; j += 2) {
                const uchar a = block[2 + j], c = block[3 + j];
                const float4 w = float4(int(a & 15) - 8, int(a >> 4) - 8,
                                        int(c & 15) - 8, int(c >> 4) - 8);
                block_acc += w * xv[j / 2];
            }
            acc[r] += block_acc * scale;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r].x + acc[r].y + acc[r].z + acc[r].w;
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2)
            sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}
kernel void q4_gemv_coop_r2_seg(
const device float* x [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]],
constant uint& K [[buffer(4)]],
constant uint& nseg [[buffer(5)]],
const device uint* seg_rows [[buffer(6)]],
const device uint64_t* seg_woff [[buffer(7)]],
uint gid [[thread_position_in_grid]])
{ q4_gemv_rows_seg_body<2>(x, Wq, y, N, K, seg_rows, seg_woff, nseg, gid); }
// fp16-weights GEMV family (TM_DECODE_F16): same rows/lanes coop shape
// as the Q4 winner, minus the dequant maths — half4 weight loads, fp32
// accumulate. woff is a BYTE offset into the weight buffer; the row
// pointer rebuilds the typed half* from it (safetensors offsets are
// 8-byte aligned, so half4 loads stay aligned as long as K % 4 == 0,
// which the loader asserts).
template <uint R>
inline void f16_gemv_rows_body(const device float* x, const device half* W,
                               device float* y, uint N, uint K, uint64_t woff,
                               uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint k4 = K >> 2;
    const uint rows = min(R, N - n0);
    const device half* Wt = (const device half*)((const device uchar*)W + woff);
    float4 acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < k4; b += gemv_row_lanes) {
        const float4 xv = *(const device float4*)(x + b * 4);
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const half4 wv = *(const device half4*)(Wt + (size_t)(n0 + r) * K + b * 4);
            acc[r] += float4(wv) * xv;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r].x + acc[r].y + acc[r].z + acc[r].w;
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2)
            sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}
kernel void f16_gemv_coop_r1(
const device float* x [[buffer(0)]], const device half* W [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { f16_gemv_rows_body<1>(x, W, y, N, K, woff, gid); }
kernel void f16_gemv_coop_r2(
const device float* x [[buffer(0)]], const device half* W [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { f16_gemv_rows_body<2>(x, W, y, N, K, woff, gid); }
kernel void f16_gemv_coop_r4(
const device float* x [[buffer(0)]], const device half* W [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { f16_gemv_rows_body<4>(x, W, y, N, K, woff, gid); }
// Deeper k step (64 instead of 32): each thread dequantizes exactly one
// 32-value Q4 block per stage, so a 64x64 tile costs 2 barriers per 64 k
// values instead of 4, and 128 multiplies per barrier pair instead of 64.
// 16 KB of threadgroup memory (two groups per core).
kernel void q4_mm_tile3(
const device half* A [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device half* C [[buffer(2)]],
constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
constant uint& K [[buffer(5)]], constant uint& shift [[buffer(6)]],
threadgroup half* shmem [[threadgroup(0)]],
uint2 tgpig [[threadgroup_position_in_grid]],
ushort tiitg [[thread_index_in_threadgroup]],
ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = shmem;            // 64 W rows x 64 k
    threadgroup half* sb = shmem + 4096;     // 64 A rows x 64 k
    constexpr int NR = 64, NK = 64;
    const uint r0 = tgpig.y * NR, r1 = tgpig.x * NR;
    const short nr0 = (N - r0 < NR) ? (short)(N - r0) : NR;
    const short nr1 = (M - r1 < NR) ? (short)(M - r1) : NR;
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)(tiitg / 2) < nr1) ? (short)(tiitg / 2) : nr1 - 1;
    const short il0 = tiitg % 2;             // which 32-k half of the step
    const uint nb = K >> 5;
    const device uchar* wrow = Wq + shift + (size_t)(r0 + lr0) * nb * 18;
    const device half* arow = A + (size_t)(r1 + lr1) * K + 32 * il0;
    simdgroup_half8x8 ma[4], mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        const device uchar* blk = wrow + ((loop_k >> 5) + il0) * 18;
        const ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        const half d = as_type<half>(dbits);
        half tmp[32];
        for (short i = 0; i < 16; ++i) {
            const uchar q = blk[2 + i];
            tmp[2 * i] = (half)((int)(q & 0xF) - 8) * d;
            tmp[2 * i + 1] = (half)((int)(q >> 4) - 8) * d;
        }
        half4 av[8];
        for (short j = 0; j < 8; ++j) av[j] = *(const device half4*)(arow + loop_k + 4 * j);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 32; ++i) {
            const short k = 32 * il0 + i, sx = k / 8, ly = k % 8, sy = lr0 / 8, lx = lr0 % 8;
            sa[64 * (8 * sx + sy) + 8 * ly + lx] = tmp[i];
        }
        for (short j = 0; j < 4; ++j) {
            const short k0 = 32 * il0 + 8 * j, sx = k0 / 8, sy = lr1 / 8, ly = lr1 % 8;
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * (8 * sx + sy) + 8 * ly);
            dst[0] = av[2 * j]; dst[1] = av[2 * j + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 4 * 64 * (sgitg / 2);
        for (short ik = 0; ik < NK / 8; ++ik) {
            for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8);
            for (short i = 0; i < 4; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8);
            for (short i = 0; i < 16; ++i) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 8 * 64;
        }
    }
    const uint row0 = r1 + 32 * (sgitg / 2), col0 = r0 + 32 * (sgitg % 2);
    const ushort lane = tiitg % 32;
    threadgroup float* mine = (threadgroup float*)shmem + sgitg * 64;
    for (short i = 0; i < 16; ++i) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(mc[i], mine, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (short e = lane; e < 64; e += 32) {
            const uint row = row0 + 8 * (i / 4) + e / 8, col = col0 + 8 * (i % 4) + e % 8;
            if (row < M && col < N) C[(size_t)row * N + col] = (half)mine[e];
        }
    }
}
// Wider variant of q4_mm_tile: 64 x 64 output tile, so each SIMD group owns
// 32 x 32 as 4x4 = 16 float accumulators fed by 4 + 4 tile loads per 8-wide
// k step — 2.0 multiplies per threadgroup-memory load against 1.33 for the
// 32 x 64 tile (llama.cpp's shape). Staging and tile layout are identical
// (8x8 tiles, k-major for W so a plain simdgroup_load yields W^T); the
// epilogue converts float accumulators to half through 1 KB of scratch.
kernel void q4_mm_tile2(
const device half* A [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device half* C [[buffer(2)]],
constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
constant uint& K [[buffer(5)]], constant uint& shift [[buffer(6)]],
threadgroup half* shmem [[threadgroup(0)]],
uint2 tgpig [[threadgroup_position_in_grid]],
ushort tiitg [[thread_index_in_threadgroup]],
ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = shmem;            // 64 W rows x 32 k, 8x8 tiles, k-major
    threadgroup half* sb = shmem + 2048;     // 64 A rows x 32 k, 8x8 tiles, row-major
    threadgroup float* stage = (threadgroup float*)(shmem + 4096);   // 4 x 64 floats
    constexpr int NR = 64, NK = 32;
    const uint r0 = tgpig.y * NR;   // W rows -> output columns
    const uint r1 = tgpig.x * NR;   // A rows -> output rows
    const short nr0 = (N - r0 < NR) ? (short)(N - r0) : NR;
    const short nr1 = (M - r1 < NR) ? (short)(M - r1) : NR;
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1;
    const short il0 = tiitg % 2;
    const short lr1 = ((short)(tiitg / 2) < nr1) ? (short)(tiitg / 2) : nr1 - 1;
    const short iy = 16 * (tiitg % 2);
    const uint nb = K >> 5;
    const device uchar* wrow = Wq + shift + (size_t)(r0 + lr0) * nb * 18;
    const device half* arow = A + (size_t)(r1 + lr1) * K + iy;
    simdgroup_half8x8 ma[4], mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        const device uchar* blk = wrow + (loop_k >> 5) * 18;
        const ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        const half d = as_type<half>(dbits);
        half tmp[16];
        for (short i = 0; i < 8; ++i) {
            const uchar q = blk[2 + 8 * il0 + i];
            tmp[2 * i] = (half)((int)(q & 0xF) - 8) * d;
            tmp[2 * i + 1] = (half)((int)(q >> 4) - 8) * d;
        }
        half4 av[4];
        for (short j = 0; j < 4; ++j) av[j] = *(const device half4*)(arow + loop_k + 4 * j);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8, sy = lr0 / 8, lx = lr0 % 8, ly = i % 8;
            sa[64 * (8 * sx + sy) + 8 * ly + lx] = tmp[i];
        }
        for (short j = 0; j < 2; ++j) {   // two 8-wide k chunks, contiguous in one tile row
            const short sx = 2 * il0 + j, sy = lr1 / 8, ly = lr1 % 8;
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * (8 * sx + sy) + 8 * ly);
            dst[0] = av[2 * j]; dst[1] = av[2 * j + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 4 * 64 * (sgitg / 2);
        for (short ik = 0; ik < NK / 8; ++ik) {
            for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8);
            for (short i = 0; i < 4; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8);
            for (short i = 0; i < 16; ++i) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 8 * 64;
        }
    }
    const uint row0 = r1 + 32 * (sgitg / 2), col0 = r0 + 32 * (sgitg % 2);
    const ushort lane = tiitg % 32;
    threadgroup float* mine = stage + sgitg * 64;
    for (short i = 0; i < 16; ++i) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(mc[i], mine, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (short e = lane; e < 64; e += 32) {
            const uint row = row0 + 8 * (i / 4) + e / 8, col = col0 + 8 * (i % 4) + e % 8;
            if (row < M && col < N) C[(size_t)row * N + col] = (half)mine[e];
        }
    }
}
// Tiled Q4 GEMM on the SIMD matrix unit (2026-09-07), design after
// llama.cpp's kernel_mul_mm (ggml-metal, classic path): C[M x N] (half) =
// A[M x K] (half) . W^T with W = N rows of tensormark Q4 blocks. One
// 128-thread group per 32 (A rows) x 64 (W rows) output tile, K steps of
// 32; each thread dequantizes half a block (16 values) into registers
// BEFORE the barrier and stores it into threadgroup memory as contiguous
// 8x8 tiles (k-major for W, so a plain 64-element simdgroup_load yields the
// W^T tile), A rows likewise; 6 KB of operands; each SIMD group owns a
// 16 x 32 sub-tile as 8 float accumulators (2 A tiles x 4 W^T tiles per
// 8-wide k step). Epilogue converts to half through threadgroup memory.
// K % 32 == 0; rows past M / N are clamped on load and skipped on store.
kernel void q4_mm_tile(
const device half* A [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device half* C [[buffer(2)]],
constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
constant uint& K [[buffer(5)]], constant uint& shift [[buffer(6)]],
threadgroup half* shmem [[threadgroup(0)]],
uint2 tgpig [[threadgroup_position_in_grid]],
ushort tiitg [[thread_index_in_threadgroup]],
ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = shmem;            // 64 rows of W x 32 k, 8x8 tiles k-major
    threadgroup half* sb = shmem + 2048;     // 32 rows of A x 32 k, 8x8 tiles row-major
    constexpr int NR0 = 64, NR1 = 32, NK = 32;
    const uint r0 = tgpig.y * NR0;   // W rows (output columns)
    const uint r1 = tgpig.x * NR1;   // A rows (output rows)
    const short nr0 = (N - r0 < NR0) ? (short)(N - r0) : NR0;
    const short nr1 = (M - r1 < NR1) ? (short)(M - r1) : NR1;
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1;
    const short il0 = tiitg % 2;
    const short lr1 = ((short)(tiitg / 4) < nr1) ? (short)(tiitg / 4) : nr1 - 1;
    const short iy = 8 * (tiitg % 4);
    const uint nb = K >> 5;
    const device uchar* wrow = Wq + shift + (size_t)(r0 + lr0) * nb * 18;
    const device half* arow = A + (size_t)(r1 + lr1) * K + iy;
    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        // dequantize this thread's half block into registers (before the barrier)
        const device uchar* blk = wrow + (loop_k >> 5) * 18;
        const ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        const half d = as_type<half>(dbits);
        half tmp[16];
        for (short i = 0; i < 8; ++i) {
            const uchar q = blk[2 + 8 * il0 + i];
            tmp[2 * i] = (half)((int)(q & 0xF) - 8) * d;
            tmp[2 * i + 1] = (half)((int)(q >> 4) - 8) * d;
        }
        // A: 8 consecutive k of one row
        const half4 a0 = *(const device half4*)(arow + loop_k);
        const half4 a1 = *(const device half4*)(arow + loop_k + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;          // k tile 0..3
            const short sy = lr0 / 8;                  // W row tile 0..7
            const short lx = lr0 % 8;                  // W row within tile
            const short ly = i % 8;                    // k within tile
            sa[64 * (8 * sx + sy) + 8 * ly + lx] = tmp[i];
        }
        {
            const short sx = tiitg % 4;                // k tile
            const short sy = lr1 / 8;                  // A row tile 0..3
            const short ly = lr1 % 8;
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * (4 * sx + sy) + 8 * ly);
            dst[0] = a0; dst[1] = a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < NK / 8; ++ik) {
            for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8);
            for (short i = 0; i < 2; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8);
            for (short i = 0; i < 8; ++i) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    // epilogue: float -> half via threadgroup memory (per SIMD group 16 x 32 floats)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* stage = (threadgroup float*)shmem + sgitg * (16 * 32);
    for (short i = 0; i < 8; ++i) simdgroup_store(mc[i], stage + 8 * (i % 4) + 32 * 8 * (i / 4), 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = r1 + 16 * (sgitg / 2), col0 = r0 + 32 * (sgitg % 2);
    const ushort lane = tiitg % 32;
    for (short e = lane; e < 16 * 32; e += 32) {
        const short rr = e / 32, cc = e % 32;
        const uint row = row0 + rr, col = col0 + cc;
        if (row < M && col < N) C[(size_t)row * N + col] = (half)stage[rr * 32 + cc];
    }
}
// Fused Q4_0-dequant GEMM: C[M x N] = A[M x K] * W^T, where W (N x K) is
// stored as Q4_0 blocks (18B: fp16 d + 16B nibbles, 32 values/block),
// row-major: Wq[n*nb + b], nb = K/32. K must be a multiple of 32.
// The 32-wide K-step aligns exactly with one block per (n, k-tile), so the
// dequantized B tile costs nb/32 x 18B of traffic -- 16x less than fp16.
kernel void sgemm_q4(
const device float* A   [[buffer(0)]],
const device uchar* Wq  [[buffer(1)]],
device float*       C   [[buffer(2)]],
constant uint&      M   [[buffer(3)]],
constant uint&      N   [[buffer(4)]],
constant uint&      K   [[buffer(5)]],
uint2 tgp [[threadgroup_position_in_grid]],
uint2 tpt [[thread_position_in_threadgroup]])
{
const uint nb = K >> 5;
threadgroup float As[32][64];   // A tile: k x m
threadgroup float Ws[32][64];   // W tile: k x n (dequantized fp32)
const uint li = tpt.y * 16 + tpt.x;
const uint r0 = tgp.y * 64 + tpt.y * 4;
const uint c0 = tgp.x * 64 + tpt.x * 4;
float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
for (uint kt = 0; kt < K; kt += 32) {
for (uint j = 0; j < 8; ++j) {          // As: 2048 floats / 256 thr
const uint idx = li + 256 * j;      // lanes coalesced
const uint kr = idx >> 6, mc = idx & 63;
const uint gm = tgp.y * 64 + mc;
As[kr][mc] = (gm < M && kt + kr < K)
? A[(size_t)gm * K + kt + kr] : 0.0f;
}
if (li < 64) {                          // one Q4 block per n-column
const uint n = tgp.x * 64 + li;
if (n < N) {
const device uchar* blk =
Wq + ((size_t)n * nb + (kt >> 5)) * 18;
const float d = (float)(*(device const half*)blk);
for (uint j = 0; j < 16; ++j) {
const uint q = blk[2 + j];
Ws[2 * j][li]     = (float)((int)(q & 0xF) - 8) * d;
Ws[2 * j + 1][li] = (float)((int)(q >> 4) - 8) * d;
}
} else {
for (uint k2 = 0; k2 < 32; ++k2) Ws[k2][li] = 0.0f;
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint k = 0; k < 32; ++k) {
const float4 wv = *(threadgroup float4*)&Ws[k][tpt.x * 4];
a0 = fma(As[k][tpt.y * 4 + 0], wv, a0);
a1 = fma(As[k][tpt.y * 4 + 1], wv, a1);
a2 = fma(As[k][tpt.y * 4 + 2], wv, a2);
a3 = fma(As[k][tpt.y * 4 + 3], wv, a3);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (r0 + 3 < M && c0 + 3 < N) {
*(device float4*)&C[(r0 + 0) * N + c0] = a0;
*(device float4*)&C[(r0 + 1) * N + c0] = a1;
*(device float4*)&C[(r0 + 2) * N + c0] = a2;
*(device float4*)&C[(r0 + 3) * N + c0] = a3;
} else {
for (uint r = 0; r < 4 && r0 + r < M; ++r)
for (uint j = 0; j < 4 && c0 + j < N; ++j) {
const float4 av = r == 0 ? a0 : r == 1 ? a1 : r == 2 ? a2 : a3;
C[(r0 + r) * N + c0 + j] = av[j];
}
}
}

// Small-M Q4 GEMM (M <= 32; 2026-09-07): the 64x64-tile kernel above pads a
// 21-row prompt to 64 rows and launches only N/64 threadgroups (333 ms for
// a 21-token TinyLlama turn). Here a threadgroup owns 32 output columns;
// lane n of simdgroup s streams column n's Q4 blocks b = s, s+8, ... once,
// dequantizes into registers and dots them against every A row (A rows are
// broadcast loads within the simdgroup), keeping M accumulators in
// registers; the 8 K-slices reduce through threadgroup memory in chunks of
// 16 rows. N/32 threadgroups keep the GPU busy at small M.
kernel void sgemm_q4_small(
const device float* A   [[buffer(0)]],
const device uchar* Wq  [[buffer(1)]],
device float*       C   [[buffer(2)]],
constant uint&      M   [[buffer(3)]],
constant uint&      N   [[buffer(4)]],
constant uint&      K   [[buffer(5)]],
uint tg [[threadgroup_position_in_grid]],
uint sg [[simdgroup_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint tid [[thread_index_in_threadgroup]])
{
const uint nb = K >> 5;
const uint n = tg * 32 + lane;
threadgroup float red[8][32][16];
float acc[32];
for (uint m = 0; m < 32; ++m) acc[m] = 0.0f;
if (n < N) {
for (uint b = sg; b < nb; b += 8) {
const device uchar* blk = Wq + ((size_t)n * nb + b) * 18;
const float d = (float)(*(device const half*)blk);
float4 w[8];
for (uint j = 0; j < 8; ++j) {          // 4 bytes -> 8 weights (even/odd interleaved)
const uint q0 = blk[2 + 2 * j], q1 = blk[2 + 2 * j + 1];
w[j] = float4((float)((int)(q0 & 0xF) - 8), (float)((int)(q0 >> 4) - 8),
              (float)((int)(q1 & 0xF) - 8), (float)((int)(q1 >> 4) - 8)) * d;
}
const device float4* a = (const device float4*)(A + b * 32);
const uint K4 = K >> 2;
for (uint m = 0; m < 32; ++m) {
if (m >= M) break;
const device float4* am = a + (size_t)m * K4;
float s4 = 0.0f;
for (uint j = 0; j < 8; ++j) s4 += dot(w[j], am[j]);
acc[m] += s4;
}
}
}
for (uint m0 = 0; m0 < M; m0 += 16) {
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0; i < 16; ++i) red[sg][lane][i] = acc[m0 + i];
threadgroup_barrier(mem_flags::mem_threadgroup);
// 32 columns x 16 rows = 512 sums over 8 slices, 2 per thread
for (uint t = tid; t < 512; t += 256) {
const uint cn = t & 31, mi = t >> 5, m = m0 + mi;
if (m < M && tg * 32 + cn < N) {
float v = 0.0f;
for (uint s2 = 0; s2 < 8; ++s2) v += red[s2][cn][mi];
C[(size_t)m * N + tg * 32 + cn] = v;
}
}
}
}

// Q4_0 blob -> dense fp32 rows (N x K), one thread per (row, block): the
// scratch for the MPS-GEMM prefill path (2026-09-07). Memory-bound: 18 B
// in, 128 B out per thread.
kernel void q4_dequant_f32(
const device uchar* Wq  [[buffer(0)]],
device float*       W   [[buffer(1)]],
constant uint&      N   [[buffer(2)]],
constant uint&      K   [[buffer(3)]],
constant uint&      shift [[buffer(4)]],
uint gid [[thread_position_in_grid]])
{
const uint nb = K >> 5;
if (gid >= N * nb) return;
// blocks may start at any byte offset in the .tmq mapping: the buffer is
// bound 16-byte aligned, `shift` carries the remainder, and the fp16 scale
// is assembled from two bytes (a 2-byte load at an odd address is undefined).
const device uchar* blk = Wq + shift + (size_t)gid * 18;
const ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
const float d = (float)as_type<half>(dbits);
device float* out = W + (size_t)gid * 32;     // row n = gid / nb, block b = gid % nb -> offset n*K + b*32
for (uint j = 0; j < 16; ++j) {
const uint q = blk[2 + j];
out[2 * j]     = (float)((int)(q & 0xF) - 8) * d;
out[2 * j + 1] = (float)((int)(q >> 4) - 8) * d;
}
}

kernel void q4_dequant_f16(
const device uchar* Wq  [[buffer(0)]],
device half*        W   [[buffer(1)]],
constant uint&      N   [[buffer(2)]],
constant uint&      K   [[buffer(3)]],
constant uint&      shift [[buffer(4)]],
uint gid [[thread_position_in_grid]])
{
const uint nb = K >> 5;
if (gid >= N * nb) return;
// blocks may start at any byte offset in the .tmq mapping: the buffer is
// bound 16-byte aligned, `shift` carries the remainder, and the fp16 scale
// is assembled from two bytes (a 2-byte load at an odd address is undefined).
const device uchar* blk = Wq + shift + (size_t)gid * 18;
const ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
const float d = (float)as_type<half>(dbits);
device half* out = W + (size_t)gid * 32;
for (uint j = 0; j < 16; ++j) {
const uint q = blk[2 + j];
out[2 * j]     = (half)((float)((int)(q & 0xF) - 8) * d);
out[2 * j + 1] = (half)((float)((int)(q >> 4) - 8) * d);
}
}
)MET";
}

// Small-M Q4 GEMM pipeline (nil when the kernel failed to build). OFF by
// default: measured correct (rel 1.2e-7 vs a double reference, better than
// the tiled kernel's 1.6e-6) but SLOWER — 375 vs 226 ms for a 21-token
// TinyLlama turn — because its inner loop re-reads the activation rows from
// memory for every weight block; the tiled kernel's threadgroup staging
// wins even at 33% row utilization. TM_Q4_SMALL=1 enables it for A/B; the
// lever left on the table is staging A per K-chunk in threadgroup memory.
inline id<MTLComputePipelineState> q4_small_pso() {
    static __strong id<MTLComputePipelineState> pso = nil;
    static bool tried = false;
    if (!tried) {
        tried = true;
        id<MTLFunction> f = [libq4() newFunctionWithName:@"sgemm_q4_small"];
        NSError* err = nil;
        if (f) pso = [dev() newComputePipelineStateWithFunction:f error:&err];
        if (!pso) std::fprintf(stderr, "[metal] sgemm_q4_small unavailable: %s\n",
                               err ? err.localizedDescription.UTF8String : "no function");
    }
    return pso;
}
inline bool q4_small_off() {
    static const bool off = [] { const char* e = std::getenv("TM_Q4_SMALL"); return !(e && e[0] == '1'); }();
    return off;
}

// MPS-GEMM prefill (2026-09-07): dequantize the Q4 blob into a per-call
// fp32 scratch on the GPU, then run MPSMatrixMultiplication on it. On this
// M1 the tiled Q4 kernel above reaches ~550 GFLOP/s; MPS reaches 1.4
// TFLOP/s on the same shapes (T=2000: 7 GEMMs 120 ms vs the whole layer's
// ~370 ms before). Scratch is the largest single tensor (46 MB fp32), never
// a resident copy. TM_PREFILL_MPS=0 restores the tiled kernel.
inline bool prefill_mps_on() {
    static const bool on = [] { const char* e = std::getenv("TM_PREFILL_MPS"); return !(e && e[0] == '0'); }();
    return on;
}
inline id<MTLComputePipelineState> q4_dequant_pso() {
    static __strong id<MTLComputePipelineState> pso = nil;
    static bool tried = false;
    if (!tried) {
        tried = true;
        id<MTLFunction> f = [libq4() newFunctionWithName:@"q4_dequant_f32"];
        NSError* err = nil;
        if (f) pso = [dev() newComputePipelineStateWithFunction:f error:&err];
        if (!pso) std::fprintf(stderr, "[metal] q4_dequant_f32 unavailable: %s\n",
                               err ? err.localizedDescription.UTF8String : "no function");
    }
    return pso;
}
inline __strong id<MTLBuffer>& dq_scratch() { static id<MTLBuffer> b = nil; return b; }
// Zero-copy view of a host activation matrix (2026-09-07): page-aligned
// host memory (every large malloc on macOS) is handed to the GPU as a
// no-copy shared buffer, length rounded up to the page (malloc's large
// allocations are page-granular). Saves the memcpy in/out per GEMM — ~5 GB
// of copies per 2000-token TinyLlama prefill. Not cached: the view is
// created per call (tens of us) so a freed/reallocated vector can never be
// aliased by a stale buffer; falls back to the copy path when unaligned.
inline id<MTLBuffer> host_view(const void* p, size_t bytes) {
    static const bool off = [] { const char* e = std::getenv("TM_GPU_ZEROCOPY"); return e && e[0] == '0'; }();
    if (off) return nil;
    const size_t pg = (size_t)getpagesize();
    if (((uintptr_t)p % pg) != 0) return nil;
    const size_t len = (bytes + pg - 1) / pg * pg;
    return [dev() newBufferWithBytesNoCopy:(void*)p length:len options:MTLResourceStorageModeShared deallocator:nil];
}
// fp16 variant (TM_PREFILL_MPS_F16=1): weights dequantized to half, A
// converted to half on the GPU, MPS half GEMM (1.9 vs 1.4 TFLOP/s measured
// on TinyLlama shapes), C converted back. Opt-in: fp16 activations cost
// ~1e-3 relative per GEMM output, i.e. quant-class rather than exact.
// Policy: fp16 for M >= 1024 (2000 tokens: 4.96 -> 4.49 s; at 512 the
// conversions cost more than they save), TM_PREFILL_MPS_F16=1/0 forces.
// Gate: strict Q4 oracle 9.1e-3 (bar 0.02), greedy tokens identical to fp32.
inline bool prefill_mps_f16(unsigned M) {
    static const int mode = [] { const char* e = std::getenv("TM_PREFILL_MPS_F16"); return e && *e ? (e[0] == '1' ? 1 : 0) : -1; }();
    return mode < 0 ? M >= 1024 : mode == 1;
}
inline id<MTLComputePipelineState> pso_named(id<MTLLibrary> lib_, const char* name) {
    static std::map<std::string, __strong id<MTLComputePipelineState>> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    id<MTLFunction> f = [lib_ newFunctionWithName:[NSString stringWithUTF8String:name]];
    NSError* err = nil;
    id<MTLComputePipelineState> p = f ? [dev() newComputePipelineStateWithFunction:f error:&err] : nil;
    if (!p) std::fprintf(stderr, "[metal] pipeline %s unavailable\n", name);
    cache[name] = p;
    return p;
}
inline __strong id<MTLBuffer>& a16_scratch() { static id<MTLBuffer> b = nil; return b; }
inline __strong id<MTLBuffer>& c16_scratch() { static id<MTLBuffer> b = nil; return b; }
// Encodes dequant(blob at woff, N x K) + C(M x N) = A(M x K) . W^T into cb.
// a_half / c_half (layer-stack use): the A buffer already holds half and/or
// C must be left as half in bc — no conversion passes. Both imply the fp16
// GEMM regardless of M.
inline bool encode_mps_q4_gemm(id<MTLCommandBuffer> cb, id<MTLBuffer> bw, size_t woff,
                               id<MTLBuffer> ba, id<MTLBuffer> bc, size_t coff,
                               unsigned M, unsigned N, unsigned K, bool a_half = false, bool c_half = false) {
    const bool f16 = a_half || c_half || prefill_mps_f16(M);
    const size_t es = f16 ? 2 : 4;
    id<MTLComputePipelineState> pso = f16 ? pso_named(libq4(), "q4_dequant_f16") : q4_dequant_pso();
    if (!pso) return false;
    id<MTLBuffer> W = grow(dq_scratch(), es * (size_t)N * K);
    if (!W) return false;
    id<MTLBuffer> A = ba, C = bc; size_t cofs = coff;
    id<MTLComputePipelineState> pcvt_in = nil, pcvt_out = nil;
    if (f16) {
        if (!a_half) { pcvt_in = pso_named(lib(), "cvt_f32_f16"); A = grow(a16_scratch(), 2 * (size_t)M * K); if (!(pcvt_in && A)) return false; }
        if (!c_half) { pcvt_out = pso_named(lib(), "cvt_f16_f32"); C = grow(c16_scratch(), 2 * (size_t)M * N); if (!(pcvt_out && C)) return false; cofs = 0; }
    }
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    const unsigned shift = (unsigned)(woff & 15);
    [e setBuffer:bw offset:(woff & ~(size_t)15) atIndex:0];
    [e setBuffer:W offset:0 atIndex:1];
    [e setBytes:&N length:4 atIndex:2];
    [e setBytes:&K length:4 atIndex:3];
    [e setBytes:&shift length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake((size_t)N * (K / 32), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (f16 && !a_half) {
        const unsigned na = M * K;
        [e setComputePipelineState:pcvt_in];
        [e setBuffer:ba offset:0 atIndex:0]; [e setBuffer:A offset:0 atIndex:1]; [e setBytes:&na length:4 atIndex:2];
        [e dispatchThreads:MTLSizeMake(na, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [e endEncoding];
    struct Key { unsigned M, N, K; bool f16; };
    struct Ent { Key k; MPSMatrixMultiplication* mm; MPSMatrixDescriptor *da, *dw, *dc; };
    static std::vector<Ent> cache;
    Ent* ent = nullptr;
    for (auto& c : cache) if (c.k.M == M && c.k.N == N && c.k.K == K && c.k.f16 == f16) { ent = &c; break; }
    if (!ent) {
        if (cache.size() > 256) cache.clear();
        const MPSDataType dt = f16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        Ent c{{M, N, K, f16}, nil, nil, nil, nil};
        c.mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev() transposeLeft:NO transposeRight:YES
                                                    resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
        c.da = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * es dataType:dt];
        c.dw = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K * es dataType:dt];
        c.dc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * es dataType:dt];
        cache.push_back(c); ent = &cache.back();
    }
    MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:ent->da];
    MPSMatrix* mW = [[MPSMatrix alloc] initWithBuffer:W descriptor:ent->dw];
    MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C offset:cofs descriptor:ent->dc];
    [ent->mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mW resultMatrix:mC];
    if (f16 && !c_half) {
        const unsigned nc = M * N;
        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        [e2 setComputePipelineState:pcvt_out];
        [e2 setBuffer:C offset:0 atIndex:0]; [e2 setBuffer:bc offset:coff atIndex:1]; [e2 setBytes:&nc length:4 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(nc, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e2 endEncoding];
    }
    return true;
}
// C (half) = A (half) . W^T straight from the Q4 blocks (q4_mm_tile): no
// dequant pass, no MPS. Requires half operands and K % 32 == 0.
// variant 1 = 32x64 output tile, 2 = 64x64, 0 = pick by shape. The 64-row
// tile computes 2.0 multiplies per threadgroup-memory load against 1.33 and
// wins whenever it is fed, but it wastes half its accumulators at M <= 32
// (isolated GFLOP/s at M=32, 32x64 vs 64x64: gate 1436 vs 712, down 1343 vs
// 724) and the two are a tie at M=64. In the model the wide tile is the
// better choice from 128 rows up even where the isolated numbers favour the
// narrow one at N <= 2048 (T=128: 801 t/s all-wide vs 772 t/s mixed), so the
// rule stays coarse.
inline int q4_mm_variant(unsigned M, unsigned) { return M <= 64 ? 1 : 2; }
inline bool encode_q4_mm_tile(id<MTLCommandBuffer> cb, id<MTLBuffer> bw, size_t woff,
                              id<MTLBuffer> ba, id<MTLBuffer> bc, size_t coff,
                              unsigned M, unsigned N, unsigned K, int variant = 0) {
    if (K % 32) return false;
    if (variant < 1 || variant > 3) variant = q4_mm_variant(M, N);
    if (variant == 3 && K % 64) variant = 2;
    id<MTLComputePipelineState> pso = pso_named(libq4(), variant == 3 ? "q4_mm_tile3" : variant == 2 ? "q4_mm_tile2" : "q4_mm_tile");
    if (!pso) return false;
    if (std::getenv("TM_Q4MM_TRACE")) {
        static std::vector<std::array<unsigned, 3>> seen;
        std::array<unsigned, 3> k{M, N, K};
        if (std::find(seen.begin(), seen.end(), k) == seen.end()) {
            seen.push_back(k);
            std::fprintf(stderr, "[q4mm] M=%u N=%u K=%u variant=%d woff&15=%u\n", M, N, K, variant, (unsigned)(woff & 15));
        }
    }
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    const unsigned shift = (unsigned)(woff & 15);
    [e setBuffer:ba offset:0 atIndex:0];
    [e setBuffer:bw offset:(woff & ~(size_t)15) atIndex:1];
    [e setBuffer:bc offset:coff atIndex:2];
    [e setBytes:&M length:4 atIndex:3];
    [e setBytes:&N length:4 atIndex:4];
    [e setBytes:&K length:4 atIndex:5];
    [e setBytes:&shift length:4 atIndex:6];
    [e setThreadgroupMemoryLength:variant == 3 ? 16384 : variant == 2 ? 9216 : 8192 atIndex:0];
    const unsigned mtile = variant == 1 ? 32u : 64u;
    [e dispatchThreadgroups:MTLSizeMake((M + mtile - 1) / mtile, (N + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [e endEncoding];
    return true;
}
inline bool ensure_init() {
    if (tried()) return dev() != nil;
    tried() = true;
    @autoreleasepool {
        dev() = MTLCreateSystemDefaultDevice();
        if (!dev()) return false;
        q() = [dev() newCommandQueue];
        if (!q()) { dev() = nil; return false; }
        NSError* err = nil;
        lib() = [dev() newLibraryWithSource:kMetalSrc() options:nil error:&err];
        if (!lib()) {
            std::fprintf(stderr, "[metal] library compile failed: %s\n",
                         err.localizedDescription.UTF8String);
            dev() = nil;
            return false;
        }
        libq4() = [dev() newLibraryWithSource:kMetalSrcQ4() options:nil error:&err];
        if (!libq4()) {
            std::fprintf(stderr, "[metal] q4 library compile failed: %s\n",
                         err.localizedDescription.UTF8String);
            dev() = nil;
            return false;
        }
    }
    return true;
}
}  // namespace detail

// Row-major C[M x N] = A[M x K] * B[K x N]. Returns false if the GPU path
// was unavailable (caller then falls back to cblas).
inline bool sgemm(const float* A, const float* B, float* C,
                  unsigned M, unsigned N, unsigned K) {
    if (!detail::ensure_init()) { ++fallback_count(); return false; }
    @autoreleasepool {
        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> ba = detail::grow(bu.a, sizeof(float) * M * K);
        id<MTLBuffer> bb = detail::grow(bu.b, sizeof(float) * K * N);
        id<MTLBuffer> bc = detail::grow(bu.c, sizeof(float) * M * N);
        if (!(ba && bb && bc)) { ++fallback_count(); return false; }
        memcpy(ba.contents, A, sizeof(float) * M * K);
        memcpy(bb.contents, B, sizeof(float) * K * N);

        // Apple-tuned GEMM via MPS (spec: "sgemm (or MPSGraph GEMM)").
        MPSMatrixDescriptor* da = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K rowBytes:K * 4
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* db = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N rowBytes:N * 4
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* dc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N rowBytes:N * 4
            dataType:MPSDataTypeFloat32];
        MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:ba descriptor:da];
        MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:bb descriptor:db];
        MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:bc descriptor:dc];
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:detail::dev() transposeLeft:NO transposeRight:NO
            resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
        if (!(mm)) { ++fallback_count(); return false; }
        id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb
                    resultMatrix:mc];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ++fallback_count();
            return false;
        }
        memcpy(C, bc.contents, sizeof(float) * M * N);
    }
    return true;
}

// fp16 GEMM: weights (B) converted once and cached on the GPU (keyed by the
// B pointer + size — weights are static per layer); activations converted on
// GPU; one command buffer per call (convert A -> gemm -> convert C back).
// Measured motivation: fp32 GPU GEMM is ~190-250 GFLOPS on M1 (loses to
// AMX), fp16 roughly doubles math rate and halves traffic.
inline bool sgemm_f16(const float* A, const float* B, float* C,
                      unsigned M, unsigned N, unsigned K) {
    if (!detail::ensure_init()) { ++fallback_count(); return false; }
    @autoreleasepool {
        static std::vector<std::pair<std::pair<const void*, size_t>,
                                     __strong id<MTLBuffer>>> f16_cache;
        // per-(M,N,K) GPU object cache: PSOs, MPS objects, constant buffers
        struct ObjCache {
            unsigned M, N, K;
            __strong id<MTLComputePipelineState> pc, pu;
            __strong MPSMatrixMultiplication *mm;
            __strong id<MTLBuffer> pna, pnc;
        };
        static std::vector<ObjCache> objs;
        ObjCache* oc = nullptr;
        for (auto& o : objs)
            if (o.M == M && o.N == N && o.K == K) { oc = &o; break; }
        if (!oc) {
            objs.push_back({M, N, K, nil, nil, nil, nil, nil});
            oc = &objs.back();
        }
        const size_t nb = (size_t)K * N;

        // weights -> cached fp16 GPU buffer
        __strong id<MTLBuffer> bb16;
        for (auto& e : f16_cache)
            if (e.first == std::make_pair(B, nb)) { bb16 = e.second; break; }
        if (!bb16) {
            id<MTLBuffer> bf32 = [detail::dev()
                newBufferWithBytes:B length:nb * 4
                options:MTLResourceStorageModeShared];
            bb16 = [detail::dev() newBufferWithLength:nb * 2
                options:MTLResourceStorageModeShared];
            if (!(bf32 && bb16)) { ++fallback_count(); return false; }
            id<MTLFunction> f = [detail::lib() newFunctionWithName:@"cvt_f32_f16"];
            id<MTLComputePipelineState> p = [detail::dev()
                newComputePipelineStateWithFunction:f error:nil];
            id<MTLBuffer> pn = [detail::dev() newBufferWithBytes:&nb
                length:4 options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:bf32 offset:0 atIndex:0];
            [e setBuffer:bb16 offset:0 atIndex:1];
            [e setBuffer:pn offset:0 atIndex:2];
            [e dispatchThreads:MTLSizeMake(nb, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                ++fallback_count();
                return false;
            }
            f16_cache.push_back({{B, nb}, bb16});
        }

        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> ba32 = detail::grow(bu.a, sizeof(float) * M * K);
        id<MTLBuffer> ba16 = detail::grow(bu.b, 2ull * M * K);
        id<MTLBuffer> bc16 = detail::grow(bu.c, 2ull * M * N);
        id<MTLBuffer> bc32 = detail::grow(bu.a, sizeof(float) * M * N);
        if (!(ba32 && ba16 && bc16 && bc32)) { ++fallback_count(); return false; }
        memcpy(ba32.contents, A, sizeof(float) * M * K);

        MPSMatrix* ma32 = [[MPSMatrix alloc] initWithBuffer:ba32
            descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M
                columns:K rowBytes:K * 4 dataType:MPSDataTypeFloat32]];
        MPSMatrix* ma16 = [[MPSMatrix alloc] initWithBuffer:ba16
            descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M
                columns:K rowBytes:K * 2 dataType:MPSDataTypeFloat16]];
        MPSMatrix* mb16 = [[MPSMatrix alloc] initWithBuffer:bb16
            descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:K
                columns:N rowBytes:N * 2 dataType:MPSDataTypeFloat16]];
        MPSMatrix* mc16 = [[MPSMatrix alloc] initWithBuffer:bc16
            descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M
                columns:N rowBytes:N * 2 dataType:MPSDataTypeFloat16]];
        MPSMatrix* mc32 = [[MPSMatrix alloc] initWithBuffer:bc32
            descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M
                columns:N rowBytes:N * 4 dataType:MPSDataTypeFloat32]];
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:detail::dev() transposeLeft:NO transposeRight:NO
            resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
        if (!(ma32 && ma16 && mb16 && mc16 && mc32 && mm)) {
            ++fallback_count();
            return false;
        }
        id<MTLFunction> fc = [detail::lib() newFunctionWithName:@"cvt_f32_f16"];
        id<MTLFunction> fu = [detail::lib() newFunctionWithName:@"cvt_f16_f32"];
        if (!oc->pc) {
            oc->pc = [detail::dev()
                newComputePipelineStateWithFunction:fc error:nil];
            oc->pu = [detail::dev()
                newComputePipelineStateWithFunction:fu error:nil];
            oc->mm = mm;
            const uint na0 = M * K, nc0 = M * N;
            oc->pna = [detail::dev() newBufferWithBytes:&na0 length:4
                options:MTLResourceStorageModeShared];
            oc->pnc = [detail::dev() newBufferWithBytes:&nc0 length:4
                options:MTLResourceStorageModeShared];
        }
        id<MTLComputePipelineState> pc = oc->pc;
        id<MTLComputePipelineState> pu = oc->pu;
        id<MTLBuffer> pna = oc->pna;
        id<MTLBuffer> pnc = oc->pnc;
        if (!(pc && pu)) { ++fallback_count(); return false; }
        const uint na = M * K, nc = M * N;
        id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pc];
        [e setBuffer:ba32 offset:0 atIndex:0];
        [e setBuffer:ba16 offset:0 atIndex:1];
        [e setBuffer:pna offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(na, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [mm encodeToCommandBuffer:cb leftMatrix:ma16 rightMatrix:mb16
                    resultMatrix:mc16];
        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        [e2 setComputePipelineState:pu];
        [e2 setBuffer:bc16 offset:0 atIndex:0];
        [e2 setBuffer:bc32 offset:0 atIndex:1];
        [e2 setBuffer:pnc offset:0 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(nc, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e2 endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ++fallback_count();
            return false;
        }
        memcpy(C, bc32.contents, sizeof(float) * M * N);
    }
    return true;
}

// Fused Q4_0-dequant GEMM: C[M x N] = A[M x K] * W^T. Wq points at the raw
// Q4_0 block blob (N rows x (K/32) blocks x 18B, row-major — exactly the
// .tmq tensor layout). K must be a multiple of 32. The blob is copied to
// the GPU once and cached by (Wq pointer, size); weights are static.
inline bool sgemm_q4(const float* A, const void* Wq, float* C,
                     unsigned M, unsigned N, unsigned K) {
    if (!detail::ensure_init()) { ++fallback_count(); return false; }
    if (K % 32) { ++fallback_count(); return false; }
    @autoreleasepool {
        static std::vector<std::pair<std::pair<const void*, size_t>,
                                     __strong id<MTLBuffer>>> wq_cache;
        static __strong id<MTLComputePipelineState> pso;
        const size_t bytes = (size_t)N * (K / 32) * 18;
        __strong id<MTLBuffer> bw;
        for (auto& e : wq_cache)
            if (e.first == std::make_pair(Wq, bytes)) { bw = e.second; break; }
        if (!bw) {
            bw = [detail::dev() newBufferWithBytes:Wq length:bytes
                options:MTLResourceStorageModeShared];
            if (!bw) { ++fallback_count(); return false; }
            wq_cache.push_back({{Wq, bytes}, bw});
        }
        if (!pso) {
            id<MTLFunction> f =
                [detail::libq4() newFunctionWithName:@"sgemm_q4"];
            pso = [detail::dev()
                newComputePipelineStateWithFunction:f error:nil];
            if (!pso) { ++fallback_count(); return false; }
        }
        id<MTLComputePipelineState> pso_small = detail::q4_small_pso();
        const bool small = M <= 32 && pso_small != nil && !detail::q4_small_off();
        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> ba = detail::host_view(A, sizeof(float) * M * K);
        id<MTLBuffer> bc = detail::host_view(C, sizeof(float) * M * N);
        const bool a_view = ba != nil, c_view = bc != nil;
        if (!ba) ba = detail::grow(bu.a, sizeof(float) * M * K);
        if (!bc) bc = detail::grow(bu.c, sizeof(float) * M * N);
        if (!(ba && bc)) { ++fallback_count(); return false; }
        if (!a_view) memcpy(ba.contents, A, sizeof(float) * M * K);
        const uint mu = M, nu = N, ku = K;
        id<MTLBuffer> bm = [detail::dev() newBufferWithBytes:&mu length:4
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn = [detail::dev() newBufferWithBytes:&nu length:4
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bk = [detail::dev() newBufferWithBytes:&ku length:4
            options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
        if (detail::prefill_mps_on() && detail::encode_mps_q4_gemm(cb, bw, 0, ba, bc, 0, M, N, K)) {
            // dequant + MPS GEMM encoded
        } else {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:small ? pso_small : pso];
        [e setBuffer:ba offset:0 atIndex:0];
        [e setBuffer:bw offset:0 atIndex:1];
        [e setBuffer:bc offset:0 atIndex:2];
        [e setBuffer:bm offset:0 atIndex:3];
        [e setBuffer:bn offset:0 atIndex:4];
        [e setBuffer:bk offset:0 atIndex:5];
        if (small)
            [e dispatchThreadgroups:MTLSizeMake((N + 31) / 32, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else
            [e dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [e endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ++fallback_count();
            return false;
        }
        if (!c_view) memcpy(C, bc.contents, sizeof(float) * M * N);
    }
    return true;
}

// M=1-native Q4 GEMV (decode): y[N] = x[K] * W^T. One thread per output;
// weights streamed exactly once; no smem, no tail path. Host side mirrors
// sgemm_q4 (blob cached by pointer; consts cached per (N, K)) minus the
// per-call constant-buffer allocations.
inline bool q4_gemv(const float* x, const void* Wq, float* y,
                    unsigned N, unsigned K) {
    const uint64_t woff64 = 0;   // copy-mode wrapper: weights contiguous
    if (!detail::ensure_init()) { ++fallback_count(); return false; }
    if (K % 32) { ++fallback_count(); return false; }
    @autoreleasepool {
        static std::vector<std::pair<std::pair<const void*, size_t>,
                                     __strong id<MTLBuffer>>> wq_cache;
        static std::vector<std::pair<std::pair<unsigned, unsigned>,
                                     std::pair<__strong id<MTLBuffer>,
                                               __strong id<MTLBuffer>>>> consts;
        static __strong id<MTLComputePipelineState> pso;
        const size_t bytes = (size_t)N * (K / 32) * 18;
        __strong id<MTLBuffer> bw;
        for (auto& e : wq_cache)
            if (e.first == std::make_pair(Wq, bytes)) { bw = e.second; break; }
        if (!bw) {
            bw = [detail::dev() newBufferWithBytes:Wq length:bytes
                options:MTLResourceStorageModeShared];
            if (!bw) { ++fallback_count(); return false; }
            wq_cache.push_back({{Wq, bytes}, bw});
        }
        __strong id<MTLBuffer> pn, pk;
        for (auto& c : consts)
            if (c.first == std::make_pair(N, K)) {
                pn = c.second.first; pk = c.second.second; break;
            }
        if (!pn) {
            pk = [detail::dev() newBufferWithBytes:&K length:4
                options:MTLResourceStorageModeShared];
            pn = [detail::dev() newBufferWithBytes:&N length:4
                options:MTLResourceStorageModeShared];
            if (!(pn && pk)) { ++fallback_count(); return false; }
            consts.push_back({{N, K}, {pn, pk}});
        }
        if (!pso) {
            id<MTLFunction> f = [detail::libq4() newFunctionWithName:@"q4_gemv"];
            pso = [detail::dev()
                newComputePipelineStateWithFunction:f error:nil];
            if (!pso) { ++fallback_count(); return false; }
        }
        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> bx = detail::grow(bu.a, sizeof(float) * K);
        id<MTLBuffer> by = detail::grow(bu.c, sizeof(float) * N);
        if (!(bx && by)) { ++fallback_count(); return false; }
        memcpy(bx.contents, x, sizeof(float) * K);
        id<MTLBuffer> pwoff = [detail::dev() newBufferWithBytes:&woff64
            length:8 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bx offset:0 atIndex:0];
        [e setBuffer:bw offset:0 atIndex:1];
        [e setBuffer:by offset:0 atIndex:2];
        [e setBuffer:pn offset:0 atIndex:3];
        [e setBuffer:pk offset:0 atIndex:4];
        [e setBuffer:pwoff offset:0 atIndex:5];
        [e dispatchThreads:MTLSizeMake(N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ++fallback_count();
            return false;
        }
        memcpy(y, by.contents, sizeof(float) * N);
    }
    return true;
}

// Fused multi-matrix prefill: C[M x Ntot] = A[M x K] * [W1; W2; ...]^T.
// One dispatch for q/k/v (or gate/up) instead of 2-3 — the per-call
// commit/wait handshake dominates at prefill chunk sizes otherwise.
// The concatenated weight blob is built once and cached by the part list.
inline bool q4_prefill(const float* A, const void* const* Wqs,
                       const unsigned* Ns, unsigned nparts, float* C,
                       unsigned M, unsigned K) {
    if (!detail::ensure_init()) { ++fallback_count(); return false; }
    if (K % 32 || !nparts) { ++fallback_count(); return false; }
    @autoreleasepool {
        unsigned Ntot = 0;
        for (unsigned i = 0; i < nparts; ++i) Ntot += Ns[i];
        static std::vector<std::pair<
            std::vector<std::pair<const void*, size_t>>,
            __strong id<MTLBuffer>>> blob_cache;
        static __strong id<MTLComputePipelineState> pso;
        std::vector<std::pair<const void*, size_t>> key(nparts);
        size_t bytes = 0;
        for (unsigned i = 0; i < nparts; ++i) {
            key[i] = {Wqs[i], (size_t)Ns[i] * (K / 32) * 18};
            bytes += key[i].second;
        }
        __strong id<MTLBuffer> bw;
        for (auto& e : blob_cache)
            if (e.first == key) { bw = e.second; break; }
        if (!bw) {
            bw = [detail::dev() newBufferWithLength:bytes
                options:MTLResourceStorageModeShared];
            if (!bw) { ++fallback_count(); return false; }
            std::size_t off = 0;
            for (unsigned i = 0; i < nparts; ++i) {
                memcpy((char*)bw.contents + off, Wqs[i], key[i].second);
                off += key[i].second;
            }
            blob_cache.push_back({key, bw});
        }
        static std::vector<std::tuple<unsigned, unsigned, unsigned,
            __strong id<MTLBuffer>, __strong id<MTLBuffer>,
            __strong id<MTLBuffer>>> consts;
        __strong id<MTLBuffer> pm, pn, pk;
        for (auto& c : consts)
            if (std::get<0>(c) == M && std::get<1>(c) == Ntot &&
                std::get<2>(c) == K) {
                pm = std::get<3>(c); pn = std::get<4>(c); pk = std::get<5>(c);
                break;
            }
        if (!pm) {
            pm = [detail::dev() newBufferWithBytes:&M length:4
                options:MTLResourceStorageModeShared];
            pn = [detail::dev() newBufferWithBytes:&Ntot length:4
                options:MTLResourceStorageModeShared];
            pk = [detail::dev() newBufferWithBytes:&K length:4
                options:MTLResourceStorageModeShared];
            consts.push_back({M, Ntot, K, pm, pn, pk});
        }
        if (!pso) {
            id<MTLFunction> f =
                [detail::libq4() newFunctionWithName:@"sgemm_q4"];
            pso = [detail::dev()
                newComputePipelineStateWithFunction:f error:nil];
            if (!pso) { ++fallback_count(); return false; }
        }
        id<MTLComputePipelineState> pso_small = detail::q4_small_pso();
        const bool small = M <= 32 && pso_small != nil && !detail::q4_small_off();
        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> ba = detail::host_view(A, sizeof(float) * M * K);
        id<MTLBuffer> bc = detail::host_view(C, sizeof(float) * M * Ntot);
        const bool a_view = ba != nil, c_view = bc != nil;
        if (!ba) ba = detail::grow(bu.a, sizeof(float) * M * K);
        if (!bc) bc = detail::grow(bu.c, sizeof(float) * M * Ntot);
        if (!(ba && bc)) { ++fallback_count(); return false; }
        if (!a_view) memcpy(ba.contents, A, sizeof(float) * M * K);
        id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
        bool mps_done = false;
        if (detail::prefill_mps_on()) {
            // The parts are row blocks of one (Ntot x K) blob, i.e. column
            // blocks of C: one dequant + GEMM over the whole blob.
            mps_done = detail::encode_mps_q4_gemm(cb, bw, 0, ba, bc, 0, M, Ntot, K);
        }
        if (!mps_done) {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:small ? pso_small : pso];
        [e setBuffer:ba offset:0 atIndex:0];
        [e setBuffer:bw offset:0 atIndex:1];
        [e setBuffer:bc offset:0 atIndex:2];
        [e setBuffer:pm offset:0 atIndex:3];
        [e setBuffer:pn offset:0 atIndex:4];
        [e setBuffer:pk offset:0 atIndex:5];
        if (small)
            [e dispatchThreadgroups:MTLSizeMake((Ntot + 31) / 32, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else
            [e dispatchThreadgroups:MTLSizeMake((Ntot + 63) / 64, (M + 63) / 64, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [e endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ++fallback_count();
            return false;
        }
        if (!c_view) memcpy(C, bc.contents, sizeof(float) * M * Ntot);
        ++prefill_count();
    }
    return true;
}

// ---- gpu2: per-token single-commit decode pipeline --------------------
// Every per-token op (rmsnorm, rope, attention, silu-mul, residual,
// Q4 GEMV) is a dispatch on persistent SLOT buffers; a whole token
// (L layers x ~10 ops) encodes into ONE command buffer = ONE
// commit/wait round trip. This removes the ~129 x 1.2 ms serialized
// round trips that capped GPU v1 at 6 t/s.
//
// Slots: caller-chosen integer ids -> grow-only shared MTLBuffers that
// persist across tokens (activations, KV caches, logits). Slot memory
// is written by slot_upload (host) or by kernels (chain within the
// command buffer); ordering within one command buffer is guaranteed.
using detail::dev;
using detail::q;
using detail::libq4;
using detail::ensure_init;

inline std::vector<__strong id<MTLBuffer>>& tslots() {
    static std::vector<__strong id<MTLBuffer>> s;
    return s;
}
inline __strong id<MTLComputeCommandEncoder>& tenc();
inline id<MTLBuffer> slot_buf(int sid, size_t bytes) {
    if (sid < 0) throw std::invalid_argument("metal: negative slot id");
    auto& s = tslots();
    if ((int)s.size() <= sid) s.resize((size_t)sid + 1);
    if (!s[sid] || [s[sid] length] < bytes) {
        if (s[sid] && tenc())
            throw std::runtime_error("metal: reserve packed slots before token encoding");
        if (!bytes || bytes > dev().maxBufferLength)
            throw std::runtime_error("metal: invalid slot buffer size");
        id<MTLBuffer> b = [dev() newBufferWithLength:bytes
                          options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("metal: slot allocation failed");
        s[sid] = b;
    }
    return s[sid];
}
inline __strong id<MTLCommandBuffer>& tcb() {
    static __strong id<MTLCommandBuffer> c; return c;
}
inline __strong id<MTLComputeCommandEncoder>& tenc() {
    static __strong id<MTLComputeCommandEncoder> e; return e;
}
inline __strong id<MTLLibrary>& libops() {
    static __strong id<MTLLibrary> l; return l;
}
inline id<MTLBuffer> const_u32(unsigned v) {
    static std::vector<std::pair<unsigned, __strong id<MTLBuffer>>> c;
    for (auto& e : c) if (e.first == v) return e.second;
    id<MTLBuffer> b = [dev() newBufferWithBytes:&v length:4
        options:MTLResourceStorageModeShared];
    c.push_back({v, b});
    return b;
}
inline id<MTLBuffer> const_i32(int v) {
    static std::vector<std::pair<int, __strong id<MTLBuffer>>> c;
    for (auto& e : c) if (e.first == v) return e.second;
    id<MTLBuffer> b = [dev() newBufferWithBytes:&v length:4
        options:MTLResourceStorageModeShared];
    c.push_back({v, b});
    return b;
}
inline id<MTLBuffer> const_f32(float v) {
    static std::vector<std::pair<float, __strong id<MTLBuffer>>> c;
    for (auto& e : c) if (e.first == v) return e.second;
    id<MTLBuffer> b = [dev() newBufferWithBytes:&v length:4
        options:MTLResourceStorageModeShared];
    c.push_back({v, b});
    return b;
}
// PSOs for the ops library, cached by kernel name.
inline id<MTLComputePipelineState> ops_pso(const char* name) {
    static std::vector<std::pair<std::string,
                                 __strong id<MTLComputePipelineState>>> c;
    for (auto& e : c) if (e.first == name) return e.second;
    id<MTLFunction> f = [libops() newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> p =
        f ? [dev() newComputePipelineStateWithFunction:f error:nil] : nil;
    c.push_back({name, p});
    return p;
}
// PSO for q4_gemv (lives in libq4), shared with q4_gemv().
inline __strong id<MTLComputePipelineState>& gemv_pso() {
    static __strong id<MTLComputePipelineState> p; return p;
}

inline NSString* kMetalSrcOps() {
return @R"MET(
#include <metal_stdlib>
using namespace metal;
// out = x * rmsinv(x) * w   (one row; 256-thread group strided)
// GPU-resident greedy decode (2026-09-07): argmax of the logits into a token
// slot and the Q4 embedding row of that token into x, so the next token's
// command buffer can be committed before the current one finishes.
kernel void argmax_f32(
const device float* x [[buffer(0)]], device uint* out [[buffer(1)]],
constant uint& n [[buffer(2)]], constant uint& idx [[buffer(3)]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float bv[8]; threadgroup uint bi[8];
    float best = -INFINITY; uint besti = 0;
    for (uint i = ti; i < n; i += 256) { const float v = x[i]; if (v > best) { best = v; besti = i; } }
    for (uint off = 16; off; off >>= 1) {
        const float ov = simd_shuffle_down(best, off); const uint oi = simd_shuffle_down(besti, off);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    if (lane == 0) { bv[sg] = best; bi[sg] = besti; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti == 0) {
        for (uint g = 1; g < 8; ++g) if (bv[g] > best || (bv[g] == best && bi[g] < besti)) { best = bv[g]; besti = bi[g]; }
        out[idx] = besti;
    }
}
// Temperature sampling as one pass: argmax over logits * inv_temp + Gumbel
// noise (Gumbel-max trick samples exactly from softmax(logits / T)). Noise is
// a hash of (seed, step, index): reproducible per seed, independent per
// token. inv_temp <= 0 degenerates to plain argmax.
inline uint tm_hash32(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
kernel void sample_gumbel_f32(
const device float* x [[buffer(0)]], device uint* out [[buffer(1)]],
constant uint& n [[buffer(2)]], constant uint& idx [[buffer(3)]],
constant float& inv_temp [[buffer(4)]], constant uint& seed [[buffer(5)]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float bv[8]; threadgroup uint bi[8];
    float best = -INFINITY; uint besti = 0;
    const uint base = tm_hash32(seed * 0x9e3779b9u + idx * 0x85ebca6bu + 0x1234567u);
    for (uint i = ti; i < n; i += 256) {
        float v = x[i];
        if (inv_temp > 0.0f) {
            const uint h = tm_hash32(base ^ (i * 0x27d4eb2fu + 0x165667b1u));
            const float u = ((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f);   // (0,1)
            v = v * inv_temp - log(-log(u));
        }
        if (v > best) { best = v; besti = i; }
    }
    for (uint off = 16; off; off >>= 1) {
        const float ov = simd_shuffle_down(best, off); const uint oi = simd_shuffle_down(besti, off);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    if (lane == 0) { bv[sg] = best; bi[sg] = besti; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti == 0) {
        for (uint g = 1; g < 8; ++g) if (bv[g] > best || (bv[g] == best && bi[g] < besti)) { best = bv[g]; besti = bi[g]; }
        out[idx] = besti;
    }
}
kernel void embed_q4_row(
const device uint* tok [[buffer(0)]], constant uint& idx [[buffer(1)]],
const device uchar* Wq [[buffer(2)]], constant uint64_t& woff [[buffer(3)]],
device float* x [[buffer(4)]], constant uint& D [[buffer(5)]],
uint gid [[thread_position_in_grid]])
{
    if (gid >= D / 2) return;
    const uint row = tok[idx];
    const uint nb = D >> 5, b = gid / 16, j = gid % 16;
    const device uchar* blk = Wq + woff + ((size_t)row * nb + b) * 18;
    const ushort d16 = ushort(blk[0]) | (ushort(blk[1]) << 8);
    const float d = float(as_type<half>(d16));
    const uchar q = blk[2 + j];
    x[b * 32 + 2 * j] = float(int(q & 15) - 8) * d;
    x[b * 32 + 2 * j + 1] = float(int(q >> 4) - 8) * d;
}
kernel void embed_h_row(
const device uint* tok [[buffer(0)]], constant uint& idx [[buffer(1)]],
const device half* W [[buffer(2)]], constant uint64_t& woff [[buffer(3)]],
device float* x [[buffer(4)]], constant uint& D [[buffer(5)]],
uint gid [[thread_position_in_grid]])
{
    if (gid >= D / 4) return;
    const uint row = tok[idx];
    const device half* Wt = (const device half*)((const device uchar*)W + woff);
    const half4 wv = *(const device half4*)(Wt + (size_t)row * D + gid * 4);
    *(device float4*)(x + gid * 4) = float4(wv);
}
kernel void rmsnorm_row(
const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
device float* out [[buffer(2)]], constant uint& D [[buffer(3)]],
constant float& eps [[buffer(4)]],
uint ti [[thread_index_in_threadgroup]],
uint tpg [[threads_per_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
// Two-level SIMD reduction (2026-09-07): the former thread-0 serial sum over
// 256 partials made this single-threadgroup kernel cost 36 us per dispatch,
// 1.6 ms per token over 44 calls.
threadgroup float ss[8];
float p = 0.0f;
for (uint i = ti; i < D; i += tpg) p += x[i] * x[i];
p = simd_sum(p);
if (lane == 0) ss[sg] = p;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
float s = (lane < tpg / 32) ? ss[lane] : 0.0f;
s = simd_sum(s);
if (lane == 0) ss[0] = 1.0f / sqrt(s / (float)D + eps);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv = ss[0];
for (uint i = ti; i < D; i += tpg) out[i] = x[i] * inv * w[i];
}
// debug probe: raw byte copy, used to test GPU-side reads of NoCopy
// weight mappings (TM_DEBUG_GPUW path in llama.h)
kernel void probe_copy(
const device uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
constant uint& n [[buffer(2)]], constant ulong& off [[buffer(3)]],
uint gid [[thread_position_in_grid]])
{
if (gid < n) dst[gid] = src[off + gid];
}
kernel void add_inplace(
device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
if (gid < n) a[gid] += b[gid];
}
kernel void silu_mul(
device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
if (gid < n) { float v = g[gid]; g[gid] = v / (1.0f + exp(-v)) * u[gid]; }
}
// RoPE in place over heads*(dh/2) pairs; pairs are (r[j], r[j+dh/2])
// within each head, angle theta^(-2j/dh) * pos.
kernel void rope_inplace(
device float* q [[buffer(0)]], constant uint& heads [[buffer(1)]],
constant uint& dh [[buffer(2)]], constant int& pos [[buffer(3)]],
constant float& theta [[buffer(4)]],
uint gid [[thread_position_in_grid]])
{
uint h = gid / (dh / 2), j = gid % (dh / 2);
if (h >= heads) return;
float inv = pow(theta, -2.0f * (float)j / (float)dh);
float c = cos((float)pos * inv), s = sin((float)pos * inv);
device float* r = q + (size_t)h * dh;
float a = r[j], b = r[j + dh / 2];
r[j] = a * c - b * s;
r[j + dh / 2] = b * c + a * s;
}
// Rotate the k part (read from the qkv activation slot) and scatter it
// straight into the per-layer KV cache at row pos.
template <typename CacheT>
inline void rope_k_cache_body(const device float* ksrc, device CacheT* cache,
                              uint KVH, uint dh, uint ctx, int pos, float theta, uint gid)
{
uint kv = gid / (dh / 2), j = gid % (dh / 2);
if (kv >= KVH) return;
float inv = pow(theta, -2.0f * (float)j / (float)dh);
float c = cos((float)pos * inv), s = sin((float)pos * inv);
const device float* r = ksrc + (size_t)kv * dh;
device CacheT* dst = cache + (size_t)kv * ctx * dh + (size_t)pos * dh;
float a = r[j], b = r[j + dh / 2];
dst[j] = (CacheT)(a * c - b * s);
dst[j + dh / 2] = (CacheT)(b * c + a * s);
}
kernel void rope_k_cache(
const device float* ksrc [[buffer(0)]],
device float* cache [[buffer(1)]],
constant uint& KVH [[buffer(2)]], constant uint& dh [[buffer(3)]],
constant uint& ctx [[buffer(4)]], constant int& pos [[buffer(5)]],
constant float& theta [[buffer(6)]],
uint gid [[thread_position_in_grid]])
{ rope_k_cache_body(ksrc, cache, KVH, dh, ctx, pos, theta, gid); }
kernel void rope_k_cache_h(
const device float* ksrc [[buffer(0)]],
device half* cache [[buffer(1)]],
constant uint& KVH [[buffer(2)]], constant uint& dh [[buffer(3)]],
constant uint& ctx [[buffer(4)]], constant int& pos [[buffer(5)]],
constant float& theta [[buffer(6)]],
uint gid [[thread_position_in_grid]])
{ rope_k_cache_body(ksrc, cache, KVH, dh, ctx, pos, theta, gid); }
kernel void v_to_cache(
const device float* v [[buffer(0)]], device float* cache [[buffer(1)]],
constant uint& KVH [[buffer(2)]], constant uint& dh [[buffer(3)]],
constant uint& ctx [[buffer(4)]], constant int& pos [[buffer(5)]],
uint gid [[thread_position_in_grid]])
{
uint kv = gid / dh, l = gid % dh;
if (kv >= KVH) return;
cache[(size_t)kv * ctx * dh + (size_t)pos * dh + l] = v[gid];
}
kernel void v_to_cache_h(
const device float* v [[buffer(0)]], device half* cache [[buffer(1)]],
constant uint& KVH [[buffer(2)]], constant uint& dh [[buffer(3)]],
constant uint& ctx [[buffer(4)]], constant int& pos [[buffer(5)]],
uint gid [[thread_position_in_grid]])
{
uint kv = gid / dh, l = gid % dh;
if (kv >= KVH) return;
cache[(size_t)kv * ctx * dh + (size_t)pos * dh + l] = (half)v[gid];
}
// One 128-thread group per query head. Four SIMD groups cooperatively read
// contiguous K rows; scores/probabilities stay in threadgroup memory. All
// arithmetic is fp32. No global score buffer or inter-dispatch barriers.
template <typename CacheT>
inline void attn_decode_fused_body(
const device float* q, const device CacheT* kcache, const device CacheT* vcache,
device float* out, uint dh, uint ctx, uint allow, float scale, uint REP,
threadgroup float* probs, threadgroup float* reductions, threadgroup float* partials,
uint h, uint ti, uint lane, uint sg)
{
    // reductions[8]: disjoint max/sum slots avoid reuse races; partials[128]
    const device float* qh = q + (size_t)h * dh;
    const device CacheT* kh = kcache + (size_t)(h / REP) * ctx * dh;
    const device CacheT* vh = vcache + (size_t)(h / REP) * ctx * dh;
    for (uint s = sg; s < allow; s += 4) {
        float dot = 0.0f;
        for (uint d = lane; d < dh; d += 32)
            dot += qh[d] * (float)kh[(size_t)s * dh + d];
        dot = simd_sum(dot);
        if (lane == 0) probs[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint s = ti; s < allow; s += 128) mx = max(mx, probs[s]);
    mx = simd_max(mx);
    if (lane == 0) reductions[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = max(max(reductions[0], reductions[1]), max(reductions[2], reductions[3]));
    float sum = 0.0f;
    for (uint s = ti; s < allow; s += 128) {
        float p = exp(probs[s] - mx);
        probs[s] = p;
        sum += p;
    }
    sum = simd_sum(sum);
    if (lane == 0) reductions[4 + sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / (reductions[4] + reductions[5] + reductions[6] + reductions[7]);
    for (uint s = ti; s < allow; s += 128) probs[s] *= inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Split the value reduction across all four SIMD groups. A thread owning
    // an entire dimension serializes allow loads and leaves half the group idle
    // at dh=64; this layout keeps adjacent lanes on adjacent value elements.
    for (uint base = 0; base < dh; base += 32) {
        const uint d = base + lane;
        float result = 0.0f;
        if (d < dh)
            for (uint s = sg; s < allow; s += 4)
                result += probs[s] * (float)vh[(size_t)s * dh + d];
        partials[ti] = result;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && d < dh)
            out[(size_t)h * dh + d] = partials[lane] + partials[32 + lane]
                                     + partials[64 + lane] + partials[96 + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void attn_decode_fused(
const device float* q [[buffer(0)]],
const device float* kcache [[buffer(1)]],
const device float* vcache [[buffer(2)]],
device float* out [[buffer(3)]],
constant uint& dh [[buffer(4)]], constant uint& ctx [[buffer(5)]],
constant uint& allow [[buffer(6)]], constant float& scale [[buffer(7)]],
constant uint& REP [[buffer(8)]],
threadgroup float* probs [[threadgroup(0)]],
uint h [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    threadgroup float partials[128];
    attn_decode_fused_body(q, kcache, vcache, out, dh, ctx, allow, scale, REP, probs, reductions, partials, h, ti, lane, sg);
}
kernel void attn_decode_fused_h(
const device float* q [[buffer(0)]],
const device half* kcache [[buffer(1)]],
const device half* vcache [[buffer(2)]],
device float* out [[buffer(3)]],
constant uint& dh [[buffer(4)]], constant uint& ctx [[buffer(5)]],
constant uint& allow [[buffer(6)]], constant float& scale [[buffer(7)]],
constant uint& REP [[buffer(8)]],
threadgroup float* probs [[threadgroup(0)]],
uint h [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    threadgroup float partials[128];
    attn_decode_fused_body(q, kcache, vcache, out, dh, ctx, allow, scale, REP, probs, reductions, partials, h, ti, lane, sg);
}
// scores[h][s] = dot(q_head_h, K_kv(s)) * scale, one thread per (h, s)
kernel void attn_scores(
const device float* q [[buffer(0)]],
const device float* kcache [[buffer(1)]],
device float* probs [[buffer(2)]],
constant uint& H [[buffer(3)]], constant uint& dh [[buffer(4)]],
constant uint& ctx [[buffer(5)]], constant int& allow [[buffer(6)]],
constant float& scale [[buffer(7)]], constant uint& REP [[buffer(8)]],
uint gid [[thread_position_in_grid]])
{
uint h = gid / (uint)allow, s = gid % (uint)allow;
if (h >= H) return;
uint kv = h / REP;
const device float* qh = q + (size_t)h * dh;
const device float* Kh = kcache + (size_t)kv * ctx * dh
                       + (size_t)s * dh;
float d = 0.0f;
for (uint i = 0; i < dh; ++i) d += qh[i] * Kh[i];
probs[(size_t)h * ctx + s] = d * scale;
}
// in-place per-head softmax over allow (one thread per head)
kernel void attn_softmax(
device float* probs [[buffer(0)]], constant uint& H [[buffer(1)]],
constant uint& ctx [[buffer(2)]], constant int& allow [[buffer(3)]],
uint gid [[thread_position_in_grid]])
{
if (gid >= H) return;
device float* p = probs + (size_t)gid * ctx;
float m = p[0];
for (int s = 1; s < allow; ++s) m = max(m, p[s]);
float z = 0.0f;
for (int s = 0; s < allow; ++s) { p[s] = exp(p[s] - m); z += p[s]; }
float inv = 1.0f / z;
for (int s = 0; s < allow; ++s) p[s] *= inv;
}
// out[h*dh + l] = sum_s probs[h][s] * V_kv(s, l)   (one thread per (h,l))
kernel void attn_pv(
const device float* probs [[buffer(0)]],
const device float* vcache [[buffer(1)]],
device float* out [[buffer(2)]],
constant uint& H [[buffer(3)]], constant uint& dh [[buffer(4)]],
constant uint& ctx [[buffer(5)]], constant int& allow [[buffer(6)]],
constant uint& REP [[buffer(7)]],
uint gid [[thread_position_in_grid]])
{
uint h = gid / dh, l = gid % dh;
if (h >= H) return;
uint kv = h / REP;
const device float* p = probs + (size_t)h * ctx;
const device float* Vh = vcache + (size_t)kv * ctx * dh + l;
float a = 0.0f;
for (int s = 0; s < allow; ++s) a += p[s] * Vh[(size_t)s * dh];
out[(size_t)h * dh + l] = a;
}
// Fully unrolled q.k for the common head sizes so all row loads issue
// before the FMA chain (a runtime-count loop serializes load latency).
template <uint DH4, typename K4>
inline float gqa_dot(const threadgroup float4* q4, const device K4* k4) {
    float4 a = 0.0f, b = 0.0f;
#pragma unroll
    for (uint j = 0; j < DH4; j += 2) { a += q4[j] * float4(k4[j]); b += q4[j + 1] * float4(k4[j + 1]); }
    const float4 c = a + b;
    return c.x + c.y + c.z + c.w;
}
template <typename K4>
inline float gqa_dot_any(const threadgroup float4* q4, const device K4* k4, uint dh4) {
    if (dh4 == 16) return gqa_dot<16>(q4, k4);
    if (dh4 == 32) return gqa_dot<32>(q4, k4);
    if (dh4 == 8) return gqa_dot<8>(q4, k4);
    float4 d4 = 0.0f;
    for (uint j = 0; j < dh4; ++j) d4 += q4[j] * float4(k4[j]);
    return d4.x + d4.y + d4.z + d4.w;
}
// GQA split-K decode attention (2026-09-07). One 128-thread group per
// (kv head, position split); every query head sharing the kv head is served
// from one pass over that split's K/V rows, 32 rows per tile (= SIMD width,
// so a SIMD group owns one head row and max/sum are simd reductions), online
// softmax per head, unnormalized partials + (max, sum) per head for
// attn_decode_combine. Nothing is staged but q and the tile probabilities:
// a lane walks its contiguous K row with an unrolled q.k, and for P.V each
// thread owns one value dimension for ALL heads (acc[8]), the 128/dh
// threads sharing a dimension taking interleaved rows, so a V element is
// loaded 128/dh times instead of REP times, coalesced, with the loads
// unrolled four deep. Two barriers per tile.
// Measured TinyLlama Q4 GPU decode at ctx 2000 (forced GPU path, quiet):
// attn_decode_fused 29.7 t/s -> 55.6 (ctx 64: 66.9 -> 68.4), greedy hash
// identical. Rejected on the way: staging K transposed + V in threadgroup
// memory with serial per-head max/sum loops (41 t/s: residency + serial
// chains), staging V only with simd reductions (49.9), hoisting the K row
// into a float4[32] register array (spilled, 43). The remaining ctx cost
// is ~0.15 ms/layer for 4 MB of fp32 K/V, ~26 GB/s; a half-precision cache
// is the next lever.
// P.V for one thread's dimension over a 32-row tile: all NV = 32/nsub value
// loads are issued up front (predicated past n) so a tile costs one memory
// latency instead of NV dependent ones.
// One 32-row half of a 64-key tile: rows s0 + sub + k*nsub, k < NV, with
// all NV value loads issued before the FMAs (register array of NV floats).
template <uint NV, typename CacheT>
inline void gqa_pv_half(const device CacheT* vb, threadgroup const float* P, thread float* acc,
                        uint s0, uint sub, uint nsub, uint n, uint dh, uint REP) {
    float v[NV];
#pragma unroll
    for (uint k = 0; k < NV; ++k) {
        const uint s = s0 + sub + k * nsub;
        v[k] = (s < n) ? (float)vb[(size_t)s * dh] : 0.0f;
    }
    for (uint r = 0; r < REP; ++r) {
        float a = 0.0f;
#pragma unroll
        for (uint k = 0; k < NV; ++k) a += P[r * 128 + s0 + sub + k * nsub] * v[k];
        acc[r] += a;
    }
}
template <uint NV, typename CacheT>
inline void gqa_pv(const device CacheT* vb, threadgroup const float* P, thread float* acc,
                   uint sub, uint nsub, uint n, uint dh, uint REP) {
    gqa_pv_half<NV / 4>(vb, P, acc, 0, sub, nsub, n, dh, REP);
    if (n > 32) gqa_pv_half<NV / 4>(vb, P, acc, 32, sub, nsub, n, dh, REP);
    if (n > 64) gqa_pv_half<NV / 4>(vb, P, acc, 64, sub, nsub, n, dh, REP);
    if (n > 96) gqa_pv_half<NV / 4>(vb, P, acc, 96, sub, nsub, n, dh, REP);
}
template <typename CacheT, typename Vec4T>
inline void attn_decode_gqa_body(
const device float* q, const device CacheT* kcache, const device CacheT* vcache,
device float* part, device float* stats, uint dh, uint ctx, uint allow, float scale,
uint REP, uint nsplit, uint chunk, uint H, threadgroup float* shm,
uint2 tg, uint ti, uint lane, uint sg)
{
    const uint TS = 128;  // keys per barrier pair
    const uint kv = tg.x, sp = tg.y;
    threadgroup float* Q  = shm;                 // [REP][dh], pre-scaled
    threadgroup float* P  = Q + REP * dh;        // [REP][TS]
    threadgroup float* R  = P + REP * TS;        // [128/dh - 1][REP][dh] reduction
    threadgroup float* M  = R + REP * (128 - dh);// [REP]
    threadgroup float* Lr = M + REP;             // [REP]
    threadgroup float* A  = Lr + REP;            // [REP]
    const device CacheT* kh = kcache + (size_t)kv * ctx * dh;
    const device CacheT* vh = vcache + (size_t)kv * ctx * dh;
    const uint h0 = kv * REP, npairs = REP * dh, dh4 = dh / 4;
    const uint s0 = sp * chunk, s1 = min(allow, s0 + chunk);
    const uint nsub = 128 / dh, d = ti % dh, sub = ti / dh;
    for (uint i = ti; i < npairs; i += 128) Q[i] = q[(size_t)h0 * dh + i] * scale;
    if (ti < REP) { M[ti] = -INFINITY; Lr[ti] = 0.0f; A[ti] = 0.0f; }
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = s0; base < s1; base += TS) {
        const uint n = min(TS, s1 - base);
        for (uint r = sg; r < REP; r += 4) {
            const threadgroup float4* q4 = (const threadgroup float4*)(Q + r * dh);
            float dots[4]; float mloc = -INFINITY;
            for (uint u = 0; u < 4; ++u) {
                const uint s = lane + 32 * u;
                dots[u] = (s < n) ? gqa_dot_any(q4, (const device Vec4T*)(kh + (size_t)(base + s) * dh), dh4) : -INFINITY;
                mloc = max(mloc, dots[u]);
            }
            const float mx = max(M[r], simd_max(mloc));
            float lsum = 0.0f;
            for (uint u = 0; u < 4; ++u) {
                const uint s = lane + 32 * u;
                const float p = (s < n) ? exp(dots[u] - mx) : 0.0f;
                P[r * TS + s] = p; lsum += p;
            }
            const float l = simd_sum(lsum);
            if (lane == 0) { A[r] = exp(M[r] - mx); Lr[r] = Lr[r] * A[r] + l; M[r] = mx; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < REP; ++r) acc[r] *= A[r];
        {
            const device CacheT* vb = vh + (size_t)base * dh + d;
            if (nsub == 2) gqa_pv<64>(vb, P, acc, sub, nsub, n, dh, REP);
            else if (nsub == 1) gqa_pv<128>(vb, P, acc, sub, nsub, n, dh, REP);
            else if (nsub == 4) gqa_pv<32>(vb, P, acc, sub, nsub, n, dh, REP);
            else {
                for (uint s = sub; s < n; s += nsub) {
                    const float v = (float)vb[(size_t)s * dh];
                    for (uint r = 0; r < REP; ++r) acc[r] += P[r * TS + s] * v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sub > 0)
        for (uint r = 0; r < REP; ++r) R[((sub - 1) * REP + r) * dh + d] = acc[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0) {
        for (uint r = 0; r < REP; ++r) {
            float a = acc[r];
            for (uint u = 1; u < nsub; ++u) a += R[((u - 1) * REP + r) * dh + d];
            part[((size_t)sp * H + h0 + r) * dh + d] = a;
        }
    }
    if (ti < REP) {
        stats[((size_t)sp * H + h0 + ti) * 2] = M[ti];
        stats[((size_t)sp * H + h0 + ti) * 2 + 1] = Lr[ti];
    }
}
kernel void attn_decode_gqa(
const device float* q [[buffer(0)]],
const device float* kcache [[buffer(1)]],
const device float* vcache [[buffer(2)]],
device float* part [[buffer(3)]],
device float* stats [[buffer(4)]],
constant uint& dh [[buffer(5)]], constant uint& ctx [[buffer(6)]],
constant uint& allow [[buffer(7)]], constant float& scale [[buffer(8)]],
constant uint& REP [[buffer(9)]], constant uint& nsplit [[buffer(10)]],
constant uint& chunk [[buffer(11)]], constant uint& TS_unused [[buffer(12)]],
constant uint& H [[buffer(13)]],
threadgroup float* shm [[threadgroup(0)]],
uint2 tg [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{ attn_decode_gqa_body<float, float4>(q, kcache, vcache, part, stats, dh, ctx, allow, scale, REP, nsplit, chunk, H, shm, tg, ti, lane, sg); }
kernel void attn_decode_gqa_h(
const device float* q [[buffer(0)]],
const device half* kcache [[buffer(1)]],
const device half* vcache [[buffer(2)]],
device float* part [[buffer(3)]],
device float* stats [[buffer(4)]],
constant uint& dh [[buffer(5)]], constant uint& ctx [[buffer(6)]],
constant uint& allow [[buffer(7)]], constant float& scale [[buffer(8)]],
constant uint& REP [[buffer(9)]], constant uint& nsplit [[buffer(10)]],
constant uint& chunk [[buffer(11)]], constant uint& TS_unused [[buffer(12)]],
constant uint& H [[buffer(13)]],
threadgroup float* shm [[threadgroup(0)]],
uint2 tg [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{ attn_decode_gqa_body<half, half4>(q, kcache, vcache, part, stats, dh, ctx, allow, scale, REP, nsplit, chunk, H, shm, tg, ti, lane, sg); }
// out[h][d] = sum_i w_i part_i / sum_i w_i l_i, w_i = exp(m_i - max m).
kernel void attn_decode_combine(
const device float* part [[buffer(0)]],
const device float* stats [[buffer(1)]],
device float* out [[buffer(2)]],
constant uint& dh [[buffer(3)]], constant uint& H [[buffer(4)]],
constant uint& nsplit [[buffer(5)]],
uint gid [[thread_position_in_grid]])
{
    const uint h = gid / dh, d = gid - h * dh;
    if (h >= H) return;
    float m = -INFINITY;
    for (uint i = 0; i < nsplit; ++i) m = max(m, stats[((size_t)i * H + h) * 2]);
    float num = 0.0f, den = 0.0f;
    for (uint i = 0; i < nsplit; ++i) {
        const float w = exp(stats[((size_t)i * H + h) * 2] - m);
        num += w * part[((size_t)i * H + h) * dh + d];
        den += w * stats[((size_t)i * H + h) * 2 + 1];
    }
    out[(size_t)h * dh + d] = num / den;
}
)MET";
}

inline bool tok_begin() {
    if (!ensure_init()) return false;
    if (!libops()) {
        NSError* err = nil;
        libops() = [dev() newLibraryWithSource:kMetalSrcOps()
                                       options:nil error:&err];
        if (!libops()) {
            std::fprintf(stderr, "[metal] ops library compile failed: %s\n",
                         err ? err.localizedDescription.UTF8String : "(no error object)");
            return false;
        }
    }
    tcb() = [q() commandBuffer];
    tenc() = [tcb() computeCommandEncoder];
    return tenc() != nil;
}
// Commit the current command buffer and wait. Called once per layer
// (TM_DECODE_LAYER_CB) or once per token (default single-buffer mode).
// TM_METAL_TOK_TIME=1: accumulate GPU busy time vs host wall time per token
// command and print the running totals every 64 commands (diagnostic).
inline bool tok_flush() {
    static const bool timed = [] { const char* e = std::getenv("TM_METAL_TOK_TIME"); return e && *e == '1'; }();
    static double gpu_s = 0, wall_s = 0; static unsigned cnt = 0;
    const auto t0 = std::chrono::steady_clock::now();
    [tenc() endEncoding];
    [tcb() commit];
    [tcb() waitUntilCompleted];
    const bool ok = tcb().status == MTLCommandBufferStatusCompleted;
    if (timed && ok) {
        gpu_s += tcb().GPUEndTime - tcb().GPUStartTime;
        wall_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (++cnt % 64 == 0)
            std::fprintf(stderr, "[metal] tok cmds %u: gpu %.2f ms/cmd, host wait %.2f ms/cmd\n",
                         cnt, 1e3 * gpu_s / cnt, 1e3 * wall_s / cnt);
    }
    if (!ok)
        std::fprintf(stderr, "[metal] token command failed: %s\n",
                     tcb().error.localizedDescription.UTF8String);
    tenc() = nil; tcb() = nil;
    return ok;
}
inline bool tok_end(int outslot, float* dst, unsigned n) {
    const bool ok = tok_flush();
    if (ok && n) memcpy(dst, slot_buf(outslot, 0).contents,
                        sizeof(float) * n);
    return ok;
}
inline bool tok_upload(int slot, size_t byte_off, const void* src,
                       size_t bytes) {
    if (!ensure_init()) return false;
    id<MTLBuffer> b = slot_buf(slot, byte_off + bytes);
    memcpy((char*)b.contents + byte_off, src, bytes);
    return true;
}
// Packed outputs must be sized BEFORE any writes are encoded. Growing a slot
// mid-command replaces its buffer; earlier kernels still target the old one.
inline bool tok_reserve(int slot, size_t bytes) {
    if (!ensure_init() || tenc()) return false;
    return slot_buf(slot, bytes) != nil;
}
inline bool tok_download(int slot, size_t byte_off, void* dst,
                         size_t bytes) {
    if (!ensure_init()) return false;
    id<MTLBuffer> b = slot_buf(slot, byte_off + bytes);
    memcpy(dst, (char*)b.contents + byte_off, bytes);
    return true;
}
inline id<MTLBuffer> const_u64(unsigned long long v) {
    static std::vector<std::pair<unsigned long long,
                                 __strong id<MTLBuffer>>> c;
    for (auto& e : c) if (e.first == v) return e.second;
    id<MTLBuffer> b = [dev() newBufferWithBytes:&v length:8
        options:MTLResourceStorageModeShared];
    c.push_back({v, b});
    return b;
}
// Chained tokens: commit without waiting, keep the buffers, wait for the
// last one when the chain ends. Buffers on one queue execute in commit
// order, so a later token's embed kernel never races the earlier token.
inline std::vector<__strong id<MTLCommandBuffer>>& pending_cbs() {
    static std::vector<__strong id<MTLCommandBuffer>> v; return v;
}
inline bool tok_end_async() {
    [tenc() endEncoding];
    [tcb() commit];
    pending_cbs().push_back(tcb());
    tenc() = nil; tcb() = nil;
    return true;
}
inline bool tok_wait_pending() {
    bool ok = true;
    for (auto& cb : pending_cbs()) {
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ok = false;
            std::fprintf(stderr, "[metal] chained token command failed: %s\n",
                         cb.error.localizedDescription.UTF8String);
        }
    }
    pending_cbs().clear();
    return ok;
}
inline void tok_argmax(int xslot, int tokslot, unsigned idx, unsigned n) {
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * n);
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    [tenc() setComputePipelineState:ops_pso("argmax_f32")];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bt offset:0 atIndex:1];
    [tenc() setBuffer:const_u32(n) offset:0 atIndex:2];
    [tenc() setBuffer:const_u32(idx) offset:0 atIndex:3];
    [tenc() dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
inline void tok_sample(int xslot, int tokslot, unsigned idx, unsigned n, float inv_temp, unsigned seed) {
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * n);
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    [tenc() setComputePipelineState:ops_pso("sample_gumbel_f32")];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bt offset:0 atIndex:1];
    [tenc() setBuffer:const_u32(n) offset:0 atIndex:2];
    [tenc() setBuffer:const_u32(idx) offset:0 atIndex:3];
    [tenc() setBuffer:const_f32(inv_temp) offset:0 atIndex:4];
    [tenc() setBuffer:const_u32(seed) offset:0 atIndex:5];
    [tenc() dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
// Read one entry of a token slot without waiting (the slot is shared
// memory; a chain writes entries in order, so a caller polling entry i
// after seeding the slot with a sentinel sees tokens as they land).
inline unsigned tok_peek(int tokslot, unsigned idx) {
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    return ((volatile unsigned*)bt.contents)[idx];
}
inline void tok_embed_q4(int tokslot, unsigned idx, int wid, uint64_t woff, int xslot, unsigned D) {
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * D);
    [tenc() setComputePipelineState:ops_pso("embed_q4_row")];
    [tenc() setBuffer:bt offset:0 atIndex:0];
    [tenc() setBuffer:const_u32(idx) offset:0 atIndex:1];
    [tenc() setBuffer:bw offset:0 atIndex:2];
    [tenc() setBuffer:const_u64(woff) offset:0 atIndex:3];
    [tenc() setBuffer:bx offset:0 atIndex:4];
    [tenc() setBuffer:const_u32(D) offset:0 atIndex:5];
    [tenc() dispatchThreads:MTLSizeMake(D / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
inline void tok_embed_h(int tokslot, unsigned idx, int wid, uint64_t woff,
                        int xslot, unsigned D) {
    if (woff % 8 || D % 4)
        throw std::invalid_argument("metal: fp16 embed requires 8-aligned woff and D % 4 == 0");
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * D);
    [tenc() setComputePipelineState:ops_pso("embed_h_row")];
    [tenc() setBuffer:bt offset:0 atIndex:0];
    [tenc() setBuffer:const_u32(idx) offset:0 atIndex:1];
    [tenc() setBuffer:bw offset:0 atIndex:2];
    [tenc() setBuffer:const_u64(woff) offset:0 atIndex:3];
    [tenc() setBuffer:bx offset:0 atIndex:4];
    [tenc() setBuffer:const_u32(D) offset:0 atIndex:5];
    [tenc() dispatchThreads:MTLSizeMake(D / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
inline void slot_set(int sid, __strong id<MTLBuffer> b) {
    auto& s = tslots();
    if ((int)s.size() <= sid) s.resize((size_t)sid + 1);
    s[sid] = b;
}
// Weight-handle registry (single-model token session). NoCopy failure is
// fatal to registration: never silently duplicate a file-backed model.
// Copy mode (TinyLlama resident): one newBufferWithBytes per tensor.
// Returns the handle id; *woff_out = byte offset of the tensor inside
// its buffer (0 for copy mode).
inline int tok_wbuf(const void* tensor_ptr, size_t bytes,
                    const void* map_base, size_t map_size,
                    bool zero_copy, uint64_t& woff_out) {
    woff_out = 0;
    if (!ensure_init() || !tensor_ptr || !bytes) return -1;
    size_t map_len = 0;
    if (zero_copy) {
        const long page = sysconf(_SC_PAGESIZE);
        const auto base = reinterpret_cast<uintptr_t>(map_base);
        const auto ptr = reinterpret_cast<uintptr_t>(tensor_ptr);
        if (page <= 0 || !base || base % (size_t)page || ptr < base ||
            ptr - base > map_size || bytes > map_size - (ptr - base) ||
            map_size > std::numeric_limits<size_t>::max() - ((size_t)page - 1))
            return -1;
        map_len = ((map_size + (size_t)page - 1) / (size_t)page) * (size_t)page;
        if (map_len > dev().maxBufferLength) return -1;
    } else if (bytes > dev().maxBufferLength) {
        return -1;
    }
    static std::map<const void*, std::pair<int, uint64_t>> c;
    auto it = c.find(tensor_ptr);
    if (it != c.end()) {
        woff_out = it->second.second;
        return it->second.first;
    }
    const int wid = 1000 + (int)c.size();
    if (zero_copy) {
        // One page-rounded buffer; tensor offsets are byte offsets, not
        // typed half pointers (the file does not promise half alignment).
        static __strong id<MTLBuffer> filebuf;
        static const void* filebuf_base = nullptr;
        if (!filebuf || filebuf_base != map_base) {
            filebuf = [dev()
                newBufferWithBytesNoCopy:const_cast<void*>(map_base)
                length:map_len options:MTLResourceStorageModeShared
                deallocator:nil];
            filebuf_base = map_base;
        }
        if (filebuf) {
            slot_set(wid, filebuf);
            woff_out = (uint64_t)((const uint8_t*)tensor_ptr
                - (const uint8_t*)map_base);
            c[tensor_ptr] = {wid, woff_out};
            return wid;
        }
        std::fprintf(stderr, "[metal] NoCopy registration refused; no copy fallback\n");
        return -1;
    }
    id<MTLBuffer> b = [dev() newBufferWithBytes:tensor_ptr
        length:bytes options:MTLResourceStorageModeShared];
    if (!b) return -1;
    slot_set(wid, b);
    c[tensor_ptr] = {wid, 0};
    woff_out = 0;
    return wid;
}
inline bool tok_probe_copy(int srcslot, size_t src_off, int dstslot,
                           unsigned n) {
    if (!tok_begin()) return false;
    id<MTLBuffer> bs = slot_buf(srcslot, src_off + n);
    id<MTLBuffer> bd = slot_buf(dstslot, n);
    [tenc() setComputePipelineState:ops_pso("probe_copy")];
    [tenc() setBuffer:bs offset:0 atIndex:0];
    [tenc() setBuffer:bd offset:0 atIndex:1];
    [tenc() setBuffer:const_u32(n) offset:0 atIndex:2];
    [tenc() setBuffer:const_u64(src_off) offset:0 atIndex:3];
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return tok_end(dstslot, nullptr, 0);
}
inline unsigned& gemv_lanes() {
    static unsigned lanes = [] {
        const char* s = std::getenv("TM_METAL_GEMV_LANES");
        if (!s) return 16u; // strict oracle + interleaved end-to-end winner on M1
        char* end = nullptr;
        const unsigned long n = std::strtoul(s, &end, 10);
        if (!*s || *end || (n != 1 && n != 4 && n != 8 && n != 16 && n != 32))
            throw std::invalid_argument("TM_METAL_GEMV_LANES must be 1, 4, 8, 16, or 32");
        return (unsigned)n;
    }();
    return lanes;
}
inline bool& gemv_scale_per_block() {
    static bool enabled = [] {
        const char* s = std::getenv("TM_METAL_GEMV_BLOCK_SCALE");
        if (!s || std::strcmp(s, "0") == 0) return false;
        if (std::strcmp(s, "1") == 0) return true;
        throw std::invalid_argument("TM_METAL_GEMV_BLOCK_SCALE must be 0 or 1");
    }();
    return enabled;
}
// Rows per thread in the cooperative GEMV (TM_METAL_GEMV_ROWS: 1, 2 or 4;
// default 2, see q4_gemv_rows_body).
inline int& gemv_rows() {
    static int rows = [] {
        const int r = tmtune::get_int("TM_METAL_GEMV_ROWS", 2);   // 6/8 = mask-dot variants with 2/4 rows
        if (r != 1 && r != 2 && r != 4 && r != 6 && r != 8) throw std::invalid_argument("TM_METAL_GEMV_ROWS must be 1, 2, 4, 6 or 8");
        return r;
    }();
    return rows;
}
inline id<MTLComputePipelineState> token_gemv_pipeline(unsigned lanes) {
    if (lanes != 1 && lanes != 4 && lanes != 8 && lanes != 16 && lanes != 32)
        throw std::invalid_argument("metal: unsupported GEMV lane count");
    static std::map<unsigned, __strong id<MTLComputePipelineState>> cache;
    const bool block_scale = gemv_scale_per_block();
    const unsigned key = (lanes << 1) | (unsigned)block_scale | ((unsigned)(gemv_rows() + 1) << 8);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    NSError* error = nil;
    id<MTLFunction> function;
    if (lanes == 1) {
        function = [libq4() newFunctionWithName:@"q4_gemv"];
    } else {
        MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
        [values setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
        [values setConstantValue:&block_scale type:MTLDataTypeBool atIndex:1];
        const int rows = gemv_rows();
        NSString* fname = rows == 6 ? @"q4_gemv_mask_r2" : rows == 8 ? @"q4_gemv_mask_r4"
                        : rows == 4 ? @"q4_gemv_coop_r4" : rows == 2 ? @"q4_gemv_coop_r2" : @"q4_gemv_coop";
        function = [libq4() newFunctionWithName:fname constantValues:values error:&error];
    }
    id<MTLComputePipelineState> pso = function ? [dev()
        newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pso || pso.threadExecutionWidth != 32)
        throw std::runtime_error("metal: GEMV pipeline requires 32-lane SIMD groups");
    cache.emplace(key, pso);
    return pso;
}
    // fp16-weights variant of the token GEMV pipeline (TM_DECODE_F16).
    // Same lanes/rows/threadgroup tuning as the Q4 pipeline; only the
    // kernel family differs. TM_METAL_GEMV_ROWS 6/8 are Q4 mask-dot
    // variants and are rejected here.
    inline id<MTLComputePipelineState> token_gemv_h_pipeline(unsigned lanes) {
        if (lanes != 1 && lanes != 4 && lanes != 8 && lanes != 16 && lanes != 32)
            throw std::invalid_argument("metal: unsupported GEMV lane count");
        static std::map<unsigned, __strong id<MTLComputePipelineState>> cache;
        const int rows = gemv_rows();
        if (rows != 1 && rows != 2 && rows != 4)
            throw std::invalid_argument("metal: fp16 GEMV supports TM_METAL_GEMV_ROWS 1, 2 or 4");
        const unsigned key = (lanes << 1) | (unsigned(rows + 1) << 8);
        auto found = cache.find(key);
        if (found != cache.end()) return found->second;
        NSError* error = nil;
        id<MTLFunction> function;
        if (lanes == 1) {
            function = [libq4() newFunctionWithName:@"f16_gemv_coop_r1"];
        } else {
            MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
            [values setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
            const bool block_scale = true;   // constant 1 unused by f16 kernels
            [values setConstantValue:&block_scale type:MTLDataTypeBool atIndex:1];
            NSString* fname = rows == 4 ? @"f16_gemv_coop_r4"
                            : rows == 2 ? @"f16_gemv_coop_r2" : @"f16_gemv_coop_r1";
            function = [libq4() newFunctionWithName:fname constantValues:values error:&error];
        }
        id<MTLComputePipelineState> pso = function ? [dev()
            newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pso || pso.threadExecutionWidth != 32)
            throw std::runtime_error("metal: fp16 GEMV pipeline requires 32-lane SIMD groups");
        cache.emplace(key, pso);
        return pso;
    }
inline unsigned& gemv_threads() {
    static unsigned threads = [] {
        const char* s = tmtune::get("TM_METAL_GEMV_THREADS");
        if (!s) return 128u;   // 2026-09-07: 128 with rows 2 (was 64 with rows 1)
        char* end = nullptr;
        const unsigned long n = std::strtoul(s, &end, 10);
        if (!*s || *end || (n != 64 && n != 128 && n != 256))
            throw std::invalid_argument("TM_METAL_GEMV_THREADS must be 64, 128, or 256");
        return (unsigned)n;
    }();
    return threads;
}
inline void tok_gemv(int xslot, int wid, uint64_t woff, int yslot,
                     size_t yoff, unsigned N, unsigned K) {
    const unsigned lanes = gemv_lanes();
    const unsigned threads = gemv_threads();
    id<MTLComputePipelineState> pso = token_gemv_pipeline(lanes);
    if ((threads != 64 && threads != 128 && threads != 256) ||
        threads > pso.maxTotalThreadsPerThreadgroup)
        throw std::invalid_argument("metal: unsupported GEMV threadgroup size");
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * K);
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> by = slot_buf(yslot, yoff + sizeof(float) * N);
    [tenc() setComputePipelineState:pso];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bw offset:0 atIndex:1];
    [tenc() setBuffer:by offset:yoff atIndex:2];
    [tenc() setBuffer:const_u32(N) offset:0 atIndex:3];
    [tenc() setBuffer:const_u32(K) offset:0 atIndex:4];
    [tenc() setBuffer:const_u64(woff) offset:0 atIndex:5];
    const int r = gemv_rows();
    const unsigned rows_per_group = lanes == 1 ? 1u : (r == 6 ? 2u : r == 8 ? 4u : (unsigned)r);
    const size_t groups = ((size_t)N + rows_per_group - 1) / rows_per_group;
    [tenc() dispatchThreads:MTLSizeMake(groups * lanes, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}
// Fused multi-tensor decode GEMV (q|k|v and gate|up): one dispatch over N
// total rows split into nseg segments, each a (byte offset, row count) into
// the shared zero-copy weight buffer. Numerically identical to the separate
// tok_gemv dispatches it replaces (same per-row block order and reduction).
inline id<MTLComputePipelineState> token_gemv_seg_pipeline(unsigned lanes) {
    if (lanes != 1 && lanes != 4 && lanes != 8 && lanes != 16 && lanes != 32)
        throw std::invalid_argument("metal: unsupported GEMV lane count");
    static std::map<unsigned, __strong id<MTLComputePipelineState>> cache;
    auto found = cache.find(lanes);
    if (found != cache.end()) return found->second;
    NSError* error = nil;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
    const bool block_scale = false;   // seg body ignores the block-scale branch
    [values setConstantValue:&block_scale type:MTLDataTypeBool atIndex:1];
    id<MTLFunction> function = [libq4() newFunctionWithName:@"q4_gemv_coop_r2_seg"
                                             constantValues:values error:&error];
    id<MTLComputePipelineState> pso = function ? [dev()
        newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pso || pso.threadExecutionWidth != 32)
        throw std::runtime_error("metal: segmented GEMV pipeline requires 32-lane SIMD groups");
    cache.emplace(lanes, pso);
    return pso;
}
inline void tok_gemv_seg(int xslot, int wid, int yslot, size_t yoff,
                         unsigned N, unsigned K,
                         const uint64_t* seg_woff, const unsigned* seg_rows,
                         unsigned nseg) {
    const unsigned lanes = gemv_lanes();
    const unsigned threads = gemv_threads();
    id<MTLComputePipelineState> pso = token_gemv_seg_pipeline(lanes);
    if ((threads != 64 && threads != 128 && threads != 256) ||
        threads > pso.maxTotalThreadsPerThreadgroup)
        throw std::invalid_argument("metal: unsupported GEMV threadgroup size");
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * K);
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> by = slot_buf(yslot, yoff + sizeof(float) * N);
    // Fresh buffers (not setBytes): the seg table must be a per-dispatch
    // snapshot — in the back-to-back chain several tokens share one command
    // buffer, so a reused scratch would alias across dispatches. 8-byte
    // alignment for uint64 woffs is guaranteed by newBufferWithBytes.
    id<MTLBuffer> rows_buf = [dev() newBufferWithBytes:seg_rows
        length:nseg * sizeof(unsigned) options:MTLResourceStorageModeShared];
    id<MTLBuffer> woff_buf = [dev() newBufferWithBytes:seg_woff
        length:nseg * sizeof(uint64_t) options:MTLResourceStorageModeShared];
    [tenc() setComputePipelineState:pso];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bw offset:0 atIndex:1];
    [tenc() setBuffer:by offset:yoff atIndex:2];
    [tenc() setBuffer:const_u32(N) offset:0 atIndex:3];
    [tenc() setBuffer:const_u32(K) offset:0 atIndex:4];
    [tenc() setBuffer:const_u32(nseg) offset:0 atIndex:5];
    [tenc() setBuffer:rows_buf offset:0 atIndex:6];
    [tenc() setBuffer:woff_buf offset:0 atIndex:7];
    const unsigned rows_per_group = 2u;   // q4_gemv_coop_r2_seg
    const size_t groups = ((size_t)N + rows_per_group - 1) / rows_per_group;
    [tenc() dispatchThreads:MTLSizeMake(groups * lanes, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}
// fp16-weights twin of tok_gemv (TM_DECODE_F16): identical slot/offset
// contract, half weights at (wid, woff) — woff is a BYTE offset and the
// loader asserts 4-byte tensor alignment so the kernel's half4 loads
// stay aligned.
inline void tok_gemv_h(int xslot, int wid, uint64_t woff, int yslot,
                       size_t yoff, unsigned N, unsigned K) {
    if (woff % 4 || K % 4)
        throw std::invalid_argument("metal: fp16 GEMV requires 4-aligned woff and K");
    const unsigned lanes = gemv_lanes();
    const unsigned threads = gemv_threads();
    id<MTLComputePipelineState> pso = token_gemv_h_pipeline(lanes);
    if ((threads != 64 && threads != 128 && threads != 256) ||
        threads > pso.maxTotalThreadsPerThreadgroup)
        throw std::invalid_argument("metal: unsupported GEMV threadgroup size");
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * K);
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> by = slot_buf(yslot, yoff + sizeof(float) * N);
    [tenc() setComputePipelineState:pso];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bw offset:0 atIndex:1];
    [tenc() setBuffer:by offset:yoff atIndex:2];
    [tenc() setBuffer:const_u32(N) offset:0 atIndex:3];
    [tenc() setBuffer:const_u32(K) offset:0 atIndex:4];
    [tenc() setBuffer:const_u64(woff) offset:0 atIndex:5];
    const int rf = gemv_rows();
    const unsigned rows_h = lanes == 1 ? 1u : (unsigned)rf;
    const size_t groups_h = ((size_t)N + rows_h - 1) / rows_h;
    [tenc() dispatchThreads:MTLSizeMake(groups_h * lanes, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}
inline void tok_rmsnorm(int xslot, const void* w, int outslot,
                        unsigned D, float eps) {
    @autoreleasepool {
        static std::vector<std::pair<const void*,
                                     __strong id<MTLBuffer>>> wc;
        __strong id<MTLBuffer> bw;
        for (auto& e : wc)
            if (e.first == w) { bw = e.second; break; }
        if (!bw) {
            bw = [dev() newBufferWithBytes:w length:D * 4
                options:MTLResourceStorageModeShared];
            wc.push_back({w, bw});
        }
        id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * D);
        id<MTLBuffer> bo = slot_buf(outslot, sizeof(float) * D);
        [tenc() setComputePipelineState:ops_pso("rmsnorm_row")];
        [tenc() setBuffer:bx offset:0 atIndex:0];
        [tenc() setBuffer:bw offset:0 atIndex:1];
        [tenc() setBuffer:bo offset:0 atIndex:2];
        [tenc() setBuffer:const_u32(D) offset:0 atIndex:3];
        [tenc() setBuffer:const_f32(eps) offset:0 atIndex:4];
        [tenc() dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
inline void tok_add(int a, int b, unsigned n) {
    [tenc() setComputePipelineState:ops_pso("add_inplace")];
    [tenc() setBuffer:slot_buf(a, sizeof(float) * n) offset:0 atIndex:0];
    [tenc() setBuffer:slot_buf(b, sizeof(float) * n) offset:0 atIndex:1];
    [tenc() setBuffer:const_u32(n) offset:0 atIndex:2];
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
inline void tok_silu_mul(int g, size_t goff, int u, size_t uoff,
                         unsigned n) {
    [tenc() setComputePipelineState:ops_pso("silu_mul")];
    [tenc() setBuffer:slot_buf(g, goff + sizeof(float) * n)
      offset:goff atIndex:0];
    [tenc() setBuffer:slot_buf(u, uoff + sizeof(float) * n)
      offset:uoff atIndex:1];
    [tenc() setBuffer:const_u32(n) offset:0 atIndex:2];
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
inline void tok_rope_q(int qslot, unsigned H, unsigned dh, int pos,
                       float theta) {
    unsigned pairs = H * (dh / 2);
    [tenc() setComputePipelineState:ops_pso("rope_inplace")];
    [tenc() setBuffer:slot_buf(qslot, sizeof(float) * H * dh)
      offset:0 atIndex:0];
    [tenc() setBuffer:const_u32(H) offset:0 atIndex:1];
    [tenc() setBuffer:const_u32(dh) offset:0 atIndex:2];
    [tenc() setBuffer:const_i32(pos) offset:0 atIndex:3];
    [tenc() setBuffer:const_f32(theta) offset:0 atIndex:4];
    [tenc() dispatchThreads:MTLSizeMake(pairs, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
// GPU K/V cache element type. Half (default) halves the cache traffic the
// decode attention reads at long context; rows are rounded once when
// written (rope_k_cache_h / v_to_cache_h or the upload converter) and the
// arithmetic stays fp32. TM_METAL_KV_HALF=0 keeps fp32 rows.
inline bool& kv_half() {
    static bool enabled = [] {
        const char* s = std::getenv("TM_METAL_KV_HALF");
        if (!s || std::strcmp(s, "1") == 0) return true;
        if (std::strcmp(s, "0") == 0) return false;
        throw std::invalid_argument("TM_METAL_KV_HALF must be 0 or 1");
    }();
    return enabled;
}
inline size_t kv_elem_bytes() { return kv_half() ? 2 : 4; }
// Upload/download fp32 rows into/out of a K/V cache slot in the slot's
// element type; offsets and counts are in elements.
// Size a K/V cache slot for n elements up front (partial uploads must not
// leave a slot short of the rows the cache kernels write later).
inline bool tok_kv_reserve(int slot, size_t n) { return tok_reserve(slot, n * kv_elem_bytes()); }
inline bool tok_kv_upload(int slot, size_t elem_off, const float* src, size_t n) {
    if (!ensure_init()) return false;
    const size_t eb = kv_elem_bytes();
    id<MTLBuffer> b = slot_buf(slot, (elem_off + n) * eb);
    if (!kv_half()) { memcpy((char*)b.contents + elem_off * 4, src, n * 4); return true; }
    __fp16* dst = (__fp16*)b.contents + elem_off;
    for (size_t i = 0; i < n; ++i) dst[i] = (__fp16)src[i];
    return true;
}
inline bool tok_kv_download(int slot, size_t elem_off, float* dst, size_t n) {
    if (!ensure_init()) return false;
    const size_t eb = kv_elem_bytes();
    id<MTLBuffer> b = slot_buf(slot, (elem_off + n) * eb);
    if (!kv_half()) { memcpy(dst, (char*)b.contents + elem_off * 4, n * 4); return true; }
    const __fp16* src = (const __fp16*)b.contents + elem_off;
    for (size_t i = 0; i < n; ++i) dst[i] = (float)src[i];
    return true;
}
inline void tok_rope_k_cache(int ksrcslot, size_t ksrc_off, int cacheslot,
                             unsigned KVH, unsigned dh, unsigned ctx,
                             int pos, float theta) {
    unsigned pairs = KVH * (dh / 2);
    [tenc() setComputePipelineState:ops_pso(kv_half() ? "rope_k_cache_h" : "rope_k_cache")];
    [tenc() setBuffer:slot_buf(ksrcslot, ksrc_off + sizeof(float) * KVH * dh)
      offset:ksrc_off atIndex:0];
    [tenc() setBuffer:slot_buf(cacheslot,
         kv_elem_bytes() * KVH * ctx * dh) offset:0 atIndex:1];
    [tenc() setBuffer:const_u32(KVH) offset:0 atIndex:2];
    [tenc() setBuffer:const_u32(dh) offset:0 atIndex:3];
    [tenc() setBuffer:const_u32(ctx) offset:0 atIndex:4];
    [tenc() setBuffer:const_i32(pos) offset:0 atIndex:5];
    [tenc() setBuffer:const_f32(theta) offset:0 atIndex:6];
    [tenc() dispatchThreads:MTLSizeMake(pairs, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
inline void tok_v_cache(int vsrcslot, size_t vsrc_off, int cacheslot,
                        unsigned KVH, unsigned dh, unsigned ctx, int pos) {
    unsigned n = KVH * dh;
    [tenc() setComputePipelineState:ops_pso(kv_half() ? "v_to_cache_h" : "v_to_cache")];
    [tenc() setBuffer:slot_buf(vsrcslot, vsrc_off + sizeof(float) * n)
      offset:vsrc_off atIndex:0];
    [tenc() setBuffer:slot_buf(cacheslot,
         kv_elem_bytes() * KVH * ctx * dh) offset:0 atIndex:1];
    [tenc() setBuffer:const_u32(KVH) offset:0 atIndex:2];
    [tenc() setBuffer:const_u32(dh) offset:0 atIndex:3];
    [tenc() setBuffer:const_u32(ctx) offset:0 atIndex:4];
    [tenc() setBuffer:const_i32(pos) offset:0 atIndex:5];
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
// Default ON since 2026-09-06: strict full-model Q4 oracle 32/32 (max err
// 1.0681e-04) and paired same-window A/B vs the three-dispatch path on
// TinyLlama Q4 GPU decode: +11.0% at tg256/ctx384 (59.36 -> 65.86 t/s, 4/4),
// +36.6% at tg1024/ctx2048 (39.10 -> 53.85 t/s, p95 34.7 -> 22.1 ms, 4/4),
// identical greedy hashes. TM_METAL_ATTN_FUSED=0 restores the legacy path,
// which also remains the fallback above 4096 context or on unsupported devices.
inline bool& fused_attention() {
    static bool enabled = [] {
        const char* s = std::getenv("TM_METAL_ATTN_FUSED");
        if (!s || std::strcmp(s, "1") == 0) return true;
        if (std::strcmp(s, "0") == 0) return false;
        throw std::invalid_argument("TM_METAL_ATTN_FUSED must be 0 or 1");
    }();
    return enabled;
}
// GQA split-K decode attention (attn_decode_gqa + attn_decode_combine).
// Default ON; TM_METAL_ATTN_GQA=0 restores attn_decode_fused for A/B.
inline bool& gqa_attention() {
    static bool enabled = [] {
        const char* s = std::getenv("TM_METAL_ATTN_GQA");
        if (!s || std::strcmp(s, "1") == 0) return true;
        if (std::strcmp(s, "0") == 0) return false;
        throw std::invalid_argument("TM_METAL_ATTN_GQA must be 0 or 1");
    }();
    return enabled;
}
// Tile rows so the staged K, V, q and probability tiles fit the device's
// threadgroup memory; 0 when the shape is unsupported (REP*dh > 512 or
// dh not a multiple of 4), which falls back to the fused kernel.
// Supported shapes: dh divides 128 (64, 128), REP <= 8; 0 otherwise, which
// falls back to attn_decode_fused.
inline unsigned gqa_tile_rows(unsigned dh, unsigned REP) {
    if (!dh || 128 % dh || dh % 4 || !REP || REP > 8) return 0;
    const size_t bytes = sizeof(float) * (REP * 128 + REP * 128 + 3 * REP);
    return bytes + 64 <= dev().maxThreadgroupMemoryLength ? 128u : 0u;
}
inline id<MTLComputePipelineState> token_attention_pipeline(unsigned allow) {
    // Bound per-head scratch and keep the legacy path for long contexts and
    // unsupported devices. Round dynamic threadgroup storage to Metal's 16B unit.
    if (!allow || allow > 4096) return nil;
    auto pso = ops_pso(kv_half() ? "attn_decode_fused_h" : "attn_decode_fused");
    if (!pso) throw std::runtime_error("metal: fused attention pipeline unavailable");
    const size_t bytes = (sizeof(float) * allow + 15) & ~size_t(15);
    if (pso.threadExecutionWidth != 32 || pso.maxTotalThreadsPerThreadgroup < 128 ||
        pso.staticThreadgroupMemoryLength + bytes > dev().maxThreadgroupMemoryLength)
        return nil;
    return pso;
}
inline void tok_attn(int qslot, int kcslot, int vcslot, int probslot,
                     int outslot, unsigned QD, unsigned H, unsigned KVH,
                     unsigned dh, unsigned ctx, int allow, float scale,
                     unsigned REP) {
    if (!H || !KVH || !dh || !ctx || allow <= 0 || (unsigned)allow > ctx ||
        H % KVH || REP != H / KVH || (uint64_t)H * dh != QD ||
        !std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("metal: invalid decode attention dimensions or scale");
    @autoreleasepool {
        id<MTLBuffer> bq = slot_buf(qslot, sizeof(float) * QD);
        id<MTLBuffer> bk = slot_buf(kcslot, kv_elem_bytes() * KVH * ctx * dh);
        id<MTLBuffer> bv = slot_buf(vcslot, kv_elem_bytes() * KVH * ctx * dh);
        id<MTLBuffer> bo = slot_buf(outslot, sizeof(float) * QD);
        // Below ~128 rows the split-K pair of dispatches costs more than it
        // saves (micro-bench allow=74: 35-47 us vs fused 18-30); the fused
        // single dispatch keeps short contexts.
        const unsigned TS = gqa_attention() && allow >= 128 ? gqa_tile_rows(dh, REP) : 0;
        if (TS) {
            // Splits: enough (kv head, split) groups to cover the GPU, each
            // split at least one tile; partials live in the probs slot
            // (H*ctx floats reserved by the caller), which bounds nsplit.
            static const unsigned target_tg = [] {
                return (unsigned)tmtune::get_int("TM_METAL_ATTN_TG", 64); }();
            unsigned nsplit = (target_tg + KVH - 1) / KVH;
            nsplit = std::min<unsigned>(nsplit, ((unsigned)allow + TS - 1) / TS);
            nsplit = std::min<unsigned>(nsplit, ctx / (dh + 2));
            nsplit = std::max<unsigned>(nsplit, 1);
            const unsigned chunk = (((unsigned)allow + nsplit - 1) / nsplit + TS - 1) / TS * TS;
            nsplit = ((unsigned)allow + chunk - 1) / chunk;   // drop empty splits
            const size_t part_bytes = sizeof(float) * (size_t)nsplit * H * dh;
            const size_t stat_bytes = sizeof(float) * (size_t)nsplit * H * 2;
            // The probs slot is reserved by the caller for H*ctx floats; a
            // tiny context (ctx < dh+2) cannot hold even one split's partials
            // and a slot cannot grow mid-encoding, so such shapes take the
            // fused path below.
            auto& ts = tslots();
            const bool fits = probslot >= (int)ts.size() || !ts[probslot] ||
                              [ts[probslot] length] >= part_bytes + stat_bytes;
            if (!fits) goto fused_path;
            {
            id<MTLBuffer> bp = slot_buf(probslot, part_bytes + stat_bytes);
            auto pso = ops_pso(kv_half() ? "attn_decode_gqa_h" : "attn_decode_gqa");
            auto comb = ops_pso("attn_decode_combine");
            if (!pso || !comb) throw std::runtime_error("metal: gqa attention pipeline unavailable");
            [tenc() setComputePipelineState:pso];
            [tenc() setBuffer:bq offset:0 atIndex:0];
            [tenc() setBuffer:bk offset:0 atIndex:1];
            [tenc() setBuffer:bv offset:0 atIndex:2];
            [tenc() setBuffer:bp offset:0 atIndex:3];
            [tenc() setBuffer:bp offset:part_bytes atIndex:4];
            [tenc() setBuffer:const_u32(dh) offset:0 atIndex:5];
            [tenc() setBuffer:const_u32(ctx) offset:0 atIndex:6];
            [tenc() setBuffer:const_u32((unsigned)allow) offset:0 atIndex:7];
            [tenc() setBuffer:const_f32(scale) offset:0 atIndex:8];
            [tenc() setBuffer:const_u32(REP) offset:0 atIndex:9];
            [tenc() setBuffer:const_u32(nsplit) offset:0 atIndex:10];
            [tenc() setBuffer:const_u32(chunk) offset:0 atIndex:11];
            [tenc() setBuffer:const_u32(TS) offset:0 atIndex:12];
            [tenc() setBuffer:const_u32(H) offset:0 atIndex:13];
            // q [REP][dh] + reduction [128/dh-1][REP][dh] = REP*128 floats.
            const size_t shm = (sizeof(float) * (REP * 128 + REP * TS + 3 * REP) + 15) & ~size_t(15);
            [tenc() setThreadgroupMemoryLength:shm atIndex:0];
            [tenc() dispatchThreadgroups:MTLSizeMake(KVH, nsplit, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [tenc() setComputePipelineState:comb];
            [tenc() setBuffer:bp offset:0 atIndex:0];
            [tenc() setBuffer:bp offset:part_bytes atIndex:1];
            [tenc() setBuffer:bo offset:0 atIndex:2];
            [tenc() setBuffer:const_u32(dh) offset:0 atIndex:3];
            [tenc() setBuffer:const_u32(H) offset:0 atIndex:4];
            [tenc() setBuffer:const_u32(nsplit) offset:0 atIndex:5];
            [tenc() dispatchThreads:MTLSizeMake(QD, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
            }
        }
    fused_path:
        auto fused = fused_attention() ? token_attention_pipeline((unsigned)allow) : nil;
        if (fused) {
            [tenc() setComputePipelineState:fused];
            [tenc() setBuffer:bq offset:0 atIndex:0];
            [tenc() setBuffer:bk offset:0 atIndex:1];
            [tenc() setBuffer:bv offset:0 atIndex:2];
            [tenc() setBuffer:bo offset:0 atIndex:3];
            [tenc() setBuffer:const_u32(dh) offset:0 atIndex:4];
            [tenc() setBuffer:const_u32(ctx) offset:0 atIndex:5];
            [tenc() setBuffer:const_u32((unsigned)allow) offset:0 atIndex:6];
            [tenc() setBuffer:const_f32(scale) offset:0 atIndex:7];
            [tenc() setBuffer:const_u32(REP) offset:0 atIndex:8];
            const size_t scratch = (sizeof(float) * allow + 15) & ~size_t(15);
            [tenc() setThreadgroupMemoryLength:scratch atIndex:0];
            [tenc() dispatchThreadgroups:MTLSizeMake(H, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }
        if (kv_half())
            throw std::runtime_error("metal: half K/V cache needs the gqa or fused attention path (allow > 4096 with TM_METAL_ATTN_GQA=0)");
        id<MTLBuffer> bp = slot_buf(probslot, sizeof(float) * H * ctx);
        [tenc() setComputePipelineState:ops_pso("attn_scores")];
        [tenc() setBuffer:bq offset:0 atIndex:0];
        [tenc() setBuffer:bk offset:0 atIndex:1];
        [tenc() setBuffer:bp offset:0 atIndex:2];
        [tenc() setBuffer:const_u32(H) offset:0 atIndex:3];
        [tenc() setBuffer:const_u32(dh) offset:0 atIndex:4];
        [tenc() setBuffer:const_u32(ctx) offset:0 atIndex:5];
        [tenc() setBuffer:const_i32(allow) offset:0 atIndex:6];
        [tenc() setBuffer:const_f32(scale) offset:0 atIndex:7];
        [tenc() setBuffer:const_u32(REP) offset:0 atIndex:8];
        [tenc() dispatchThreads:MTLSizeMake((size_t)H * allow, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [tenc() setComputePipelineState:ops_pso("attn_softmax")];
        [tenc() setBuffer:bp offset:0 atIndex:0];
        [tenc() setBuffer:const_u32(H) offset:0 atIndex:1];
        [tenc() setBuffer:const_u32(ctx) offset:0 atIndex:2];
        [tenc() setBuffer:const_i32(allow) offset:0 atIndex:3];
        [tenc() dispatchThreads:MTLSizeMake(H, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [tenc() setComputePipelineState:ops_pso("attn_pv")];
        [tenc() setBuffer:bp offset:0 atIndex:0];
        [tenc() setBuffer:bv offset:0 atIndex:1];
        [tenc() setBuffer:bo offset:0 atIndex:2];
        [tenc() setBuffer:const_u32(H) offset:0 atIndex:3];
        [tenc() setBuffer:const_u32(dh) offset:0 atIndex:4];
        [tenc() setBuffer:const_u32(ctx) offset:0 atIndex:5];
        [tenc() setBuffer:const_i32(allow) offset:0 atIndex:6];
        [tenc() setBuffer:const_u32(REP) offset:0 atIndex:7];
        [tenc() dispatchThreads:MTLSizeMake(QD, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
}

}  // namespace tmgpu

// C shim consumed by llama.h via weak_import (TM_HAVE_METAL builds).
// Declared here, DEFINED in metal_shim.mm (an out-of-line definition is
// required so the symbol is always emitted for the linker).
extern "C" bool tm_metal_sgemm_q4(const float* A, const void* Wq, float* C,
                                  unsigned M, unsigned N, unsigned K);
extern "C" bool tm_metal_q4_prefill(const float* A, const void* const* Wqs,
                                    const unsigned* Ns, unsigned nparts,
                                    float* C, unsigned M, unsigned K);
extern "C" bool tm_metal_q4_gemv(const float* x, const void* Wq, float* y,
                                 unsigned N, unsigned K);
extern "C" bool tm_metal_tok_begin(void);
extern "C" bool tm_metal_tok_end(int outslot, float* dst, unsigned n);
extern "C" bool tm_metal_tok_upload(int slot, size_t byte_off,
                                    const void* src, size_t bytes);
extern "C" bool tm_metal_tok_download(int slot, size_t byte_off,
                                      void* dst, size_t bytes);
extern "C" int tm_metal_tok_wbuf(const void* tensor_ptr, size_t bytes,
                                 const void* map_base, size_t map_size,
                                 int zero_copy, uint64_t& woff_out);
extern "C" bool tm_metal_tok_probe_copy(int srcslot, size_t src_off,
                                        int dstslot, unsigned n);
extern "C" void tm_metal_tok_gemv(int xslot, int wid, uint64_t woff,
                                  int yslot, size_t yoff, unsigned N,
                                  unsigned K);
extern "C" void tm_metal_tok_gemv_seg(int xslot, int wid, int yslot,
                                      size_t yoff, unsigned N, unsigned K,
                                      const uint64_t* seg_woff,
                                      const unsigned* seg_rows, unsigned nseg);
extern "C" void tm_metal_tok_rmsnorm(int xslot, const void* w,
                                     int outslot, unsigned D, float eps);
extern "C" void tm_metal_tok_add(int a, int b, unsigned n);
extern "C" void tm_metal_tok_silu_mul(int g, size_t goff, int u,
                                      size_t uoff, unsigned n);
extern "C" void tm_metal_tok_rope_q(int qslot, unsigned H, unsigned dh,
                                    int pos, float theta);
extern "C" void tm_metal_tok_rope_k_cache(int ksrcslot, size_t ksrc_off,
                                          int cacheslot, unsigned KVH,
                                          unsigned dh, unsigned ctx,
                                          int pos, float theta);
extern "C" void tm_metal_tok_v_cache(int vsrcslot, size_t vsrc_off,
                                     int cacheslot, unsigned KVH,
                                     unsigned dh, unsigned ctx, int pos);
extern "C" void tm_metal_tok_attn(int qslot, int kcslot, int vcslot,
                                  int probslot, int outslot, unsigned QD,
                                  unsigned H, unsigned KVH, unsigned dh,
                                  unsigned ctx, int allow, float scale,
                                  unsigned REP);
#endif  // __OBJC__
