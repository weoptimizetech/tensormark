// tensormark/metal.h — M6 optional Metal backend.
//
// ObjC++ ONLY: include from .mm translation units. The C++ core must never
// include this. Build with -DTM_HAVE_METAL; without it nothing here
// compiles and the core is Metal-free (spec criterion 1).
//
// Design notes:
//  - Runtime shader compilation via newLibraryWithSource: no offline
//    metallib build step. Source of truth is the kMetalSrc* raw strings
//    below. (A stale mirror, metal_kernels.metal - a 16x16 v1 tile whose
//    "mirrors metal.h" comment was already false - is deleted.)
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
#include <Accelerate/Accelerate.h>   // vImage fp16<->fp32 conversions (item 14)
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
#include <tuple>
#include <vector>

namespace tmgpu {

[[nodiscard]] inline std::atomic<unsigned long>& fallback_count() {
    static std::atomic<unsigned long> n{0};
    return n;
}

// The ONE failure exit for the recoverable entry points: bumps the fallback
// counter (spec: silent, measurable fallback - a counter, not a print) and
// hands back false. Every failure path reads `return fail_fallback();`, so the
// bump and the refusal cannot drift apart the way 30 separate
// `fail_fallback(); return false;` pairs let them.
[[nodiscard]] inline bool fail_fallback() {
    ++fallback_count();   // the choke point's own bump; not a fail_fallback call
    return false;
}
// Successful GPU prefill GEMM dispatches (tests assert the path engaged).
[[nodiscard]] inline std::atomic<unsigned long>& prefill_count() {
    static std::atomic<unsigned long> n{0};
    return n;
}

namespace detail {
// The process-lifetime Metal objects, in ONE aggregate instead of ten separate
// function-local statics. Each singleton used to initialise independently,
// which made their lifetimes an implicit contract (a non-ARC build already
// dangled one of them); one struct instance has exactly one lifetime and the
// accessors below keep every call site unchanged. The slot vector rides along
// so token state and device state are created and destroyed together.
struct Device {
    __strong id<MTLDevice> dev = nil;
    __strong id<MTLCommandQueue> q = nil;
    __strong id<MTLLibrary> lib = nil;
    __strong id<MTLLibrary> libq4 = nil;
    __strong id<MTLLibrary> libops = nil;
    __strong id<MTLBuffer> dq_scratch = nil;
    __strong id<MTLBuffer> a16_scratch = nil;
    __strong id<MTLBuffer> c16_scratch = nil;
    __strong id<MTLCommandBuffer> tcb = nil;
    __strong id<MTLComputeCommandEncoder> tenc = nil;
    std::vector<__strong id<MTLBuffer>> tslots;
};
[[nodiscard]] inline Device& device() { static Device d; return d; }
[[nodiscard]] inline __strong id<MTLDevice>& dev() { return device().dev; }
[[nodiscard]] inline __strong id<MTLCommandQueue>& q() { return device().q; }
[[nodiscard]] inline __strong id<MTLLibrary>& lib() { return device().lib; }
[[nodiscard]] inline __strong id<MTLLibrary>& libq4() { return device().libq4; }
[[nodiscard]] inline __strong id<MTLLibrary>& libops() { return device().libops; }
// One place that decides how the runtime sources are compiled. Passing nil left
// the MSL language version - and with it which half/simdgroup/packed builtins
// exist, and how the driver lowers them - to whatever default the running OS
// chose; these kernels use MSL 2.x/3.0 constructs, so the version they were
// written against is pinned. Math mode stays at the default fast setting the
// float arithmetic was tuned on.
[[nodiscard]] inline MTLCompileOptions* compile_options() {
    static MTLCompileOptions* o = [] {
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion3_0;
        return opts;
    }();
    return o;
}

// Every Metal entry point this header can load, as a type. Each name is written
// ONCE, next to the library that holds it: a dispatch costs an array index instead
// of constructing a std::string and scanning the names loaded so far (ops_pso did
// that ~14 times per token in the GPU-resident chain), and the pipelines live in
// one array sized by the enum instead of a node-per-kernel map. Entries are still
// created lazily, so a path that never dispatches a kernel never pays for it.
enum class Kern : uint8_t {
    embed_kq_row,
    f16_gemv_coop_r1,
    f16_gemv_coop_r2,
    f16_gemv_coop_r4,
    q4_1_gemv_coop_r1,
    q4_1_gemv_coop_r2,
    q4_1_gemv_coop_r4,
    q4_1_gemv_coop_r2_seg,
    q4_dequant_f16,
    q4_dequant_f32,
    q4_gemv,
    q4_gemv_coop,
    q4_gemv_coop_r2,
    q4_gemv_coop_r2_seg,
    q4_gemv_coop_r4,
    q4_gemv_mask_r2,
    q4_gemv_mask_r4,
    q4_mm_tile,
    q4_mm_tile2,
    q4_mm_tile3,
    q5k_gemv_coop_r1,
    q5k_gemv_coop_r2,
    q5k_gemv_coop_r4,
    q5k_gemv_coop_r2_seg,
    q6k_gemv_coop_r1,
    q6k_gemv_coop_r2,
    q6k_gemv_coop_r4,
    q6k_gemv_coop_r2_seg,
    sgemm_q4,
    sgemm_q4_small,
    tok_bias_qkv,
    add_inplace,
    argmax_f32,
    attn_decode_combine,
    attn_decode_fused,
    attn_decode_fused_h,
    attn_decode_gqa,
    attn_decode_gqa_h,
    attn_pv,
    attn_scores,
    attn_softmax,
    embed_h_row,
    embed_q4_row,
    gdn_conv_silu,
    gdn_decay,
    gdn_head_rms,
    gdn_l2_norm,
    gdn_silu_gate,
    gdn_step,
    gdn_step_f4,
    probe_copy,
    rmsnorm_row,
    rope_inplace,
    rope_k_cache,
    rope_k_cache_h,
    sample_gumbel_f32,
    silu_mul,
    v_to_cache,
    v_to_cache_h,
    cvt_f16_f32,
    cvt_f32_f16,
    Count,
};
struct KernInfo { const char* name; uint8_t lib; };   // 0 = libq4, 1 = libops, 2 = lib
inline constexpr KernInfo kKernInfo[] = {
    {"embed_kq_row", 0},
    {"f16_gemv_coop_r1", 0},
    {"f16_gemv_coop_r2", 0},
    {"f16_gemv_coop_r4", 0},
    {"q4_1_gemv_coop_r1", 0},
    {"q4_1_gemv_coop_r2", 0},
    {"q4_1_gemv_coop_r4", 0},
    {"q4_1_gemv_coop_r2_seg", 0},
    {"q4_dequant_f16", 0},
    {"q4_dequant_f32", 0},
    {"q4_gemv", 0},
    {"q4_gemv_coop", 0},
    {"q4_gemv_coop_r2", 0},
    {"q4_gemv_coop_r2_seg", 0},
    {"q4_gemv_coop_r4", 0},
    {"q4_gemv_mask_r2", 0},
    {"q4_gemv_mask_r4", 0},
    {"q4_mm_tile", 0},
    {"q4_mm_tile2", 0},
    {"q4_mm_tile3", 0},
    {"q5k_gemv_coop_r1", 0},
    {"q5k_gemv_coop_r2", 0},
    {"q5k_gemv_coop_r4", 0},
    {"q5k_gemv_coop_r2_seg", 0},
    {"q6k_gemv_coop_r1", 0},
    {"q6k_gemv_coop_r2", 0},
    {"q6k_gemv_coop_r4", 0},
    {"q6k_gemv_coop_r2_seg", 0},
    {"sgemm_q4", 0},
    {"sgemm_q4_small", 0},
    {"tok_bias_qkv", 0},
    {"add_inplace", 1},
    {"argmax_f32", 1},
    {"attn_decode_combine", 1},
    {"attn_decode_fused", 1},
    {"attn_decode_fused_h", 1},
    {"attn_decode_gqa", 1},
    {"attn_decode_gqa_h", 1},
    {"attn_pv", 1},
    {"attn_scores", 1},
    {"attn_softmax", 1},
    {"embed_h_row", 1},
    {"embed_q4_row", 1},
    {"gdn_conv_silu", 1},
    {"gdn_decay", 1},
    {"gdn_head_rms", 1},
    {"gdn_l2_norm", 1},
    {"gdn_silu_gate", 1},
    {"gdn_step", 1},
    {"gdn_step_f4", 1},
    {"probe_copy", 1},
    {"rmsnorm_row", 1},
    {"rope_inplace", 1},
    {"rope_k_cache", 1},
    {"rope_k_cache_h", 1},
    {"sample_gumbel_f32", 1},
    {"silu_mul", 1},
    {"v_to_cache", 1},
    {"v_to_cache_h", 1},
    {"cvt_f16_f32", 2},
    {"cvt_f32_f16", 2},
};
inline constexpr size_t kKernCount = sizeof(kKernInfo) / sizeof(kKernInfo[0]);
static_assert(kKernCount == (size_t)Kern::Count,
              "kKernInfo must name every Kern exactly once, in enum order");
[[nodiscard]] inline id<MTLLibrary> kern_lib(uint8_t which) {
    return which == 0 ? libq4() : which == 1 ? libops() : lib();
}
[[nodiscard]] inline __strong id<MTLComputePipelineState>& kern_slot(Kern k) {
    static __strong id<MTLComputePipelineState> slots[kKernCount];
    return slots[(size_t)k];
}
[[nodiscard]] inline id<MTLComputePipelineState> kern(Kern k) {
    __strong id<MTLComputePipelineState>& p = kern_slot(k);
    if (p) return p;
    // A miss is cached like a hit (the name-keyed caches did the same) so a
    // missing kernel does not re-load and re-report on every dispatch.
    static bool tried[kKernCount] = {};
    if (tried[(size_t)k]) return nil;
    tried[(size_t)k] = true;
    const KernInfo& info = kKernInfo[(size_t)k];
    id<MTLFunction> f = [kern_lib(info.lib)
        newFunctionWithName:[NSString stringWithUTF8String:info.name]];
    if (f) p = [dev() newComputePipelineStateWithFunction:f error:nil];
    if (!p) std::fprintf(stderr, "[metal] pipeline %s unavailable\n", info.name);
    return p;
}
// Name of an entry point, for diagnostics and tests (linear over the 55 names, so
// this is not a dispatch path).
[[nodiscard]] inline const char* kern_name(Kern k) { return kKernInfo[(size_t)k].name; }
[[nodiscard]] inline int kern_index(const char* name) {
    for (size_t i = 0; i < kKernCount; ++i)
        if (std::strcmp(kKernInfo[i].name, name) == 0) return (int)i;
    return -1;
}
[[nodiscard]] inline bool& tried() { static bool t = false; return t; }

// Cached, grow-only buffers: ba/bb inputs, bc output.
struct Bufs {
    __strong id<MTLBuffer> a, b, c;
};
[[nodiscard]] inline Bufs& bufs() { static Bufs b; return b; }

[[nodiscard]] inline id<MTLBuffer> grow(__strong id<MTLBuffer>& buf, size_t bytes) {
    if (!buf || [buf length] < bytes)
        buf = [dev() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return buf;
}

[[nodiscard]] inline NSString* kMetalSrc() {
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
)MET";
}

[[nodiscard]] inline NSString* kMetalSrcQ4() {
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
    uint N, uint K, constant uint* seg_rows,
    constant uint64_t* seg_woff, uint nseg, uint gid)
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
constant uint* seg_rows [[buffer(6)]],
constant uint64_t* seg_woff [[buffer(7)]],
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
// K-quant weight GEMV family (Q4_1 / Q5_K / Q6_K) — the dtypes a GGUF K-quant source
// keeps in the .tmq (quant.h), and exactly the ones the hybrid decode lane was
// missing: on the shipped Qwen3.5-0.8B, `linear_attn.out_proj` is Q5_K,
// `mlp.down_proj` is Q4_1 on layers 0-2, and lm_head/embed_tokens are Q6_K — the
// largest per-token read in the model. Same rows/lanes coop shape as the Q4 winner,
// so one lane can mix all four dtypes.
//
// These read fp32 activations, because that is what the token lane keeps in x; the
// affine dtypes (Q4_1, Q5_K) therefore carry a sum(x) term where the CPU kernels use
// the precomputed int8 activation sums. PER-ROW ACCUMULATION ORDER is lane-strided
// over super-blocks followed by a simd tree — the same shape as q4_gemv_rows_body. It
// cannot be bit-identical to the CPU kernels (which reduce int8 activations in a
// different order), so test_kquant_gemv BOUNDS the difference rather than asserting
// equality; the layout is what must be exact, and that is mirrored deliberately:
//   Q4_0 (above) is stored INTERLEAVED  — qs[j].lo -> element 2j,   .hi -> 2j+1
//   Q4_1 (here)  is stored SPLIT        — qs[j].lo -> element j,    .hi -> j+16
// Both are what quant.h's dequantizers and the CPU kernels do; the two conventions
// genuinely differ inside one container, and guessing wrong is silently wrong maths.
[[nodiscard]] inline float kq_f16(const device uchar* p) {   // little-endian f16 -> f32
    // The intermediate ushort is load-bearing: as_type<> takes a bit pattern, and an
    // expression that promotes to int is rejected outright ("cast from 'int' to
    // 'half' is not allowed"). Same shape as the Q4 kernels' scale load.
    const ushort bits = ushort(ushort(p[0]) | (ushort(p[1]) << 8));
    return float(as_type<half>(bits));
}

// Q4_1: 20-byte block, 32 values, w = q*d + m (d at 0, m at 2, qs at 4).
template <uint R>
inline void q4_1_gemv_rows_body(const device float* x, const device uchar* Wq,
                                device float* y, uint N, uint K, uint64_t woff, uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint nb = K >> 5;
    const uint rows = min(R, N - n0);
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nb; b += gemv_row_lanes) {
        const device float4* xb = (const device float4*)(x + b * 32);
        float4 xv[8]; float sumx = 0.0f;
        for (uint j = 0; j < 8; ++j) {
            xv[j] = xb[j];
            sumx += xv[j].x + xv[j].y + xv[j].z + xv[j].w;
        }
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* blk = Wq + woff + ((size_t)(n0 + r) * nb + b) * 20;
            const float d = kq_f16(blk);
            const float m = kq_f16(blk + 2);
            const device uchar* qs = blk + 4;
            float4 lo = 0.0f, hi = 0.0f;
            for (uint j = 0; j < 16; j += 4) {
                const uchar a0 = qs[j], a1 = qs[j + 1], a2 = qs[j + 2], a3 = qs[j + 3];
                lo += float4(int(a0 & 15), int(a1 & 15), int(a2 & 15), int(a3 & 15)) * xv[j / 4];
                hi += float4(int(a0 >> 4), int(a1 >> 4), int(a2 >> 4), int(a3 >> 4))
                      * xv[4 + j / 4];
            }
            acc[r] += d * (lo.x + lo.y + lo.z + lo.w + hi.x + hi.y + hi.z + hi.w) + m * sumx;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r];
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2) sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}

// The 6-bit scale/min of a Q5_K sub-block (ggml get_scale_min_k4, as quant.h's
// k_scale_min transcribes it): the first four sub-blocks take theirs from the low 6
// bits of scales[j] / scales[j+4], the last four from the low nibble of scales[j+4]
// with the top 2 bits of their own scale/min packed into the high bits of scales[j-4]
// and scales[j].
[[nodiscard]] inline uint2 kq5_scale_min(uint j, const device uchar* q) {
    if (j < 4u) return uint2(q[j] & 63u, q[j + 4] & 63u);
    return uint2((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4),
                 (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

// Four consecutive 5-bit codes as floats (0..31, no bias — Q5_K is affine). ql points
// at the 32-byte half of qs this sub-block's nibbles live in; the sub-block index
// picks the nibble and the qh bit, which is one bit per 64-value step.
[[nodiscard]] inline float4 kq5_codes(const device uchar* qg, const device uchar* qh, uint u, uint s) {
    float4 q;
    for (uint t = 0; t < 4; ++t) {
        const uchar qb = qg[u + t];
        const int nib = (s & 1u) ? int(qb >> 4) : int(qb & 0x0Fu);
        const int hb = int((qh[u + t] >> s) & 1u);
        q[t] = float(nib | (hb << 4));
    }
    return q;
}

// Q5_K: 176-byte super-block, 256 values in 8 sub-blocks of 32, w = d*sc*code - dmin*mn
// (d at 0, dmin at 2, scales at 4, qh at 16, qs at 48).
template <uint R>
inline void q5k_gemv_rows_body(const device float* x, const device uchar* Wq,
                               device float* y, uint N, uint K, uint64_t woff, uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint nbs = K >> 8;
    const uint rows = min(R, N - n0);
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nbs; b += gemv_row_lanes) {
        const device float* xb = x + b * 256;
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* blk = Wq + woff + ((size_t)(n0 + r) * nbs + b) * 176;
            const float d = kq_f16(blk);
            const float dmin = kq_f16(blk + 2);
            const device uchar* scales = blk + 4;
            const device uchar* qh = blk + 16;
            const device uchar* qs = blk + 48;
            float sum = 0.0f;
            for (uint s = 0; s < 8; ++s) {
                const uint2 scm = kq5_scale_min(s, scales);
                const device uchar* qg = qs + 32 * (s >> 1);
                const device float* xg = xb + 32 * s;
                float dot = 0.0f, sx = 0.0f;
                for (uint u = 0; u < 32; u += 4) {
                    const float4 xv = *(const device float4*)(xg + u);
                    sx += xv.x + xv.y + xv.z + xv.w;
                    const float4 q = kq5_codes(qg, qh, u, s);
                    dot += q.x * xv.x + q.y * xv.y + q.z * xv.z + q.w * xv.w;
                }
                // w = (d*sc)*code - dmin*mn, so the super-block scale multiplies the
                // CODE term only. Factoring `d` out of both terms is a 2%-scale error,
                // not a rounding difference — the CPU kernel and ggml both keep them
                // separate.
                sum += d * float(scm.x) * dot - dmin * float(scm.y) * sx;
            }
            acc[r] += sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r];
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2) sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}

// Four consecutive 6-bit codes as floats, centred (code - 32). `ql` already points at
// the right 32-byte half of ql for this quarter; J picks the quarter within the
// 128-value half and therefore the nibble half and the qh bit pair.
[[nodiscard]] inline float4 kq6_codes(const device uchar* ql, const device uchar* qh, uint l, uint J) {
    float4 q;
    for (uint t = 0; t < 4; ++t) {
        const uchar qlv = ql[l + t];
        const int nib = (J < 2u) ? int(qlv & 0x0Fu) : int(qlv >> 4);
        const int hi = int((qh[l + t] >> (2u * J)) & 3u);
        q[t] = float(nib | (hi << 4)) - 32.0f;
    }
    return q;
}

// Q6_K: 210-byte super-block, 256 values in 16 sub-blocks of 16, w = d*sc*(code-32)
// (ql at 0, qh at 128, int8 scales at 192, d at 208). The codes are permuted within
// each 128-value half exactly as quant.h's dequantize_row_q6_K and the CPU's
// kq6k_dot16 pin it: quarter J covers values [32J, 32J+16) of the half, reads the
// nibble half (J & 1) and the qh bits (2J), and takes sub-block scale 2J + (l/16).
template <uint R>
inline void q6k_gemv_rows_body(const device float* x, const device uchar* Wq,
                               device float* y, uint N, uint K, uint64_t woff, uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint nbs = K >> 8;
    const uint rows = min(R, N - n0);
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nbs; b += gemv_row_lanes) {
        const device float* xb = x + b * 256;
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* blk = Wq + woff + ((size_t)(n0 + r) * nbs + b) * 210;
            const float d = kq_f16(blk + 208);
            float sum = 0.0f;
            // `half` is a reserved type name in MSL, so this index is `hf`.
            for (uint hf = 0; hf < 2; ++hf) {
                const device uchar* qlh = blk + hf * 64;
                const device uchar* qhh = blk + 128 + hf * 32;
                const device uchar* sch = blk + 192 + hf * 8;
                const device float* xh = xb + hf * 128;
                for (uint g16 = 0; g16 < 2; ++g16) {   // l = 0..15, then 16..31
                    const uint l0 = g16 * 16;
                    for (uint j = 0; j < 4; ++j) {
                        const device uchar* ql = qlh + 32 * (j & 1u);
                        const device float* xj = xh + 32 * j + l0;
                        float4 a = 0.0f;
                        for (uint l = 0; l < 16; l += 4)
                            a += kq6_codes(ql, qhh, l0 + l, j)
                                 * *(const device float4*)(xj + l);
                        // scales[16] is signed int8 (`char` is signed in MSL) and the
                        // sub-block index within the half is 2J + l/16, exactly as
                        // dequantize_row_q6_K's `sc[is + 2J]` (is = l/16) has it.
                        sum += float(int((char)sch[2 * j + g16])) * (a.x + a.y + a.z + a.w);
                    }
                }
            }
            acc[r] += d * sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r];
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2) sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}
kernel void q4_1_gemv_coop_r1(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q4_1_gemv_rows_body<1>(x, Wq, y, N, K, woff, gid); }
kernel void q4_1_gemv_coop_r2(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q4_1_gemv_rows_body<2>(x, Wq, y, N, K, woff, gid); }
kernel void q4_1_gemv_coop_r4(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q4_1_gemv_rows_body<4>(x, Wq, y, N, K, woff, gid); }
kernel void q5k_gemv_coop_r1(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q5k_gemv_rows_body<1>(x, Wq, y, N, K, woff, gid); }
kernel void q5k_gemv_coop_r2(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q5k_gemv_rows_body<2>(x, Wq, y, N, K, woff, gid); }
kernel void q5k_gemv_coop_r4(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q5k_gemv_rows_body<4>(x, Wq, y, N, K, woff, gid); }
kernel void q6k_gemv_coop_r1(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q6k_gemv_rows_body<1>(x, Wq, y, N, K, woff, gid); }
kernel void q6k_gemv_coop_r2(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q6k_gemv_rows_body<2>(x, Wq, y, N, K, woff, gid); }
kernel void q6k_gemv_coop_r4(
const device float* x [[buffer(0)]], const device uchar* Wq [[buffer(1)]], device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint64_t& woff [[buffer(5)]],
uint gid [[thread_position_in_grid]]) { q6k_gemv_rows_body<4>(x, Wq, y, N, K, woff, gid); }
// Segmented variant: identical maths and summation order to the row body above,
// only the row->weight mapping comes from the per-dispatch segment table.
template <uint R>
inline void q4_1_gemv_rows_body_seg(const device float* x, const device uchar* Wq,
                                device float* y, uint N, uint K, constant uint* seg_rows, constant uint64_t* seg_woff,
                               uint nseg, uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint nb = K >> 5;
    uint rows = min(R, N - n0);
    uint local_n0 = n0;
    uint64_t woff = 0;
    if (nseg) {   // segmented dispatch: rows live in R-aligned segments,
                  // each with its own byte offset into the shared buffer
        uint seg = 0, seg_start = 0;
        for (uint s2 = 0; s2 < nseg; ++s2) {
            if (n0 < seg_start + seg_rows[s2]) { seg = s2; break; }
            seg_start += seg_rows[s2];
        }
        local_n0 = n0 - seg_start;
        woff = seg_woff[seg];
        rows = min(R, seg_rows[seg] - local_n0);
    }
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nb; b += gemv_row_lanes) {
        const device float4* xb = (const device float4*)(x + b * 32);
        float4 xv[8]; float sumx = 0.0f;
        for (uint j = 0; j < 8; ++j) {
            xv[j] = xb[j];
            sumx += xv[j].x + xv[j].y + xv[j].z + xv[j].w;
        }
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* blk = Wq + woff + ((size_t)(local_n0 + r) * nb + b) * 20;
            const float d = kq_f16(blk);
            const float m = kq_f16(blk + 2);
            const device uchar* qs = blk + 4;
            float4 lo = 0.0f, hi = 0.0f;
            for (uint j = 0; j < 16; j += 4) {
                const uchar a0 = qs[j], a1 = qs[j + 1], a2 = qs[j + 2], a3 = qs[j + 3];
                lo += float4(int(a0 & 15), int(a1 & 15), int(a2 & 15), int(a3 & 15)) * xv[j / 4];
                hi += float4(int(a0 >> 4), int(a1 >> 4), int(a2 >> 4), int(a3 >> 4))
                      * xv[4 + j / 4];
            }
            acc[r] += d * (lo.x + lo.y + lo.z + lo.w + hi.x + hi.y + hi.z + hi.w) + m * sumx;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r];
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2) sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}

kernel void q4_1_gemv_coop_r2_seg(
const device float* x [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]],
constant uint& K [[buffer(4)]],
constant uint& nseg [[buffer(5)]],
constant uint* seg_rows [[buffer(6)]],
constant uint64_t* seg_woff [[buffer(7)]],
uint gid [[thread_position_in_grid]])
{ q4_1_gemv_rows_body_seg<2>(x, Wq, y, N, K, seg_rows, seg_woff, nseg, gid); }

// Segmented variant: identical maths and summation order to the row body above,
// only the row->weight mapping comes from the per-dispatch segment table.
template <uint R>
inline void q5k_gemv_rows_body_seg(const device float* x, const device uchar* Wq,
                               device float* y, uint N, uint K, constant uint* seg_rows, constant uint64_t* seg_woff,
                               uint nseg, uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint nbs = K >> 8;
    uint rows = min(R, N - n0);
    uint local_n0 = n0;
    uint64_t woff = 0;
    if (nseg) {   // segmented dispatch: rows live in R-aligned segments,
                  // each with its own byte offset into the shared buffer
        uint seg = 0, seg_start = 0;
        for (uint s2 = 0; s2 < nseg; ++s2) {
            if (n0 < seg_start + seg_rows[s2]) { seg = s2; break; }
            seg_start += seg_rows[s2];
        }
        local_n0 = n0 - seg_start;
        woff = seg_woff[seg];
        rows = min(R, seg_rows[seg] - local_n0);
    }
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nbs; b += gemv_row_lanes) {
        const device float* xb = x + b * 256;
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* blk = Wq + woff + ((size_t)(local_n0 + r) * nbs + b) * 176;
            const float d = kq_f16(blk);
            const float dmin = kq_f16(blk + 2);
            const device uchar* scales = blk + 4;
            const device uchar* qh = blk + 16;
            const device uchar* qs = blk + 48;
            float sum = 0.0f;
            for (uint s = 0; s < 8; ++s) {
                const uint2 scm = kq5_scale_min(s, scales);
                const device uchar* qg = qs + 32 * (s >> 1);
                const device float* xg = xb + 32 * s;
                float dot = 0.0f, sx = 0.0f;
                for (uint u = 0; u < 32; u += 4) {
                    const float4 xv = *(const device float4*)(xg + u);
                    sx += xv.x + xv.y + xv.z + xv.w;
                    const float4 q = kq5_codes(qg, qh, u, s);
                    dot += q.x * xv.x + q.y * xv.y + q.z * xv.z + q.w * xv.w;
                }
                // w = (d*sc)*code - dmin*mn, so the super-block scale multiplies the
                // CODE term only. Factoring `d` out of both terms is a 2%-scale error,
                // not a rounding difference — the CPU kernel and ggml both keep them
                // separate.
                sum += d * float(scm.x) * dot - dmin * float(scm.y) * sx;
            }
            acc[r] += sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r];
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2) sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}

kernel void q5k_gemv_coop_r2_seg(
const device float* x [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]],
constant uint& K [[buffer(4)]],
constant uint& nseg [[buffer(5)]],
constant uint* seg_rows [[buffer(6)]],
constant uint64_t* seg_woff [[buffer(7)]],
uint gid [[thread_position_in_grid]])
{ q5k_gemv_rows_body_seg<2>(x, Wq, y, N, K, seg_rows, seg_woff, nseg, gid); }

// Segmented variant: identical maths and summation order to the row body above,
// only the row->weight mapping comes from the per-dispatch segment table.
template <uint R>
inline void q6k_gemv_rows_body_seg(const device float* x, const device uchar* Wq,
                               device float* y, uint N, uint K, constant uint* seg_rows, constant uint64_t* seg_woff,
                               uint nseg, uint gid) {
    const uint g = gid / gemv_row_lanes, lane = gid % gemv_row_lanes;
    const uint n0 = g * R;
    if (n0 >= N) return;
    const uint nbs = K >> 8;
    uint rows = min(R, N - n0);
    uint local_n0 = n0;
    uint64_t woff = 0;
    if (nseg) {   // segmented dispatch: rows live in R-aligned segments,
                  // each with its own byte offset into the shared buffer
        uint seg = 0, seg_start = 0;
        for (uint s2 = 0; s2 < nseg; ++s2) {
            if (n0 < seg_start + seg_rows[s2]) { seg = s2; break; }
            seg_start += seg_rows[s2];
        }
        local_n0 = n0 - seg_start;
        woff = seg_woff[seg];
        rows = min(R, seg_rows[seg] - local_n0);
    }
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = lane; b < nbs; b += gemv_row_lanes) {
        const device float* xb = x + b * 256;
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            const device uchar* blk = Wq + woff + ((size_t)(local_n0 + r) * nbs + b) * 210;
            const float d = kq_f16(blk + 208);
            float sum = 0.0f;
            // `half` is a reserved type name in MSL, so this index is `hf`.
            for (uint hf = 0; hf < 2; ++hf) {
                const device uchar* qlh = blk + hf * 64;
                const device uchar* qhh = blk + 128 + hf * 32;
                const device uchar* sch = blk + 192 + hf * 8;
                const device float* xh = xb + hf * 128;
                for (uint g16 = 0; g16 < 2; ++g16) {   // l = 0..15, then 16..31
                    const uint l0 = g16 * 16;
                    for (uint j = 0; j < 4; ++j) {
                        const device uchar* ql = qlh + 32 * (j & 1u);
                        const device float* xj = xh + 32 * j + l0;
                        float4 a = 0.0f;
                        for (uint l = 0; l < 16; l += 4)
                            a += kq6_codes(ql, qhh, l0 + l, j)
                                 * *(const device float4*)(xj + l);
                        // scales[16] is signed int8 (`char` is signed in MSL) and the
                        // sub-block index within the half is 2J + l/16, exactly as
                        // dequantize_row_q6_K's `sc[is + 2J]` (is = l/16) has it.
                        sum += float(int((char)sch[2 * j + g16])) * (a.x + a.y + a.z + a.w);
                    }
                }
            }
            acc[r] += d * sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float sum = acc[r];
        for (uint hop = gemv_row_lanes / 2; hop; hop /= 2) sum += simd_shuffle_down(sum, hop);
        if (lane == 0 && r < rows) y[n0 + r] = sum;
    }
}

kernel void q6k_gemv_coop_r2_seg(
const device float* x [[buffer(0)]],
const device uchar* Wq [[buffer(1)]],
device float* y [[buffer(2)]],
constant uint& N [[buffer(3)]],
constant uint& K [[buffer(4)]],
constant uint& nseg [[buffer(5)]],
constant uint* seg_rows [[buffer(6)]],
constant uint64_t* seg_woff [[buffer(7)]],
uint gid [[thread_position_in_grid]])
{ q6k_gemv_rows_body_seg<2>(x, Wq, y, N, K, seg_rows, seg_woff, nseg, gid); }
// One row of a K-quant embedding table, dequantized into x. A decode step reads exactly
// ONE row per token, so a per-element gather is the right shape: there is no reduction to
// amortise, and the row is ~4 KB against the hundreds of MB the layer stack streams. It
// exists because the token id stays ON THE GPU (the chain's argmax writes it into the
// token slot), so a CPU row read would need a download per token and would end the chain
// — and the chain is where the GPU lane's advantage has always lived. Buffer order and
// tok[idx] addressing mirror embed_h_row; the element layout mirrors the GEMV bodies.
kernel void embed_kq_row(
const device uint* tok [[buffer(0)]], constant uint& idx [[buffer(1)]],
const device uchar* Wq [[buffer(2)]], constant uint64_t& woff [[buffer(3)]],
device float* x [[buffer(4)]], constant uint& D [[buffer(5)]],
constant uint& dtype [[buffer(6)]],
uint gid [[thread_position_in_grid]]) {
    if (gid >= D) return;
    const uint row = tok[idx];
    if (row == 0xFFFFFFFFu) { x[gid] = 0.0f; return; }   // unwritten sentinel
    const device uchar* base = Wq + woff;
    const uint k = gid;
    if (dtype == 4u) {                                   // Q4_1: 20 B block, 32 values
        const uint nb = D >> 5;
        const device uchar* blk = base + (size_t)row * nb * 20 + (size_t)(k >> 5) * 20;
        const uint j = k & 31u;
        const uchar qb = blk[4 + (j < 16u ? j : j - 16u)];
        const int q = (j < 16u) ? int(qb & 0x0Fu) : int(qb >> 4);
        x[k] = float(q) * kq_f16(blk) + kq_f16(blk + 2);
        return;
    }
    if (dtype == 5u) {                                   // Q5_K: 176 B, 256 values
        const uint nb = D >> 8;
        const device uchar* blk = base + (size_t)row * nb * 176 + (size_t)(k >> 8) * 176;
        const uint s = (k & 255u) >> 5, u = k & 31u;
        const uint2 scm = kq5_scale_min(s, blk + 4);
        const uchar qb = blk[48 + 32 * (s >> 1) + u];
        const int nib = (s & 1u) ? int(qb >> 4) : int(qb & 0x0Fu);
        const int hb = int((blk[16 + u] >> s) & 1u);
        x[k] = kq_f16(blk) * float(scm.x) * float(nib | (hb << 4)) -
               kq_f16(blk + 2) * float(scm.y);
        return;
    }
    // Q6_K: 210 B, 256 values — ql 64 B, qh 32 B and scales 8 B per 128-value half, d
    // last; the codes are permuted within the half exactly as the GEMV body has it.
    const uint nb = D >> 8;
    const device uchar* blk = base + (size_t)row * nb * 210 + (size_t)(k >> 8) * 210;
    const uint rr = k & 255u, hf = rr >> 7, p = rr & 127u;
    const uint J = p >> 5, l = p & 31u;
    const device uchar* ql = blk + hf * 64 + 32 * (J & 1u);
    const device uchar* qh = blk + 128 + hf * 32;
    const device uchar* sc = blk + 192 + hf * 8;
    const uchar qb = ql[l];
    const int nib = (J < 2u) ? int(qb & 0x0Fu) : int(qb >> 4);
    const int hi = int((qh[l] >> (2u * J)) & 3u);
    x[k] = kq_f16(blk + 208) * float(int((char)sc[2 * J + (l >> 4)])) *
           (float(nib | (hi << 4)) - 32.0f);
}
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
// Qwen2-family attention bias: y[i] += bias[i] over the fused q|k|v output
// (one contiguous [q | k | v] row, N = QD + 2*KVD). bias is bound at the
// per-layer byte offset; one thread per element.
kernel void tok_bias_qkv(
device float*       y     [[buffer(0)]],
const device float* bias  [[buffer(1)]],
constant uint&      N     [[buffer(2)]],
uint gid [[thread_position_in_grid]])
{
    if (gid < N) y[gid] += bias[gid];
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
[[nodiscard]] inline id<MTLComputePipelineState> q4_small_pso() {
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
[[nodiscard]] inline bool q4_small_off() {
    static const bool off = [] { const char* e = std::getenv("TM_Q4_SMALL"); return !(e && e[0] == '1'); }();
    return off;
}

// MPS-GEMM prefill (2026-09-07): dequantize the Q4 blob into a per-call
// fp32 scratch on the GPU, then run MPSMatrixMultiplication on it. On this
// M1 the tiled Q4 kernel above reaches ~550 GFLOP/s; MPS reaches 1.4
// TFLOP/s on the same shapes (T=2000: 7 GEMMs 120 ms vs the whole layer's
// ~370 ms before). Scratch is the largest single tensor (46 MB fp32), never
// a resident copy. TM_PREFILL_MPS=0 restores the tiled kernel.
[[nodiscard]] inline bool prefill_mps_on() {
    static const bool on = [] { const char* e = std::getenv("TM_PREFILL_MPS"); return !(e && e[0] == '0'); }();
    return on;
}
[[nodiscard]] inline id<MTLComputePipelineState> q4_dequant_pso() {
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
[[nodiscard]] inline __strong id<MTLBuffer>& dq_scratch() { return device().dq_scratch; }
// Zero-copy view of a host activation matrix (2026-09-07): page-aligned
// host memory (every large malloc on macOS) is handed to the GPU as a
// no-copy shared buffer, length rounded up to the page (malloc's large
// allocations are page-granular). Saves the memcpy in/out per GEMM — ~5 GB
// of copies per 2000-token TinyLlama prefill. Not cached: the view is
// created per call (tens of us) so a freed/reallocated vector can never be
// aliased by a stale buffer; falls back to the copy path when unaligned.
[[nodiscard]] inline id<MTLBuffer> host_view(const void* p, size_t bytes) {
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
[[nodiscard]] inline bool prefill_mps_f16(unsigned M) {
    static const int mode = [] { const char* e = std::getenv("TM_PREFILL_MPS_F16"); return e && *e ? (e[0] == '1' ? 1 : 0) : -1; }();
    return mode < 0 ? M >= 1024 : mode == 1;
}
[[nodiscard]] inline __strong id<MTLBuffer>& a16_scratch() { return device().a16_scratch; }
[[nodiscard]] inline __strong id<MTLBuffer>& c16_scratch() { return device().c16_scratch; }
// Encodes dequant(blob at woff, N x K) + C(M x N) = A(M x K) . W^T into cb.
// a_half / c_half (layer-stack use): the A buffer already holds half and/or
// C must be left as half in bc — no conversion passes. Both imply the fp16
// GEMM regardless of M.
[[nodiscard]] inline bool encode_mps_q4_gemm(id<MTLCommandBuffer> cb, id<MTLBuffer> bw, size_t woff,
                               id<MTLBuffer> ba, id<MTLBuffer> bc, size_t coff,
                               unsigned M, unsigned N, unsigned K, bool a_half = false, bool c_half = false) {
    const bool f16 = a_half || c_half || prefill_mps_f16(M);
    const size_t es = f16 ? 2 : 4;
    id<MTLComputePipelineState> pso = f16 ? kern(Kern::q4_dequant_f16) : q4_dequant_pso();
    if (!pso) return false;
    id<MTLBuffer> W = grow(dq_scratch(), es * (size_t)N * K);
    if (!W) return false;
    id<MTLBuffer> A = ba, C = bc; size_t cofs = coff;
    id<MTLComputePipelineState> pcvt_in = nil, pcvt_out = nil;
    if (f16) {
        if (!a_half) { pcvt_in = kern(Kern::cvt_f32_f16); A = grow(a16_scratch(), 2 * (size_t)M * K); if (!(pcvt_in && A)) return false; }
        if (!c_half) { pcvt_out = kern(Kern::cvt_f16_f32); C = grow(c16_scratch(), 2 * (size_t)M * N); if (!(pcvt_out && C)) return false; cofs = 0; }
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
[[nodiscard]] inline int q4_mm_variant(unsigned M, unsigned) { return M <= 64 ? 1 : 2; }
[[nodiscard]] inline bool encode_q4_mm_tile(id<MTLCommandBuffer> cb, id<MTLBuffer> bw, size_t woff,
                              id<MTLBuffer> ba, id<MTLBuffer> bc, size_t coff,
                              unsigned M, unsigned N, unsigned K, int variant = 0) {
    if (K % 32) return false;
    if (variant < 1 || variant > 3) variant = q4_mm_variant(M, N);
    if (variant == 3 && K % 64) variant = 2;
    id<MTLComputePipelineState> pso = kern(variant == 3 ? Kern::q4_mm_tile3 : variant == 2 ? Kern::q4_mm_tile2 : Kern::q4_mm_tile);
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
[[nodiscard]] inline bool ensure_init() {
    if (tried()) return dev() != nil;
    tried() = true;
    @autoreleasepool {
        dev() = MTLCreateSystemDefaultDevice();
        if (!dev()) return false;
        q() = [dev() newCommandQueue];
        if (!q()) { dev() = nil; return false; }
        NSError* err = nil;
        lib() = [dev() newLibraryWithSource:kMetalSrc() options:compile_options() error:&err];
        if (!lib()) {
            std::fprintf(stderr, "[metal] library compile failed: %s\n",
                         err.localizedDescription.UTF8String);
            dev() = nil;
            return false;
        }
        libq4() = [dev() newLibraryWithSource:kMetalSrcQ4() options:compile_options() error:&err];
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
[[nodiscard]] inline bool sgemm(const float* A, const float* B, float* C,
                  unsigned M, unsigned N, unsigned K) {
    if (!detail::ensure_init()) { return fail_fallback(); }
    @autoreleasepool {
        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> ba = detail::grow(bu.a, sizeof(float) * M * K);
        id<MTLBuffer> bb = detail::grow(bu.b, sizeof(float) * K * N);
        id<MTLBuffer> bc = detail::grow(bu.c, sizeof(float) * M * N);
        if (!(ba && bb && bc)) { return fail_fallback(); }
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
        if (!(mm)) { return fail_fallback(); }
        id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb
                    resultMatrix:mc];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            (void)fail_fallback();
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
[[nodiscard]] inline bool sgemm_f16(const float* A, const float* B, float* C,
                      unsigned M, unsigned N, unsigned K) {
    if (!detail::ensure_init()) { return fail_fallback(); }
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
            if (!(bf32 && bb16)) { return fail_fallback(); }
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
                (void)fail_fallback();
                return false;
            }
            f16_cache.push_back({{B, nb}, bb16});
        }

        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> ba32 = detail::grow(bu.a, sizeof(float) * M * K);
        id<MTLBuffer> ba16 = detail::grow(bu.b, 2ull * M * K);
        id<MTLBuffer> bc16 = detail::grow(bu.c, 2ull * M * N);
        id<MTLBuffer> bc32 = detail::grow(bu.a, sizeof(float) * M * N);
        if (!(ba32 && ba16 && bc16 && bc32)) { return fail_fallback(); }
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
            (void)fail_fallback();
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
        if (!(pc && pu)) { return fail_fallback(); }
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
            (void)fail_fallback();
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
[[nodiscard]] inline bool sgemm_q4(const float* A, const void* Wq, float* C,
                     unsigned M, unsigned N, unsigned K) {
    if (!detail::ensure_init()) { return fail_fallback(); }
    if (K % 32) { return fail_fallback(); }
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
            if (!bw) { return fail_fallback(); }
            wq_cache.push_back({{Wq, bytes}, bw});
        }
        if (!pso) {
            id<MTLFunction> f =
                [detail::libq4() newFunctionWithName:@"sgemm_q4"];
            pso = [detail::dev()
                newComputePipelineStateWithFunction:f error:nil];
            if (!pso) { return fail_fallback(); }
        }
        id<MTLComputePipelineState> pso_small = detail::q4_small_pso();
        const bool small = M <= 32 && pso_small != nil && !detail::q4_small_off();
        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> ba = detail::host_view(A, sizeof(float) * M * K);
        id<MTLBuffer> bc = detail::host_view(C, sizeof(float) * M * N);
        const bool a_view = ba != nil, c_view = bc != nil;
        if (!ba) ba = detail::grow(bu.a, sizeof(float) * M * K);
        if (!bc) bc = detail::grow(bu.c, sizeof(float) * M * N);
        if (!(ba && bc)) { return fail_fallback(); }
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
            (void)fail_fallback();
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
[[nodiscard]] inline bool q4_gemv(const float* x, const void* Wq, float* y,
                    unsigned N, unsigned K) {
    const uint64_t woff64 = 0;   // copy-mode wrapper: weights contiguous
    if (!detail::ensure_init()) { return fail_fallback(); }
    if (K % 32) { return fail_fallback(); }
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
            if (!bw) { return fail_fallback(); }
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
            if (!(pn && pk)) { return fail_fallback(); }
            consts.push_back({{N, K}, {pn, pk}});
        }
        if (!pso) {
            id<MTLFunction> f = [detail::libq4() newFunctionWithName:@"q4_gemv"];
            pso = [detail::dev()
                newComputePipelineStateWithFunction:f error:nil];
            if (!pso) { return fail_fallback(); }
        }
        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> bx = detail::grow(bu.a, sizeof(float) * K);
        id<MTLBuffer> by = detail::grow(bu.c, sizeof(float) * N);
        if (!(bx && by)) { return fail_fallback(); }
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
            (void)fail_fallback();
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
[[nodiscard]] inline bool q4_prefill(const float* A, const void* const* Wqs,
                       const unsigned* Ns, unsigned nparts, float* C,
                       unsigned M, unsigned K) {
    if (!detail::ensure_init()) { return fail_fallback(); }
    if (K % 32 || !nparts) { return fail_fallback(); }
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
            if (!bw) { return fail_fallback(); }
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
            if (!pso) { return fail_fallback(); }
        }
        id<MTLComputePipelineState> pso_small = detail::q4_small_pso();
        const bool small = M <= 32 && pso_small != nil && !detail::q4_small_off();
        detail::Bufs& bu = detail::bufs();
        id<MTLBuffer> ba = detail::host_view(A, sizeof(float) * M * K);
        id<MTLBuffer> bc = detail::host_view(C, sizeof(float) * M * Ntot);
        const bool a_view = ba != nil, c_view = bc != nil;
        if (!ba) ba = detail::grow(bu.a, sizeof(float) * M * K);
        if (!bc) bc = detail::grow(bu.c, sizeof(float) * M * Ntot);
        if (!(ba && bc)) { return fail_fallback(); }
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
            (void)fail_fallback();
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
using detail::libops;
using detail::compile_options;
using detail::Kern;
using detail::kern;
using detail::kern_name;
using detail::kern_index;
using detail::ensure_init;
using detail::device;

[[nodiscard]] inline std::vector<__strong id<MTLBuffer>>& tslots() { return device().tslots; }
[[nodiscard]] inline __strong id<MTLComputeCommandEncoder>& tenc();
[[nodiscard]] inline id<MTLBuffer> slot_buf(int sid, size_t bytes) {
    if (sid < 0) throw std::invalid_argument("metal: negative slot id");
    auto& s = tslots();
    if ((int)s.size() <= sid) s.resize((size_t)sid + 1);
    if (!s[sid] || [s[sid] length] < bytes) {
        if (s[sid] && tenc())
            throw std::runtime_error("metal: reserve packed slots before token encoding");
        if (!bytes || bytes > dev().maxBufferLength)
            throw std::runtime_error("metal: invalid slot buffer size: slot " +
                                     std::to_string(sid) + " requests " +
                                     std::to_string(bytes) + " bytes (max " +
                                     std::to_string((unsigned long long)dev().maxBufferLength) +
                                     ")");
        id<MTLBuffer> b = [dev() newBufferWithLength:bytes
                          options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("metal: slot allocation failed");
        s[sid] = b;
    }
    return s[sid];
}
[[nodiscard]] inline __strong id<MTLCommandBuffer>& tcb() { return device().tcb; }
[[nodiscard]] inline __strong id<MTLComputeCommandEncoder>& tenc() { return device().tenc; }
// Scalar kernel arguments (pos, allow, chunk, dims, byte offsets, the chained
// token index) are bound with setBytes, which COPIES the value into the command
// buffer at encode time: each dispatch keeps its own snapshot even when a whole
// token's dispatches share one command buffer. The earlier revision cached one
// 4- or 8-byte MTLBuffer per distinct value in a grow-only vector found by a
// linear scan, so a per-token value (pos, allow, chunk, the chain index) made
// the scan grow with the context and left one MTLBuffer alive per distinct value
// for the lifetime of the process.
template <class T>
inline void set_scalar(T v, NSUInteger index) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "scalar kernel arguments are 32- or 64-bit");
    [tenc() setBytes:&v length:sizeof(T) atIndex:index];
}
// PSO for q4_gemv (lives in libq4), shared with q4_gemv().
[[nodiscard]] inline __strong id<MTLComputePipelineState>& gemv_pso() {
    static __strong id<MTLComputePipelineState> p; return p;
}

// The attention decode kernels (attn_decode_fused{,_h}, attn_decode_gqa{,_h},
// attn_decode_combine) share one 32-byte argument struct - AttnArgs, generated
// from the TM_ATTN_ARGS X-macro near tok_attn. attn_args_msl() renders the MSL
// mirror of the same X-macro and it is prepended to this source, so the host
// setBytes layout and the kernel signature cannot drift field-by-field.
[[nodiscard]] NSString* attn_args_msl();  // defined near tok_attn

[[nodiscard]] inline NSString* kMetalSrcOps() {
return [attn_args_msl() stringByAppendingString:@R"MET(
#include <metal_stdlib>
using namespace metal;
// Attention decode specialization: dh and REP are baked per pipeline by
// attention_pso() (function constants 0/1); every other scalar rides the
// AttnArgs struct, so a per-shape PSO costs one cache key, not one source.
constant uint fc_dh [[function_constant(0)]];
constant uint fc_rep [[function_constant(1)]];
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
[[nodiscard]] inline uint tm_hash32(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
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
// RoPE in place over heads*(dh/2) pairs; pairs are (r[j], r[j+dh/2]) within
// each head. The angle is NOT computed here: it is read from the host-built
// (cos, sin) table at row*stride + j, the same table ll_rope/ll_rope_h read, so
// the decode lane and the prefill lane rotate a position by the same bits.
kernel void rope_inplace(
device float* q [[buffer(0)]], constant uint& heads [[buffer(1)]],
constant uint& dh [[buffer(2)]], constant int& pos [[buffer(3)]],
const device float2* tbl [[buffer(4)]], constant uint& tbl_n [[buffer(5)]],
uint gid [[thread_position_in_grid]])
{
uint h = gid / (dh / 2), j = gid % (dh / 2);
if (h >= heads) return;
const uint stride = dh / 2;
if (pos < 0 || (uint)pos * stride + j >= tbl_n) return;
const float2 cs = tbl[(size_t)(uint)pos * stride + j];
device float* r = q + (size_t)h * dh;
float a = r[j], b = r[j + dh / 2];
r[j] = a * cs.x - b * cs.y;
r[j + dh / 2] = b * cs.x + a * cs.y;
}
// Rotate the k part (read from the qkv activation slot) and scatter it
// straight into the per-layer KV cache at row pos.
template <typename CacheT>
inline void rope_k_cache_body(const device float* ksrc, device CacheT* cache,
                              uint KVH, uint dh, uint ctx, int pos, const device float2* tbl, uint tbl_n, uint gid)
{
uint kv = gid / (dh / 2), j = gid % (dh / 2);
if (kv >= KVH) return;
const uint stride = dh / 2;
if (pos < 0 || (uint)pos * stride + j >= tbl_n) return;
const float2 cs = tbl[(size_t)(uint)pos * stride + j];
const device float* r = ksrc + (size_t)kv * dh;
device CacheT* dst = cache + (size_t)kv * ctx * dh + (size_t)pos * dh;
float a = r[j], b = r[j + dh / 2];
dst[j] = (CacheT)(a * cs.x - b * cs.y);
dst[j + dh / 2] = (CacheT)(b * cs.x + a * cs.y);
}
kernel void rope_k_cache(
const device float* ksrc [[buffer(0)]],
device float* cache [[buffer(1)]],
constant uint& KVH [[buffer(2)]], constant uint& dh [[buffer(3)]],
constant uint& ctx [[buffer(4)]], constant int& pos [[buffer(5)]],
const device float2* tbl [[buffer(6)]], constant uint& tbl_n [[buffer(7)]],
uint gid [[thread_position_in_grid]])
{ rope_k_cache_body(ksrc, cache, KVH, dh, ctx, pos, tbl, tbl_n, gid); }
kernel void rope_k_cache_h(
const device float* ksrc [[buffer(0)]],
device half* cache [[buffer(1)]],
constant uint& KVH [[buffer(2)]], constant uint& dh [[buffer(3)]],
constant uint& ctx [[buffer(4)]], constant int& pos [[buffer(5)]],
const device float2* tbl [[buffer(6)]], constant uint& tbl_n [[buffer(7)]],
uint gid [[thread_position_in_grid]])
{ rope_k_cache_body(ksrc, cache, KVH, dh, ctx, pos, tbl, tbl_n, gid); }
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
constant AttnArgs& a [[buffer(4)]],
threadgroup float* probs [[threadgroup(0)]],
uint h [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    threadgroup float partials[128];
    attn_decode_fused_body(q, kcache, vcache, out, fc_dh, a.ctx, a.allow, a.scale, fc_rep, probs, reductions, partials, h, ti, lane, sg);
}
kernel void attn_decode_fused_h(
const device float* q [[buffer(0)]],
const device half* kcache [[buffer(1)]],
const device half* vcache [[buffer(2)]],
device float* out [[buffer(3)]],
constant AttnArgs& a [[buffer(4)]],
threadgroup float* probs [[threadgroup(0)]],
uint h [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    threadgroup float partials[128];
    attn_decode_fused_body(q, kcache, vcache, out, fc_dh, a.ctx, a.allow, a.scale, fc_rep, probs, reductions, partials, h, ti, lane, sg);
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
[[nodiscard]] inline float gqa_dot(const threadgroup float4* q4, const device K4* k4) {
    float4 a = 0.0f, b = 0.0f;
#pragma unroll
    for (uint j = 0; j < DH4; j += 2) { a += q4[j] * float4(k4[j]); b += q4[j + 1] * float4(k4[j + 1]); }
    const float4 c = a + b;
    return c.x + c.y + c.z + c.w;
}
template <typename K4>
[[nodiscard]] inline float gqa_dot_any(const threadgroup float4* q4, const device K4* k4, uint dh4) {
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
constant AttnArgs& a [[buffer(5)]],
threadgroup float* shm [[threadgroup(0)]],
uint2 tg [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{ attn_decode_gqa_body<float, float4>(q, kcache, vcache, part, stats, fc_dh, a.ctx, a.allow, a.scale, fc_rep, a.nsplit, a.chunk, a.H, shm, tg, ti, lane, sg); }
kernel void attn_decode_gqa_h(
const device float* q [[buffer(0)]],
const device half* kcache [[buffer(1)]],
const device half* vcache [[buffer(2)]],
device float* part [[buffer(3)]],
device float* stats [[buffer(4)]],
constant AttnArgs& a [[buffer(5)]],
threadgroup float* shm [[threadgroup(0)]],
uint2 tg [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{ attn_decode_gqa_body<half, half4>(q, kcache, vcache, part, stats, fc_dh, a.ctx, a.allow, a.scale, fc_rep, a.nsplit, a.chunk, a.H, shm, tg, ti, lane, sg); }
// out[h][d] = sum_i w_i part_i / sum_i w_i l_i, w_i = exp(m_i - max m).
kernel void attn_decode_combine(
const device float* part [[buffer(0)]],
const device float* stats [[buffer(1)]],
device float* out [[buffer(2)]],
constant AttnArgs& a [[buffer(3)]],
uint gid [[thread_position_in_grid]])
{
    const uint h = gid / a.dh, d = gid - h * a.dh;
    if (h >= a.H) return;
    float m = -INFINITY;
    for (uint i = 0; i < a.nsplit; ++i) m = max(m, stats[((size_t)i * a.H + h) * 2]);
    float num = 0.0f, den = 0.0f;
    for (uint i = 0; i < a.nsplit; ++i) {
        const float w = exp(stats[((size_t)i * a.H + h) * 2] - m);
        num += w * part[((size_t)i * a.H + h) * a.dh + d];
        den += w * stats[((size_t)i * a.H + h) * 2 + 1];
    }
    out[(size_t)h * a.dh + d] = num / den;
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (qwen35 linear attention) — one decode token.
//
// The layer's five projections (qkv, gate, alpha, beta, out) reuse the existing
// GEMV kernels. These five cover the rest: the causal depthwise conv, the
// per-head L2 norm, the per-head log-decay, the delta-rule recurrence and the
// gated normaliser.
//
// The recurrent state S is stored TRANSPOSED relative to the CPU kernel —
// S[hv][dv][dk] — so that each thread owns a contiguous Dk row. One thread per
// (value head, value dim) then needs no cross-thread communication at all:
// contraction and update are both strided over dk in its own row.
// ---------------------------------------------------------------------------

// Causal depthwise conv over `channels` of the fused q|k|v stream, then SiLU.
// `state` holds the previous K-1 rows (oldest first) and is rolled in place.
kernel void gdn_conv_silu(
const device float* in [[buffer(0)]],
device float* state [[buffer(1)]],
const device float* kern [[buffer(2)]],
device float* out [[buffer(3)]],
constant uint& channels [[buffer(4)]], constant uint& K [[buffer(5)]],
uint c [[thread_position_in_grid]])
{
    if (c >= channels || K < 2) return;
    const uint P = K - 1;
    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        const float xi = (j < P) ? state[(size_t)j * channels + c]
                                 : in[(size_t)(j - P) * channels + c];
        acc += kern[(size_t)j * channels + c] * xi;
    }
    for (uint j = 0; j + 1 < P; ++j)
        state[(size_t)j * channels + c] = state[(size_t)(j + 1) * channels + c];
    state[(size_t)(P - 1) * channels + c] = in[c];
    out[c] = acc / (1.0f + exp(-acc));
}

// Per-head L2 normalisation (DeltaNet q/k norm): x / max(||x||, eps), with an
// optional scale folded into the reciprocal (q gets 1/sqrt(Dk), k gets 1).
// One threadgroup per head, two-level SIMD reduction, same shape as rmsnorm_row.
kernel void gdn_l2_norm(
const device float* src [[buffer(0)]],
device float* dst [[buffer(1)]],
constant uint& H [[buffer(2)]], constant uint& D [[buffer(3)]],
constant uint& src_off [[buffer(4)]],
constant float& scale [[buffer(5)]], constant float& eps [[buffer(6)]],
constant uint& dst_off [[buffer(7)]],
uint tg [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint tpg [[threads_per_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
    if (tg >= H) return;
    const device float* x = src + src_off + (size_t)tg * D;
    // dst_off, not 0: the production call normalises q|k in place inside one
    // packed scratch, so k lives at qn_l and a fixed offset-0 write would clobber
    // q. In-place is safe: every element is read into the sum before the barrier,
    // and y[i] = x[i] * inv only ever touches its own element afterwards.
    device float* y = dst + dst_off + (size_t)tg * D;
    threadgroup float ss[8];
    float p = 0.0f;
    for (uint i = ti; i < D; i += tpg) p += x[i] * x[i];
    p = simd_sum(p);
    if (lane == 0) ss[sg] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < tpg / 32) ? ss[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) ss[0] = scale / max(sqrt(v), eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = ss[0];
    for (uint i = ti; i < D; i += tpg) y[i] = x[i] * inv;
}

// Per value head decay = exp(ssm_a * softplus(alpha + dt)). `a` is the RAW
// ssm_a from the gguf, which is already -exp(A_log).
kernel void gdn_decay(
const device float* alpha [[buffer(0)]],
const device float* dt [[buffer(1)]],
const device float* a [[buffer(2)]],
device float* decay [[buffer(3)]],
constant uint& Hv [[buffer(4)]],
uint gid [[thread_position_in_grid]])
{
    if (gid >= Hv) return;
    const float x = alpha[gid] + dt[gid];
    const float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    decay[gid] = exp(a[gid] * sp);
}

// The gated delta rule, one thread per (value head hv, value dim dv):
//     S[hv] *= decay;  sk = S[hv] . k[hk];
//     delta = beta[hv] * (v[hv,dv] - sk);
//     S[hv] += k[hk] (x) delta;  o[hv,dv] = S[hv] . q[hk]
// with hk = hv % Hk (ggml_repeat tiles the shared key heads).
kernel void gdn_step(
device float* S [[buffer(0)]],
const device float* q [[buffer(1)]],
const device float* k [[buffer(2)]],
const device float* v [[buffer(3)]],
const device float* beta [[buffer(4)]],
const device float* decay [[buffer(5)]],
device float* o [[buffer(6)]],
constant uint& Hk [[buffer(7)]], constant uint& Hv [[buffer(8)]],
constant uint& Dk [[buffer(9)]], constant uint& Dv [[buffer(10)]],
uint gid [[thread_position_in_grid]])
{
    const uint hv = gid / Dv, dv = gid - hv * Dv;
    if (hv >= Hv || dv >= Dv || Dk == 0) return;
    const uint hk = hv % Hk;
    const device float* kh = k + (size_t)hk * Dk;
    const device float* qh = q + (size_t)hk * Dk;
    device float* row = S + ((size_t)hv * Dv + dv) * (size_t)Dk;
    const float dec = decay[hv];
    float sk = 0.0f;
    for (uint dk = 0; dk < Dk; ++dk) { row[dk] *= dec; sk += row[dk] * kh[dk]; }
    const float delta = beta[hv] * (v[(size_t)hv * Dv + dv] - sk);
    float acc = 0.0f;
    for (uint dk = 0; dk < Dk; ++dk) { row[dk] += kh[dk] * delta; acc += row[dk] * qh[dk]; }
    o[(size_t)hv * Dv + dv] = acc;
}

// Same recurrence, four state elements per lane op. Only valid when Dk % 4 == 0
// (every row then starts on a 16-byte boundary) and the q/k slot offsets are
// 16-byte aligned - the host wrapper checks both and falls back to gdn_step.
// The dot products accumulate four lanes and reduce once, so the summation
// order differs from the scalar kernel by at most one partial-sum level; the
// oracle's 3e-3 step tolerance absorbs it (measured below).
kernel void gdn_step_f4(
device float* S [[buffer(0)]],
const device float* q [[buffer(1)]],
const device float* k [[buffer(2)]],
const device float* v [[buffer(3)]],
const device float* beta [[buffer(4)]],
const device float* decay [[buffer(5)]],
device float* o [[buffer(6)]],
constant uint& Hk [[buffer(7)]], constant uint& Hv [[buffer(8)]],
constant uint& Dk [[buffer(9)]], constant uint& Dv [[buffer(10)]],
uint gid [[thread_position_in_grid]])
{
    const uint hv = gid / Dv, dv = gid - hv * Dv;
    if (hv >= Hv || dv >= Dv) return;
    const uint hk = hv % Hk;
    const device float4* kh4 = (const device float4*)(k + (size_t)hk * Dk);
    const device float4* qh4 = (const device float4*)(q + (size_t)hk * Dk);
    device float4* row4 = (device float4*)(S + ((size_t)hv * Dv + dv) * (size_t)Dk);
    const float dec = decay[hv];
    float4 sk4 = 0.0f;
    const uint n4 = Dk / 4;
    for (uint i = 0; i < n4; ++i) {
        const float4 r = row4[i] * dec;
        row4[i] = r;
        sk4 += r * kh4[i];
    }
    const float sk = sk4.x + sk4.y + sk4.z + sk4.w;
    const float delta = beta[hv] * (v[(size_t)hv * Dv + dv] - sk);
    float4 acc4 = 0.0f;
    for (uint i = 0; i < n4; ++i) {
        const float4 r = row4[i] + kh4[i] * delta;
        row4[i] = r;
        acc4 += r * qh4[i];
    }
    o[(size_t)hv * Dv + dv] = acc4.x + acc4.y + acc4.z + acc4.w;
}

// Gated normaliser epilogue: x[i] *= z[i] / (1 + exp(-z[i])) — i.e. the
// rms_norm output times silu(z). (silu_mul applies silu to its FIRST argument,
// which is the wrong side here.)
kernel void gdn_silu_gate(
device float* x [[buffer(0)]], const device float* z [[buffer(1)]],
constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) { const float g = z[gid]; x[gid] *= g / (1.0f + exp(-g)); }
}

// Per-head RMS norm with a WEIGHT (gdn_l2_norm has none) — the gated normaliser's
// first half. One threadgroup per head, same two-level SIMD reduction.
kernel void gdn_head_rms(
const device float* x [[buffer(0)]],
const device float* w [[buffer(1)]],
device float* out [[buffer(2)]],
constant uint& H [[buffer(3)]], constant uint& D [[buffer(4)]],
constant float& eps [[buffer(5)]],
uint tg [[threadgroup_position_in_grid]],
uint ti [[thread_index_in_threadgroup]],
uint tpg [[threads_per_threadgroup]],
uint lane [[thread_index_in_simdgroup]],
uint sg [[simdgroup_index_in_threadgroup]])
{
    if (tg >= H) return;
    const device float* xh = x + (size_t)tg * D;
    device float* oh = out + (size_t)tg * D;
    threadgroup float ss[8];
    float p = 0.0f;
    for (uint i = ti; i < D; i += tpg) p += xh[i] * xh[i];
    p = simd_sum(p);
    if (lane == 0) ss[sg] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < tpg / 32) ? ss[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) ss[0] = 1.0f / sqrt(v / (float)D + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = ss[0];
    for (uint i = ti; i < D; i += tpg) oh[i] = xh[i] * inv * w[i];
}
)MET"];
}

[[nodiscard]] inline bool tok_begin() {
    if (!ensure_init()) return false;
    if (!libops()) {
        NSError* err = nil;
        libops() = [dev() newLibraryWithSource:kMetalSrcOps()
                                       options:compile_options() error:&err];
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
// TM_METAL_TOK_TIME=1 accumulates, per token command, the GPU busy time
// (GPUStartTime..GPUEndTime) against the host wall time spent around it, and
// prints running totals every 64 commands. Both decode paths feed the same
// counters - the sync one below and the chained one in tok_wait_pending - so a
// lane's numbers are comparable across the two.
inline void tok_time_note(id<MTLCommandBuffer> cb, double wall_s) {
    static const bool timed = [] { const char* e = std::getenv("TM_METAL_TOK_TIME"); return e && *e == '1'; }();
    if (!timed) return;
    static double gpu_s = 0, wall_tot = 0; static unsigned cnt = 0;
    if (cb.status == MTLCommandBufferStatusCompleted)
        gpu_s += cb.GPUEndTime - cb.GPUStartTime;
    wall_tot += wall_s;
    if (++cnt % 64 == 0)
        std::fprintf(stderr, "[metal] tok cmds %u: gpu %.2f ms/cmd, host %.2f ms/cmd, %.0f%% gpu busy\n",
                     cnt, 1e3 * gpu_s / cnt, 1e3 * wall_tot / cnt,
                     100.0 * gpu_s / (wall_tot > 0 ? wall_tot : 1e-9));
}
[[nodiscard]] inline bool tok_flush() {
    const auto t0 = std::chrono::steady_clock::now();
    [tenc() endEncoding];
    [tcb() commit];
    [tcb() waitUntilCompleted];
    const bool ok = tcb().status == MTLCommandBufferStatusCompleted;
    tok_time_note(tcb(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (!ok)
        std::fprintf(stderr, "[metal] token command failed: %s\n",
                     tcb().error.localizedDescription.UTF8String);
    tenc() = nil; tcb() = nil;
    return ok;
}
[[nodiscard]] inline bool tok_end(int outslot, float* dst, unsigned n) {
    const bool ok = tok_flush();
    if (ok && n) memcpy(dst, slot_buf(outslot, 0).contents,
                        sizeof(float) * n);
    return ok;
}
[[nodiscard]] inline bool tok_upload(int slot, size_t byte_off, const void* src,
                       size_t bytes) {
    if (!ensure_init()) return false;
    id<MTLBuffer> b = slot_buf(slot, byte_off + bytes);
    memcpy((char*)b.contents + byte_off, src, bytes);
    return true;
}
// Packed outputs must be sized BEFORE any writes are encoded. Growing a slot
// mid-command replaces its buffer; earlier kernels still target the old one.
[[nodiscard]] inline bool tok_reserve(int slot, size_t bytes) {
    if (!ensure_init() || tenc()) return false;
    return slot_buf(slot, bytes) != nil;
}
[[nodiscard]] inline bool tok_download(int slot, size_t byte_off, void* dst,
                         size_t bytes) {
    if (!ensure_init()) return false;
    id<MTLBuffer> b = slot_buf(slot, byte_off + bytes);
    memcpy(dst, (char*)b.contents + byte_off, bytes);
    return true;
}
// Chained tokens: commit without waiting, keep the buffers, wait for the
// last one when the chain ends. Buffers on one queue execute in commit
// order, so a later token's embed kernel never races the earlier token.
[[nodiscard]] inline std::vector<__strong id<MTLCommandBuffer>>& pending_cbs() {
    static std::vector<__strong id<MTLCommandBuffer>> v; return v;
}
[[nodiscard]] inline bool tok_end_async() {
    [tenc() endEncoding];
    [tcb() commit];
    pending_cbs().push_back(tcb());
    tenc() = nil; tcb() = nil;
    return true;
}
[[nodiscard]] inline bool tok_wait_pending() {
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    for (auto& cb : pending_cbs()) {
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ok = false;
            std::fprintf(stderr, "[metal] chained token command failed: %s\n",
                         cb.error.localizedDescription.UTF8String);
        }
    }
    // The chained path never waits per token, so its cost is invisible to
    // tok_flush. Reported as GPU busy time per token: if that sits near the
    // lane's measured ms/token, the GPU is really executing and the kernels are
    // the target; if it is far below, the host cannot feed the GPU and the
    // dispatch count is the target.
    static const bool timed = [] { const char* e = std::getenv("TM_METAL_TOK_TIME"); return e && *e == '1'; }();
    if (timed && !pending_cbs().empty()) {
        const auto n = pending_cbs().size();
        double gpu = 0;
        for (auto& cb : pending_cbs())
            if (cb.status == MTLCommandBufferStatusCompleted)
                gpu += cb.GPUEndTime - cb.GPUStartTime;
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[metal] chain %zu cmds: gpu %.2f ms total (%.2f ms/token), drain %.2f ms\n",
                     n, 1e3 * gpu, 1e3 * gpu / (double)n, 1e3 * wall);
    }
    pending_cbs().clear();
    return ok;
}
inline void tok_argmax(int xslot, int tokslot, unsigned idx, unsigned n) {
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * n);
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    [tenc() setComputePipelineState:kern(Kern::argmax_f32)];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bt offset:0 atIndex:1];
    set_scalar(n, 2);
    set_scalar(idx, 3);
    [tenc() dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
inline void tok_sample(int xslot, int tokslot, unsigned idx, unsigned n, float inv_temp, unsigned seed) {
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * n);
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    [tenc() setComputePipelineState:kern(Kern::sample_gumbel_f32)];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bt offset:0 atIndex:1];
    set_scalar(n, 2);
    set_scalar(idx, 3);
    set_scalar(inv_temp, 4);
    set_scalar(seed, 5);
    [tenc() dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
// Read one entry of a token slot without waiting (the slot is shared
// memory; a chain writes entries in order, so a caller polling entry i
// after seeding the slot with a sentinel sees tokens as they land).
[[nodiscard]] inline unsigned tok_peek(int tokslot, unsigned idx) {
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    // The GPU stores this entry non-atomically (argmax_f32 / sample_gumbel_f32)
    // while the host polls it, and the caller reads the token slot's other
    // contents right after a change is observed, so the load has to be
    // synchronising, not merely non-elidable: an acquire atomic_ref pairs with
    // the kernel's store, a `volatile` read does not (it orders nothing
    // against the surrounding non-volatile accesses).
    return std::atomic_ref<unsigned>(static_cast<unsigned*>(bt.contents)[idx])
        .load(std::memory_order_acquire);
}
inline void tok_embed_q4(int tokslot, unsigned idx, int wid, uint64_t woff, int xslot, unsigned D) {
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * D);
    [tenc() setComputePipelineState:kern(Kern::embed_q4_row)];
    [tenc() setBuffer:bt offset:0 atIndex:0];
    set_scalar(idx, 1);
    [tenc() setBuffer:bw offset:0 atIndex:2];
    set_scalar(woff, 3);
    [tenc() setBuffer:bx offset:0 atIndex:4];
    set_scalar(D, 5);
    [tenc() dispatchThreads:MTLSizeMake(D / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
inline void tok_embed_h(int tokslot, unsigned idx, int wid, uint64_t woff,
                        int xslot, unsigned D) {
    if (woff % 8 || D % 4)
        throw std::invalid_argument("metal: fp16 embed requires 8-aligned woff and D % 4 == 0");
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * D);
    [tenc() setComputePipelineState:kern(Kern::embed_h_row)];
    [tenc() setBuffer:bt offset:0 atIndex:0];
    set_scalar(idx, 1);
    [tenc() setBuffer:bw offset:0 atIndex:2];
    set_scalar(woff, 3);
    [tenc() setBuffer:bx offset:0 atIndex:4];
    set_scalar(D, 5);
    [tenc() dispatchThreads:MTLSizeMake(D / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
// K-quant twin of tok_embed_h / tok_embed_q4: same slot/offset contract, dtype carried.
inline void tok_embed_kq(int tokslot, unsigned idx, int wid, uint64_t woff,
                         int xslot, unsigned D, unsigned dtype) {
    id<MTLBuffer> bt = slot_buf(tokslot, sizeof(unsigned) * (idx + 1));
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * D);
    [tenc() setComputePipelineState:kern(Kern::embed_kq_row)];
    [tenc() setBuffer:bt offset:0 atIndex:0];
    set_scalar(idx, 1);
    [tenc() setBuffer:bw offset:0 atIndex:2];
    set_scalar(woff, 3);
    [tenc() setBuffer:bx offset:0 atIndex:4];
    set_scalar(D, 5);
    set_scalar(dtype, 6);
    [tenc() dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
[[nodiscard]] inline int tok_wbuf(const void* tensor_ptr, size_t bytes,
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
[[nodiscard]] inline bool tok_probe_copy(int srcslot, size_t src_off, int dstslot,
                           unsigned n) {
    if (!tok_begin()) return false;
    id<MTLBuffer> bs = slot_buf(srcslot, src_off + n);
    id<MTLBuffer> bd = slot_buf(dstslot, n);
    [tenc() setComputePipelineState:kern(Kern::probe_copy)];
    [tenc() setBuffer:bs offset:0 atIndex:0];
    [tenc() setBuffer:bd offset:0 atIndex:1];
    set_scalar(n, 2);
    set_scalar(src_off, 3);
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return tok_end(dstslot, nullptr, 0);
}
[[nodiscard]] inline unsigned& gemv_lanes() {
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
[[nodiscard]] inline bool& gemv_scale_per_block() {
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
[[nodiscard]] inline int& gemv_rows() {
    static int rows = [] {
        const int r = tmtune::get_int("TM_METAL_GEMV_ROWS", 2);   // 6/8 = mask-dot variants with 2/4 rows
        if (r != 1 && r != 2 && r != 4 && r != 6 && r != 8) throw std::invalid_argument("TM_METAL_GEMV_ROWS must be 1, 2, 4, 6 or 8");
        return r;
    }();
    return rows;
}
[[nodiscard]] inline id<MTLComputePipelineState> token_gemv_pipeline(unsigned lanes) {
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
[[nodiscard]] inline unsigned& gemv_threads() {
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
    set_scalar(N, 3);
    set_scalar(K, 4);
    set_scalar(woff, 5);
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
[[nodiscard]] inline id<MTLComputePipelineState> token_gemv_seg_pipeline(unsigned lanes) {
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
// The segmented fused GEMV's dtype contract, reported rather than assumed: the
// seg kernel (q4_gemv_coop_r2_seg) decodes q4_0 blocks only, so today exactly
// one dtype is served and every other weight dtype still costs one GEMV
// dispatch per projection (7 per layer for the K-quant lanes). The templated
// seg body over q4_1/q5_k/q6_k/f16 is the audit's phase-2 lever; when it lands,
// this is the only predicate that has to widen. llama.h's fused-segment
// decisions (qkv_seg_ok and the gate|up call sites) will consult this instead
// of hardcoding q4_0 - deferred to the cpu-inference lane.
[[nodiscard]] inline bool tok_gemv_seg_supported(unsigned dtype) {
    // q4_0, q4_1, q5_k, q6_k. f16 (dtype 3) stays out: its host lane does not
    // exist yet (F16_METAL_PLAN phase 2), so a seg kernel for it would be dead
    // code with no twin to verify against.
    return dtype == 1u || dtype == 4u || dtype == 5u || dtype == 6u;
}
inline void tok_gemv_seg(int xslot, int wid, int yslot, size_t yoff,
                         unsigned N, unsigned K,
                         const uint64_t* seg_woff, const unsigned* seg_rows,
                         unsigned nseg) {
    // setBytes carries at most 4 KB per argument; validate the table before
    // anything is encoded so an over-large one is a caller error, not a
    // half-encoded command buffer.
    if (nseg * sizeof(uint64_t) > 4096)
        throw std::invalid_argument("metal: segment table does not fit setBytes");
    const unsigned lanes = gemv_lanes();
    const unsigned threads = gemv_threads();
    id<MTLComputePipelineState> pso = token_gemv_seg_pipeline(lanes);
    if ((threads != 64 && threads != 128 && threads != 256) ||
        threads > pso.maxTotalThreadsPerThreadgroup)
        throw std::invalid_argument("metal: unsupported GEMV threadgroup size");
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * K);
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> by = slot_buf(yslot, yoff + sizeof(float) * N);
    // setBytes IS a per-dispatch snapshot (it copies the bytes into the command
    // buffer while the dispatch is encoded), so the back-to-back chain — several
    // tokens sharing one command buffer — still reads the rows and offsets each
    // dispatch was encoded with. The previous comment claimed the opposite and
    // allocated two fresh MTLBuffers per dispatch: two Metal resources per fused
    // projection per token, for a table that never changes within a layer.
    [tenc() setComputePipelineState:pso];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bw offset:0 atIndex:1];
    [tenc() setBuffer:by offset:yoff atIndex:2];
    set_scalar(N, 3);
    set_scalar(K, 4);
    set_scalar(nseg, 5);
    [tenc() setBytes:seg_rows length:(NSUInteger)(nseg * sizeof(unsigned)) atIndex:6];
    [tenc() setBytes:seg_woff length:(NSUInteger)(nseg * sizeof(uint64_t)) atIndex:7];
    const unsigned rows_per_group = 2u;   // q4_gemv_coop_r2_seg
    const size_t groups = ((size_t)N + rows_per_group - 1) / rows_per_group;
    [tenc() dispatchThreads:MTLSizeMake(groups * lanes, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}
// Segmented fused GEMV pipeline for the K-quant families: same shape as
// token_gemv_seg_pipeline (function constants lanes@0, block_scale=false@1),
// dtype selects the seg kernel. R is fixed at 2, like the q4_0 seg kernel.
[[nodiscard]] inline id<MTLComputePipelineState> token_gemv_seg_dt_pipeline(unsigned lanes, unsigned dtype) {
    if (lanes != 1 && lanes != 4 && lanes != 8 && lanes != 16 && lanes != 32)
        throw std::invalid_argument("metal: unsupported GEMV lane count");
    NSString* fam = dtype == 1u ? @"q4" : dtype == 4u ? @"q4_1"
                  : dtype == 5u ? @"q5k" : dtype == 6u ? @"q6k" : nil;
    if (!fam) throw std::invalid_argument("metal: no segmented GEMV kernel for that dtype");
    static std::map<unsigned, __strong id<MTLComputePipelineState>> cache;
    const unsigned key = (lanes << 8) | dtype;
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    NSError* error = nil;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
    const bool block_scale = false;   // the seg bodies have no per-block-scale branch
    [values setConstantValue:&block_scale type:MTLDataTypeBool atIndex:1];
    NSString* fname = [NSString stringWithFormat:@"%@_gemv_coop_r2_seg", fam];
    id<MTLFunction> function = [libq4() newFunctionWithName:fname
                                            constantValues:values error:&error];
    id<MTLComputePipelineState> pso = function ? [dev()
        newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pso || pso.threadExecutionWidth != 32)
        throw std::runtime_error("metal: segmented GEMV pipeline requires 32-lane SIMD groups");
    cache.emplace(key, pso);
    return pso;
}
inline void tok_gemv_seg_dt(int xslot, int wid, int yslot, size_t yoff,
                            unsigned N, unsigned K, unsigned dtype,
                            const uint64_t* seg_woff, const unsigned* seg_rows,
                            unsigned nseg) {
    if (nseg * sizeof(uint64_t) > 4096)
        throw std::invalid_argument("metal: segment table does not fit setBytes");
    const unsigned lanes = gemv_lanes();
    const unsigned threads = gemv_threads();
    id<MTLComputePipelineState> pso = token_gemv_seg_dt_pipeline(lanes, dtype);
    if ((threads != 64 && threads != 128 && threads != 256) ||
        threads > pso.maxTotalThreadsPerThreadgroup)
        throw std::invalid_argument("metal: unsupported GEMV threadgroup size");
    id<MTLBuffer> bx = slot_buf(xslot, sizeof(float) * K);
    id<MTLBuffer> bw = tslots()[wid];
    id<MTLBuffer> by = slot_buf(yslot, yoff + sizeof(float) * N);
    // Same setBytes snapshot contract as tok_gemv_seg: the table is copied at
    // encode time, so back-to-back dispatches in one command buffer each read
    // the rows and offsets they were encoded with.
    [tenc() setComputePipelineState:pso];
    [tenc() setBuffer:bx offset:0 atIndex:0];
    [tenc() setBuffer:bw offset:0 atIndex:1];
    [tenc() setBuffer:by offset:yoff atIndex:2];
    set_scalar(N, 3);
    set_scalar(K, 4);
    set_scalar(nseg, 5);
    [tenc() setBytes:seg_rows length:(NSUInteger)(nseg * sizeof(unsigned)) atIndex:6];
    [tenc() setBytes:seg_woff length:(NSUInteger)(nseg * sizeof(uint64_t)) atIndex:7];
    const unsigned rows_per_group = 2u;   // *_gemv_coop_r2_seg
    const size_t groups = ((size_t)N + rows_per_group - 1) / rows_per_group;
    [tenc() dispatchThreads:MTLSizeMake(groups * lanes, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}
// K-quant token GEMV (Q4_1 / Q5_K / Q6_K weights) — the token-lane twin of
// tok_gemv, carrying the dtype because the three kernels differ only in how a block
// is unpacked. woff is a BYTE offset into the shared zero-copy weight buffer, the
// same contract tok_gemv has, so one slot table serves every dtype in a layer.
[[nodiscard]] inline id<MTLComputePipelineState> token_gemv_kq_pipeline(unsigned lanes, unsigned dtype) {
    if (lanes != 1 && lanes != 4 && lanes != 8 && lanes != 16 && lanes != 32)
        throw std::invalid_argument("metal: unsupported GEMV lane count");
    const int rows = gemv_rows();
    if (rows != 1 && rows != 2 && rows != 4)
        throw std::invalid_argument("metal: K-quant GEMV supports TM_METAL_GEMV_ROWS 1, 2 or 4");
    NSString* fam = dtype == 4u ? @"q4_1" : dtype == 5u ? @"q5k"
                  : dtype == 6u ? @"q6k" : nil;
    if (!fam) throw std::invalid_argument("metal: no K-quant GEMV kernel for that dtype");
    static std::map<unsigned, __strong id<MTLComputePipelineState>> cache;
    const unsigned key = (lanes << 8) | (unsigned(rows + 1) << 4) | dtype;
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    NSError* error = nil;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
    NSString* fname = [NSString stringWithFormat:@"%@_gemv_coop_r%d", fam, rows];
    id<MTLFunction> function = [libq4() newFunctionWithName:fname
                                            constantValues:values error:&error];
    id<MTLComputePipelineState> pso = function ? [dev()
        newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pso || pso.threadExecutionWidth != 32)
        throw std::runtime_error("metal: K-quant GEMV pipeline requires 32-lane SIMD groups");
    cache.emplace(key, pso);
    return pso;
}
inline void tok_gemv_kq(int xslot, int wid, uint64_t woff, int yslot, size_t yoff,
                        unsigned N, unsigned K, unsigned dtype) {
    const unsigned lanes = gemv_lanes();
    const unsigned threads = gemv_threads();
    id<MTLComputePipelineState> pso = token_gemv_kq_pipeline(lanes, dtype);
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
    set_scalar(N, 3);
    set_scalar(K, 4);
    set_scalar(woff, 5);
    const unsigned rows_per_group = lanes == 1 ? 1u : (unsigned)gemv_rows();
    const size_t groups = ((size_t)N + rows_per_group - 1) / rows_per_group;
    [tenc() dispatchThreads:MTLSizeMake(groups * lanes, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}
// Qwen2 attention-bias add over the fused q|k|v row (N = QD + 2*KVD).
// bias is bound at the per-layer byte offset; one thread per element.
inline void tok_bias_qkv(int yslot, int biasslot, size_t bias_off, unsigned N) {
    static __strong id<MTLComputePipelineState> pso;
    if (!pso) {
        NSError* error = nil;
        id<MTLFunction> function = [libq4() newFunctionWithName:@"tok_bias_qkv"];
        pso = function ? [dev() newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pso) throw std::runtime_error("metal: tok_bias_qkv pipeline failed");
    }
    id<MTLBuffer> by = slot_buf(yslot, sizeof(float) * N);
    id<MTLBuffer> bb = slot_buf(biasslot, bias_off + sizeof(float) * N);
    [tenc() setComputePipelineState:pso];
    [tenc() setBuffer:by offset:0 atIndex:0];
    [tenc() setBuffer:bb offset:bias_off atIndex:1];
    set_scalar(N, 2);
    [tenc() dispatchThreads:MTLSizeMake(N, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
// fp16-weights twin of tok_gemv (TM_DECODE_F16): identical slot/offset
// contract, half weights at (wid, woff) — woff is a BYTE offset and the
// loader asserts 4-byte tensor alignment so the kernel's half4 loads
// stay aligned.
inline void tok_gemv_h(int xslot, int wid, uint64_t woff, int yslot,
                       size_t yoff, unsigned N, unsigned K) {
    if (woff % 4 || K % 4)
        throw std::invalid_argument(
            "metal: fp16 GEMV requires 4-aligned woff and K: wid " +
            std::to_string(wid) + " woff " + std::to_string(woff) +
            " N " + std::to_string(N) + " K " + std::to_string(K));
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
    set_scalar(N, 3);
    set_scalar(K, 4);
    set_scalar(woff, 5);
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
        [tenc() setComputePipelineState:kern(Kern::rmsnorm_row)];
        [tenc() setBuffer:bx offset:0 atIndex:0];
        [tenc() setBuffer:bw offset:0 atIndex:1];
        [tenc() setBuffer:bo offset:0 atIndex:2];
        set_scalar(D, 3);
        set_scalar(eps, 4);
        [tenc() dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
inline void tok_add(int a, int b, unsigned n) {
    [tenc() setComputePipelineState:kern(Kern::add_inplace)];
    [tenc() setBuffer:slot_buf(a, sizeof(float) * n) offset:0 atIndex:0];
    [tenc() setBuffer:slot_buf(b, sizeof(float) * n) offset:0 atIndex:1];
    set_scalar(n, 2);
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
inline void tok_silu_mul(int g, size_t goff, int u, size_t uoff,
                         unsigned n) {
    [tenc() setComputePipelineState:kern(Kern::silu_mul)];
    [tenc() setBuffer:slot_buf(g, goff + sizeof(float) * n)
      offset:goff atIndex:0];
    [tenc() setBuffer:slot_buf(u, uoff + sizeof(float) * n)
      offset:uoff atIndex:1];
    set_scalar(n, 2);
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
// ---- Gated DeltaNet host wrappers (qwen35 linear attention) ----------------
// All buffers are integer slots; see slot_buf/tok_reserve. Sizes are the exact
// byte spans each kernel reads or writes so a missing reserve is a hard error
// rather than silent corruption.
// `taps`, not `kern`: the parameter would shadow the kern() accessor.
inline void tok_gdn_conv(int in_slot, int state_slot, const void* taps,
                         int out_slot, unsigned channels, unsigned K) {
    const size_t kbytes = sizeof(float) * (size_t)K * channels;
    static std::vector<std::tuple<const void*, size_t, __strong id<MTLBuffer>>> kc;
    __strong id<MTLBuffer> bk;
    for (auto& e : kc)
        if (std::get<0>(e) == taps && std::get<1>(e) == kbytes) { bk = std::get<2>(e); break; }
    if (!bk) {
        bk = [dev() newBufferWithBytes:taps length:kbytes
            options:MTLResourceStorageModeShared];
        kc.push_back({taps, kbytes, bk});
    }
    [tenc() setComputePipelineState:kern(Kern::gdn_conv_silu)];
    [tenc() setBuffer:slot_buf(in_slot, sizeof(float) * channels) offset:0 atIndex:0];
    [tenc() setBuffer:slot_buf(state_slot, sizeof(float) * (K - 1) * channels) offset:0 atIndex:1];
    [tenc() setBuffer:bk offset:0 atIndex:2];
    [tenc() setBuffer:slot_buf(out_slot, sizeof(float) * channels) offset:0 atIndex:3];
    set_scalar(channels, 4);
    set_scalar(K, 5);
    [tenc() dispatchThreads:MTLSizeMake(channels, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(channels < 64 ? channels : 64, 1, 1)];
}
inline void tok_gdn_l2(int src_slot, int dst_slot, unsigned H, unsigned D,
                       unsigned src_off, unsigned dst_off, float scale, float eps) {
    [tenc() setComputePipelineState:kern(Kern::gdn_l2_norm)];
    // src_off/dst_off are bytes; see tok_gdn_step's note.
    [tenc() setBuffer:slot_buf(src_slot, src_off + sizeof(float) * (size_t)H * D)
      offset:0 atIndex:0];
    [tenc() setBuffer:slot_buf(dst_slot, dst_off + sizeof(float) * (size_t)H * D)
      offset:0 atIndex:1];
    set_scalar(H, 2);
    set_scalar(D, 3);
    set_scalar(src_off, 4);
    set_scalar(scale, 5);
    set_scalar(eps, 6);
    set_scalar(dst_off, 7);
    [tenc() dispatchThreadgroups:MTLSizeMake(H, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
inline void tok_gdn_head_rms(int x_slot, const void* w, int out_slot,
                             unsigned H, unsigned D, float eps) {
    static std::vector<std::tuple<const void*, size_t, __strong id<MTLBuffer>>> wc;
    const size_t wbytes = sizeof(float) * D;
    __strong id<MTLBuffer> bw;
    for (auto& e : wc)
        if (std::get<0>(e) == w && std::get<1>(e) == wbytes) { bw = std::get<2>(e); break; }
    if (!bw) {
        bw = [dev() newBufferWithBytes:w length:wbytes options:MTLResourceStorageModeShared];
        wc.push_back({w, wbytes, bw});
    }
    [tenc() setComputePipelineState:kern(Kern::gdn_head_rms)];
    [tenc() setBuffer:slot_buf(x_slot, sizeof(float) * (size_t)H * D) offset:0 atIndex:0];
    [tenc() setBuffer:bw offset:0 atIndex:1];
    [tenc() setBuffer:slot_buf(out_slot, sizeof(float) * (size_t)H * D) offset:0 atIndex:2];
    set_scalar(H, 3);
    set_scalar(D, 4);
    set_scalar(eps, 5);
    [tenc() dispatchThreadgroups:MTLSizeMake(H, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
inline void tok_gdn_decay(int alpha_slot, const void* dt, const void* a,
                          int decay_slot, unsigned Hv) {
    const size_t nbytes = sizeof(float) * Hv;
    static std::vector<std::tuple<const void*, size_t, __strong id<MTLBuffer>>> dc;
    auto buf_for = [&](const void* ptr) {
        for (auto& e : dc)
            if (std::get<0>(e) == ptr && std::get<1>(e) == nbytes) return std::get<2>(e);
        id<MTLBuffer> b = [dev() newBufferWithBytes:ptr length:nbytes
            options:MTLResourceStorageModeShared];
        dc.push_back({ptr, nbytes, b});
        return b;
    };
    [tenc() setComputePipelineState:kern(Kern::gdn_decay)];
    [tenc() setBuffer:slot_buf(alpha_slot, sizeof(float) * Hv) offset:0 atIndex:0];
    [tenc() setBuffer:buf_for(dt) offset:0 atIndex:1];
    [tenc() setBuffer:buf_for(a) offset:0 atIndex:2];
    [tenc() setBuffer:slot_buf(decay_slot, sizeof(float) * Hv) offset:0 atIndex:3];
    set_scalar(Hv, 4);
    [tenc() dispatchThreads:MTLSizeMake(Hv, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}
inline void tok_gdn_step(int s_slot, int q_slot, int k_slot, int v_slot,
                         int beta_slot, int decay_slot, int o_slot,
                         unsigned Hk, unsigned Hv, unsigned Dk, unsigned Dv,
                         unsigned q_off, unsigned k_off, unsigned v_off,
                         unsigned o_off) {
    // float4 path needs every state row 16-byte aligned (Dk % 4 == 0 makes the
    // row stride a multiple of 16 bytes) and the q/k slot offsets 16-byte
    // aligned (the real caller reads q and k out of ONE conv slot at byte
    // offsets 0 / qn_l / 2*qn_l, which are 16-byte multiples exactly when the
    // head dims are). Anything else takes the scalar kernel.
    const bool f4 = Dk % 4 == 0 && q_off % 16 == 0 && k_off % 16 == 0;
    [tenc() setComputePipelineState:kern(f4 ? Kern::gdn_step_f4 : Kern::gdn_step)];
    [tenc() setBuffer:slot_buf(s_slot, sizeof(float) * (size_t)Hv * Dv * Dk) offset:0 atIndex:0];
    // The offsets are BYTES (MTL setBuffer offsets are bytes; the real caller
    // slices one conv slot at byte offsets 0 / qn_l / 2*qn_l). The size checks
    // used to fold them into the element count and multiply by 4 - over-
    // reserving ~3x on every non-zero offset and throwing for an exactly-sized
    // slot. Same bug in tok_gdn_l2 below.
    [tenc() setBuffer:slot_buf(q_slot, q_off + sizeof(float) * (size_t)Hk * Dk) offset:q_off atIndex:1];
    [tenc() setBuffer:slot_buf(k_slot, k_off + sizeof(float) * (size_t)Hk * Dk) offset:k_off atIndex:2];
    [tenc() setBuffer:slot_buf(v_slot, v_off + sizeof(float) * (size_t)Hv * Dv) offset:v_off atIndex:3];
    [tenc() setBuffer:slot_buf(beta_slot, sizeof(float) * Hv) offset:0 atIndex:4];
    [tenc() setBuffer:slot_buf(decay_slot, sizeof(float) * Hv) offset:0 atIndex:5];
    [tenc() setBuffer:slot_buf(o_slot, o_off + sizeof(float) * (size_t)Hv * Dv) offset:o_off atIndex:6];
    set_scalar(Hk, 7);
    set_scalar(Hv, 8);
    set_scalar(Dk, 9);
    set_scalar(Dv, 10);
    const unsigned n = Hv * Dv;
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(n < 64 ? n : 64, 1, 1)];
}
inline void tok_gdn_silu_gate(int x_slot, int z_slot, unsigned n) {
    [tenc() setComputePipelineState:kern(Kern::gdn_silu_gate)];
    [tenc() setBuffer:slot_buf(x_slot, sizeof(float) * n) offset:0 atIndex:0];
    [tenc() setBuffer:slot_buf(z_slot, sizeof(float) * n) offset:0 atIndex:1];
    set_scalar(n, 2);
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// RoPE angles: ONE host-built (cos, sin) table per (theta, dh), read by every
// rope kernel in the process — rope_inplace and rope_k_cache(_h) in the decode
// lane, ll_rope(_h) in the prefill lane. The angle of pair j at position p is
// theta^(-2j/dh) * p, evaluated here with the same float expressions and the
// same libm the CPU rope_head uses. Before this the decode kernels used Metal's
// fast-math pow/cos/sin and the prefill kernels precise::pow/cos/sin, so the
// K row for one position depended on which lane wrote it — the last bits
// differed. Table entries are a position-ordered prefix of each other, so
// growing appends; a rebuild allocates a fresh buffer and Metal retains the old
// one for dispatches already encoded against it.
struct RopeTable {
    __strong id<MTLBuffer> buf;
    unsigned dh = 0, cap = 0;   // cap counts positions, not float2 entries
    float theta = 0;
};
// One angle pair as the HOST sees it: the kernels index the table as float2, so
// the layout has to be two tightly packed 32-bit floats and nothing else.
struct RopeAngle { float c, s; };
static_assert(sizeof(RopeAngle) == 8, "rope angle table is an array of float2");
[[nodiscard]] inline RopeTable& rope_table() { static RopeTable t; return t; }
[[nodiscard]] inline bool ensure_rope_table(unsigned dh, unsigned need, float theta) {
    if (!dh || (dh & 1u)) throw std::invalid_argument("metal: odd rope head dim");
    if (!need) need = 1;
    RopeTable& t = rope_table();
    const bool same = t.buf && t.dh == dh && t.theta == theta;
    if (same && t.cap >= need) return true;
    unsigned cap = same ? t.cap : 0;
    while (cap < need) cap = cap ? cap * 2 : 64;
    const unsigned half_ = dh / 2;
    id<MTLBuffer> b = [dev() newBufferWithLength:(size_t)cap * half_ * sizeof(RopeAngle)
        options:MTLResourceStorageModeShared];
    if (!b) return false;
    RopeAngle* tbl = (RopeAngle*)b.contents;
    for (unsigned j = 0; j < half_; ++j) {
        const float inv = std::pow(theta, -2.0f * (float)j / (float)dh);
        for (unsigned p = 0; p < cap; ++p) {
            const float ang = (float)p * inv;
            // The PAIRED libm spelling, exactly as the CPU's rope_head/rope_qk
            // (llama.h) write this angle. The two spellings can differ by 1 ULP
            // per pair when clang does not fuse them (isolated probe: 4128 of
            // 67200 pairs, first at p=34 j=0) - but in THIS build both spellings
            // already compile to one sincos call (`nm -u`: one symbol, no separate
            // cosf/sinf), so spelling it out moves no bits and no hashes. It is
            // kept so the pairing is explicit in builds that do not fuse (the CPU
            // rope's own comment records clang not combining the pair under
            // sanitizer instrumentation), and test_metal_token pins this function.
            RopeAngle cs;
            ::__sincosf(ang, &cs.s, &cs.c);
            tbl[(size_t)p * half_ + j] = cs;
        }
    }
    t.buf = b; t.dh = dh; t.cap = cap; t.theta = theta;
    return true;
}
[[nodiscard]] inline unsigned rope_table_entries(unsigned dh) {
    return (unsigned)((size_t)rope_table().cap * (dh / 2));
}
inline void tok_rope_q(int qslot, unsigned H, unsigned dh, int pos,
                       float theta) {
    unsigned pairs = H * (dh / 2);
    // This signature carries no ctx, so ask for exactly the rows needed (the
    // table grows geometrically and only appends).
    if (!ensure_rope_table(dh, (unsigned)(pos > 0 ? pos + 1 : 1), theta))
        throw std::runtime_error("metal: rope angle table allocation failed");
    [tenc() setComputePipelineState:kern(Kern::rope_inplace)];
    [tenc() setBuffer:slot_buf(qslot, sizeof(float) * H * dh)
      offset:0 atIndex:0];
    set_scalar(H, 1);
    set_scalar(dh, 2);
    set_scalar(pos, 3);
    [tenc() setBuffer:rope_table().buf offset:0 atIndex:4];
    set_scalar(rope_table_entries(dh), 5);
    [tenc() dispatchThreads:MTLSizeMake(pairs, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
// GPU K/V cache element type. Half (default) halves the cache traffic the
// decode attention reads at long context; rows are rounded once when
// written (rope_k_cache_h / v_to_cache_h or the upload converter) and the
// arithmetic stays fp32. TM_METAL_KV_HALF=0 keeps fp32 rows.
[[nodiscard]] inline bool& kv_half() {
    static bool enabled = [] {
        const char* s = std::getenv("TM_METAL_KV_HALF");
        if (!s || std::strcmp(s, "1") == 0) return true;
        if (std::strcmp(s, "0") == 0) return false;
        throw std::invalid_argument("TM_METAL_KV_HALF must be 0 or 1");
    }();
    return enabled;
}
[[nodiscard]] inline size_t kv_elem_bytes() { return kv_half() ? 2 : 4; }
// Upload/download fp32 rows into/out of a K/V cache slot in the slot's
// element type; offsets and counts are in elements.
// Size a K/V cache slot for n elements up front (partial uploads must not
// leave a slot short of the rows the cache kernels write later).
[[nodiscard]] inline bool tok_kv_reserve(int slot, size_t n) { return tok_reserve(slot, n * kv_elem_bytes()); }
[[nodiscard]] inline bool tok_kv_upload(int slot, size_t elem_off, const float* src, size_t n) {
    if (!ensure_init()) return false;
    const size_t eb = kv_elem_bytes();
    id<MTLBuffer> b = slot_buf(slot, (elem_off + n) * eb);
    if (!kv_half()) { memcpy((char*)b.contents + elem_off * 4, src, n * 4); return true; }
    __fp16* dst = (__fp16*)b.contents + elem_off;
    // vImage's NEON F-to-16F conversion (round-to-nearest-even, the same
    // rule the scalar cast uses, so the bits are identical - pinned by
    // test_metal_token's edge-value round trip). The hybrid lane runs this
    // per layer per token for the CPU tail's K/V rows.
    const vImage_Buffer vs{const_cast<float*>(src), 1, n, n * 4};
    const vImage_Buffer vd{dst, 1, n, n * 2};
    if (vImageConvert_PlanarFtoPlanar16F(&vs, &vd, 0) != kvImageNoError)
        return false;
    return true;
}
[[nodiscard]] inline bool tok_kv_download(int slot, size_t elem_off, float* dst, size_t n) {
    if (!ensure_init()) return false;
    const size_t eb = kv_elem_bytes();
    id<MTLBuffer> b = slot_buf(slot, (elem_off + n) * eb);
    if (!kv_half()) { memcpy(dst, (char*)b.contents + elem_off * 4, n * 4); return true; }
    const __fp16* src = (const __fp16*)b.contents + elem_off;
    const vImage_Buffer vs{const_cast<__fp16*>(src), 1, n, n * 2};
    const vImage_Buffer vd{dst, 1, n, n * 4};
    if (vImageConvert_Planar16FtoPlanarF(&vs, &vd, 0) != kvImageNoError)
        return false;
    return true;
}
inline void tok_rope_k_cache(int ksrcslot, size_t ksrc_off, int cacheslot,
                             unsigned KVH, unsigned dh, unsigned ctx,
                             int pos, float theta) {
    unsigned pairs = KVH * (dh / 2);
    if (!ensure_rope_table(dh, ctx, theta))
        throw std::runtime_error("metal: rope angle table allocation failed");
    [tenc() setComputePipelineState:kern(kv_half() ? Kern::rope_k_cache_h : Kern::rope_k_cache)];
    [tenc() setBuffer:slot_buf(ksrcslot, ksrc_off + sizeof(float) * KVH * dh)
      offset:ksrc_off atIndex:0];
    [tenc() setBuffer:slot_buf(cacheslot,
         kv_elem_bytes() * KVH * ctx * dh) offset:0 atIndex:1];
    set_scalar(KVH, 2);
    set_scalar(dh, 3);
    set_scalar(ctx, 4);
    set_scalar(pos, 5);
    [tenc() setBuffer:rope_table().buf offset:0 atIndex:6];
    set_scalar(rope_table_entries(dh), 7);
    [tenc() dispatchThreads:MTLSizeMake(pairs, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
inline void tok_v_cache(int vsrcslot, size_t vsrc_off, int cacheslot,
                        unsigned KVH, unsigned dh, unsigned ctx, int pos) {
    unsigned n = KVH * dh;
    [tenc() setComputePipelineState:kern(kv_half() ? Kern::v_to_cache_h : Kern::v_to_cache)];
    [tenc() setBuffer:slot_buf(vsrcslot, vsrc_off + sizeof(float) * n)
      offset:vsrc_off atIndex:0];
    [tenc() setBuffer:slot_buf(cacheslot,
         kv_elem_bytes() * KVH * ctx * dh) offset:0 atIndex:1];
    set_scalar(KVH, 2);
    set_scalar(dh, 3);
    set_scalar(ctx, 4);
    set_scalar(pos, 5);
    [tenc() dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
// Default ON since 2026-09-06: strict full-model Q4 oracle 32/32 (max err
// 1.0681e-04) and paired same-window A/B vs the three-dispatch path on
// TinyLlama Q4 GPU decode: +11.0% at tg256/ctx384 (59.36 -> 65.86 t/s, 4/4),
// +36.6% at tg1024/ctx2048 (39.10 -> 53.85 t/s, p95 34.7 -> 22.1 ms, 4/4),
// identical greedy hashes. TM_METAL_ATTN_FUSED=0 restores the legacy path,
// which also remains the fallback above 4096 context or on unsupported devices.
[[nodiscard]] inline bool& fused_attention() {
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
[[nodiscard]] inline bool& gqa_attention() {
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
[[nodiscard]] inline unsigned gqa_tile_rows(unsigned dh, unsigned REP) {
    if (!dh || 128 % dh || dh % 4 || !REP || REP > 8) return 0;
    const size_t bytes = sizeof(float) * (REP * 128 + REP * 128 + 3 * REP);
    return bytes + 64 <= dev().maxThreadgroupMemoryLength ? 128u : 0u;
}
// Specialized decode-attention PSO cache: dh and REP are MSL function
// constants baked per pipeline, keyed (dh, REP, kv_half, family) the way
// token_gemv_pipeline keys (lanes, block_scale). A nil miss is cached like a
// hit so an unsupported shape does not re-attempt compilation per dispatch.
[[nodiscard]] inline id<MTLComputePipelineState> attention_pso(
    unsigned dh, unsigned REP, bool kvh, bool gqa) {
    const uint32_t key = (dh << 11) | (REP << 2) | ((unsigned)kvh << 1) | (unsigned)gqa;
    static std::map<uint32_t, __strong id<MTLComputePipelineState>> cache;
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&dh type:MTLDataTypeUInt atIndex:0];
    [values setConstantValue:&REP type:MTLDataTypeUInt atIndex:1];
    const char* name = gqa ? (kvh ? "attn_decode_gqa_h" : "attn_decode_gqa")
                           : (kvh ? "attn_decode_fused_h" : "attn_decode_fused");
    NSError* err = nil;
    id<MTLFunction> f = [libops() newFunctionWithName:[NSString stringWithUTF8String:name]
                                      constantValues:values error:&err];
    id<MTLComputePipelineState> pso =
        f ? [dev() newComputePipelineStateWithFunction:f error:&err] : nil;
    if (!pso) {
        const char* detail = err.localizedDescription.UTF8String;
        std::fprintf(stderr, "[metal] %s pipeline unavailable: %s\n",
                     name, detail ? detail : "(no detail)");
    }
    cache.emplace(key, pso);
    return pso;
}
[[nodiscard]] inline id<MTLComputePipelineState> token_attention_pipeline(
    unsigned allow, unsigned dh, unsigned REP) {
    // Bound per-head scratch and keep the legacy path for long contexts and
    // unsupported devices. Round dynamic threadgroup storage to Metal's 16B unit.
    if (!allow || allow > 4096) return nil;
    auto pso = attention_pso(dh, REP, kv_half(), false);
    if (!pso) throw std::runtime_error("metal: fused attention pipeline unavailable");
    const size_t bytes = (sizeof(float) * allow + 15) & ~size_t(15);
    if (pso.threadExecutionWidth != 32 || pso.maxTotalThreadsPerThreadgroup < 128 ||
        pso.staticThreadgroupMemoryLength + bytes > dev().maxThreadgroupMemoryLength)
        return nil;
    return pso;
}
// The scalar arguments of the attention decode kernels, defined ONCE and shared
// with the MSL: the X-macro below generates both the host struct (setBytes, 32
// bytes, one binding instead of 8/3/5) and the MSL struct declaration injected
// into the ops source. Field order is the ABI; all fields are 4 bytes, so the
// layouts cannot diverge as long as the list is the same on both sides - and
// both sides assert the 32-byte size.
#define TM_ATTN_ARGS(F) \
    F(U32, dh) F(U32, ctx) F(U32, allow) F(F32, scale) \
    F(U32, rep) F(U32, nsplit) F(U32, chunk) F(U32, H)
struct AttnArgs {
    using U32 = unsigned;
    using F32 = float;
#define TM_ATTN_FIELD(T, N) T N;
    TM_ATTN_ARGS(TM_ATTN_FIELD)
#undef TM_ATTN_FIELD
};
static_assert(sizeof(AttnArgs) == 32, "AttnArgs is the MSL ABI, 8 x 4 bytes");
[[nodiscard]] inline NSString* attn_args_msl() {
    std::string s = "struct AttnArgs {\n";
#define TM_ATTN_MSL(T, N) \
    s += (std::string_view(#T) == "U32" ? "uint " : "float ") + std::string(#N) + ";\n";
    TM_ATTN_ARGS(TM_ATTN_MSL)
#undef TM_ATTN_MSL
    s += "};\nstatic_assert(sizeof(AttnArgs) == 32, \"AttnArgs MSL ABI\");\n";
    return [NSString stringWithUTF8String:s.c_str()];
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
            auto pso = attention_pso(dh, REP, kv_half(), true);
            auto comb = kern(Kern::attn_decode_combine);
            if (!pso || !comb) throw std::runtime_error("metal: gqa attention pipeline unavailable");
            [tenc() setComputePipelineState:pso];
            [tenc() setBuffer:bq offset:0 atIndex:0];
            [tenc() setBuffer:bk offset:0 atIndex:1];
            [tenc() setBuffer:bv offset:0 atIndex:2];
            [tenc() setBuffer:bp offset:0 atIndex:3];
            [tenc() setBuffer:bp offset:part_bytes atIndex:4];
            AttnArgs aa{};
            aa.dh = dh; aa.ctx = ctx; aa.allow = (unsigned)allow;
            aa.scale = scale; aa.rep = REP; aa.nsplit = nsplit;
            aa.chunk = chunk; aa.H = H;
            [tenc() setBytes:&aa length:sizeof(AttnArgs) atIndex:5];
            // q [REP][dh] + reduction [128/dh-1][REP][dh] = REP*128 floats.
            const size_t shm = (sizeof(float) * (REP * 128 + REP * TS + 3 * REP) + 15) & ~size_t(15);
            [tenc() setThreadgroupMemoryLength:shm atIndex:0];
            [tenc() dispatchThreadgroups:MTLSizeMake(KVH, nsplit, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [tenc() setComputePipelineState:comb];
            [tenc() setBuffer:bp offset:0 atIndex:0];
            [tenc() setBuffer:bp offset:part_bytes atIndex:1];
            [tenc() setBuffer:bo offset:0 atIndex:2];
            [tenc() setBytes:&aa length:sizeof(AttnArgs) atIndex:3];
            [tenc() dispatchThreads:MTLSizeMake(QD, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
            }
        }
    fused_path:
        auto fused = fused_attention() ? token_attention_pipeline((unsigned)allow, dh, REP) : nil;
        if (fused) {
            [tenc() setComputePipelineState:fused];
            [tenc() setBuffer:bq offset:0 atIndex:0];
            [tenc() setBuffer:bk offset:0 atIndex:1];
            [tenc() setBuffer:bv offset:0 atIndex:2];
            [tenc() setBuffer:bo offset:0 atIndex:3];
            AttnArgs fa{};
            fa.dh = dh; fa.ctx = ctx; fa.allow = (unsigned)allow;
            fa.scale = scale; fa.rep = REP;
            [tenc() setBytes:&fa length:sizeof(AttnArgs) atIndex:4];
            const size_t scratch = (sizeof(float) * allow + 15) & ~size_t(15);
            [tenc() setThreadgroupMemoryLength:scratch atIndex:0];
            [tenc() dispatchThreadgroups:MTLSizeMake(H, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }
        if (kv_half())
            throw std::runtime_error("metal: half K/V cache needs the gqa or fused attention path (allow > 4096 with TM_METAL_ATTN_GQA=0)");
        id<MTLBuffer> bp = slot_buf(probslot, sizeof(float) * H * ctx);
        [tenc() setComputePipelineState:kern(Kern::attn_scores)];
        [tenc() setBuffer:bq offset:0 atIndex:0];
        [tenc() setBuffer:bk offset:0 atIndex:1];
        [tenc() setBuffer:bp offset:0 atIndex:2];
        set_scalar(H, 3);
        set_scalar(dh, 4);
        set_scalar(ctx, 5);
        set_scalar(allow, 6);
        set_scalar(scale, 7);
        set_scalar(REP, 8);
        [tenc() dispatchThreads:MTLSizeMake((size_t)H * allow, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [tenc() setComputePipelineState:kern(Kern::attn_softmax)];
        [tenc() setBuffer:bp offset:0 atIndex:0];
        set_scalar(H, 1);
        set_scalar(ctx, 2);
        set_scalar(allow, 3);
        [tenc() dispatchThreads:MTLSizeMake(H, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [tenc() setComputePipelineState:kern(Kern::attn_pv)];
        [tenc() setBuffer:bp offset:0 atIndex:0];
        [tenc() setBuffer:bv offset:0 atIndex:1];
        [tenc() setBuffer:bo offset:0 atIndex:2];
        set_scalar(H, 3);
        set_scalar(dh, 4);
        set_scalar(ctx, 5);
        set_scalar(allow, 6);
        set_scalar(REP, 7);
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
