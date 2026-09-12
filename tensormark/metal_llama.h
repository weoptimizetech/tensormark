// tensormark/metal_llama.h — Llama prefill (T > 1) as a GPU layer stack
// (2026-09-07). Included by metal_shim.mm (ObjC++); the C++ runtime reaches
// it through the tm_metal_llama_* C shim under TM_HAVE_METAL.
//
// Why: with the projections on MPS the CPU stages between them (RMSNorm,
// RoPE, split, KV append, blocked attention, SiLU, residuals) and the four
// GPU round trips per layer were ~60% of a 2000-token TinyLlama prefill.
// Here one command buffer per layer holds everything: rmsnorm -> q/k/v
// GEMMs (dequant-to-scratch + MPS, fp16 for M >= 1024) -> RoPE -> K/V rows
// appended straight into the CPU cache (zero-copy) -> attention per (KV
// group, 128-query block) as two MPS GEMMs around a masked-softmax kernel
// -> o_proj -> residual -> rmsnorm -> gate/up GEMMs -> silu*up -> down ->
// residual. Command buffers are committed as they are encoded and only the
// last is waited on, so CPU encoding overlaps GPU execution. Weights are
// read from a zero-copy view of the .tmq mapping. Decode is untouched.
#pragma once
#include "metal.h"
#include "tm_tune.h"
#include "llama_gpu_api.h"
#include <unistd.h>
#include <atomic>
#include <memory>
#include <map>
#include <vector>

namespace tmgpu {
namespace llama_gpu {

inline NSString* kSrc() {
    return @R"MET(
#include <metal_stdlib>
using namespace metal;

kernel void ll_rmsnorm(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       const device float* w [[buffer(2)]], constant uint& D [[buffer(3)]],
                       constant float& eps [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                       uint tpg [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const uint nsg = (tpg + 31) / 32;
    const device float* xr = x + row * D;
    float s = 0.0f;
    for (uint i = tid; i < D; i += tpg) s += xr[i] * xr[i];
    s = simd_sum(s);
    if (lane == 0) red[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < nsg; ++i) tot += red[i];
    const float r = 1.0f / sqrt(tot / D + eps);
    device float* yr = y + row * D;
    for (uint i = tid; i < D; i += tpg) yr[i] = xr[i] * r * w[i];
}

// RoPE in place on rows of `nheads` heads (row stride = nheads*dh), pairs
// (j, j+dh/2), position pos0 + row — the CPU rope_head convention.
kernel void ll_rope(device float* q [[buffer(0)]], constant uint& nheads [[buffer(1)]],
                    constant uint& dh [[buffer(2)]], constant uint& pos0 [[buffer(3)]],
                    constant float& theta [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint half_ = dh / 2;
    const uint j = gid % half_;
    const uint h = (gid / half_) % nheads;
    const uint t = gid / (half_ * nheads);
    device float* r = q + (size_t)t * nheads * dh + h * dh;
    const float inv = precise::pow(theta, -2.0f * (float)j / (float)dh);
    const float ang = (float)(pos0 + t) * inv;
    const float c = precise::cos(ang), s = precise::sin(ang);
    const float a = r[j], b = r[j + half_];
    r[j] = a * c - b * s;
    r[j + half_] = b * c + a * s;
}

// k, v rows (T x KVH*dh) -> caches [KVH][ctx][dh] at rows pos0 + t
kernel void ll_kv_append(const device float* k [[buffer(0)]], const device float* v [[buffer(1)]],
                         device float* kc [[buffer(2)]], device float* vc [[buffer(3)]],
                         constant uint& KVD [[buffer(4)]], constant uint& dh [[buffer(5)]],
                         constant uint& ctx [[buffer(6)]], constant uint& pos0 [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint t = gid / KVD, c = gid % KVD, g = c / dh, j = c % dh;
    const size_t dst = (size_t)g * ctx * dh + (size_t)(pos0 + t) * dh + j;
    kc[dst] = k[gid];
    vc[dst] = v[gid];
}

// q rows t0..t0+bt of group g (REP heads, columns g*REP*dh ..) -> qb (bt*REP x dh)
kernel void ll_gather_q(const device float* q [[buffer(0)]], device float* qb [[buffer(1)]],
                        constant uint& QD [[buffer(2)]], constant uint& REP [[buffer(3)]],
                        constant uint& dh [[buffer(4)]], constant uint& t0 [[buffer(5)]],
                        constant uint& g [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    // gid over bt*REP*dh: row = t*REP + h
    const uint j = gid % dh, hr = (gid / dh) % REP, t = gid / (dh * REP);
    qb[gid] = q[(size_t)(t0 + t) * QD + (g * REP + hr) * dh + j];
}
// ob (bt*REP x dh) -> o rows t0.., columns g*REP*dh..
kernel void ll_scatter_o(const device float* ob [[buffer(0)]], device float* o [[buffer(1)]],
                         constant uint& D [[buffer(2)]], constant uint& REP [[buffer(3)]],
                         constant uint& dh [[buffer(4)]], constant uint& t0 [[buffer(5)]],
                         constant uint& g [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    const uint j = gid % dh, hr = (gid / dh) % REP, t = gid / (dh * REP);
    o[(size_t)(t0 + t) * D + (g * REP + hr) * dh + j] = ob[gid];
}

// Causal masked softmax on S rows (ld floats apart): row r of the block sees
// keys 0 .. pos0 + t0 + r/REP; masked entries become 0.
kernel void ll_softmax(device float* S [[buffer(0)]], constant uint& ld [[buffer(1)]],
                       constant uint& allow_max [[buffer(2)]], constant uint& base [[buffer(3)]],
                       constant uint& REP [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                       uint tpg [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const uint nsg = (tpg + 31) / 32;
    device float* s = S + (size_t)row * ld;
    const uint allow = base + row / REP + 1;
    float m = -1e30f;
    for (uint i = tid; i < allow; i += tpg) m = max(m, s[i]);
    m = simd_max(m);
    if (lane == 0) red[sgid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -1e30f;
    for (uint i = 0; i < nsg; ++i) M = max(M, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float z = 0.0f;
    for (uint i = tid; i < allow; i += tpg) { const float e = exp(s[i] - M); s[i] = e; z += e; }
    z = simd_sum(z);
    if (lane == 0) red[sgid] = z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float Z = 0.0f;
    for (uint i = 0; i < nsg; ++i) Z += red[i];
    const float inv = 1.0f / Z;
    for (uint i = tid; i < allow; i += tpg) s[i] *= inv;
    for (uint i = allow + tid; i < allow_max; i += tpg) s[i] = 0.0f;
}

kernel void ll_silu_mul(device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    const float x = g[gid];
    g[gid] = x / (1.0f + exp(-x)) * u[gid];
}
kernel void ll_add(device float* x [[buffer(0)]], const device float* a [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    x[gid] += a[gid];
}

// ---- half-activation variants (fp16 mode, M >= 1024): the residual stream
// x and the KV caches stay fp32; everything between GEMMs is half. ----
kernel void ll_rmsnorm_h(const device float* x [[buffer(0)]], device half* y [[buffer(1)]],
                         const device float* w [[buffer(2)]], constant uint& D [[buffer(3)]],
                         constant float& eps [[buffer(4)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const uint nsg = (tpg + 31) / 32;
    const device float* xr = x + row * D;
    float s = 0.0f;
    for (uint i = tid; i < D; i += tpg) s += xr[i] * xr[i];
    s = simd_sum(s);
    if (lane == 0) red[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < nsg; ++i) tot += red[i];
    const float r = 1.0f / sqrt(tot / D + eps);
    device half* yr = y + row * D;
    for (uint i = tid; i < D; i += tpg) yr[i] = (half)(xr[i] * r * w[i]);
}
kernel void ll_rope_h(device half* q [[buffer(0)]], constant uint& nheads [[buffer(1)]],
                      constant uint& dh [[buffer(2)]], constant uint& pos0 [[buffer(3)]],
                      constant float& theta [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint half_ = dh / 2;
    const uint j = gid % half_;
    const uint h = (gid / half_) % nheads;
    const uint t = gid / (half_ * nheads);
    device half* r = q + (size_t)t * nheads * dh + h * dh;
    const float inv = precise::pow(theta, -2.0f * (float)j / (float)dh);
    const float ang = (float)(pos0 + t) * inv;
    const float c = precise::cos(ang), s = precise::sin(ang);
    const float a = (float)r[j], b = (float)r[j + half_];
    r[j] = (half)(a * c - b * s);
    r[j + half_] = (half)(b * c + a * s);
}
// half k, v rows -> fp32 caches, and half copies of the group caches for the
// attention GEMMs (k16/v16: [KVH][ctx][dh] half, rows [0, pos0+T))
kernel void ll_kv_append_h(const device half* k [[buffer(0)]], const device half* v [[buffer(1)]],
                           device float* kc [[buffer(2)]], device float* vc [[buffer(3)]],
                           device half* k16 [[buffer(4)]], device half* v16 [[buffer(5)]],
                           constant uint& KVD [[buffer(6)]], constant uint& dh [[buffer(7)]],
                           constant uint& ctx [[buffer(8)]], constant uint& pos0 [[buffer(9)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint t = gid / KVD, c = gid % KVD, g = c / dh, j = c % dh;
    const size_t dst = (size_t)g * ctx * dh + (size_t)(pos0 + t) * dh + j;
    kc[dst] = (float)k[gid]; vc[dst] = (float)v[gid];
    k16[dst] = k[gid]; v16[dst] = v[gid];
}
// fp32 cache rows [0, n) of every group -> half copies (for pos0 > 0 prefills)
kernel void ll_cache_to_h(const device float* kc [[buffer(0)]], const device float* vc [[buffer(1)]],
                          device half* k16 [[buffer(2)]], device half* v16 [[buffer(3)]],
                          constant uint& dh [[buffer(4)]], constant uint& ctx [[buffer(5)]],
                          constant uint& n [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    const uint g = gid / (n * dh), r = gid % (n * dh);
    const size_t i = (size_t)g * ctx * dh + r;
    k16[i] = (half)kc[i]; v16[i] = (half)vc[i];
}
kernel void ll_gather_q_h(const device half* q [[buffer(0)]], device half* qb [[buffer(1)]],
                          constant uint& QD [[buffer(2)]], constant uint& REP [[buffer(3)]],
                          constant uint& dh [[buffer(4)]], constant uint& t0 [[buffer(5)]],
                          constant uint& g [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    const uint j = gid % dh, hr = (gid / dh) % REP, t = gid / (dh * REP);
    qb[gid] = q[(size_t)(t0 + t) * QD + (g * REP + hr) * dh + j];
}
kernel void ll_scatter_o_h(const device half* ob [[buffer(0)]], device half* o [[buffer(1)]],
                           constant uint& D [[buffer(2)]], constant uint& REP [[buffer(3)]],
                           constant uint& dh [[buffer(4)]], constant uint& t0 [[buffer(5)]],
                           constant uint& g [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    const uint j = gid % dh, hr = (gid / dh) % REP, t = gid / (dh * REP);
    o[(size_t)(t0 + t) * D + (g * REP + hr) * dh + j] = ob[gid];
}
kernel void ll_softmax_h(device half* S [[buffer(0)]], constant uint& ld [[buffer(1)]],
                         constant uint& allow_max [[buffer(2)]], constant uint& base [[buffer(3)]],
                         constant uint& REP [[buffer(4)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const uint nsg = (tpg + 31) / 32;
    device half* s = S + (size_t)row * ld;
    const uint allow = base + row / REP + 1;
    float m = -1e30f;
    for (uint i = tid; i < allow; i += tpg) m = max(m, (float)s[i]);
    m = simd_max(m);
    if (lane == 0) red[sgid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -1e30f;
    for (uint i = 0; i < nsg; ++i) M = max(M, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float z = 0.0f;
    for (uint i = tid; i < allow; i += tpg) z += exp((float)s[i] - M);
    z = simd_sum(z);
    if (lane == 0) red[sgid] = z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float Z = 0.0f;
    for (uint i = 0; i < nsg; ++i) Z += red[i];
    const float inv = 1.0f / Z;
    for (uint i = tid; i < allow; i += tpg) s[i] = (half)(exp((float)s[i] - M) * inv);
    for (uint i = allow + tid; i < allow_max; i += tpg) s[i] = (half)0.0f;
}
kernel void ll_silu_mul_h(device half* g [[buffer(0)]], const device half* u [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    const float x = (float)g[gid];
    g[gid] = (half)(x / (1.0f + exp(-x)) * (float)u[gid]);
}
kernel void ll_add_h(device float* x [[buffer(0)]], const device half* a [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    x[gid] += (float)a[gid];
}

// Fused causal attention for prefill (2026-09-07), half in / float
// accumulate on the SIMD matrix unit. One 128-thread group per (row block
// of 64 (query, head) rows of one kv group, in gathered-Q order row =
// t*REP + hr); each SIMD group owns 16 rows with its Q tiles in registers,
// walks the shared K/V tiles (32 keys) with an online softmax kept in
// threadgroup memory (S float, P half, per-row max/sum), rescales the O
// accumulators through a diagonal matrix product, and writes O/l as half.
// dh = 64 only; the MPS path stays for other head sizes.
kernel void ll_flash_h(const device half* qb [[buffer(0)]],
                       const device half* kc16 [[buffer(1)]], const device half* vc16 [[buffer(2)]],
                       device half* ob [[buffer(3)]],
                       constant uint& rows_total [[buffer(4)]], constant uint& REP [[buffer(5)]],
                       constant uint& ctx [[buffer(6)]], constant uint& pos0 [[buffer(7)]],
                       constant uint& T [[buffer(8)]], constant float& scale [[buffer(9)]],
                       constant uint& group [[buffer(10)]], constant uint& dbg [[buffer(11)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint ti [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    // 32 keys staged per step (two threadgroup barriers), processed as two
    // 16-key sub-steps through a 16x16 S^T tile per SIMD group. S^T = K Q^T:
    // plain row loads for K and V, Q^T transposed once into registers,
    // softmax lanes read consecutive addresses, P^T aliases S^T.
    constexpr uint DH = 64, BK = 32, SUB = 16, KS = 72, VS = 72, SS = 16, PS = 24;
    threadgroup half Ks[BK * KS];
    threadgroup half Vs[BK * VS];
    threadgroup float Ss[4][16 * SS];
    threadgroup half Ds[4][64];
    threadgroup float Mr[4][16], Lr[4][16], Ar[4][16];
    threadgroup uint Rs[4];
    const uint g = group, r0 = tg.x * 64, rs = r0 + sg * 16;
    const device half* kh = kc16 + (size_t)g * ctx * DH;
    const device half* vh = vc16 + (size_t)g * ctx * DH;
    const uint last_row = min(r0 + 63, rows_total - 1);
    const uint kend = min(pos0 + T, pos0 + last_row / REP + 1);
    simdgroup_half8x8 qt[2][8];
    for (uint i = 0; i < 2; ++i) {
        const uint rr = rs + i * 8;
        const device half* src = qb + (size_t)(rr + 8 <= rows_total ? rr : 0) * DH;
        for (uint k = 0; k < 8; ++k) simdgroup_load(qt[i][k], src + k * 8, DH, ulong2(0, 0), true);
    }
    simdgroup_float8x8 o[2][8];
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 8; ++j) o[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (lane < 16) { Mr[sg][lane] = -INFINITY; Lr[sg][lane] = 0.0f; }
    const uint row = lane >> 1, half_ = lane & 1;
    const uint qrow = rs + row;
    const uint qpos = pos0 + (qrow < rows_total ? qrow / REP : 0);
    threadgroup half* Pp = (threadgroup half*)&Ss[sg][0];
    for (uint kb = 0; kb < kend; kb += BK) {
        // 1. stage 32 keys: thread -> (key ti/4, 16 dims)
        {
            const uint kr = ti >> 2, c0 = (ti & 3) * 16, key = kb + kr;
            const bool ok = key < kend;
            const device half4* ksrc = (const device half4*)(kh + (size_t)key * DH + c0);
            const device half4* vsrc = (const device half4*)(vh + (size_t)key * DH + c0);
            threadgroup half4* kd = (threadgroup half4*)(Ks + kr * KS + c0);
            threadgroup half4* vd = (threadgroup half4*)(Vs + kr * VS + c0);
            for (uint u = 0; u < 4; ++u) { kd[u] = ok ? ksrc[u] : half4(0.0h); vd[u] = ok ? vsrc[u] : half4(0.0h); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub = 0; sub < BK; sub += SUB) {
            // 2. S^T = K Q^T for keys sub..sub+15: four independent
            // accumulator chains (the two-chain form was latency-bound:
            // 205 ms vs 90 ms for the same FLOPs in P.V).
            {
                simdgroup_float8x8 a00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), a01 = a00, a10 = a00, a11 = a00;
                for (uint k = 0; k < 8; ++k) {
                    simdgroup_half8x8 k0, k1;
                    simdgroup_load(k0, Ks + (sub + 0) * KS + k * 8, KS);
                    simdgroup_load(k1, Ks + (sub + 8) * KS + k * 8, KS);
                    simdgroup_multiply_accumulate(a00, k0, qt[0][k], a00);
                    simdgroup_multiply_accumulate(a01, k0, qt[1][k], a01);
                    simdgroup_multiply_accumulate(a10, k1, qt[0][k], a10);
                    simdgroup_multiply_accumulate(a11, k1, qt[1][k], a11);
                }
                simdgroup_store(a00, &Ss[sg][0], SS);
                simdgroup_store(a01, &Ss[sg][8], SS);
                simdgroup_store(a10, &Ss[sg][8 * SS], SS);
                simdgroup_store(a11, &Ss[sg][8 * SS + 8], SS);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // 3. online softmax; P^T over S^T
            if (dbg & 1) {
                for (uint e = lane; e < 16 * PS; e += 32) Pp[e] = (half)0.01h;
                if (lane == 0) Rs[sg] = 0u;
            } else {
                const threadgroup float* scol = &Ss[sg][(half_ * 8) * SS + row];
                float v[8]; float mx = -INFINITY;
                for (uint c = 0; c < 8; ++c) {
                    const uint key = kb + sub + half_ * 8 + c;
                    float x = scol[c * SS] * scale;
                    if (key > qpos || key >= kend) x = -INFINITY;
                    v[c] = x; mx = max(mx, x);
                }
                mx = max(mx, simd_shuffle_xor(mx, 1));
                const float m_old = Mr[sg][row];
                const float m_new = max(m_old, mx);
                const bool dead = m_new == -INFINITY;
                const float alpha = dead ? 1.0f : exp(m_old - m_new);
                float p[8]; float sum = 0.0f;
                for (uint c = 0; c < 8; ++c) { p[c] = dead ? 0.0f : exp(v[c] - m_new); sum += p[c]; }
                sum += simd_shuffle_xor(sum, 1);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup half* pcol = Pp + (half_ * 8) * PS + row;
                for (uint c = 0; c < 8; ++c) pcol[c * PS] = (half)p[c];
                if (half_ == 0) { Mr[sg][row] = m_new; Lr[sg][row] = Lr[sg][row] * alpha + sum; Ar[sg][row] = alpha; }
                const bool need = simd_any(alpha != 1.0f);
                if (lane == 0) Rs[sg] = need ? 1u : 0u;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // 4. rescale O rows when any max moved
            if (Rs[sg] && !(dbg & 2)) {
                for (uint i = 0; i < 2; ++i) {
                    for (uint e = lane; e < 64; e += 32) {
                        const uint rr = e >> 3, cc = e & 7;
                        Ds[sg][e] = (rr == cc) ? (half)Ar[sg][i * 8 + rr] : (half)0.0h;
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_half8x8 dm; simdgroup_load(dm, Ds[sg], 8);
                    for (uint j = 0; j < 8; ++j) { simdgroup_float8x8 t; simdgroup_multiply(t, dm, o[i][j]); o[i][j] = t; }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            // 5. O += P V for keys sub..sub+15
            if (!(dbg & 4))
            for (uint kk = 0; kk < 2; ++kk) {
                simdgroup_half8x8 p0, p1;
                simdgroup_load(p0, Pp + (kk * 8) * PS + 0, PS, ulong2(0, 0), true);
                simdgroup_load(p1, Pp + (kk * 8) * PS + 8, PS, ulong2(0, 0), true);
                for (uint j = 0; j < 8; ++j) {
                    simdgroup_half8x8 vt;
                    simdgroup_load(vt, Vs + (sub + kk * 8) * VS + j * 8, VS);
                    simdgroup_multiply_accumulate(o[0][j], p0, vt, o[0][j]);
                    simdgroup_multiply_accumulate(o[1][j], p1, vt, o[1][j]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);   // P^T consumed before the next S^T store
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 6. normalize and write O
    for (uint pass = 0; pass < 4; ++pass) {
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 2; ++j)
                simdgroup_store(o[i][pass * 2 + j], &Ss[sg][(i * 8) * SS + j * 8], SS);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 16 * 16; e += 32) {
            const uint rr = e >> 4, cc = e & 15;
            const uint grow = rs + rr;
            if (grow < rows_total) {
                const float l = Lr[sg][rr];
                ob[(size_t)grow * DH + pass * 16 + cc] = (half)(l > 0.0f ? Ss[sg][rr * SS + cc] / l : 0.0f);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)MET";
}

struct Layer {
    size_t q, k, v, o, gate, up, down;   // byte offsets of the Q4 block streams in the mapping
    id<MTLBuffer> rms1, rms2, kc, vc;    // norm weights (copied, 8 KB); caches zero-copy
};
struct Ctx {
    int L, D, H, KVH, dh, F, ctx, REP, QD, KVD;
    float theta, eps;
    id<MTLLibrary> lib;
    std::map<std::string, id<MTLComputePipelineState>> pso;
    id<MTLBuffer> map;                   // the .tmq mapping (zero-copy)
    std::vector<Layer> layers;
    id<MTLBuffer> x, h, q, k, v, o, h2, g, u, dn, qb, S, ob;
    id<MTLBuffer> h16, q16, k16, v16, o16, h216, g16, u16, dn16, qb16, S16, ob16, kc16, vc16;   // fp16 mode
    size_t cap_T = 0;
    struct Mm { MPSMatrixMultiplication* mm; MPSMatrixDescriptor *da, *db, *dc; };
    std::map<std::string, Mm> mm;        // attention GEMM objects by shape
};

inline id<MTLComputePipelineState> pso(Ctx& c, const char* name) {
    auto it = c.pso.find(name);
    if (it != c.pso.end()) return it->second;
    id<MTLFunction> fn = [c.lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    NSError* err = nil;
    id<MTLComputePipelineState> p = fn ? [detail::dev() newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!p) std::fprintf(stderr, "[metal-llama] pipeline %s failed: %s\n", name,
                         err ? err.localizedDescription.UTF8String : "no function");
    c.pso[name] = p;
    return p;
}

inline bool ensure_scratch(Ctx& c, int T) {
    if ((size_t)T <= c.cap_T) return true;
    auto mk = [&](size_t bytes) { return [detail::dev() newBufferWithLength:bytes options:MTLResourceStorageModeShared]; };
    const size_t TD = (size_t)T * c.D * 4;
    c.x = mk(TD); c.h = mk(TD); c.o = mk(TD); c.h2 = mk(TD); c.dn = mk(TD);
    c.q = mk((size_t)T * c.QD * 4); c.k = mk((size_t)T * c.KVD * 4); c.v = mk((size_t)T * c.KVD * 4);
    c.g = mk((size_t)T * c.F * 4); c.u = mk((size_t)T * c.F * 4);
    const size_t rows = 128 * (size_t)c.REP;
    c.qb = mk(rows * c.dh * 4); c.ob = mk(rows * c.dh * 4);
    c.S = mk(rows * ((size_t)c.ctx + 4) * 4);
    if (!(c.x && c.h && c.o && c.h2 && c.dn && c.q && c.k && c.v && c.g && c.u && c.qb && c.ob && c.S)) return false;
    c.h16 = mk(TD / 2); c.o16 = mk(TD / 2); c.h216 = mk(TD / 2); c.dn16 = mk(TD / 2);
    c.q16 = mk((size_t)T * c.QD * 2); c.k16 = mk((size_t)T * c.KVD * 2); c.v16 = mk((size_t)T * c.KVD * 2);
    c.g16 = mk((size_t)T * c.F * 2); c.u16 = mk((size_t)T * c.F * 2);
    const size_t frows = std::max(rows, (size_t)T * c.REP);   // flash path gathers every row
    c.qb16 = mk(frows * c.dh * 2); c.ob16 = mk(frows * c.dh * 2); c.S16 = mk(rows * ((size_t)c.ctx + 8) * 2);
    if (!c.kc16) { c.kc16 = mk((size_t)c.KVD * c.ctx * 2); c.vc16 = mk((size_t)c.KVD * c.ctx * 2); }
    if (!(c.h16 && c.o16 && c.h216 && c.dn16 && c.q16 && c.k16 && c.v16 && c.g16 && c.u16 && c.qb16 && c.ob16 && c.S16 && c.kc16 && c.vc16)) return false;
    c.cap_T = T;
    return true;
}

inline Ctx::Mm& attn_mm(Ctx& c, const char* kind, unsigned M, unsigned N, unsigned K, unsigned ldA, unsigned ldB, unsigned ldC, bool transB, float alpha, bool half = false) {
    char key[128];
    std::snprintf(key, sizeof key, "%s/%u/%u/%u/%u/%u/%u/%d/%d", kind, M, N, K, ldA, ldB, ldC, (int)transB, (int)half);
    const MPSDataType dt = half ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
    const unsigned es = half ? 2 : 4;
    auto it = c.mm.find(key);
    if (it != c.mm.end()) return it->second;
    if (c.mm.size() > 4096) c.mm.clear();
    Ctx::Mm m;
    m.mm = [[MPSMatrixMultiplication alloc] initWithDevice:detail::dev() transposeLeft:NO transposeRight:(transB ? YES : NO)
                                                resultRows:M resultColumns:N interiorColumns:K alpha:alpha beta:0.0];
    m.da = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:ldA * es dataType:dt];
    m.db = transB ? [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:ldB * es dataType:dt]
                  : [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:ldB * es dataType:dt];
    m.dc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:ldC * es dataType:dt];
    return c.mm[key] = m;
}

inline void set_u32(id<MTLComputeCommandEncoder> e, unsigned v, int idx) { [e setBytes:&v length:4 atIndex:idx]; }

// Encode one layer into cb. `hmode`: half activations between GEMMs.
// Mp >= T: row count handed to the projection GEMMs. MPS picks a slow kernel
// for M = 5..7 (T=5: 294 ms per TinyLlama pass vs 115 at 8), so those are
// padded to 8 rows (TM_LLAMA_GPU_MPAD); the extra rows are never read.
inline bool encode_layer(Ctx& c, id<MTLCommandBuffer> cb, int l, int T, int pos0, bool hmode, int Mp) {
    Layer& L = c.layers[l];
    const unsigned D = c.D, QD = c.QD, KVD = c.KVD, F = c.F, dh = c.dh, REP = c.REP, ctx = c.ctx, TD = (unsigned)T * D;
    const float scale = 1.0f / std::sqrt((float)dh);
    const char* sfx = hmode ? "_h" : "";
    auto K = [&](const char* base) { static thread_local std::string n; n = base; n += sfx; return pso(c, n.c_str()); };
    auto enc = [&]() { return [cb computeCommandEncoder]; };
    auto elem = [&](id<MTLComputeCommandEncoder> e, unsigned n) {
        [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    id<MTLBuffer> H = hmode ? c.h16 : c.h, Q = hmode ? c.q16 : c.q, Kb = hmode ? c.k16 : c.k, Vb = hmode ? c.v16 : c.v;
    id<MTLBuffer> O = hmode ? c.o16 : c.o, H2 = hmode ? c.h216 : c.h2, G = hmode ? c.g16 : c.g, U = hmode ? c.u16 : c.u;
    id<MTLBuffer> DN = hmode ? c.dn16 : c.dn, QB = hmode ? c.qb16 : c.qb, S = hmode ? c.S16 : c.S, OB = hmode ? c.ob16 : c.ob;
    id<MTLBuffer> KC = hmode ? c.kc16 : L.kc, VC = hmode ? c.vc16 : L.vc;
    const size_t kes = hmode ? 2 : 4;
    auto rms = [&](id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> w) {
        id<MTLComputeCommandEncoder> e = enc();
        [e setComputePipelineState:K("ll_rmsnorm")];
        [e setBuffer:in offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1]; [e setBuffer:w offset:0 atIndex:2];
        set_u32(e, D, 3); [e setBytes:&c.eps length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
    };
    // Fused Q4 GEMM (metal.h q4_mm_tile2) for short prompts: it reads the Q4
    // blocks straight into the SIMD matrix unit, so the dequant pass and its
    // N*K half scratch disappear. In isolation it beats the dequant+MPS path
    // at every prefill shape (M=128 1.22-1.36x, M=512 1.06-1.11x, M=2000 a
    // wash), but in the layer pipeline MPS's dequant overlaps with the
    // neighbouring independent GEMMs and is nearly free, so the win survives
    // only while the GEMMs are too small to hide it. Measured in the model
    // (prefill t/s, MPS -> fused): T=64 419 -> 671, T=128 558 -> 724,
    // T=256 683 -> 748, T=320 734 -> 759, T=384 748 -> 749, T=512 767 -> 758.
    // That measurement predated q4_mm_tile2 and the per-row-count variant
    // choice. Re-measured 2026-09-07 with both in place, the fused path wins
    // at every length, quiet and loaded (prefill t/s, MPS above 384 rows ->
    // always fused): quiet T=256 856 -> 857, T=512 800 -> 867, T=1024 816 ->
    // 845, T=2000 785 -> 789; 4 burners T=2000 755 -> 774, because the
    // dequant pass writes and re-reads N*K halves and memory bandwidth is the
    // one resource CPU work actually contends for (register-bound CPU threads
    // leave the GPU at 2531 ms against 2534 idle; memory-heavy ones cost 6%).
    // Hence the default threshold 0 = no limit. TM_LLAMA_Q4MM=0 disables the
    // fused path entirely; the threshold is in autotune.py's search space, so
    // a machine where MPS wins again will find it.
    // 2 (default) = fused with the per-shape tile choice, 1 = force the 32x64
    // tile, 0 = MPS only.
    static const int q4mm = tmtune::get_int("TM_LLAMA_Q4MM", 2);
    static const unsigned q4mm_max_m = (unsigned)tmtune::get_int("TM_LLAMA_Q4MM_MAX_M", 0);
    const bool use_q4mm = q4mm && hmode && (q4mm_max_m == 0 || (unsigned)Mp <= q4mm_max_m);
    auto gemm = [&](size_t woff, id<MTLBuffer> A, id<MTLBuffer> C, unsigned N, unsigned Kd) {
        if (use_q4mm && detail::encode_q4_mm_tile(cb, c.map, woff, A, C, 0, (unsigned)Mp, N, Kd, q4mm == 2 ? 0 : q4mm))
            return true;
        return detail::encode_mps_q4_gemm(cb, c.map, woff, A, C, 0, (unsigned)Mp, N, Kd, hmode, hmode);
    };
    // 1. norm, q/k/v
    rms(c.x, H, L.rms1);
    if (!gemm(L.q, H, Q, QD, D) || !gemm(L.k, H, Kb, KVD, D) || !gemm(L.v, H, Vb, KVD, D)) return false;
    // 2. rope + kv append (+ half cache copies in hmode)
    {
        id<MTLComputeCommandEncoder> e = enc();
        [e setComputePipelineState:K("ll_rope")];
        [e setBuffer:Q offset:0 atIndex:0]; set_u32(e, c.H, 1); set_u32(e, dh, 2); set_u32(e, (unsigned)pos0, 3);
        [e setBytes:&c.theta length:4 atIndex:4];
        elem(e, (unsigned)T * c.H * (dh / 2));
        [e setBuffer:Kb offset:0 atIndex:0]; set_u32(e, c.KVH, 1);
        elem(e, (unsigned)T * c.KVH * (dh / 2));
        if (hmode) {
            if (pos0 > 0) {   // earlier rows of the fp32 cache -> half copies
                [e setComputePipelineState:K("ll_cache_to")];
                [e setBuffer:L.kc offset:0 atIndex:0]; [e setBuffer:L.vc offset:0 atIndex:1];
                [e setBuffer:c.kc16 offset:0 atIndex:2]; [e setBuffer:c.vc16 offset:0 atIndex:3];
                set_u32(e, dh, 4); set_u32(e, ctx, 5); set_u32(e, (unsigned)pos0, 6);
                elem(e, (unsigned)c.KVH * pos0 * dh);
            }
            [e setComputePipelineState:K("ll_kv_append")];
            [e setBuffer:Kb offset:0 atIndex:0]; [e setBuffer:Vb offset:0 atIndex:1];
            [e setBuffer:L.kc offset:0 atIndex:2]; [e setBuffer:L.vc offset:0 atIndex:3];
            [e setBuffer:c.kc16 offset:0 atIndex:4]; [e setBuffer:c.vc16 offset:0 atIndex:5];
            set_u32(e, KVD, 6); set_u32(e, dh, 7); set_u32(e, ctx, 8); set_u32(e, (unsigned)pos0, 9);
        } else {
            [e setComputePipelineState:K("ll_kv_append")];
            [e setBuffer:Kb offset:0 atIndex:0]; [e setBuffer:Vb offset:0 atIndex:1];
            [e setBuffer:L.kc offset:0 atIndex:2]; [e setBuffer:L.vc offset:0 atIndex:3];
            set_u32(e, KVD, 4); set_u32(e, dh, 5); set_u32(e, ctx, 6); set_u32(e, (unsigned)pos0, 7);
        }
        elem(e, (unsigned)T * KVD);
        [e endEncoding];
    }
    // 3. attention per (group, 128-query block)
    const int B = 128;
    static const int diag = [] { const char* e = std::getenv("TM_LLAMA_GPU_DIAG"); return e ? std::atoi(e) : 0; }();   // 1: skip attention (timing only)
    static const bool flash_ok = [] { const char* e = std::getenv("TM_LLAMA_FLASH"); return !(e && e[0] == '0'); }();
    const bool flash = flash_ok && hmode && dh == 64 && diag != 1;
    if (flash) {
        // gather all (t, head) rows of every group, one fused kernel per
        // group over 64-row blocks, scatter back
        const unsigned rows_all = (unsigned)T * REP;
        for (int g = 0; g < c.KVH; ++g) {
            id<MTLComputeCommandEncoder> e = enc();
            [e setComputePipelineState:K("ll_gather_q")];
            [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:QB offset:0 atIndex:1];
            set_u32(e, QD, 2); set_u32(e, REP, 3); set_u32(e, dh, 4); set_u32(e, 0u, 5); set_u32(e, (unsigned)g, 6);
            elem(e, rows_all * dh);
            [e setComputePipelineState:pso(c, "ll_flash_h")];
            [e setBuffer:QB offset:0 atIndex:0]; [e setBuffer:c.kc16 offset:0 atIndex:1]; [e setBuffer:c.vc16 offset:0 atIndex:2];
            [e setBuffer:OB offset:0 atIndex:3];
            set_u32(e, rows_all, 4); set_u32(e, REP, 5); set_u32(e, ctx, 6); set_u32(e, (unsigned)pos0, 7); set_u32(e, (unsigned)T, 8);
            [e setBytes:&scale length:4 atIndex:9]; set_u32(e, (unsigned)g, 10);
            static const unsigned fdbg = [] { const char* e = std::getenv("TM_LLAMA_FLASH_DBG"); return e && *e ? (unsigned)std::atoi(e) : 0u; }();
            set_u32(e, fdbg, 11);
            [e dispatchThreadgroups:MTLSizeMake((rows_all + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e setComputePipelineState:K("ll_scatter_o")];
            [e setBuffer:OB offset:0 atIndex:0]; [e setBuffer:O offset:0 atIndex:1];
            set_u32(e, D, 2); set_u32(e, REP, 3); set_u32(e, dh, 4); set_u32(e, 0u, 5); set_u32(e, (unsigned)g, 6);
            elem(e, rows_all * dh);
            [e endEncoding];
        }
    }
    for (int g = 0; g < ((diag == 1 || flash) ? 0 : c.KVH); ++g) {
        for (int t0 = 0; t0 < T; t0 += B) {
            const int bt = std::min(B, T - t0);
            const unsigned rows = (unsigned)bt * REP;
            const unsigned allow_max = (unsigned)(pos0 + t0 + bt);
            const unsigned ld = (allow_max + 7) / 8 * 8;
            id<MTLComputeCommandEncoder> e = enc();
            [e setComputePipelineState:K("ll_gather_q")];
            [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:QB offset:0 atIndex:1];
            set_u32(e, QD, 2); set_u32(e, REP, 3); set_u32(e, dh, 4); set_u32(e, (unsigned)t0, 5); set_u32(e, (unsigned)g, 6);
            elem(e, rows * dh);
            [e endEncoding];
            Ctx::Mm& m1 = attn_mm(c, "qk", rows, allow_max, dh, dh, dh, ld, true, scale, hmode);
            MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:QB descriptor:m1.da];
            MPSMatrix* mK = [[MPSMatrix alloc] initWithBuffer:KC offset:(size_t)g * ctx * dh * kes descriptor:m1.db];
            MPSMatrix* mS = [[MPSMatrix alloc] initWithBuffer:S descriptor:m1.dc];
            [m1.mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mK resultMatrix:mS];
            e = enc();
            [e setComputePipelineState:K("ll_softmax")];
            [e setBuffer:S offset:0 atIndex:0]; set_u32(e, ld, 1); set_u32(e, allow_max, 2);
            set_u32(e, (unsigned)(pos0 + t0), 3); set_u32(e, REP, 4);
            [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            Ctx::Mm& m2 = attn_mm(c, "pv", rows, dh, allow_max, ld, dh, dh, false, 1.0f, hmode);
            MPSMatrix* mP = [[MPSMatrix alloc] initWithBuffer:S descriptor:m2.da];
            MPSMatrix* mV = [[MPSMatrix alloc] initWithBuffer:VC offset:(size_t)g * ctx * dh * kes descriptor:m2.db];
            MPSMatrix* mO = [[MPSMatrix alloc] initWithBuffer:OB descriptor:m2.dc];
            [m2.mm encodeToCommandBuffer:cb leftMatrix:mP rightMatrix:mV resultMatrix:mO];
            e = enc();
            [e setComputePipelineState:K("ll_scatter_o")];
            [e setBuffer:OB offset:0 atIndex:0]; [e setBuffer:O offset:0 atIndex:1];
            set_u32(e, D, 2); set_u32(e, REP, 3); set_u32(e, dh, 4); set_u32(e, (unsigned)t0, 5); set_u32(e, (unsigned)g, 6);
            elem(e, rows * dh);
            [e endEncoding];
        }
    }
    // 4. o_proj + residual, norm, gate/up, silu, down + residual
    if (!gemm(L.o, O, H2, D, D)) return false;
    {
        id<MTLComputeCommandEncoder> e = enc();
        [e setComputePipelineState:K("ll_add")];
        [e setBuffer:c.x offset:0 atIndex:0]; [e setBuffer:H2 offset:0 atIndex:1];
        elem(e, TD);
        [e endEncoding];
    }
    rms(c.x, H, L.rms2);
    if (!gemm(L.gate, H, G, F, D) || !gemm(L.up, H, U, F, D)) return false;
    {
        id<MTLComputeCommandEncoder> e = enc();
        [e setComputePipelineState:K("ll_silu_mul")];
        [e setBuffer:G offset:0 atIndex:0]; [e setBuffer:U offset:0 atIndex:1];
        elem(e, (unsigned)T * F);
        [e endEncoding];
    }
    if (!gemm(L.down, G, DN, D, F)) return false;
    {
        id<MTLComputeCommandEncoder> e = enc();
        [e setComputePipelineState:K("ll_add")];
        [e setBuffer:c.x offset:0 atIndex:0]; [e setBuffer:DN offset:0 atIndex:1];
        elem(e, TD);
        [e endEncoding];
    }
    return true;
}

}  // namespace llama_gpu
}  // namespace tmgpu

namespace tmgpu {
namespace llama_gpu {

inline void* create(const tm_llama_gpu_desc* d) {
    if (!d || !detail::ensure_init()) return nullptr;
    @autoreleasepool {
        // unique_ptr guard: every failure path below must also release the
        // half-built context (kernel objects, buffers) exactly once.
        std::unique_ptr<Ctx> c(new Ctx());
        c->L = d->L; c->D = d->D; c->H = d->H; c->KVH = d->KVH; c->dh = d->dh; c->F = d->F; c->ctx = d->ctx;
        c->REP = d->H / d->KVH; c->QD = d->H * d->dh; c->KVD = d->KVH * d->dh;
        c->theta = d->theta; c->eps = d->eps;
        if (c->dh % 2 || c->ctx > 4096 || c->H % c->KVH) return nullptr;
        NSError* err = nil;
        c->lib = [detail::dev() newLibraryWithSource:kSrc() options:nil error:&err];
        if (!c->lib) {
            std::fprintf(stderr, "[metal-llama] library compile failed: %s\n", err.localizedDescription.UTF8String);
            return nullptr;
        }
        for (const char* k : {"ll_rmsnorm", "ll_rope", "ll_kv_append", "ll_gather_q", "ll_scatter_o", "ll_softmax", "ll_silu_mul", "ll_add",
                              "ll_rmsnorm_h", "ll_rope_h", "ll_kv_append_h", "ll_cache_to_h", "ll_gather_q_h", "ll_scatter_o_h",
                              "ll_softmax_h", "ll_silu_mul_h", "ll_add_h"})
            if (!pso(*c, k)) return nullptr;
        // the whole mapping, zero-copy (page-aligned mmap; length rounded up)
        c->map = detail::host_view(d->map_base, d->map_size);
        if (!c->map) { std::fprintf(stderr, "[metal-llama] mapping not page-shareable; GPU prefill off\n"); return nullptr; }
        c->layers.resize(c->L);
        const auto off = [&](const void* p) { return (size_t)((const uint8_t*)p - (const uint8_t*)d->map_base); };
        for (int l = 0; l < c->L; ++l) {
            const tm_llama_gpu_layer& s = d->layers[l];
            Layer& L = c->layers[l];
            L.q = off(s.q); L.k = off(s.k); L.v = off(s.v); L.o = off(s.o); L.gate = off(s.gate); L.up = off(s.up); L.down = off(s.down);
            for (size_t o : {L.q, L.k, L.v, L.o, L.gate, L.up, L.down})
                if (o >= d->map_size) return nullptr;
            L.rms1 = [detail::dev() newBufferWithBytes:s.rms1 length:4 * (size_t)c->D options:MTLResourceStorageModeShared];
            L.rms2 = [detail::dev() newBufferWithBytes:s.rms2 length:4 * (size_t)c->D options:MTLResourceStorageModeShared];
            L.kc = detail::host_view(s.kc, 4 * (size_t)c->KVD * c->ctx);
            L.vc = detail::host_view(s.vc, 4 * (size_t)c->KVD * c->ctx);
            if (!(L.rms1 && L.rms2 && L.kc && L.vc)) {
                std::fprintf(stderr, "[metal-llama] KV cache not page-shareable; GPU prefill off\n");
                return nullptr;
            }
        }
        return c.release();
    }
}

inline void destroy(void* h) { delete static_cast<Ctx*>(h); }

    // x: T x D fp32 in/out (residual stream after all layers).
    // Hybrid prefill (2026-09-07): the GPU takes the leading rows and the CPU the
    // tail, one layer apart. begin() encodes and commits every layer without
    // waiting; wait_layer(l) blocks until layer l's K/V rows are visible to the
    // CPU (its command buffer completed), so the caller can run its own rows of
    // layer l while the GPU is already on l+1; end() drains the rest.
    //
    // Completion protocol (2026-09-08): begin() registers a completion handler
    // per command buffer and the hybrid interleave polls per-layer flags — NO
    // MTLCommandBuffer reference is held between begin() and wait_layer(). The
    // old pending_layers() vector stored the buffers as __strong, which retains
    // only when the TU is built with -fobjc-arc; hybrid_bench (built without
    // it) let the buffers die at begin()'s autorelease-pool pop and crashed in
    // objc_msgSend from wait_layer (EXC_BAD_ACCESS on 0x48c8, 2026-09-08), or
    // hung waiting on a zombie at T=2048. Flags carry no object identity, so
    // ARC/MRC is irrelevant. 0 = running, 2 = completed, 3 = failed, 1 = timed
    // out (bounded spin: past it the K/V rows are not visible and the prefill
    // fails — the caller surfaces the error instead of hanging).
    struct LayerFlags {
        std::unique_ptr<std::atomic<int>[]> a;
        size_t n = 0;
        // Arrays retired while a buffer may still be in flight (wait_layer
        // timed out and gave up): the handler writes into the array, so it
        // must outlive the buffer. Bounded to 8 — draining only on the next
        // successful begin() bounds memory; on a GPU wedged hard enough to
        // stall 8 prefills the process is unusable anyway.
        std::vector<std::unique_ptr<std::atomic<int>[]>> retired;
    };
    inline LayerFlags& layer_flags() { static LayerFlags f; return f; }
    inline std::atomic<double>& prefill_gpu_busy_ms() {
        static std::atomic<double> v{0.0}; return v;
    }
    inline bool prefill_begin(void* h, const float* x, int T, int pos0);
    inline bool prefill_wait_layer(int l) {
        auto& f = layer_flags();
        if (l < 0 || (size_t)l >= f.n) return false;
        std::atomic<int>& fl = f.a[(size_t)l];
        // ~30 s cap: a healthy layer finishes in < 3 s even at T=1848 on the
        // 8 GB M1; past that the stack is stalled — an error beats a hang.
        for (int waited = 0; fl.load(std::memory_order_acquire) == 0; ++waited) {
            if (waited > 600000) { fl.store(1, std::memory_order_relaxed); return false; }
            usleep(50);
        }
        return fl.load(std::memory_order_relaxed) == 2;
    }
    inline double& prefill_gpu_ms() { static double v = 0; return v; }
    inline bool prefill_end() {
        auto& f = layer_flags();
        bool ok = true;
        for (size_t l = 0; l < f.n; ++l)
            if (!prefill_wait_layer((int)l)) ok = false;
        prefill_gpu_ms() = prefill_gpu_busy_ms().exchange(0.0);  // GPU busy ms, for the split controller
        if (f.a) {  // retire, don't free — an in-flight handler may still write it
            f.retired.push_back(std::move(f.a));
            if (f.retired.size() > 8) f.retired.erase(f.retired.begin());
        }
        f.n = 0;
        return ok;
    }
    inline bool prefill(void* h, float* x, int T, int pos0) {
        auto* c = static_cast<Ctx*>(h);
        if (!c || T < 2 || pos0 < 0 || pos0 + T > c->ctx) return false;
        @autoreleasepool {
            static const int mpad = [] { const char* e = std::getenv("TM_LLAMA_GPU_MPAD"); return e && *e ? std::atoi(e) : 8; }();
            const int Mp = (mpad > 0 && T < mpad) ? mpad : T;
            if (!ensure_scratch(*c, Mp)) return false;
            std::memcpy(c->x.contents, x, sizeof(float) * (size_t)T * c->D);
            // Half activations between GEMMs at every length (measured 2026-09-07:
            // 128 tok 434 -> 573 t/s, 512 566 -> 757, 2000 638 -> 707; logits within
            // 1.2e-2 of fp32, greedy identical). TM_LLAMA_GPU_HALF=0 keeps fp32 glue
            // (then the GEMMs follow the M >= 1024 policy).
            static const bool half_ok = [] { const char* e = std::getenv("TM_LLAMA_GPU_HALF"); return !(e && e[0] == '0'); }();
            const bool hmode = half_ok;
            id<MTLCommandBuffer> last = nil;
            static const bool verbose = [] { const char* e = std::getenv("TM_LLAMA_GPU_VERBOSE"); return e && e[0] == '1'; }();
            std::vector<id<MTLCommandBuffer>> cbs;
            std::vector<double> enc_ms;
            const auto t_enc0 = std::chrono::steady_clock::now();
            for (int l = 0; l < c->L; ++l) {
                const auto t0 = std::chrono::steady_clock::now();
                id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
                if (!encode_layer(*c, cb, l, T, pos0, hmode, Mp)) return false;
                [cb commit];
                last = cb;
                if (verbose) { cbs.push_back(cb); enc_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
            }
            const double enc_total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc0).count();
            [last waitUntilCompleted];
            if (last.status != MTLCommandBufferStatusCompleted) {
                std::fprintf(stderr, "[metal-llama] command buffer failed (status %ld)\n", (long)last.status);
                return false;
            }
            if (verbose) {
                // GPU busy per layer, idle gaps between layers (host encode not
                // keeping up shows as a gap), host encode per layer.
                double busy = 0, gaps = 0;
                for (size_t i = 0; i < cbs.size(); ++i) {
                    busy += cbs[i].GPUEndTime - cbs[i].GPUStartTime;
                    if (i) gaps += std::max(0.0, cbs[i].GPUStartTime - cbs[i - 1].GPUEndTime);
                }
                const double first_start = cbs.front().GPUStartTime, total = cbs.back().GPUEndTime - first_start;
                std::fprintf(stderr, "[metal-llama] T=%d pos0=%d: gpu busy %.1f ms, gaps %.1f ms, span %.1f ms; host encode %.1f ms total (layer 0 %.1f ms, mean %.2f ms); layer 0 gpu %.1f ms, layer 1 %.1f ms\n",
                             T, pos0, busy * 1e3, gaps * 1e3, total * 1e3, enc_total, enc_ms[0], enc_total / (double)c->L,
                             (cbs[0].GPUEndTime - cbs[0].GPUStartTime) * 1e3, cbs.size() > 1 ? (cbs[1].GPUEndTime - cbs[1].GPUStartTime) * 1e3 : 0.0);
            }
            std::memcpy(x, c->x.contents, sizeof(float) * (size_t)T * c->D);
            ++prefill_count();
        }
        return true;
    }
    inline bool prefill_begin(void* h, const float* x, int T, int pos0) {
        auto* c = static_cast<Ctx*>(h);
        if (!c || T < 2 || pos0 < 0 || pos0 + T > c->ctx) return false;
        @autoreleasepool {
            static const int mpad = [] { const char* e = std::getenv("TM_LLAMA_GPU_MPAD"); return e && *e ? std::atoi(e) : 8; }();
            const int Mp = (mpad > 0 && T < mpad) ? mpad : T;
            if (!ensure_scratch(*c, Mp)) return false;
            std::memcpy(c->x.contents, x, sizeof(float) * (size_t)T * c->D);
            // Half activations between GEMMs at every length (measured 2026-09-07:
            // 128 tok 434 -> 573 t/s, 512 566 -> 757, 2000 638 -> 707; logits within
            // 1.2e-2 of fp32, greedy identical). TM_LLAMA_GPU_HALF=0 keeps fp32 glue
            // (then the GEMMs follow the M >= 1024 policy).
            static const bool half_ok = [] { const char* e = std::getenv("TM_LLAMA_GPU_HALF"); return !(e && e[0] == '0'); }();
            const bool hmode = half_ok;
            id<MTLCommandBuffer> last = nil;
            static const bool verbose = [] { const char* e = std::getenv("TM_LLAMA_GPU_VERBOSE"); return e && e[0] == '1'; }();
            // Drain any stragglers a previously failed begin() left committed (a
            // fresh flags array would otherwise be freed under their handlers).
            if (layer_flags().n) prefill_end();
            auto& flags = layer_flags();
            flags.a = std::make_unique<std::atomic<int>[]>((size_t)c->L);
            flags.n = (size_t)c->L;
            prefill_gpu_busy_ms().store(0.0);
            std::vector<double> enc_ms;
            const auto t_enc0 = std::chrono::steady_clock::now();
            for (int l = 0; l < c->L; ++l) {
                const auto t0 = std::chrono::steady_clock::now();
                id<MTLCommandBuffer> cb = [detail::q() commandBuffer];
                if (!encode_layer(*c, cb, l, T, pos0, hmode, Mp)) return false;
                std::atomic<int>* flag = &flags.a[(size_t)l];
                [cb addCompletedHandler:^(id<MTLCommandBuffer> d) {
                    const double ms = (d.GPUEndTime - d.GPUStartTime) * 1e3;
                    if (ms > 0) prefill_gpu_busy_ms().fetch_add(ms);
                    flag->store(d.status == MTLCommandBufferStatusCompleted ? 2 : 3,
                                std::memory_order_release);
                }];
                [cb commit];
                last = cb;
                if (verbose) enc_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            (void)last;
            ++prefill_count();
        }
        return true;
    }

    }  // namespace llama_gpu
    }  // namespace tmgpu
