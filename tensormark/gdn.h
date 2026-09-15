// Gated DeltaNet (qwen35 / qwen3_5_text linear attention) — recurrent kernel.
//
// The exact per-token recurrence, transcribed from llama.cpp's
// src/models/delta-net-base.cpp + ggml/src/ggml-cpu/ops.cpp
// (ggml_compute_forward_gated_delta_net_one_chunk, pinned 2026-09-13):
//
//   q'    = q * (1/sqrt(S_v))
//   decay = exp(g)                       // per value head, g = log-decay
//   S_t   = decay * S_{t-1} + k_t (x) ( beta_t * (v_t - S_{t-1}^T k_t) )
//   o_t   = S_t^T q'
//
// S is the recurrent state [D_k, D_v] PER VALUE HEAD — O(1) in context length,
// replacing the KV cache for linear-attention layers. H_v is a multiple of H_k;
// the reference broadcasts the shared key heads with ggml_repeat, i.e. value
// head `hv` reads key head `hv % H_k` (NOT hv / (H_v/H_k)).
//
// The decay is the per head SCALAR exp(g) (the reference's GDA path, kda==false:
// `ggml_vec_scale_f32(S_v*S_v, s_out, expf(g_d[0]))`).
#ifndef TENSORMARK_GDN_H
#define TENSORMARK_GDN_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace tmgdn {

// One decode token. All buffers are flat, per-head contiguous.
//   S     in/out [H_v * D_k * D_v]   recurrent state (row-major [dk][dv])
//   q     in     [H_k * D_k]         (already scaled by 1/sqrt(D_k))
//   k     in     [H_k * D_k]
//   v     in     [H_v * D_v]
//   beta  in     [H_v]               (ssm_beta; the delta gate)
//   decay in     [H_v]               (= exp(g); the log-space decay exponentiated)
//   o     out    [H_v * D_v]
inline void step(float* S, const float* q, const float* k, const float* v,
                 const float* beta, const float* decay, float* o,
                 int H_k, int H_v, int D_k, int D_v) {
    std::vector<float> sk((std::size_t)D_v), d((std::size_t)D_v),
                       od((std::size_t)D_v);
    // Every loop below is element-wise over D_v, so it vectorizes — but only
    // once the compiler can prove the scratch does not alias the recurrent
    // state. Without the __restrict qualifiers it does not, and each loop
    // runs at one scalar FMA per cycle: measured 3.2 GFLOPs, which made the
    // attention section ~15% of a decode token.
    //
    // Element-wise work has no reassociation, so vectorising is bit-exact
    // here; each dv accumulator is independent. Verified by comparing
    // logits and greedy tokens against the pre-change build.
    float* __restrict skp = sk.data();
    float* __restrict dp  = d.data();
    float* __restrict odp = od.data();
    for (int hv = 0; hv < H_v; ++hv) {
        const int hk = hv % H_k;   // ggml_repeat: shared key heads broadcast
        float* __restrict Shv = S + (std::size_t)hv * D_k * D_v;
        const float* __restrict qh = q + (std::size_t)hk * D_k;
        const float* __restrict kh = k + (std::size_t)hk * D_k;
        const float* __restrict vh = v + (std::size_t)hv * D_v;
        const float b = beta[hv], dec = decay[hv];
        // S *= decay
        for (int i = 0; i < D_k * D_v; ++i) Shv[i] *= dec;
        // sk[dv] = sum_dk S[dk][dv] * k[dk]
        std::memset(skp, 0, sizeof(float) * (std::size_t)D_v);
        for (int dk = 0; dk < D_k; ++dk) {
            const float kk = kh[dk];
            const float* __restrict Srow = Shv + (std::size_t)dk * D_v;
            for (int dv = 0; dv < D_v; ++dv) skp[dv] += Srow[dv] * kk;
        }
        // d[dv] = beta * (v[dv] - sk[dv]);  S += k (x) d;  o = S^T q
        for (int dv = 0; dv < D_v; ++dv) dp[dv] = b * (vh[dv] - skp[dv]);
        std::memset(odp, 0, sizeof(float) * (std::size_t)D_v);
        for (int dk = 0; dk < D_k; ++dk) {
            const float kk = kh[dk], qq = qh[dk];
            float* __restrict Srow = Shv + (std::size_t)dk * D_v;
            for (int dv = 0; dv < D_v; ++dv) {
                // Same arithmetic, staged through a local so the vectoriser
                // is not reading back a value it just stored.
                const float s = Srow[dv] + kk * dp[dv];
                Srow[dv] = s;
                odp[dv] += s * qq;
            }
        }
        std::memcpy(o + (std::size_t)hv * D_v, odp,
                    sizeof(float) * (std::size_t)D_v);
    }
}

}  // namespace tmgdn

// ---------------------------------------------------------------------------
// Supporting kernels for the same (qwen35 linear-attention) pipeline. The
// conv/L2 forms are pinned; the gated-RMS-normaliser form follows Qwen3-Next's
// `build_norm_gated` = rms_norm(x) * silu(gate).
namespace tmgdn {

// Causal depthwise conv1d over `channels` rows of `T` tokens, kernel `K`,
// followed by SiLU. `state` holds the previous K-1 tokens (row-major
// [K-1, channels]); it is updated in place to the last K-1 rows of the input.
// in/out are row-major [T, channels]; kernel is [K, channels] with tap 0 the
// OLDEST sample (ggml_compute_forward_ssm_conv_f32 reads the window oldest-first).
inline void conv1d_silu(float* out, const float* in, float* state,
                        const float* kernel, int T, int channels, int K) {
    const int P = K - 1;
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < channels; ++c) {
            float acc = 0.f;
            for (int j = 0; j < K; ++j) {
                // tap j reads token (t - P + j); negative -> conv state
                int tt = t - P + j;
                const float xi =
                    tt < 0 ? state[(std::size_t)(tt + P) * channels + c]
                           : in[(std::size_t)tt * (std::size_t)channels + c];
                acc += kernel[(std::size_t)j * channels + c] * xi;
            }
            // SiLU
            out[(std::size_t)t * channels + c] = acc / (1.f + std::exp(-acc));
        }
    }
    for (int j = 0; j < P; ++j) {
        const int t = T - P + j;
        for (int c = 0; c < channels; ++c)
            state[(std::size_t)j * channels + c] =
                t < 0 ? 0.f : in[(std::size_t)t * channels + c];
    }
}

// Per-head L2 normalisation (DeltaNet q/k norm): x / max(||x||_2, eps) —
// identical to ggml_compute_forward_l2_norm_f32.
inline void l2_norm(float* x, int n_heads, int head_dim, float eps) {
    for (int h = 0; h < n_heads; ++h) {
        float* v = x + (std::size_t)h * head_dim;
        float ss = 0.f;
        for (int d = 0; d < head_dim; ++d) ss += v[d] * v[d];
        const float inv = 1.f / std::max(std::sqrt(ss), eps);
        for (int d = 0; d < head_dim; ++d) v[d] *= inv;
    }
}

// Elementwise nonlinearities used to build the per-head decay and gates.
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float softplus(float x) {
    // numerically stable: log(1+e^x)
    return x > 20.f ? x : std::log1p(std::exp(x));
}
// Per value head decay. `ssm_a` is the RAW loaded ssm_a tensor, which the GGUF
// already stores as -exp(A_log) (verified against Unsloth Qwen3.5-0.8B-Q4_0:
// values in [-10.58, -0.105]); the reference computes
//   g = softplus(alpha + dt) * ssm_a        // == -A_log.exp() * softplus
//   decay = exp(g)
// alpha[] is the ssm_alpha projection, dt[] its bias, all length n_v_heads.
inline void gdn_decay(float* decay, const float* alpha, const float* dt,
                      const float* ssm_a, int n_v_heads) {
    for (int h = 0; h < n_v_heads; ++h)
        decay[h] = std::exp(ssm_a[h] * softplus(alpha[h] + dt[h]));
}

// Gated RMS normalisation per value head: rms_norm(x; w) * silu(gate).
// `w` is the per-head weight (length head_dim, shared across heads) and `gate`
// is the value head's slice of the wqkv_gate projection.
inline void gated_rms_norm(float* x, const float* w, const float* gate,
                           int n_heads, int head_dim, float eps) {
    for (int h = 0; h < n_heads; ++h) {
        float* v = x + (std::size_t)h * head_dim;
        const float* g = gate + (std::size_t)h * head_dim;
        float ss = 0.f;
        for (int d = 0; d < head_dim; ++d) ss += v[d] * v[d];
        const float r = 1.f / std::sqrt(ss / (float)head_dim + eps);
        for (int d = 0; d < head_dim; ++d) {
            const float silu = g[d] / (1.f + std::exp(-g[d]));
            v[d] = v[d] * r * w[d] * silu;
        }
    }
}

// Full-attention output gate (qwen35 `attn_q` emits q|gate INTERLEAVED per
// head; the reference applies `ggml_sigmoid(gate)` to the attention output).
inline void sigmoid_gate(float* o, const float* gate, int n) {
    for (int i = 0; i < n; ++i)
        o[i] *= 1.f / (1.f + std::exp(-gate[i]));
}

// Partial RoPE over the first `n_rot` of `head_dim` dims of one head, in the
// NORM (half-split) pairing: pair j rotates (r[j], r[j + n_rot/2]) by
// theta^(-2j/n_rot). Only wqkv/dense-attn q,k at position `pos` are rotated;
// the remaining head_dim - n_rot dims are left untouched, matching
// ggml_rope_multi with mrope sections [11,11,10,0] on text-only positions
// (all three MRoPE position ids equal -> plain partial RoPE).
inline void rope_partial_head(float* r, int head_dim, int n_rot, float theta,
                              int pos) {
    if (n_rot < 2 || n_rot > head_dim) return;
    const int half = n_rot / 2;
    for (int j = 0; j < half; ++j) {
        const float inv = std::pow(theta, -2.f * j / (float)n_rot);
        const float c = std::cos((float)pos * inv), s = std::sin((float)pos * inv);
        const float a = r[j], b = r[j + half];
        r[j] = a * c - b * s;
        r[j + half] = b * c + a * s;
    }
}

}  // namespace tmgdn

#endif  // TENSORMARK_GDN_H
