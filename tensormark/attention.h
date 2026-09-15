// tensormark/attention.h — fused multi-head causal self-attention (tape op).
//
// One op for scores -> causal mask -> softmax -> value-weighted sum, forward
// and backward. Two properties make it worth fusing rather than composing it
// out of matmul/softmax/slice:
//
//   * the upper triangle is never materialised and never multiplied. Scores
//     are produced in row tiles of `attn_tile` rows; a tile of rows [i0,i1)
//     only needs columns [0,i1), so the GEMM shrinks with the tile index.
//     At T=256, tile=64 that is 0.63x the FLOPs of the dense form.
//   * heads are strided views, not copies. Q/K/V are (B*T, D) row-major with
//     head-major columns, so head h of batch b is the sub-matrix starting at
//     (b*T)*D + h*dh with leading dimension D — every step below is a single
//     strided cblas_sgemm: no packing, no permute, no per-head tensor.
//
// (b, head) pairs are independent and write disjoint memory, so the whole
// thing is one fork-join over B*H tasks with serial BLAS inside.
#pragma once
#include "autograd.h"

namespace tmg {

// Row-block height for the causal scan. Bigger = fewer BLAS calls, smaller =
// less masked-out work. Tunable through tm_set_param("attn_tile", n);
// storage/definition is in autograd.h (tm_attn_tile).

namespace attn_detail {

// Mask + row-softmax, in place, for rows [i0, i0+rows) of one (b,h) score
// block. `S` points at row i0 of a (T x T) buffer with leading dimension T.
// Row r holds query position i0+r: columns [0, i0+r] are live, columns
// (i0+r, cols) hold scores the GEMM computed but the mask kills, and columns
// >= cols are never read downstream.
inline void softmax_causal_tile(float* S, int T, int i0, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = S + std::size_t(r) * T;
        int len = i0 + r + 1;
        float m; vDSP_maxv(row, 1, &m, vDSP_Length(len));
        const float negm = -m;
        vDSP_vsadd(row, 1, &negm, row, 1, vDSP_Length(len));
        vvexpf(row, row, &len);
        float z; vDSP_sve(row, 1, &z, vDSP_Length(len));
        const float inv = 1.f / z;
        vDSP_vsmul(row, 1, &inv, row, 1, vDSP_Length(len));
        if (len < cols) vDSP_vclr(row + len, 1, vDSP_Length(cols - len));
    }
}

// dS = P * (dP - rowsum(dP * P)), in place on dP. Same tile geometry.
inline void softmax_causal_tile_bwd(float* dP, const float* P, int T, int i0,
                                    int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* dr = dP + std::size_t(r) * T;
        const float* pr = P + std::size_t(r) * T;
        const int len = i0 + r + 1;
        float dot; vDSP_dotpr(dr, 1, pr, 1, &dot, vDSP_Length(len));
        const float ndot = -dot;
        vDSP_vsadd(dr, 1, &ndot, dr, 1, vDSP_Length(len));
        vDSP_vmul(dr, 1, pr, 1, dr, 1, vDSP_Length(len));
        if (len < cols) vDSP_vclr(dr + len, 1, vDSP_Length(cols - len));
    }
}

}  // namespace attn_detail

// Fused causal multi-head attention.
//   Q, K, V : (B*T, D) row-major, D = H * dh, head-major columns
//   returns : (B*T, D)
inline ValueP mha_causal(ValueP Q, ValueP K, ValueP V, int B, int T, int H) {
    const int D = Q->shape()[1];
    const int dh = D / H;
    const float sc = 1.f / std::sqrt(float(dh));
    const int BH = B * H;
    const int tile = std::max(1, tm_attn_tile());

    Tensor out = Tensor::zeros({B * T, D});
    // Attention probabilities, (B*H, T, T), kept for backward. Every column a
    // later GEMM reads is either written by the score product or explicitly
    // zeroed by the mask, so this buffer needs no pre-fill.
    auto P = pooled(std::size_t(BH) * T * T, /*zero=*/false);

    {
        const float* q = Q->t->data();
        const float* k = K->t->data();
        const float* v = V->t->data();
        float* o = out.data();
        float* Pd = P->data();
        const int nc = std::max(1, traink::nchunk());
        traink::par_chunks(BH, std::max(1, (BH + nc - 1) / nc), [&](int s0, int s1) {
            for (int s = s0; s < s1; ++s) {
                const int b = s / H, h = s % H;
                const std::size_t base = std::size_t(b) * T * D + std::size_t(h) * dh;
                const float* qb = q + base;
                const float* kb = k + base;
                const float* vb = v + base;
                float* ob = o + base;
                float* Pb = Pd + std::size_t(s) * T * T;
                for (int i0 = 0; i0 < T; i0 += tile) {
                    const int i1 = std::min(T, i0 + tile);
                    const int rows = i1 - i0, cols = i1;
                    float* S = Pb + std::size_t(i0) * T;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, cols,
                                dh, sc, qb + std::size_t(i0) * D, D, kb, D, 0.f, S, T);
                    attn_detail::softmax_causal_tile(S, T, i0, rows, cols);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, dh,
                                cols, 1.f, S, T, vb, D, 0.f, ob + std::size_t(i0) * D, D);
                }
            }
        });
    }

    auto y = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled &&
        (Q->requires_grad || K->requires_grad || V->requires_grad)) {
        Node n;
        n.inputs = {Q, K, V};
        n.output = y;
        n.backward = [Q, K, V, P, B, T, H, D, dh, sc, tile](const Tensor& g) {
            for (auto& p : {Q, K, V})
                if (!p->grad) p->grad = std::make_shared<Tensor>(Tensor::zeros(p->shape()));
            const float* q = Q->t->data();
            const float* k = K->t->data();
            const float* v = V->t->data();
            float* dq = Q->grad->data();
            float* dk = K->grad->data();
            float* dv = V->grad->data();
            const float* go = g.data();
            const float* Pd = P->data();
            const int BH = B * H;
            const int nc = std::max(1, traink::nchunk());
            traink::par_chunks(BH, std::max(1, (BH + nc - 1) / nc), [&](int s0, int s1) {
                std::vector<float> dS(std::size_t(tile) * T);
                for (int s = s0; s < s1; ++s) {
                    const int b = s / H, h = s % H;
                    const std::size_t base = std::size_t(b) * T * D + std::size_t(h) * dh;
                    const float* qb = q + base;
                    const float* kb = k + base;
                    const float* vb = v + base;
                    const float* gb = go + base;
                    float* dqb = dq + base;
                    float* dkb = dk + base;
                    float* dvb = dv + base;
                    const float* Pb = Pd + std::size_t(s) * T * T;
                    for (int i0 = 0; i0 < T; i0 += tile) {
                        const int i1 = std::min(T, i0 + tile);
                        const int rows = i1 - i0, cols = i1;
                        const float* Pt = Pb + std::size_t(i0) * T;
                        // dV[0:cols] += P_tile^T @ dO_tile
                        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, cols, dh,
                                    rows, 1.f, Pt, T, gb + std::size_t(i0) * D, D,
                                    1.f, dvb, D);
                        // dP_tile = dO_tile @ V[0:cols]^T
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, cols,
                                    dh, 1.f, gb + std::size_t(i0) * D, D, vb, D,
                                    0.f, dS.data(), T);
                        attn_detail::softmax_causal_tile_bwd(dS.data(), Pt, T, i0, rows, cols);
                        // dQ_tile += scale * dS @ K[0:cols]
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, dh,
                                    cols, sc, dS.data(), T, kb, D, 1.f,
                                    dqb + std::size_t(i0) * D, D);
                        // dK[0:cols] += scale * dS^T @ Q_tile
                        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, cols, dh,
                                    rows, sc, dS.data(), T, qb + std::size_t(i0) * D, D,
                                    1.f, dkb, D);
                    }
                }
            });
        };
        y->node = current_tape().push(std::move(n));
    }
    return y;
}

}  // namespace tmg
