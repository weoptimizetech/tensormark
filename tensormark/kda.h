// tensormark/kda.h — gated delta-rule linear attention (KDA), tape op.
//
// Forward and backward for the fine-grained per-key-channel gated delta rule:
//
//     S'_t = (I - beta_t k_t k_t^T) S'_{t-1} + k_t (beta_t v_t / P_t)^T
//     o_t  = (q_t * P_t) @ S'_t        P_t = prod_{r<=t} exp(g_r)   (global)
//
// Semantics: a write made at step s contributes to a read at t scaled by
// P_t / P_s — memory decays per key channel from the moment it is written;
// erasure uses raw keys.  The state is carried in compensated coordinates
// (writes pre-divided by P_t).
//
// Forward is the chunked UT transform (Yang et al. NeurIPS 2024, Fig. 8):
// per chunk of C steps, one C x C triangular forward substitution plus small
// GEMMs. Backward is sequential per step: it
// recomputes the per-step states forward, then walks back — d^2-sized
// gemv/axpy/gemm work, no pool dispatches, all inside ONE fork-join over
// B*H, same shape as mha_causal.
//
// Layout (same as mha_causal):
//   Q, K, V : (B*T, D) row-major, D = H * dh, head-major columns
//   G       : (B*T, D) log-decay per channel, <= 0
//   Beta    : (B*T, H) gates in (0, 1)   (sigmoid applied upstream)
//   returns : (B*T, D)
#pragma once
#include "autograd.h"

namespace tmg {

// Steps per chunk for the UT-transform scan. Tunable via
// tm_set_param("kda_chunk", n); storage/definition is in autograd.h
// (tmg_kda_chunk).
inline int& tm_kda_chunk() { return tmg_kda_chunk(); }

namespace kda_detail {

// Tmat = (I - tril(diag(beta) K K^T, -1))^{-1}, cn x cn row-major written
// with leading dimension ldt (the caller's chunk-buffer stride C, which
// may exceed cn for a short final chunk).  k is (cn, dh); beta is (cn,).
inline void ut_transform(const float* k, const float* beta, int cn, int ldt,
                         int dh, float* Tm) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, cn, cn, dh, 1.f,
                k, dh, k, dh, 0.f, Tm, ldt);
    for (int i = 0; i < cn; ++i) {
        const float bi = beta[i];
        float* Ti = Tm + std::size_t(i) * ldt;
        for (int j = 0; j < cn; ++j) Ti[j] = (j < i ? -bi * Ti[j] : 0.f);
    }
    // NOTE: numpy updates the whole row slice at once, so the substitution
    // must read the PRE-update row (snapshot), not the partially updated one.
    // The identity diagonal is added only AFTER the whole substitution
    // (ref does `T += eye` last) — otherwise T[r,r]=1 leaks into the sums.
    for (int i = 1; i < cn; ++i) {
        float* Ti0 = Tm + std::size_t(i) * ldt;
        const std::vector<float> row(Ti0, Ti0 + cn);
        for (int j = 0; j < i; ++j) {
            float acc = 0.f;
            for (int r = 0; r < i; ++r) acc += row[r] * Tm[std::size_t(r) * ldt + j];
            Ti0[j] += acc;
        }
    }
    for (int i = 0; i < cn; ++i) Tm[std::size_t(i) * ldt + i] = 1.f;
    Tm[0] = 1.f;
}

}  // namespace kda_detail

inline ValueP kda_causal(ValueP Q, ValueP K, ValueP V, ValueP G, ValueP Beta,
                         int B, int T, int H) {
    const int D = Q->shape()[1];
    const int dh = D / H;
    const int BH = B * H;
    const int C = std::min(std::max(1, tm_kda_chunk()), T);

    Tensor out = Tensor::zeros({B * T, D});
    // Per-step global prefix products P (B*H, T, dh), kept for backward.
    auto PP = pooled(std::size_t(BH) * T * dh, /*zero=*/false);

    {
        const float* q = Q->t->data();
        const float* k = K->t->data();
        const float* v = V->t->data();
        const float* g = G->t->data();
        const float* beta = Beta->t->data();
        float* o = out.data();
        float* Pd = PP->data();
        const int nc = std::max(1, traink::nchunk());
        traink::par_chunks(BH, std::max(1, (BH + nc - 1) / nc), [&](int s0, int s1) {
            std::vector<float> Tm(std::size_t(C) * C), Pl(std::size_t(C) * dh),
                qh(std::size_t(C) * dh), vbc(std::size_t(C) * dh),
                kbc(std::size_t(C) * dh), kbk(std::size_t(C) * dh),
                bc(static_cast<std::size_t>(C)),
                Wc(std::size_t(C) * dh), Uc(std::size_t(C) * dh),
                uc(std::size_t(C) * dh), Ac(std::size_t(C) * C),
                on(std::size_t(C) * dh), st(std::size_t(dh) * dh);
            for (int s = s0; s < s1; ++s) {
                const int b = s / H, h = s % H;
                const std::size_t base = std::size_t(b) * T * D + std::size_t(h) * dh;
                const std::size_t pb = std::size_t(s) * T * dh;
                const float* qb = q + base;
                const float* kb_ = k + base;
                const float* vb = v + base;
                const float* gb = g + base;
                const float* bb = beta + std::size_t(b) * T * H + h;
                float* ob = o + base;
                float* pp = Pd + pb;
                std::fill(st.begin(), st.end(), 0.f);
                for (int i0 = 0; i0 < T; i0 += C) {
                    const int c1 = std::min(T, i0 + C), cn = c1 - i0;
                    for (int i = 0; i < cn; ++i) {
                        const float* pprev = (i == 0)
                            ? (i0 == 0 ? nullptr : pp + std::size_t(i0 - 1) * dh)
                            : pp + std::size_t(i0 + i - 1) * dh;
                        const float* gr = gb + std::size_t(i0 + i) * D;
                        float* dst = Pl.data() + std::size_t(i) * dh;
                        if (i == 0 && i0 == 0)
                            for (int j = 0; j < dh; ++j) dst[j] = std::exp(gr[j]);
                        else
                            for (int j = 0; j < dh; ++j) dst[j] = pprev[j] * std::exp(gr[j]);
                        std::copy_n(dst, dh, pp + std::size_t(i0 + i) * dh);
                    }
                    for (int i = 0; i < cn; ++i) {
                        const int t = i0 + i;
                        bc[i] = bb[std::size_t(t) * H];
                        const float bi = bc[i];
                        const float* P = Pl.data() + std::size_t(i) * dh;
                        std::copy_n(kb_ + std::size_t(t) * D, dh,
                                    kbc.data() + std::size_t(i) * dh);
                        float* vc = vbc.data() + std::size_t(i) * dh;
                        float* qc = qh.data() + std::size_t(i) * dh;
                        const float* vr = vb + std::size_t(t) * D;
                        const float* qr = qb + std::size_t(t) * D;
                        for (int j = 0; j < dh; ++j) {
                            vc[j] = vr[j] * bi / P[j];
                            qc[j] = qr[j] * P[j];
                        }
                    }
                    kda_detail::ut_transform(kbc.data(), bc.data(), cn, C, dh, Tm.data());
                    // A = tril((Q*P) @ K_raw^T, 0)  — raw keys
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, cn, cn, dh,
                                1.f, qh.data(), dh, kbc.data(), dh, 0.f, Ac.data(), C);
                    for (int i = 0; i < cn; ++i)
                        for (int j = i + 1; j < cn; ++j)
                            Ac[std::size_t(i) * C + j] = 0.f;
                    // kbk = beta * K_raw  (only W uses beta-scaled keys)
                    for (int i = 0; i < cn; ++i)
                        for (int j = 0; j < dh; ++j)
                            kbk[std::size_t(i) * dh + j] =
                                kbc[std::size_t(i) * dh + j] * bc[i];
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cn, dh, cn,
                                1.f, Tm.data(), C, kbk.data(), dh, 0.f, Wc.data(), dh);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cn, dh, cn,
                                1.f, Tm.data(), C, vbc.data(), dh, 0.f, Uc.data(), dh);
                    // u = U - W @ s
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cn, dh, dh,
                                1.f, Wc.data(), dh, st.data(), dh, 0.f, uc.data(), dh);
                    for (std::size_t i = 0; i < std::size_t(cn) * dh; ++i)
                        uc[i] = Uc[i] - uc[i];
                    // o = A @ u + (Q*P) @ s
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cn, dh, cn,
                                1.f, Ac.data(), C, uc.data(), dh, 0.f, on.data(), dh);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cn, dh, dh,
                                1.f, qh.data(), dh, st.data(), dh, 1.f, on.data(), dh);
                    for (int i = 0; i < cn; ++i)
                        std::copy_n(on.data() + std::size_t(i) * dh, dh,
                                    ob + std::size_t(i0 + i) * D);
                    // s += K_raw^T @ u
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dh, dh, cn,
                                1.f, kbc.data(), dh, uc.data(), dh, 1.f, st.data(), dh);
                }
            }
        });
    }

    auto y = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled &&
        (Q->requires_grad || K->requires_grad || V->requires_grad ||
         G->requires_grad || Beta->requires_grad)) {
        Node n;
        n.inputs = {Q, K, V, G, Beta};
        n.output = y;
        n.backward = [Q, K, V, G, Beta, B, T, H, D, dh, BH, PP](const Tensor& gt) {
            for (auto& p : {Q, K, V, G, Beta})
                if (p->requires_grad && !p->grad)
                    p->grad = std::make_shared<Tensor>(Tensor::zeros(p->shape()));
            const float* q = Q->t->data();
            const float* k = K->t->data();
            const float* v = V->t->data();
            const float* gv = G->t->data();
            const float* beta = Beta->t->data();
            const float* go = gt.data();
            const float* Pd = PP->data();
            float* dq = Q->requires_grad ? Q->grad->data() : nullptr;
            float* dk = K->requires_grad ? K->grad->data() : nullptr;
            float* dv = V->requires_grad ? V->grad->data() : nullptr;
            float* dg = G->requires_grad ? G->grad->data() : nullptr;
            float* db = Beta->requires_grad ? Beta->grad->data() : nullptr;
            const int nc = std::max(1, traink::nchunk());
            traink::par_chunks(BH, std::max(1, (BH + nc - 1) / nc), [&](int s0, int s1) {
                for (int s = s0; s < s1; ++s) {
                    const int b = s / H, h = s % H;
                    const std::size_t base = std::size_t(b) * T * D + std::size_t(h) * dh;
                    const std::size_t pb = std::size_t(s) * T * dh;
                    const float* qb = q + base;
                    const float* kb_ = k + base;
                    const float* vb = v + base;
                    const float* gb = gv + base;
                    const float* bb = beta + std::size_t(b) * T * H + h;
                    const float* gob = go + base;
                    const float* pp = Pd + pb;
                    float* dqb = dq ? dq + base : nullptr;
                    float* dkb = dk ? dk + base : nullptr;
                    float* dvb = dv ? dv + base : nullptr;
                    float* dgb = dg ? dg + base : nullptr;
                    float* dbb = db ? db + std::size_t(b) * T * H + h : nullptr;
                    const bool wq = dq || dk || db, wv = dv || db, wb = db, wg = dg || db;

                    std::vector<float> Sx((std::size_t(T) + 1) * dh * dh);
                    std::vector<float> dS(std::size_t(dh) * dh, 0.f),
                        dM(std::size_t(dh) * dh), ktS(dh), sd(dh), gma(dh),
                        cst(dh), vpt(dh), tmp(dh), dMk(dh), dMtk(dh), qt(dh);
                    float* dp = Sx.data() + std::size_t(T) * dh * dh;  // reuse? no
                    (void)dp;
                    std::vector<float> dP(dh, 0.f);
                    // ---- pass 1: recompute per-step states S'_t
                    // (T+1 slots: the t=T-1 update writes slot T)
                    {
                        float* st = Sx.data();
                        std::fill(st, st + (std::size_t(T) + 1) * dh * dh, 0.f);
                        for (int t = 0; t < T; ++t) {
                            float* nxt = st + std::size_t(dh) * dh;
                            std::copy_n(st, std::size_t(dh) * dh, nxt);
                            const float* P = pp + std::size_t(t) * dh;
                            const float* kr = kb_ + std::size_t(t) * D;
                            const float* vr = vb + std::size_t(t) * D;
                            const float bi = bb[std::size_t(t) * H];
                            // M @ s : s[i,:] -= bi*k[i]*(k^T s)
                            cblas_sgemv(CblasRowMajor, CblasTrans, dh, dh, 1.f, nxt,
                                        dh, kr, 1, 0.f, ktS.data(), 1);
                            for (int i = 0; i < dh; ++i)
                                cblas_saxpy(dh, -bi * kr[i], ktS.data(), 1,
                                           nxt + std::size_t(i) * dh, 1);
                            // + k (beta v / P)^T
                            for (int j = 0; j < dh; ++j) cst[j] = vr[j] * bi / P[j];
                            for (int i = 0; i < dh; ++i)
                                cblas_saxpy(dh, kr[i], cst.data(), 1,
                                           nxt + std::size_t(i) * dh, 1);
                            st = nxt;
                        }
                    }
                    // ---- pass 2: descend
                    for (int t = T - 1; t >= 0; --t) {
                        // pass 1 stores the post-update state of step t at
                        // slot t+1 (slot 0 = S'_0 = 0)
                        const float* st = Sx.data() + std::size_t(t + 1) * dh * dh;
                        const float* sprev = Sx.data() + std::size_t(t) * dh * dh;
                        const float* P = pp + std::size_t(t) * dh;
                        const float* kr = kb_ + std::size_t(t) * D;
                        const float* vr = vb + std::size_t(t) * D;
                        const float* qr = qb + std::size_t(t) * D;
                        const float* gr = gb + std::size_t(t) * D;
                        const float* dor = gob + std::size_t(t) * D;
                        const float bi = bb[std::size_t(t) * H];
                        for (int j = 0; j < dh; ++j) {
                            gma[j] = std::exp(gr[j]);
                            qt[j] = qr[j] * P[j];
                            cst[j] = vr[j] * bi / P[j];
                            vpt[j] = vr[j] / P[j];
                        }
                        // dS += (q*P) (do)^T
                        for (int i = 0; i < dh; ++i)
                            cblas_saxpy(dh, qt[i], dor, 1, dS.data() + std::size_t(i) * dh, 1);
                        // sd = S'_t @ do
                        cblas_sgemv(CblasRowMajor, CblasNoTrans, dh, dh, 1.f, st, dh,
                                    dor, 1, 0.f, sd.data(), 1);
                        if (dqb)
                            for (int j = 0; j < dh; ++j)
                                dqb[std::size_t(t) * D + j] += sd[j] * P[j];
                        if (wb) for (int j = 0; j < dh; ++j) dP[j] += sd[j] * qr[j];
                        // ktS = dS^T k   (used by dv, dbeta, dpw, and the flow)
                        cblas_sgemv(CblasRowMajor, CblasTrans, dh, dh, 1.f, dS.data(),
                                    dh, kr, 1, 0.f, ktS.data(), 1);
                        // write-term key gradient: dk += dS @ (beta v / P)
                        if (dkb) {
                            cblas_sgemv(CblasRowMajor, CblasNoTrans, dh, dh, 1.f,
                                        dS.data(), dh, cst.data(), 1, 0.f, tmp.data(), 1);
                            for (int j = 0; j < dh; ++j)
                                dkb[std::size_t(t) * D + j] += tmp[j];
                        }
                        if (dvb) {
                            cblas_sgemv(CblasRowMajor, CblasNoTrans, dh, dh, 1.f,
                                        dS.data(), dh, cst.data(), 1, 0.f, tmp.data(), 1);
                            for (int j = 0; j < dh; ++j)
                                dvb[std::size_t(t) * D + j] +=
                                    ktS[j] * bi / P[j];
                            (void)tmp.data();
                        }
                        if (wb) {
                            cblas_sgemv(CblasRowMajor, CblasNoTrans, dh, dh, 1.f,
                                        dS.data(), dh, vpt.data(), 1, 0.f, tmp.data(), 1);
                            float acc = cblas_sdot(dh, kr, 1, tmp.data(), 1);
                            dbb[std::size_t(t) * H] += acc;
                        }
                        if (wg) for (int j = 0; j < dh; ++j)
                            dP[j] += -ktS[j] * bi * vpt[j] / P[j];
                        // dM = dS @ S'_{t-1}^T
                        if (sprev) {
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dh,
                                        dh, dh, 1.f, dS.data(), dh, sprev, dh, 0.f,
                                        dM.data(), dh);
                            cblas_sgemv(CblasRowMajor, CblasNoTrans, dh, dh, 1.f,
                                        dM.data(), dh, kr, 1, 0.f, dMk.data(), 1);
                            cblas_sgemv(CblasRowMajor, CblasTrans, dh, dh, 1.f,
                                        dM.data(), dh, kr, 1, 0.f, dMtk.data(), 1);
                            if (dkb)
                                for (int j = 0; j < dh; ++j)
                                    dkb[std::size_t(t) * D + j] +=
                                        -bi * (dMk[j] + dMtk[j]);
                            if (wb) dbb[std::size_t(t) * H] -= cblas_sdot(dh, kr, 1, dMk.data(), 1);
                        }
                        if (wg) {
                            for (int j = 0; j < dh; ++j)
                                dgb[std::size_t(t) * D + j] += dP[j] * P[j];
                            for (int j = 0; j < dh; ++j) dP[j] *= gma[j];
                        }
                        // flow: dS <- (I - beta k k^T) dS  (= M^T dS, M symmetric)
                        if (t > 0)
                            for (int i = 0; i < dh; ++i)
                                cblas_saxpy(dh, -bi * kr[i], ktS.data(), 1,
                                           dS.data() + std::size_t(i) * dh, 1);
                    }
                }
            });
        };
        y->node = current_tape().push(std::move(n));
    }
    return y;
}

}  // namespace tmg
