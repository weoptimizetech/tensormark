// neural_demo.cpp — layers, FastNet and the CLI entry.
//
// The engine (Tensor/Rng/pool/traink/BufferPool/tmprof) lives in
// tensormark/train_engine.h, which this file includes for every existing
// includer. This file is still the include target; nothing else includes
// the header directly.
#include "tensormark/train_engine.h"

// This file is included both directly (by tests and mains) and indirectly
// (fast_kernels.h and kernel/registry.h include it), so the layer half needs
// its own guard — #pragma once in the header does not cover this file.
#ifndef TM_NEURAL_DEMO_CPP
#define TM_NEURAL_DEMO_CPP

// ============================================================================
// Layer abstraction + concrete layers.
// ============================================================================
struct Layer {
    virtual ~Layer() = default;
    // Sink parameter, not a const&: Sequential hands each layer the buffer the
    // previous layer produced, and a layer that keeps its input (every layer
    // here does — it is the backward mask or the activation the backward reads
    // again) can take that buffer over instead of copying it. The returned
    // activation still has to be a separate live buffer, so callers see no
    // aliasing. Bit-identical: this moves ownership, never reorders arithmetic.
    virtual Tensor forward(Tensor x, Rng& rng, bool training) = 0;
    virtual Tensor backward(const Tensor& grad_out, float lr) = 0;
    virtual long num_params() const = 0;
};

struct AdamState {
    Tensor m, v;
    int t = 0;
};

static inline void adam_update(Tensor& param, const Tensor& grad,
                               AdamState& st, float lr, float wd) {
    tmprof::Scope _prof(tmprof::slot(tmprof::kAdamw));
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    st.t += 1;
    const int n = param.numel();
    float* p = param.raw().data();
    const float* gptr = grad.raw().data();
    float* m = st.m.raw().data();
    float* v = st.v.raw().data();
    const float bc1 = 1.0f - std::pow(beta1, st.t);
    const float bc2 = 1.0f - std::pow(beta2, st.t);
    const float lr_bc1 = lr / bc1;
    // Parallel over parameters (disjoint slots); bias corrections hoisted
    // out of the loop. Tensor layouts are flat, so chunking is safe.
    traink::par_chunks(n, std::max(1, n / 8), [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            float g = gptr[i] + wd * p[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            p[i] -= lr_bc1 * m[i] / (std::sqrt(v[i] / bc2) + eps);
        }
    });
}

// (placeholder removed)

// ---- Dense (fully-connected) -------------------------------------------------
class Dense : public Layer {
   public:
    Dense(int in_dim, int out_dim)
        : in_dim_(in_dim), out_dim_(out_dim),
          W_(Tensor({out_dim, in_dim})), b_(Tensor::zeros({out_dim, 1})) {}

    // Fast-inference accessors (added for fast_kernels.h).
    int in_dim_v()  const { return in_dim_; }
    int out_dim_v() const { return out_dim_; }
    const Tensor& weight_tensor() const { return W_; }
    const Tensor& bias_tensor()   const { return b_; }

    void init(Rng& rng) {
        float s = std::sqrt(2.0f / float(in_dim_));
        for (auto& v : W_.raw()) v = s * rng.normal();
        for (auto& v : b_.raw()) v = 0.0f;
        m_W_ = AdamState{Tensor::zeros(W_.shape()), Tensor::zeros(W_.shape()), 0};
        m_b_ = AdamState{Tensor::zeros(b_.shape()), Tensor::zeros(b_.shape()), 0};
    }
    void set_weight_decay(float wd) { weight_decay_ = wd; }

    int in_dim() const { return in_dim_; }
    int out_dim() const { return out_dim_; }

    // x: (in_dim, B)
    Tensor forward(Tensor x, Rng&, bool) override {
        const int B = x.dim(1);
        Tensor y(Tensor::uninit({out_dim_, B}));
        // y (out x B) = W (out x in) * x (in x B) via SGEMM, then + bias.
        traink::psgemm_nn_m(out_dim_, B, in_dim_, W_.data(), x.data(), y.data(), 1.0f);
        for (int i = 0; i < out_dim_; ++i)
            for (int b = 0; b < B; ++b) y[i * B + b] += b_[i];  // b_ shape (out,1)
        x_cache_ = std::move(x);  // the input is dead here; take the buffer over
        return y;
    }

    Tensor backward(const Tensor& grad_out, float lr) override {
        const int B = x_cache_.dim(1);
        Tensor dW(Tensor::uninit(W_.shape()));
        Tensor db(Tensor::zeros(b_.shape()));
        // dW (out x in) = (1/B) * grad_out (out x B) * x^T (B x in).
        // x_cache_ is (in, B) row-major, so x^T is sgemm_nt against it.
        traink::psgemm_nt(out_dim_, in_dim_, B, grad_out.data(), x_cache_.data(),
                           dW.data(), 1.0f / float(B));
        for (int i = 0; i < out_dim_; ++i) {
            float bs = 0.0f;
            for (int b = 0; b < B; ++b) bs += grad_out[i * B + b];
            db[i] = bs / float(B);
        }
        // grad_in (in x B) = W^T (in x out) * grad_out (out x B).
        // W_ is (out, in) row-major; A^T form reads it as (in, out).
        Tensor grad_in(Tensor::uninit({in_dim_, B}));
        traink::psgemm_tn(in_dim_, B, out_dim_, W_.data(), grad_out.data(),
                           grad_in.data(), 1.0f);
        adam_update(W_, dW, m_W_, lr, weight_decay_);
        adam_update(b_, db, m_b_, lr, 0.0f);
        dW_last_ = dW;  // exposed for gradient checks
        return grad_in;
    }
    // Test/debug accessors: the (1/B)-scaled parameter gradients computed by
    // the most recent backward pass.
    const Tensor& last_dW() const { return dW_last_; }
    long num_params() const override { return W_.numel() + b_.numel(); }
   private:
    int in_dim_, out_dim_;
    Tensor W_, b_;
    AdamState m_W_, m_b_;
    Tensor x_cache_;
    float weight_decay_ = 0.0f;
    Tensor dW_last_;
};

// ---- ReLU --------------------------------------------------------------------
class ReLU : public Layer {
   public:
    Tensor forward(Tensor x, Rng&, bool) override {
        // The mask the backward pass needs IS the sign pattern of the output,
        // so cache the activated buffer and hand a copy to the next layer.
        Tensor y = std::move(x);
        for (auto& v : y.raw()) v = std::max(v, 0.0f);
        cache_ = y;
        return y;
    }
    Tensor backward(const Tensor& grad_out, float) override {
        Tensor g = grad_out;
        for (std::size_t i = 0; i < g.raw().size(); ++i)
            g.raw()[i] = (cache_[int(i)] > 0.0f) ? grad_out.raw()[i] : 0.0f;
        return g;
    }
    long num_params() const override { return 0; }
   private:
    Tensor cache_;
};

// ---- Dropout -----------------------------------------------------------------
class Dropout : public Layer {
   public:
    explicit Dropout(float p_keep) : p_keep_(p_keep) {}
    Tensor forward(Tensor x, Rng& rng, bool training) override {
        if (!training) return x;         // eval: no copy at all, was one
        // The dropped activation is dead once masked, and x is ours by value,
        // so own it and mask in place. The RNG draw order is unchanged, so a
        // fixed-seed run reproduces this mask bit for bit.
        Tensor y = std::move(x);
        mask_ = y;                       // one copy, was two
        for (auto& v : mask_.raw()) v = (rng.uniform() < p_keep_) ? 1.0f / p_keep_ : 0.0f;
        for (std::size_t i = 0; i < y.raw().size(); ++i)
            y.raw()[i] = y.raw()[i] * mask_.raw()[i];
        return y;
    }
    Tensor backward(const Tensor& grad_out, float) override {
        Tensor g = grad_out;
        const std::size_t n = g.raw().size();
        const float* src = grad_out.raw().data();
        const float* msk = mask_.raw().data();
        float* dst = g.raw().data();
        traink::par_chunks(int(n), std::max(1, int(n) / 8), [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i) dst[i] = src[i] * msk[i];
        });
        return g;
    }
    long num_params() const override { return 0; }
   private:
    float p_keep_;
    Tensor mask_;
};

// ---- BatchNorm2D (per-channel, training/eval modes) --------------------------
// Input (B, C, H, W). Normalises over (B, H, W) per channel in training; uses
// running mean/var in eval. gamma/beta are learnable per channel.
class BatchNorm2D : public Layer {
   public:
    explicit BatchNorm2D(int channels, float momentum = 0.1f, float eps = 1e-5f,
                         bool fuse_relu = false)
        // Initializer order matches member-declaration order (fuse_relu_ is
        // declared last), which is what -Wreorder-ctor wants.
        : C_(channels), momentum_(momentum), eps_(eps),
          gamma_(Tensor({channels}, 1.0f)),
          beta_(Tensor({channels}, 0.0f)),
          running_mean_(Tensor::zeros({channels})),
          running_var_(Tensor::ones({channels})),
          fuse_relu_(fuse_relu) {}

    // Fast-inference accessors.
    bool fuse_relu() const { return fuse_relu_; }
    int channels_v() const { return C_; }
    float eps_v() const { return eps_; }
    const Tensor& gamma_data() const { return gamma_; }
    const Tensor& beta_data()  const { return beta_; }
    const Tensor& rm_data()    const { return running_mean_; }
    const Tensor& rv_data()    const { return running_var_; }

    void init(Rng& rng) {
        m_gamma_ = AdamState{Tensor::zeros(gamma_.shape()), Tensor::zeros(gamma_.shape()), 0};
        m_beta_  = AdamState{Tensor::zeros(beta_.shape()),  Tensor::zeros(beta_.shape()),  0};
        (void)rng;
    }
    void set_weight_decay(float wd) { weight_decay_ = wd; }

    int channels() const { return C_; }

    Tensor forward(Tensor x, Rng&, bool training) override {
        // x: (B, C, H, W). Flatten spatial+b into a single dim per channel.
        const int B = x.dim(0), H = x.dim(2), W = x.dim(3);
        const int N = B * H * W;
        if (int(cache_mean_.size()) != C_) {
            cache_mean_.assign(C_, 0.0f);
            cache_ivar_.assign(C_, 0.0f);
        }
        if (fuse_relu_) cache_z_ = Tensor::uninit(x.shape());
        Tensor y(Tensor::uninit(x.shape()));
        traink::par_chunks(C_, std::max(1, C_ / 8), [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            float mean, var;
            if (training) {
                // Sum and sumsq over the channel via vDSP_sve. A channel's
                // (B,H,W) slice is B contiguous H*W runs, so each run is one
                // vector reduce; the per-run results accumulate in the same
                // ascending order the serial pass used. NOT bit-identical to
                // the old strided scalar loop (sve reorders within a run) —
                // measured max |delta| on the BN2D output is recorded in the
                // commit message.
                float s = 0.0f, s2 = 0.0f;
                for (int n = 0; n < B; ++n) {
                    const float* run = x.data() + std::size_t(n * C_ + c) * H * W;
                    float s1, sq;
                    vDSP_sve(run, 1, &s1, vDSP_Length(H * W));
                    vDSP_svesq(run, 1, &sq, vDSP_Length(H * W));
                    s += s1;
                    s2 += sq;
                }
                mean = s / float(N);
                var = s2 / float(N) - mean * mean + eps_;
                running_mean_[c] = (1.0f - momentum_) * running_mean_[c] + momentum_ * mean;
                running_var_[c]  = (1.0f - momentum_) * running_var_[c]  + momentum_ * var;
            } else {
                mean = running_mean_[c];
                var  = running_var_[c];
            }
            cache_mean_[c] = mean;
            cache_ivar_[c] = 1.0f / std::sqrt(var);
            float g = gamma_[c], b = beta_[c];
            float iv = cache_ivar_[c];
            if (fuse_relu_) {
                // Fused BN + ReLU: write the activated output once and
                // cache the pre-activation for the backward mask.
                for (int n = 0; n < B; ++n)
                    for (int i = 0; i < H; ++i)
                        for (int j = 0; j < W; ++j) {
                            float z = (x[((n * C_ + c) * H + i) * W + j] - mean) * iv * g + b;
                            cache_z_[((n * C_ + c) * H + i) * W + j] = z;
                            y[((n * C_ + c) * H + i) * W + j] = z > 0.0f ? z : 0.0f;
                        }
            } else {
            for (int n = 0; n < B; ++n)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        float xc = x[((n * C_ + c) * H + i) * W + j];
                        y[((n * C_ + c) * H + i) * W + j] = (xc - mean) * iv * g + b;
                    }
            }
        }
            });
        cache_x_ = std::move(x);  // all reads are done; keep the buffer, not a copy
        return y;
    }

    Tensor backward(const Tensor& grad_out, float lr) override {
        const int B = cache_x_.dim(0), H = cache_x_.dim(2), W = cache_x_.dim(3);
        const int N = B * H * W;
        Tensor grad_in(Tensor::uninit(cache_x_.shape()));
        // Per-channel dgamma/dbeta, plus the two BN-backward reduction sums.
        std::vector<float> dg(C_, 0.0f), db(C_, 0.0f);
        std::vector<float> sum_dxhat(C_, 0.0f);
        std::vector<float> sum_dxhat_xhat(C_, 0.0f);
        traink::par_chunks(C_, std::max(1, C_ / 8), [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            float gsum = 0.0f, bsum = 0.0f, sd = 0.0f, sdx = 0.0f;
            float iv = cache_ivar_[c];
            float g = gamma_[c];
            for (int n = 0; n < B; ++n)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        float go = grad_out[((n * C_ + c) * H + i) * W + j];
                        if (fuse_relu_ && cache_z_[((n * C_ + c) * H + i) * W + j] <= 0.0f)
                            continue;  // ReLU masked
                        float xc = cache_x_[((n * C_ + c) * H + i) * W + j];
                        float xhat = (xc - cache_mean_[c]) * iv;
                        float dxhat = go * g;
                        gsum += go * xhat;
                        bsum += go;
                        sd += dxhat;
                        sdx += dxhat * xhat;
                    }
            dg[c] = gsum / float(N);
            db[c] = bsum / float(N);
            sum_dxhat[c] = sd;
            sum_dxhat_xhat[c] = sdx;
        }
        });
        Tensor dgamma({C_}), dbeta({C_});
        for (int c = 0; c < C_; ++c) { dgamma[c] = dg[c]; dbeta[c] = db[c]; }
        adam_update(gamma_, dgamma, m_gamma_, lr, weight_decay_);
        adam_update(beta_,  dbeta,  m_beta_,  lr, 0.0f);

        traink::par_chunks(C_, std::max(1, C_ / 8), [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            float iv = cache_ivar_[c];
            float g = gamma_[c];
            for (int n = 0; n < B; ++n)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        float go = grad_out[((n * C_ + c) * H + i) * W + j];
                        if (fuse_relu_ && cache_z_[((n * C_ + c) * H + i) * W + j] <= 0.0f) {
                            grad_in[((n * C_ + c) * H + i) * W + j] = 0.0f;
                            continue;
                        }
                        float xc = cache_x_[((n * C_ + c) * H + i) * W + j];
                        float xhat = (xc - cache_mean_[c]) * iv;
                        float dxhat = go * g;
                        // Standard BN backward (affine form):
                        // dx = gamma * ivar * (dxhat - sum_dxhat/N - xhat * sum_dxhat_xhat/N)
                        float dx = g * iv * (dxhat - sum_dxhat[c] / float(N)
                                                   - xhat * sum_dxhat_xhat[c] / float(N));
                        grad_in[((n * C_ + c) * H + i) * W + j] = dx;
                    }
        }
        });
        return grad_in;
    }

    long num_params() const override { return 2 * C_; }
   private:
    int C_;
    float momentum_, eps_;
    Tensor gamma_, beta_;
    Tensor running_mean_, running_var_;
    AdamState m_gamma_, m_beta_;
    Tensor cache_x_;
    Tensor cache_z_;  // pre-activation, only when fuse_relu_
    std::vector<float> cache_mean_, cache_ivar_;
    float weight_decay_ = 0.0f;
    bool fuse_relu_ = false;
};

// Conv2D::col_cache_ is retained across the forward/backward pair by default;
// TM_CONV_COLCACHE=0 restores rebuilding the col in backward.
inline bool tm_conv_colcache() {
    static const bool on = [] {
        const char* v = std::getenv("TM_CONV_COLCACHE");
        return !(v && v[0] == '0');
    }();
    return on;
}

// ---- Conv2D (valid, stride 1) -------------------------------------------------
// Weights stored as (C_out, C_in, kH, kW). Input: (B, C_in, H, W).
class Conv2D : public Layer {
   public:
    Conv2D(int c_in, int c_out, int kH, int kW)
        : c_in_(c_in), c_out_(c_out), kH_(kH), kW_(kW),
          W_(Tensor::zeros({c_out, c_in, kH, kW})),
          b_(Tensor::zeros({c_out, 1, 1})) {}

    // Fast-inference accessors (added for fast_kernels.h).
    int c_in_v()  const { return c_in_; }
    int c_out_v() const { return c_out_; }
    int kH_v()    const { return kH_; }
    int kW_v()    const { return kW_; }
    const Tensor& weight_tensor() const { return W_; }
    const Tensor& bias_tensor()   const { return b_; }

    void init(Rng& rng) {
        float fan_in = float(c_in_) * kH_ * kW_;
        float s = std::sqrt(2.0f / fan_in);
        for (auto& v : W_.raw()) v = s * rng.normal();
        for (auto& v : b_.raw()) v = 0.0f;
        m_W_ = AdamState{Tensor::zeros(W_.shape()), Tensor::zeros(W_.shape()), 0};
        m_b_ = AdamState{Tensor::zeros(b_.shape()), Tensor::zeros(b_.shape()), 0};
    }
    void set_weight_decay(float wd) { weight_decay_ = wd; }

    int c_in() const { return c_in_; }
    int c_out() const { return c_out_; }
    int kH() const { return kH_; }
    int kW() const { return kW_; }

    // Stride: assumes contiguous, computes strides from shape at call time.
    static int stride_for(const std::vector<int>& shape, int axis) {
        int s = 1;
        for (std::size_t k = axis + 1; k < shape.size(); ++k) s *= shape[k];
        return s;
    }
    static int offset(const std::vector<int>& shape,
                      int B, int C, int H, int W) {
        // shape: (B,C,H,W)
        int sW = 1, sH = shape[3], sC = shape[3] * shape[2], sB = shape[3] * shape[2] * shape[1];
        return B * sB + C * sC + H * sH + W * sW;
    }

    Tensor forward(Tensor x, Rng&, bool) override {
        const int B = x.dim(0), H = x.dim(2), W = x.dim(3);
        const int oH = H - kH_ + 1, oW = W - kW_ + 1;
        Tensor y(Tensor::uninit({B, c_out_, oH, oW}));
        // im2col + SGEMM (Accelerate multithreads internally):
        // y_gemm (Cout x N) = W (Cout x K) * col^T (K x N), N = B*oH*oW.
        const int K = c_in_ * kH_ * kW_, N = B * oH * oW;
        // Built into the cache the backward pass reuses. col is a pure function
        // of x_cache_, so hand it over instead of rebuilding it there - that
        // rebuild was 7.3% of an epoch in the profile. The reuse is
        // bit-identical: im2col overwrites every element it owns.
        if (std::size_t(col_cache_.numel()) != std::size_t(N) * K) col_cache_ = Tensor::uninit({N, K});
        Tensor& col = col_cache_;
        const auto _t_im2col = tmprof::start();
        traink::im2col(x.data(), col.data(), B, c_in_, H, W, kH_, kW_, oH, oW);
        tmprof::stop(tmprof::slot(tmprof::kIm2colF), _t_im2col);
        // Serial GEMM (Accelerate threads large sizes internally); chunked
        // GEMM under dispatch oversubscribed and measured slower.
        Tensor y_gemm(Tensor::uninit({c_out_, N}));
        const auto _t_cgemm = tmprof::start();
        traink::psgemm_nt(c_out_, N, K, W_.data(), col.data(), y_gemm.data(), 1.0f);
        tmprof::stop(tmprof::slot(tmprof::kGemmF), _t_cgemm);
        const auto _t_bias = tmprof::start();
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
            for (int n = n0; n < n1; ++n)
                for (int oc = 0; oc < c_out_; ++oc)
                    for (int p = 0; p < oH * oW; ++p)
                        y[(n * c_out_ + oc) * oH * oW + p] =
                            y_gemm[oc * N + n * oH * oW + p] + b_[oc];
        });
        tmprof::stop(tmprof::slot(tmprof::kBiasF), _t_bias);
        // Retained only if the backward pass that consumes it is coming.
        if (!tm_conv_colcache()) col_cache_ = Tensor();
        const auto _t_xc = tmprof::start();
        x_cache_ = std::move(x);  // all reads are done; keep the buffer, not a copy
        tmprof::stop(tmprof::slot(tmprof::kCacheX), _t_xc);
        return y;
    }

    Tensor backward(const Tensor& grad_out, float lr) override {
        const int B = x_cache_.dim(0), H = x_cache_.dim(2), W = x_cache_.dim(3);
        const int oH = grad_out.dim(2), oW = grad_out.dim(3);
        const int K = c_in_ * kH_ * kW_;
        const int N = B * oH * oW;

        // Permute grad_out (B, Cout, oH, oW) -> go_mat (Cout, N), row-major.
        // Parallel over batch samples; disjoint source blocks, disjoint
        // destination column ranges.
        Tensor go_mat(Tensor::uninit({c_out_, N}));
        const auto _t_perm = tmprof::start();
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
            for (int n = n0; n < n1; ++n)
                for (int oc = 0; oc < c_out_; ++oc)
                    for (int p = 0; p < oH * oW; ++p)
                        go_mat[oc * N + n * oH * oW + p] =
                            grad_out[(n * c_out_ + oc) * oH * oW + p];
        });
        tmprof::stop(tmprof::slot(tmprof::kBwdPerm), _t_perm);

        // Reuse the im2col the forward pass built for this same x_cache_ when
        // its shape still matches; otherwise rebuild it, as before.
        if (std::size_t(col_cache_.numel()) != std::size_t(N) * K) {
            col_cache_ = Tensor::uninit({N, K});
            const auto _t_im2colb = tmprof::start();
            traink::im2col(x_cache_.data(), col_cache_.data(), B, c_in_, H, W,
                           kH_, kW_, oH, oW);
            tmprof::stop(tmprof::slot(tmprof::kIm2colB), _t_im2colb);
        }
        const Tensor& col = col_cache_;

        Tensor dW(Tensor::uninit(W_.shape()));
        Tensor db(Tensor::zeros(b_.shape()));
        // dW (Cout x K) = (1/B) * go_mat (Cout x N) * col (N x K).
        // col is already (N x K) row-major, so this is a plain NN GEMM.
        const auto _t_dw = tmprof::start();
        traink::psgemm_nn_k(c_out_, K, N, go_mat.data(), col.data(),
                             dW.data(), 1.0f / float(B));
        tmprof::stop(tmprof::slot(tmprof::kBwdGemmW), _t_dw);
        // Per-channel bias gradient: parallel over output channels with
        // per-channel partials written to disjoint slots.
        std::vector<float> db_par(std::size_t(c_out_), 0.0f);
        traink::par_chunks(c_out_, std::max(1, c_out_ / 8), [&](int oc0, int oc1) {
            for (int oc = oc0; oc < oc1; ++oc) {
                float s = 0.0f;
                for (int n = 0; n < N; ++n) s += go_mat[oc * N + n];
                db_par[oc] = s / float(B);
            }
        });
        for (int oc = 0; oc < c_out_; ++oc) db[oc] = db_par[oc];

        // grad_col (N x K) = go_mat^T (N x Cout) * W (Cout x K).
        // sgemm_tn reads A stored (K_c x M) = (Cout x N), i.e. go_mat as-is.
        Tensor grad_col(Tensor::uninit({N, K}));
        const auto _t_gc = tmprof::start();
        traink::psgemm_tn(N, K, c_out_, go_mat.data(), W_.data(),
                           grad_col.data(), 1.0f);
        tmprof::stop(tmprof::slot(tmprof::kBwdGemmCol), _t_gc);

        // Scatter back into input layout.
        Tensor grad_in(Tensor::zeros(x_cache_.shape()));
        const auto _t_c2i = tmprof::start();
        traink::col2im(grad_col.data(), grad_in.data(), B, c_in_, H, W,
                       kH_, kW_, oH, oW);
        tmprof::stop(tmprof::slot(tmprof::kCol2im), _t_c2i);

        adam_update(W_, dW, m_W_, lr, weight_decay_);
        adam_update(b_, db, m_b_, lr, 0.0f);
        dW_last_ = dW;  // exposed for gradient checks
        // Retention is one layer deep (forward -> backward), so the resident
        // cost stays a single col buffer instead of one per conv layer.
        col_cache_ = Tensor();
        return grad_in;
    }
    // Test/debug accessor: the (1/B)-scaled weight gradients computed by
    // the most recent backward pass.
    const Tensor& last_dW() const { return dW_last_; }
    const Tensor& input_cache() const { return x_cache_; }
    long num_params() const override { return W_.numel() + b_.numel(); }
   private:
    int c_in_, c_out_, kH_, kW_;
    Tensor W_, b_;
    AdamState m_W_, m_b_;
    Tensor x_cache_;
    float weight_decay_ = 0.0f;
    Tensor dW_last_;
    Tensor col_cache_;  // im2col output, retained across the fwd/bwd pair
};

// ---- MaxPool2D (stride = kernel size) ----------------------------------------
class MaxPool2D : public Layer {
   public:
    explicit MaxPool2D(int k) : k_(k) {}

    Tensor forward(Tensor x, Rng&, bool) override {
        const int B = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
        const int oH = H / k_, oW = W / k_;
        Tensor y(Tensor::uninit({B, C, oH, oW}));
        argmax_ = Tensor::zeros({B, C, oH, oW, k_, k_});  // 0/1 mask
        // Parallel over images; each sample touches disjoint y/argmax rows.
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
        for (int c = 0; c < C; ++c)
            for (int oy = 0; oy < oH; ++oy)
                for (int ox = 0; ox < oW; ++ox) {
                    float best = -1e30f; int bIdx = 0;
                    for (int ky = 0; ky < k_; ++ky)
                        for (int kx = 0; kx < k_; ++kx) {
                            float v = x[((n * C + c) * H + (oy * k_ + ky)) * W + (ox * k_ + kx)];
                            int idx = ((((n * C + c) * oH + oy) * oW + ox) * k_ + ky) * k_ + kx;
                            if (v > best) { best = v; bIdx = idx; }
                        }
                    y[((n * C + c) * oH + oy) * oW + ox] = best;
                    argmax_[bIdx] = 1.0f;
                }
        });
        x_cache_shape_ = x.shape();
        return y;
    }

    Tensor backward(const Tensor& grad_out, float) override {
        const int B = grad_out.dim(0), C = grad_out.dim(1), oH = grad_out.dim(2), oW = grad_out.dim(3);
        Tensor grad_in(Tensor::zeros(x_cache_shape_));
        const int H = x_cache_shape_[2], W = x_cache_shape_[3];
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
        for (int c = 0; c < C; ++c)
            for (int oy = 0; oy < oH; ++oy)
                for (int ox = 0; ox < oW; ++ox) {
                    float g = grad_out[((n * C + c) * oH + oy) * oW + ox];
                    for (int ky = 0; ky < k_; ++ky)
                        for (int kx = 0; kx < k_; ++kx) {
                            int idx = ((((n * C + c) * oH + oy) * oW + ox) * k_ + ky) * k_ + kx;
                            if (argmax_[idx] > 0.5f)
                                grad_in[((n * C + c) * H + (oy * k_ + ky)) * W + (ox * k_ + kx)] += g;
                        }
                }
        });
        return grad_in;
    }
    long num_params() const override { return 0; }
   private:
    int k_;
    Tensor argmax_;
    std::vector<int> x_cache_shape_;
};

// ---- BatchNorm1D (for 2D inputs (C, B), e.g. Dense outputs) -------------------
class BatchNorm1D : public Layer {
   public:
    explicit BatchNorm1D(int features, float momentum = 0.1f, float eps = 1e-5f)
        : F_(features), momentum_(momentum), eps_(eps),
          gamma_(Tensor({features}, 1.0f)),
          beta_(Tensor({features}, 0.0f)),
          running_mean_(Tensor::zeros({features})),
          running_var_(Tensor::ones({features})) {}

    // Fast-inference accessors.
    int features_v() const { return F_; }
    float eps_v() const { return eps_; }
    const Tensor& gamma_data() const { return gamma_; }
    const Tensor& beta_data()  const { return beta_; }
    const Tensor& rm_data()    const { return running_mean_; }
    const Tensor& rv_data()    const { return running_var_; }

    void init(Rng&) {
        m_gamma_ = AdamState{Tensor::zeros(gamma_.shape()), Tensor::zeros(gamma_.shape()), 0};
        m_beta_  = AdamState{Tensor::zeros(beta_.shape()),  Tensor::zeros(beta_.shape()),  0};
    }
    void set_weight_decay(float wd) { weight_decay_ = wd; }
    int features() const { return F_; }

    Tensor forward(Tensor x, Rng&, bool training) override {
        // x: (F, B)
        const int B = x.dim(1);
        Tensor y(Tensor::uninit(x.shape()));
        cache_mean_.assign(F_, 0.0f);
        cache_ivar_.assign(F_, 0.0f);
        // Parallel over features; each feature owns disjoint rows and its
        // running-stat update.
        traink::par_chunks(F_, std::max(1, F_ / 8), [&](int f0, int f1) {
        for (int f = f0; f < f1; ++f) {
            float mean, var;
            if (training) {
                float s = 0.0f;
                for (int b = 0; b < B; ++b) s += x[f * B + b];
                mean = s / float(B);
                float s2 = 0.0f;
                for (int b = 0; b < B; ++b) {
                    float d = x[f * B + b] - mean;
                    s2 += d * d;
                }
                var = s2 / float(B) + eps_;
                running_mean_[f] = (1.0f - momentum_) * running_mean_[f] + momentum_ * mean;
                running_var_[f]  = (1.0f - momentum_) * running_var_[f]  + momentum_ * var;
            } else {
                mean = running_mean_[f];
                var  = running_var_[f];
            }
            cache_mean_[f] = mean;
            cache_ivar_[f] = 1.0f / std::sqrt(var);
            float iv = cache_ivar_[f];
            float g = gamma_[f];
            float beta_v = beta_[f];
            for (int bn = 0; bn < B; ++bn) {
                float xc = x[f * B + bn];
                y[f * B + bn] = (xc - mean) * iv * g + beta_v;
            }
        }
        });
        cache_x_ = std::move(x);  // all reads are done; keep the buffer, not a copy
        return y;
    }

    Tensor backward(const Tensor& grad_out, float lr) override {
        const int B = grad_out.dim(1);
        Tensor grad_in(grad_out.shape());
        std::vector<float> dg(F_, 0.0f), db(F_, 0.0f);
        traink::par_chunks(F_, std::max(1, F_ / 8), [&](int f0, int f1) {
        for (int f = f0; f < f1; ++f) {
            float gsum = 0.0f, bsum = 0.0f;
            float iv = cache_ivar_[f];
            for (int b = 0; b < B; ++b) {
                float go = grad_out[f * B + b];
                float xc = cache_x_[f * B + b];
                gsum += go * (xc - cache_mean_[f]) * iv;
                bsum += go;
            }
            dg[f] = gsum / float(B);
            db[f] = bsum / float(B);
        }
        });
        Tensor dgamma({F_}), dbeta({F_});
        for (int f = 0; f < F_; ++f) { dgamma[f] = dg[f]; dbeta[f] = db[f]; }
        adam_update(gamma_, dgamma, m_gamma_, lr, weight_decay_);
        adam_update(beta_,  dbeta,  m_beta_,  lr, 0.0f);

        traink::par_chunks(F_, std::max(1, F_ / 8), [&](int f0, int f1) {
        for (int f = f0; f < f1; ++f) {
            float iv = cache_ivar_[f];
            float g = gamma_[f];
            for (int bn = 0; bn < B; ++bn) {
                float go = grad_out[f * B + bn];
                grad_in[f * B + bn] = iv * g * (go - dg[f] - db[f] / float(B));
            }
        }
        });
        return grad_in;
    }

    long num_params() const override { return 2 * F_; }
   private:
    int F_;
    float momentum_, eps_;
    Tensor gamma_, beta_;
    Tensor running_mean_, running_var_;
    AdamState m_gamma_, m_beta_;
    Tensor cache_x_;
    std::vector<float> cache_mean_, cache_ivar_;
    float weight_decay_ = 0.0f;
};

// ---- Flatten (B,C,H,W) -> (C*H*W, B) -----------------------------------------
class Flatten : public Layer {
   public:
    Tensor forward(Tensor x, Rng&, bool) override {
        cache_shape_ = x.shape();
        const int B = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
        Tensor y({C * H * W, B});
        // Parallel over batch samples; each writes a disjoint column of y.
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        y[(c * H * W + i * W + j) * B + n] =
                            x[((n * C + c) * H + i) * W + j];
        });
        return y;
    }
    Tensor backward(const Tensor& grad_out, float) override {
        const int B = cache_shape_[0], C = cache_shape_[1], H = cache_shape_[2], W = cache_shape_[3];
        Tensor g(cache_shape_);
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        g[((n * C + c) * H + i) * W + j] =
                            grad_out[(c * H * W + i * W + j) * B + n];
        });
        return g;
    }
    long num_params() const override { return 0; }
   private:
    std::vector<int> cache_shape_;
};

// ---- Softmax + cross-entropy (combined, numerically stable) ------------------
class SoftmaxCrossEntropy {
   public:
    Tensor forward(const Tensor& logits, const Tensor& y_onehot) {
        const int C = logits.dim(0), B = logits.dim(1);
        probs_ = logits;  // same shape
        // numerical-stability: subtract per-sample max
        for (int b = 0; b < B; ++b) {
            float m = -1e30f;
            for (int c = 0; c < C; ++c) m = std::max(m, logits[c * B + b]);
            float s = 0.0f;
            for (int c = 0; c < C; ++c) {
                float e = std::exp(logits[c * B + b] - m);
                probs_[c * B + b] = e;
                s += e;
            }
            for (int c = 0; c < C; ++c) probs_[c * B + b] /= s;
        }
        float loss = 0.0f;
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c) {
                float p = std::max(probs_[c * B + b], 1e-12f);
                loss += -y_onehot[c * B + b] * std::log(p);
            }
        loss_ = loss / float(B);
        return probs_;
    }
    Tensor backward(const Tensor& y_onehot) {
        const int C = probs_.dim(0), B = probs_.dim(1);
        Tensor g(Tensor::uninit(probs_.shape()));
        for (int i = 0; i < C * B; ++i) g[i] = (probs_[i] - y_onehot[i]) / float(B);
        return g;
    }
    float loss() const { return loss_; }
   private:
    Tensor probs_;
    float loss_ = 0.0f;
};

// ---- Sequential container ----------------------------------------------------
class Sequential {
   public:
    Sequential() : rng_(std::make_shared<Rng>(42)) {}
    template <typename... Layers>
    explicit Sequential(Layers&&... ls) : rng_(std::make_shared<Rng>(42)) {
        (add(std::forward<Layers>(ls)), ...);
    }

    // Lightweight RNG owned by the Sequential. Used by FastNet and by callers
    // who want a stable per-network rng without managing one themselves.
    Rng& rng_ref() { return *rng_; }

    template <typename T>
    void add(T layer) {
        layers_.push_back(std::make_shared<T>(std::move(layer)));
    }

    Tensor forward(const Tensor& x, Rng& rng, bool training) {
        Tensor cur = x;
        int i = 0;
        for (auto& l : layers_) {
            tmprof::Scope _prof(tmprof::layer(i++, false));
            // Sink: the layer takes the buffer over, so nothing is copied here.
            cur = l->forward(std::move(cur), rng, training);
        }
        return cur;
    }

    Tensor backward(const Tensor& grad, float lr) {
        Tensor cur = grad;
        int i = (int)layers_.size();
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            tmprof::Scope _prof(tmprof::layer(--i, true));
            cur = (*it)->backward(cur, lr);
        }
        return cur;
    }

    long num_params() const {
        long n = 0;
        for (auto& l : layers_) n += l->num_params();
        return n;
    }
    template <typename T>
    T& get(std::size_t i) { return *std::static_pointer_cast<T>(layers_[i]); }
    Layer& at(std::size_t i) { return *layers_[i]; }

    std::size_t size() const { return layers_.size(); }

    // Fast-inference accessors (added for fast_kernels.h).
    const std::vector<std::shared_ptr<Layer>>& layers() const { return layers_; }

    void init_all(Rng& rng) {
        for (auto& l : layers_) {
            if (auto* d = dynamic_cast<Dense*>(l.get())) d->init(rng);
            else if (auto* c = dynamic_cast<Conv2D*>(l.get())) c->init(rng);
            else if (auto* bn = dynamic_cast<BatchNorm2D*>(l.get())) bn->init(rng);
            else if (auto* bn = dynamic_cast<BatchNorm1D*>(l.get())) bn->init(rng);
        }
    }
    void set_weight_decay(float wd) {
        for (auto& l : layers_) {
            if (auto* d = dynamic_cast<Dense*>(l.get())) d->set_weight_decay(wd);
            else if (auto* c = dynamic_cast<Conv2D*>(l.get())) c->set_weight_decay(wd);
            else if (auto* bn = dynamic_cast<BatchNorm2D*>(l.get())) bn->set_weight_decay(wd);
            else if (auto* bn = dynamic_cast<BatchNorm1D*>(l.get())) bn->set_weight_decay(wd);
        }
    }

   private:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::shared_ptr<Rng> rng_;
};

// ============================================================================
// XOR demo: a small MLP (kept as the smoke test).
// ============================================================================
class XORNet {
   public:
    XORNet() {
        net_.add(Dense(2, 8));
        net_.add(ReLU());
        net_.add(Dense(8, 8));
        net_.add(ReLU());
        net_.add(Dense(8, 1));
        net_.init_all(rng_);
    }
    float train_one(float x0, float x1, float y, float lr) {
        Tensor in({2, 1}); in[0] = x0; in[1] = x1;
        Tensor yt({1, 1}); yt[0] = y;
        Tensor out = net_.forward(in, rng_, false);
        float d = out[0] - y;
        Tensor g({1, 1}); g[0] = 2.0f * d;
        net_.backward(g, lr);
        return d * d;
    }
    float predict(float x0, float x1) {
        Tensor in({2, 1}); in[0] = x0; in[1] = x1;
        return net_.forward(in, rng_, false)[0];
    }
   private:
    Rng rng_{12345};
    Sequential net_;
};

// ============================================================================
// MNIST training utilities.
// ============================================================================
static inline Tensor make_image_batch(const std::vector<float>& X, int N, int H, int W) {
    Tensor t({N, 1, H, W});
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < H * W; ++k)
            t[((n * 1) * H + (k / W)) * W + (k % W)] = X[n * H * W + k];
    return t;
}

static inline Tensor make_onehot(const std::vector<int>& Y, int N, int C, float smooth = 0.0f) {
    // Label smoothing: targets are uniform(C) * smooth + (1-smooth) * one_hot.
    // smooth=0 reproduces hard targets.
    Tensor t({C, N});
    float base = smooth / float(C);
    float peak = 1.0f - smooth + base;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) t[c * N + n] = base;
        t[Y[n] * N + n] = peak;
    }
    return t;
}

float static lr_at_oracle(int step, int total_steps, float lr_max, int warmup_steps) {
    if (step < warmup_steps) return lr_max * float(step + 1) / float(warmup_steps);
    float t = float(step - warmup_steps) / float(std::max(1, total_steps - warmup_steps));
    return lr_max * 0.5f * (1.0f + std::cos(3.14159265358979323846f * t));
}

void static run_xor_oracle(const Args& a) {
    std::cout << "==== XOR (2 -> 8 -> 8 -> 1) ====\n";
    XORNet net;
    struct Pt { float x0, x1, y; };
    std::vector<Pt> data = {{0,0,0},{0,1,1},{1,0,1},{1,1,0}};
    float lr = 0.1f;
    for (int ep = 0; ep < a.xor_epochs; ++ep) {
        float loss = 0;
        for (const auto& d : data) loss += net.train_one(d.x0, d.x1, d.y, lr);
        if (ep % std::max(1, a.xor_epochs / 10) == 0 || ep + 1 == a.xor_epochs)
            std::printf("  epoch %5d  loss=%.6f\n", ep, loss / float(data.size()));
    }
    std::cout << "  predictions:\n";
    for (const auto& d : data)
        std::printf("    [%.0f, %.0f] -> %.4f  (target %.0f)\n",
                         d.x0, d.x1, net.predict(d.x0, d.x1), d.y);
}

void static run_mnist_oracle(const Args& a) {
    std::cout << "==== MNIST CNN ====\n";
    auto m = Mnist::load(a.mnist_dir, a.mnist_url);
    if (!m) { std::cerr << "MNIST unavailable\n"; return; }
    const int Ntr = int(m->train_x().size()), Nte = int(m->test_x().size());

    std::vector<float> Xtr(Ntr * 784), Xte(Nte * 784);
    std::vector<int> Ytr(Ntr), Yte(Nte);
    for (int i = 0; i < Ntr; ++i) {
        for (int k = 0; k < 784; ++k) Xtr[i * 784 + k] = m->train_x()[i][k];
        Ytr[i] = m->train_y()[i];
    }
    for (int i = 0; i < Nte; ++i) {
        for (int k = 0; k < 784; ++k) Xte[i * 784 + k] = m->test_x()[i][k];
        Yte[i] = m->test_y()[i];
    }

    // Standardise.
    std::vector<float> mean(784, 0.0f), var(784, 0.0f);
    for (int i = 0; i < Ntr; ++i)
        for (int k = 0; k < 784; ++k) mean[k] += Xtr[i * 784 + k];
    for (auto& v : mean) v /= float(Ntr);
    for (int i = 0; i < Ntr; ++i)
        for (int k = 0; k < 784; ++k) {
            float d = Xtr[i * 784 + k] - mean[k];
            var[k] += d * d;
        }
    for (auto& v : var) v = v / float(Ntr) + 1e-6f;
    for (int i = 0; i < Ntr; ++i)
        for (int k = 0; k < 784; ++k)
            Xtr[i * 784 + k] = (Xtr[i * 784 + k] - mean[k]) / std::sqrt(var[k]);
    for (int i = 0; i < Nte; ++i)
        for (int k = 0; k < 784; ++k)
            Xte[i * 784 + k] = (Xte[i * 784 + k] - mean[k]) / std::sqrt(var[k]);

    Rng rng(42);

    // Architecture (input 1x28x28):
    //   Conv(1->16,3)  -> 26x26
    //   Conv(16->16,3) -> 24x24
    //   MaxPool 2      -> 12x12
    //   Conv(16->32,3) -> 10x10
    //   Conv(32->32,3) -> 8x8
    //   MaxPool 2      -> 4x4
    //   Flatten        -> 32*4*4 = 512
    //   Dense(128)     -> ReLU -> Dropout(0.5) -> Dense(10)
    Sequential net(
        Conv2D(1, 32, 3, 3), BatchNorm2D(32, 0.1f, 1e-5f, /*fuse_relu=*/true),
        Conv2D(32, 32, 3, 3), BatchNorm2D(32, 0.1f, 1e-5f, /*fuse_relu=*/true), MaxPool2D(2),
        Conv2D(32, 64, 3, 3), BatchNorm2D(64, 0.1f, 1e-5f, /*fuse_relu=*/true),
        Conv2D(64, 64, 3, 3), BatchNorm2D(64, 0.1f, 1e-5f, /*fuse_relu=*/true), MaxPool2D(2),
        Flatten(),
        Dense(64 * 4 * 4, 256), BatchNorm1D(256, 0.1f, 1e-5f), ReLU(), Dropout(0.5f),
        Dense(256, 10));
    net.init_all(rng);
    net.set_weight_decay(a.weight_decay);

    std::cout << "  network params: " << net.num_params() << "\n";

    const int batch = a.batch_size;
    const int epochs = a.mnist_epochs;
    int steps_per_epoch = (Ntr + batch - 1) / batch;
    int total_steps = steps_per_epoch * epochs;
    int warmup_steps = std::max(1, total_steps / 20);

    SoftmaxCrossEntropy ce;
    int global_step = 0;

    for (int ep = 0; ep < epochs; ++ep) {
        const auto ep_t0 = std::chrono::steady_clock::now();
        std::vector<int> idx(Ntr);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng.engine());
        float ep_loss = 0; float ep_acc = 0; int nb = 0;
        for (int off = 0; off < Ntr; off += batch) {
            int bs = std::min(batch, Ntr - off);
            const auto _t_prep = tmprof::start();
            std::vector<float> xb(bs * 784);
            std::vector<int> yb(bs);
            for (int n = 0; n < bs; ++n) {
                int i = idx[off + n];
                // Random shift augmentation: ±2 pixels in each direction, zero-padded.
                int sx = int(rng.uniform() * 5.0f) - 2;
                int sy = int(rng.uniform() * 5.0f) - 2;
                for (int r = 0; r < 28; ++r) {
                    int sr = r - sy;
                    for (int c = 0; c < 28; ++c) {
                        int sc = c - sx;
                        float v = 0.0f;
                        if (sr >= 0 && sr < 28 && sc >= 0 && sc < 28)
                            v = Xtr[i * 784 + sr * 28 + sc];
                        xb[n * 784 + r * 28 + c] = v;
                    }
                }
                yb[n] = Ytr[i];
            }
            Tensor x = make_image_batch(xb, bs, 28, 28);
            Tensor yoh = make_onehot(yb, bs, 10, a.label_smooth);
            tmprof::stop(tmprof::slot(tmprof::kPrep), _t_prep);

            float lr_now = lr_at_oracle(global_step, total_steps, a.lr_max, warmup_steps);
            Tensor logits = net.forward(x, rng, /*training=*/true);
            const auto _t_loss = tmprof::start();
            Tensor probs = ce.forward(logits, yoh);
            // Accuracy on this batch.
            int correct = 0;
            for (int n = 0; n < bs; ++n) {
                int pc = 0; float pv = probs[n];
                for (int c = 1; c < 10; ++c) if (probs[c * bs + n] > pv) { pv = probs[c * bs + n]; pc = c; }
                if (pc == yb[n]) ++correct;
            }
            tmprof::stop(tmprof::slot(tmprof::kLoss), _t_loss);
            const auto _t_grad = tmprof::start();
            Tensor grad = ce.backward(yoh);
            // Gradient clipping (global L2).
            float n2 = 0.0f;
            for (int i = 0; i < grad.numel(); ++i) n2 += grad[i] * grad[i];
            float gn = std::sqrt(n2);
            const float clip = 1.0f;
            if (gn > clip) for (int i = 0; i < grad.numel(); ++i) grad[i] *= clip / gn;
            tmprof::stop(tmprof::slot(tmprof::kGrad), _t_grad);
            net.backward(grad, lr_now);

            ep_loss += ce.loss();
            ep_acc += float(correct) / float(bs);
            ++nb;
            ++global_step;
        }
        // Eval. B=256 thrashes L2 in the im2col/GEMM path (measured 50 vs
        // 88 samples/s on the naive path); 128 keeps the working set hot.
        int te_correct = 0;
        int eval_batch = 128;
        const auto _t_eval = tmprof::start();
        for (int off = 0; off < Nte; off += eval_batch) {
            int bs = std::min(eval_batch, Nte - off);
            Tensor x = make_image_batch(std::vector<float>(
                Xte.begin() + off * 784, Xte.begin() + (off + bs) * 784), bs, 28, 28);
            Tensor logits = net.forward(x, rng, /*training=*/false);
            for (int n = 0; n < bs; ++n) {
                int pc = 0; float pv = logits[n];
                for (int c = 1; c < 10; ++c) if (logits[c * bs + n] > pv) { pv = logits[c * bs + n]; pc = c; }
                if (pc == Yte[off + n]) ++te_correct;
            }
        }
        tmprof::stop(tmprof::slot(tmprof::kEval), _t_eval);
        std::printf(
            "  epoch %2d  lr=%.5f  train_loss=%.4f  train_acc=%.2f%%  "
            "test_acc=%.2f%%  epoch_s=%.2f  samples_s=%.0f\n",
            ep, double(lr_at_oracle(global_step - 1, total_steps, a.lr_max, warmup_steps)),
            double(ep_loss / float(nb)), double(ep_acc / float(nb) * 100.0f),
            double(float(te_correct) / float(Nte) * 100.0f),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - ep_t0).count(),
            double(float(Ntr) / std::chrono::duration<double>(std::chrono::steady_clock::now() - ep_t0).count()));
        if (tmprof::on())
            tmprof::dump(std::chrono::duration<double>(std::chrono::steady_clock::now() - ep_t0).count());
    }
}

// ============================================================================
// CLI.
// ============================================================================
Args static parse_args_oracle(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view k = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (k == "--mode")           a.mode = next("--mode");
        else if (k == "--xor-epochs")     a.xor_epochs = std::stoi(next("--xor-epochs"));
        else if (k == "--mnist-epochs")   a.mnist_epochs = std::stoi(next("--mnist-epochs"));
        else if (k == "--batch-size")     a.batch_size = std::stoi(next("--batch-size"));
        else if (k == "--lr")             a.lr_max = std::stof(next("--lr"));
        else if (k == "--weight-decay")   a.weight_decay = std::stof(next("--weight-decay"));
        else if (k == "--label-smooth")   a.label_smooth = std::stof(next("--label-smooth"));
        else if (k == "--mnist-dir")      a.mnist_dir = next("--mnist-dir");
        else if (k == "--mnist-url")      a.mnist_url = next("--mnist-url");
        else if (k == "-h" || k == "--help") {
            std::cout <<
                "Usage: neural_demo [--mode xor|mnist|all]\n"
                "                   [--xor-epochs N] [--mnist-epochs N]\n"
                "                   [--batch-size N] [--lr F] [--weight-decay F]\n"
                "                   [--mnist-dir PATH] [--mnist-url URL]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown flag: " << k << "\n";
            std::exit(2);
        }
    }
    return a;
}

// [[maybe_unused]]: tests/test_fastpath.cpp calls this, but the other TUs that
// include this file (the Metal oracles) do not - without the attribute each of
// those builds warns -Wunused-function for a live entry point.
[[maybe_unused]] int static main_oracle(int argc, char** argv) {
    Args a = parse_args_oracle(argc, argv);
    if (a.mode == "xor" || a.mode == "all") run_xor_oracle(a);
    if (a.mode == "mnist" || a.mode == "all") run_mnist_oracle(a);
    return 0;
}

#endif  // TM_NEURAL_DEMO_CPP
