// tensormark/optim.h — AdamW over tape parameters, engine-side.
//
// Kept in C++ rather than driven from Python because the update is a
// bandwidth-bound pass over every parameter and its two moment buffers: at
// 1.6M parameters that is 19 MB touched per step, which is real time if it
// goes through per-tensor Python calls. Decoupled weight decay (Loshchilov &
// Hutter) — the decay term is applied to the weight, not folded into the
// gradient, so it does not enter the second moment.
#pragma once
#include "autograd.h"

namespace tmg {

struct AdamW {
    struct Slot {
        ValueP p;
        std::vector<float> m, v;
        bool decay;
    };
    std::vector<Slot> slots;
    float beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weight_decay = 0.1f;
    long t = 0;

    void add(ValueP p, bool decay) {
        slots.push_back({p, std::vector<float>(std::size_t(p->numel()), 0.f),
                         std::vector<float>(std::size_t(p->numel()), 0.f), decay});
    }

    void zero_grad() {
        for (auto& s : slots)
            if (s.p->grad) std::fill(s.p->grad->raw().begin(), s.p->grad->raw().end(), 0.f);
    }

    long num_params() const {
        long n = 0;
        for (auto& s : slots) n += s.p->numel();
        return n;
    }

    // Global L2 norm over all gradients; scales them down in place if it
    // exceeds max_norm. Returns the pre-clip norm (the number worth logging).
    float clip_grad_norm(float max_norm) {
        float sq = 0.f;
        for (auto& s : slots) {
            if (!s.p->grad) continue;
            float d;
            vDSP_svesq(s.p->grad->data(), 1, &d, vDSP_Length(s.p->numel()));
            sq += d;
        }
        const float norm = std::sqrt(sq);
        if (max_norm > 0.f && norm > max_norm) {
            const float k = max_norm / (norm + 1e-6f);
            for (auto& s : slots) {
                if (!s.p->grad) continue;
                float* g = s.p->grad->data();
                const int n = s.p->numel();
                vDSP_vsmul(g, 1, &k, g, 1, vDSP_Length(n));
            }
        }
        return norm;
    }

    void step(float lr) {
        ++t;
        const float bc1 = 1.f - std::pow(beta1, float(t));
        const float bc2 = 1.f - std::pow(beta2, float(t));
        const float b1 = beta1, b2 = beta2, ep = eps, wd = weight_decay;
        for (auto& s : slots) {
            if (!s.p->grad) continue;
            float* w = s.p->t->data();
            const float* g = s.p->grad->data();
            float* m = s.m.data();
            float* v = s.v.data();
            const float decay = s.decay ? lr * wd : 0.f;
            const auto update = [=](int i0, int i1) {
                for (int i = i0; i < i1; ++i) {
                    m[i] = b1 * m[i] + (1.f - b1) * g[i];
                    v[i] = b2 * v[i] + (1.f - b2) * g[i] * g[i];
                    const float mh = m[i] / bc1;
                    const float vh = v[i] / bc2;
                    w[i] -= lr * mh / (std::sqrt(vh) + ep) + decay * w[i];
                }
            };
            // AdamW's short elementwise update cannot amortize a pool
            // fork/join at the generic 64K-element threshold on M1.
            // Keep larger slots parallel and honor higher global cutoffs.
            if (s.p->numel() < (1 << 18)) update(0, s.p->numel());
            else tm_par(s.p->numel(), update);
        }
    }
};

// Warmup-then-cosine schedule, the nanoGPT default shape.
inline float lr_at(long step, long warmup, long total, float lr_max, float lr_min) {
    if (step < warmup) return lr_max * float(step + 1) / float(warmup);
    if (step >= total) return lr_min;
    const float r = float(step - warmup) / float(std::max(1L, total - warmup));
    return lr_min + (lr_max - lr_min) * 0.5f * (1.f + std::cos(3.14159265358979f * r));
}

}  // namespace tmg
