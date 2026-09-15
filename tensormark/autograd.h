// tensormark/autograd.h — dynamic autograd tape (tier-3 core).
//
// Design: every op on TTensorValue records a Node (inputs + backward
// closure) into the current Tape. backward() walks the tape in reverse,
// accumulating gradients into leaf tensors (weights). GIL is released
// around kernel execution by the Python bindings, not here (pure C++).
//
// Scope: Apple M-series, Accelerate CPU, float32 only.
#pragma once
#include "engine.h"
#include <functional>
#include <memory>
#include <unordered_map>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TM_HAVE_NEON 1
#endif
#include "rowops.h"

namespace tmg {

using TMTensor = std::shared_ptr<Tensor>;

// ---------------------------------------------------------------------------
// Value: a tensor + tape linkage. Leaves (weights, inputs) have requires_grad.
// ---------------------------------------------------------------------------
struct Value : std::enable_shared_from_this<Value> {
    TMTensor t;                 // data
    TMTensor grad;              // accumulated gradient (lazily allocated)
    bool requires_grad = false;
    long node = -1;             // tape index; -1 = leaf/constant

    explicit Value(Tensor tensor, bool rg = false)
        : t(std::make_shared<Tensor>(std::move(tensor))), requires_grad(rg) {}
    Value(Tensor tensor, bool rg, long tape_node)
        : t(std::make_shared<Tensor>(std::move(tensor))), requires_grad(rg), node(tape_node) {}

    int numel() const { return t->numel(); }
    const std::vector<int>& shape() const { return t->shape(); }
};

using ValueP = std::shared_ptr<Value>;

// ---------------------------------------------------------------------------
// Tape: recorded ops in creation order.
// ---------------------------------------------------------------------------
struct Node {
    // backward: given d(loss)/d(output), ADD d(loss)/d(input) into each
    // input's grad. Captures what it needs by value (output, inputs).
    std::function<void(const Tensor&)> backward;
    std::vector<ValueP> inputs;   // keep alive
    ValueP output;
};

struct Tape {
    std::vector<Node> nodes;
    bool enabled = true;

    long push(Node n) {
        nodes.push_back(std::move(n));
        return long(nodes.size() - 1);
    }
    void clear() { nodes.clear(); }

    // Reverse-mode backward from `loss` (a scalar 0-d/1x1 value).
    void backward(ValueP loss) {
        // seed: dL/dL = 1
        loss->grad = std::make_shared<Tensor>(Tensor::ones(loss->shape()));
        // walk in reverse; nodes with grad-seeded outputs propagate
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
            if (!it->output || !it->output->grad) continue;
            it->backward(*it->output->grad);
        }
    }
};

// global current tape (single-threaded training step model, like torch's
// default graph; thread_local so sessions don't collide)
inline Tape& current_tape() {
    static thread_local Tape tape;
    return tape;
}

// ---------------------------------------------------------------------------
// Elementwise execution helpers.
//
// A pool fork-join costs tens of microseconds, so tensors below a threshold
// run on the calling thread and larger ones are split one chunk per worker.
// The threshold is a tunable knob, like the GEMM one, so the autotune sweep
// can own it. Inside a chunk we call Accelerate's vDSP/vForce rather than
// scalar loops: they are the platform's own vector math, already linked, and
// zero new dependencies.
// ---------------------------------------------------------------------------
inline int& tm_elem_par_threshold() { static int v = 1 << 16; return v; }

// Bandwidth-bound operations use a higher parallel cutoff than compute-bound
// operations to limit scheduling overhead and memory-bandwidth contention.
inline int& tm_bw_par_threshold() { static int v = 1 << 21; return v; }

template <class F>
inline void tm_par(int n, F&& body) {
    if (n < tm_elem_par_threshold()) { body(0, n); return; }
    const int nc = std::max(1, traink::nchunk());
    traink::par_chunks(n, (n + nc - 1) / nc, std::forward<F>(body));
}

// Same split, higher cutoff: for ops limited by memory bandwidth, not ALU.
template <class F>
inline void tm_par_bw(int n, F&& body) {
    if (n < tm_bw_par_threshold()) { body(0, n); return; }
    const int nc = std::max(1, traink::nchunk());
    traink::par_chunks(n, (n + nc - 1) / nc, std::forward<F>(body));
}

// Streaming elementwise kernels: below the stream cutoff use the vDSP body
// through tm_par_bw; at or above it use the loop body serially. The cutoff
// separates cache-oriented vector execution from large streaming operations.
inline int& tm_stream_threshold() { static int v = 1 << 21; return v; }
template <class VDSP, class LOOP>
inline void tm_elem_stream(int n, VDSP&& vdsp, LOOP&& loop) {
    if (n >= tm_stream_threshold()) { loop(0, n); return; }
    tm_par_bw(n, std::forward<VDSP>(vdsp));
}
inline void tm_loop_add(const float* __restrict a, const float* __restrict b,
                        float* __restrict o, int n) {
    for (int i = 0; i < n; ++i) o[i] = a[i] + b[i];
}
inline void tm_loop_mul(const float* __restrict a, const float* __restrict b,
                        float* __restrict o, int n) {
    for (int i = 0; i < n; ++i) o[i] = a[i] * b[i];
}
inline void tm_loop_acc(const float* __restrict g, float* __restrict o, int n) {  // o += g
    for (int i = 0; i < n; ++i) o[i] += g[i];
}
inline void tm_loop_fma_acc(const float* __restrict g, const float* __restrict b,
                            float* __restrict o, int n) {  // o += g*b
    for (int i = 0; i < n; ++i) o[i] += g[i] * b[i];
}
inline void tm_loop_scale(const float* __restrict a, float s, float* __restrict o, int n) {
    for (int i = 0; i < n; ++i) o[i] = a[i] * s;
}
inline void tm_loop_axpy_acc(const float* __restrict g, float s, float* __restrict o, int n) {  // o += g*s
    for (int i = 0; i < n; ++i) o[i] += g[i] * s;
}
inline void tm_loop_sadd_acc(float s, float* __restrict o, int n) {  // o += s
    for (int i = 0; i < n; ++i) o[i] += s;
}
inline void tm_loop_add_rowvec(const float* __restrict x, const float* __restrict b,
                               float* __restrict o, int rows, int cols) {  // o[r,:] = x[r,:] + b
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (std::size_t)r * cols;
        float* orow = o + (std::size_t)r * cols;
        for (int c = 0; c < cols; ++c) orow[c] = xr[c] + b[c];
    }
}

// Same, but the unit of work is a row of a (rows x cols) matrix.
template <class F>
inline void tm_par_rows(int rows, int cols, F&& body) {
    if (rows < 2 || rows * cols < tm_elem_par_threshold()) { body(0, rows); return; }
    const int nc = std::max(1, traink::nchunk());
    traink::par_chunks(rows, (rows + nc - 1) / nc, std::forward<F>(body));
}

// Row-unit variant of the bandwidth-bound dispatch.
template <class F>
inline void tm_par_rows_bw(int rows, int cols, F&& body) {
    if (rows < 2 || std::int64_t(rows) * cols < tm_bw_par_threshold()) { body(0, rows); return; }
    const int nc = std::max(1, traink::nchunk());
    traink::par_chunks(rows, (rows + nc - 1) / nc, std::forward<F>(body));
}

// Per-thread staging buffers for vForce.
inline std::vector<float>& tm_scratch(std::size_t n, int slot = 0) {
    static thread_local std::vector<float> s[4];
    auto& v = s[std::size_t(slot) & 3u];
    if (v.size() < n) v.resize(n);
    return v;
}

// Op scratch that lives as long as a tape node (attention probabilities,
// dropout masks): pool-backed, so a training loop stops handing pages back to
// the OS every step. `zero = false` means "the op fills it before reading it".
struct PooledBuf {
    std::vector<float> v;
    PooledBuf(std::size_t n, bool zero)
        : v(zero ? buffer_pool().take(n) : buffer_pool().take_raw(n)) {}
    ~PooledBuf() { buffer_pool().give(std::move(v)); }
    PooledBuf(const PooledBuf&) = delete;
    PooledBuf& operator=(const PooledBuf&) = delete;
    // deducing this: one definition covers both const and non-const access.
    template <class Self>
    decltype(std::declval<Self&>().v.data()) data(this Self&& self) {
        return self.v.data();
    }
};
using PooledBufP = std::shared_ptr<PooledBuf>;
inline PooledBufP pooled(std::size_t n, bool zero = true) {
    return std::make_shared<PooledBuf>(n, zero);
}

// A vector of ones, long enough for `n` (column reductions as one GEMV).
inline const float* tm_ones(int n) {
    static thread_local std::vector<float> v;
    if (int(v.size()) < n) v.assign(std::size_t(n), 1.f);
    return v.data();
}

// Lazily allocate a zeroed gradient buffer for a value.
inline Tensor& tm_grad(const ValueP& p) {
    if (!p->grad) p->grad = std::make_shared<Tensor>(Tensor::zeros(p->shape()));
    return *p->grad;
}
// First-writer variant: an op whose backward can WRITE its whole contribution
// (GEMM beta=0, plain assignment) asks for the buffer here; `fresh` is true
// when it is the first contributor and receives uninitialized storage it must
// fully overwrite, false when it must accumulate into an existing gradient.
// Skipping the zero-fill matters: a mini-GPT step zeroed ~240 MB of gradient
// buffers it then overwrote (5-6 ms of memset per step, sample-profiled).
inline Tensor& tm_grad_fresh(const ValueP& p, bool& fresh) {
    fresh = !p->grad;
    if (fresh) p->grad = std::make_shared<Tensor>(Tensor::uninit(p->shape()));
    return *p->grad;
}
// Pass-through: an op whose gradient w.r.t. `p` IS the incoming gradient
// (add, bias add) lets `p` share the output's gradient buffer instead of
// copying it — the same trick PyTorch's add backward uses. Safe because the
// tape runs every consumer of the output before the output's own node, so
// nothing reads the output's gradient afterwards; a later accumulation into
// p->grad therefore only touches a dead buffer. At most ONE input per op may
// alias (a second sharer would see the first one's later accumulations).
// Returns true when aliased; false means the caller must write/accumulate.
inline bool tm_grad_alias(const ValueP& p, const std::weak_ptr<Value>& out, const Tensor& g) {
    if (p->grad) return false;
    auto o = out.lock();
    if (!o || !o->grad || o->grad->data() != g.data() || o->grad->numel() != p->numel()) return false;
    p->grad = o->grad;
    return true;
}

// ---------------------------------------------------------------------------
// Ops. Each: allocate output, record Node with closure.
// ---------------------------------------------------------------------------

// C = A(M,K) @ B(K,N) row-major matmul (engine SGEMM, row-major).
inline ValueP matmul(ValueP A, ValueP B) {
    const int M = A->shape()[0], K = A->shape()[1], N = B->shape()[1];
    // beta = 0 overwrites every element of C: zeroing first is a wasted
    // bandwidth pass at exactly the size where GEMM is bandwidth-sensitive.
    Tensor out = Tensor::uninit({M, N});
    // engine sgemm_nt computes A(M,K) * B^T with B stored (N,K);
    // we have B stored (K,N) -> use nn variant.
    traink::psgemm_nn_m(M, N, K, A->t->data(), B->t->data(), out.data(), 1.0f);
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && (A->requires_grad || B->requires_grad)) {
        Node n;
        n.inputs = {A, B};
        n.output = v;
        n.backward = [A, B, M, K, N](const Tensor& g) {
            // beta = 1 accumulates straight into the gradient buffer: no temp
            // allocation and no second pass over it.
            if (A->requires_grad) {  // dA = g(M,N) @ B(K,N)^T
                bool fresh; float* ad = tm_grad_fresh(A, fresh).data();
                traink::psgemm_nt(M, K, N, g.data(), B->t->data(), ad, 1.0f, fresh ? 0.0f : 1.0f);
            }
            if (B->requires_grad) {  // dB = A(M,K)^T @ g(M,N)
                bool fresh; float* bd = tm_grad_fresh(B, fresh).data();
                traink::psgemm_tn(K, N, M, A->t->data(), g.data(), bd, 1.0f, fresh ? 0.0f : 1.0f);
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// C = A + B (elementwise, same shape)
inline ValueP add(ValueP A, ValueP B) {
    Tensor out = Tensor::uninit(A->t->shape());
    {
        const float* a = A->t->data();
        const float* b = B->t->data();
        float* o = out.data();
        tm_elem_stream(out.numel(), [=](int i0, int i1) {
            vDSP_vadd(a + i0, 1, b + i0, 1, o + i0, 1, vDSP_Length(i1 - i0));
        }, [=](int i0, int i1) { tm_loop_add(a + i0, b + i0, o + i0, i1 - i0); });
    }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && (A->requires_grad || B->requires_grad)) {
        Node n;
        n.inputs = {A, B};
        n.output = v;
        std::weak_ptr<Value> wout = v;
        n.backward = [A, B, wout](const Tensor& g) {
            bool aliased = false;
            for (auto& p : {A, B}) {
                if (!p->requires_grad) continue;
                if (!aliased && tm_grad_alias(p, wout, g)) { aliased = true; continue; }
                bool fresh; float* pd = tm_grad_fresh(p, fresh).data();
                const float* gd = g.data();
                if (fresh) { std::memcpy(pd, gd, sizeof(float) * (std::size_t)p->numel()); continue; }
                tm_elem_stream(p->numel(), [=](int i0, int i1) {
                    vDSP_vadd(pd + i0, 1, gd + i0, 1, pd + i0, 1, vDSP_Length(i1 - i0));
                }, [=](int i0, int i1) { tm_loop_acc(gd + i0, pd + i0, i1 - i0); });
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// ReLU elementwise
inline ValueP relu(ValueP A) {
    Tensor o = Tensor::uninit(A->t->shape());
    for (int i = 0; i < o.numel(); ++i) o[i] = std::max(0.f, (*A->t)[i]);
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n;
        n.inputs = {A};
        n.output = v;
        n.backward = [A](const Tensor& g) {
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            for (int i = 0; i < A->numel(); ++i)
                if ((*A->t)[i] > 0.f) (*A->grad)[i] += g[i];
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// MSE loss over equal-shape A, B: mean((A-B)^2). Returns scalar value.
inline ValueP mse(ValueP A, ValueP B) {
    Tensor out = Tensor::zeros({1});
    float acc = 0.f;
    for (int i = 0; i < A->numel(); ++i) {
        float d = (*A->t)[i] - (*B->t)[i];
        acc += d * d;
    }
    out[0] = acc / float(A->numel());
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && (A->requires_grad || B->requires_grad)) {
        Node n;
        n.inputs = {A, B};
        n.output = v;
        n.backward = [A, B](const Tensor& g) {
            const float s = 2.f * g[0] / float(A->numel());
            for (auto& p : {A, B}) {
                if (!p->requires_grad) continue;
                if (!p->grad) p->grad = std::make_shared<Tensor>(Tensor::zeros(p->shape()));
                float sign = (p == A) ? 1.f : -1.f;
                for (int i = 0; i < p->numel(); ++i)
                    (*p->grad)[i] += sign * s * ((*A->t)[i] - (*B->t)[i]);
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}


// ---------------------------------------------------------------------------
// Batch-2 ops: gelu, layernorm, softmax (rows), sum, mean, transpose, reshape.
// ---------------------------------------------------------------------------

// GELU (tanh approximation).
inline float gelu_f(float x) {
    return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}
inline float gelu_grad_f(float x) {
    const float c = 0.7978845608f;
    float x3 = x * x * x;
    float t = std::tanh(c * (x + 0.044715f * x3));
    float sech2 = 1.f - t * t;
    return 0.5f * (1.f + t) + 0.5f * x * sech2 * c * (1.f + 3.f * 0.044715f * x * x);
}
// GELU over a range, staging the tanh through vForce (a scalar std::tanh
// loop over a 4M-element MLP activation is the single most expensive
// elementwise op in a GPT block).
inline constexpr int TM_VBLK = 2048;   // 8 KB of floats: fits L1 comfortably
inline void tm_gelu_range(const float* x, float* out, int n) {
    auto& u = tm_scratch(TM_VBLK, 0);
    const float c = 0.7978845608f;
    for (int base = 0; base < n; base += TM_VBLK) {
        int m = std::min(TM_VBLK, n - base);
        const float* xb = x + base;
        for (int i = 0; i < m; ++i)
            u[i] = c * (xb[i] + 0.044715f * xb[i] * xb[i] * xb[i]);
        vvtanhf(u.data(), u.data(), &m);
        for (int i = 0; i < m; ++i) out[base + i] = 0.5f * xb[i] * (1.f + u[i]);
    }
}
inline ValueP gelu(ValueP A) {
    Tensor o = Tensor::uninit(A->t->shape());
    {
        const float* a = A->t->data();
        float* od = o.data();
        tm_par(o.numel(), [=](int i0, int i1) { tm_gelu_range(a + i0, od + i0, i1 - i0); });
    }
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A](const Tensor& g) {
            bool fresh; float* ad = tm_grad_fresh(A, fresh).data();
            const float* x = A->t->data();
            const float* gd = g.data();
            tm_par(A->numel(), [=](int lo, int hi) {
                auto& u = tm_scratch(TM_VBLK, 0);
                const float c = 0.7978845608f;
                for (int base = lo; base < hi; base += TM_VBLK) {
                    int m = std::min(TM_VBLK, hi - base);
                    for (int i = 0; i < m; ++i) {
                        const float xi = x[base + i];
                        u[i] = c * (xi + 0.044715f * xi * xi * xi);
                    }
                    vvtanhf(u.data(), u.data(), &m);
                    for (int i = 0; i < m; ++i) {
                        const float xi = x[base + i], t = u[i];
                        const float d = gd[base + i] *
                            (0.5f * (1.f + t) +
                             0.5f * xi * (1.f - t * t) * c *
                                 (1.f + 3.f * 0.044715f * xi * xi));
                        if (fresh) ad[base + i] = d; else ad[base + i] += d;
                    }
                }
            });
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// LayerNorm over the LAST dimension. gamma/beta are Values shaped (cols,).
//
// Backward is split into two parallel passes rather than one serial one:
// dx is row-parallel (each row is independent), dgamma/dbeta are
// column-parallel (each column reduces over all rows). Neither pass has a
// write conflict, so no per-thread partials and no locks.
inline ValueP layernorm(ValueP A, ValueP gamma, ValueP beta, float eps = 1e-5f) {
    const int cols = A->shape().back(), rows = A->numel() / cols;
    Tensor o = Tensor::uninit(A->t->shape());
    auto stats = std::make_shared<std::vector<float>>(std::size_t(rows) * 2);  // mean, rstd
    {
        const float* a = A->t->data();
        const float* gm = gamma->t->data();
        const float* bt = beta->t->data();
        float* od = o.data();
        float* st = stats->data();
        tm_par_rows(rows, cols, [=](int r0, int r1) {
            for (int r = r0; r < r1; ++r) {
                const float* row = a + std::size_t(r) * cols;
                float* orow = od + std::size_t(r) * cols;
                const float m = tm_row_sum(row, cols) / float(cols);
                const float var = tm_row_sumsq_dev(row, m, cols);
                const float rstd = 1.f / std::sqrt(var / float(cols) + eps);
                st[2 * r] = m; st[2 * r + 1] = rstd;
                for (int c = 0; c < cols; ++c)
                    orow[c] = (row[c] - m) * rstd * gm[c] + bt[c];
            }
        });
    }
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled &&
        (A->requires_grad || gamma->requires_grad || beta->requires_grad)) {
        Node n; n.inputs = {A, gamma, beta}; n.output = v;
        n.backward = [A, gamma, beta, stats, rows, cols](const Tensor& g) {
            const float* a = A->t->data();
            const float* gm = gamma->t->data();
            const float* gd = g.data();
            const float* st = stats->data();
            if (A->requires_grad) {
                bool fresh; float* ad = tm_grad_fresh(A, fresh).data();
                tm_par_rows(rows, cols, [=](int r0, int r1) {
                    for (int r = r0; r < r1; ++r) {
                        const float* row = a + std::size_t(r) * cols;
                        const float* gr = gd + std::size_t(r) * cols;
                        float* ar = ad + std::size_t(r) * cols;
                        const float m = st[2 * r], rstd = st[2 * r + 1];
                        // sums of dnorm and dnorm * xhat over the row
                        float s1, s2;
                        tm_row_ln_sums(gr, gm, row, m, rstd, cols, s1, s2);
                        const float inv = 1.f / float(cols);
                        if (fresh)
                            for (int c = 0; c < cols; ++c) {
                                const float dn = gr[c] * gm[c];
                                const float xh = (row[c] - m) * rstd;
                                ar[c] = rstd * (dn - inv * s1 - inv * s2 * xh);
                            }
                        else
                            for (int c = 0; c < cols; ++c) {
                                const float dn = gr[c] * gm[c];
                                const float xh = (row[c] - m) * rstd;
                                ar[c] += rstd * (dn - inv * s1 - inv * s2 * xh);
                            }
                    }
                });
            }
            if (gamma->requires_grad || beta->requires_grad) {
                // Row-major traversal into per-block partials, reduced after:
                // a column-parallel version reads with a `cols`-float stride
                // and misses every cache line it touches.
                const int nc = std::max(1, traink::nchunk());
                const int chunk = std::max(1, (rows + nc - 1) / nc);
                const int nblk = (rows + chunk - 1) / chunk;
                std::vector<float> part(std::size_t(nblk) * 2 * cols, 0.f);
                float* pd = part.data();
                traink::par_chunks(rows, chunk, [=](int r0, int r1) {
                    float* pg = pd + std::size_t(r0 / chunk) * 2 * cols;
                    float* pb = pg + cols;
                    for (int r = r0; r < r1; ++r) {
                        const float* gr = gd + std::size_t(r) * cols;
                        const float* row = a + std::size_t(r) * cols;
                        const float m = st[2 * r], rstd = st[2 * r + 1];
                        for (int c = 0; c < cols; ++c) {
                            pg[c] += gr[c] * (row[c] - m) * rstd;
                            pb[c] += gr[c];
                        }
                    }
                });
                float* dgm = gamma->requires_grad ? tm_grad(gamma).data() : nullptr;
                float* dbt = beta->requires_grad ? tm_grad(beta).data() : nullptr;
                for (int b = 0; b < nblk; ++b) {
                    const float* pg = pd + std::size_t(b) * 2 * cols;
                    if (dgm) vDSP_vadd(dgm, 1, pg, 1, dgm, 1, vDSP_Length(cols));
                    if (dbt) vDSP_vadd(dbt, 1, pg + cols, 1, dbt, 1, vDSP_Length(cols));
                }
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// Softmax over the LAST dimension (rows).
// Row-softmax of a (rows x cols) block: max-shift, vForce exp, normalise.
inline void tm_softmax_rows(const float* x, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* in = x + std::size_t(r) * cols;
        float* o = out + std::size_t(r) * cols;
        float m; vDSP_maxv(in, 1, &m, vDSP_Length(cols));
        const float negm = -m;
        vDSP_vsadd(in, 1, &negm, o, 1, vDSP_Length(cols));
        int n = cols; vvexpf(o, o, &n);
        float s; vDSP_sve(o, 1, &s, vDSP_Length(cols));
        const float inv = 1.f / s;
        vDSP_vsmul(o, 1, &inv, o, 1, vDSP_Length(cols));
    }
}
inline ValueP softmax(ValueP A) {
    const int rows = A->numel() / A->shape().back(), cols = A->shape().back();
    Tensor o = Tensor::uninit(A->t->shape());
    {
        const float* a = A->t->data(); float* od = o.data();
        tm_par_rows(rows, cols, [=](int r0, int r1) {
            tm_softmax_rows(a + std::size_t(r0) * cols, od + std::size_t(r0) * cols,
                            r1 - r0, cols);
        });
    }
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        auto out_copy = v->t;   // shared_ptr: output stays alive with the node
        n.backward = [A, out_copy, rows, cols](const Tensor& g) {
            float* ad = tm_grad(A).data();
            const float* y = out_copy->data();
            const float* gd = g.data();
            tm_par_rows(rows, cols, [=](int r0, int r1) {
                for (int r = r0; r < r1; ++r) {
                    const float* yr = y + std::size_t(r) * cols;
                    const float* gr = gd + std::size_t(r) * cols;
                    float* ar = ad + std::size_t(r) * cols;
                    float dot; vDSP_dotpr(gr, 1, yr, 1, &dot, vDSP_Length(cols));
                    for (int c = 0; c < cols; ++c) ar[c] += yr[c] * (gr[c] - dot);
                }
            });
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// Sum of n floats: vDSP_sve per chunk (parallel above the bandwidth cutoff,
// chunk-major combine), replacing a serial scalar accumulator chain that ran
// at ~1 GB/s on 8 MB tensors.
inline float tm_sum_range(const float* a, int n) {
    if (n < tm_bw_par_threshold()) {
        float s = 0.f; vDSP_sve(a, 1, &s, vDSP_Length(n)); return s;
    }
    const int nc = std::max(1, traink::nchunk());
    const int chunk = (n + nc - 1) / nc;
    std::vector<float> part(nc, 0.f);
    float* pp = part.data();
    traink::par_chunks(n, chunk, [=](int i0, int i1) {
        vDSP_sve(a + i0, 1, pp + i0 / chunk, vDSP_Length(i1 - i0));
    });
    double s = 0.0;
    for (float x : part) s += x;
    return float(s);
}

// sum / mean over ALL elements -> scalar value (shared reduce impl)
inline ValueP _reduce_scalar(ValueP A, Tensor out, float scale) {
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A, scale](const Tensor& g) {
            float* ad = tm_grad(A).data();
            const float gs = g[0] * scale;
            tm_elem_stream(A->numel(), [=](int i0, int i1) {
                vDSP_vsadd(ad + i0, 1, &gs, ad + i0, 1, vDSP_Length(i1 - i0));
            }, [=](int i0, int i1) { tm_loop_sadd_acc(gs, ad + i0, i1 - i0); });
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}
inline ValueP sum(ValueP A) {
    Tensor out = Tensor::zeros({1});
    out[0] = tm_sum_range(A->t->data(), A->numel());
    return _reduce_scalar(A, std::move(out), 1.f);
}
inline ValueP mean(ValueP A) {
    Tensor out = Tensor::zeros({1});
    out[0] = tm_sum_range(A->t->data(), A->numel()) / float(A->numel());
    return _reduce_scalar(A, std::move(out), 1.f / A->numel());
}

// transpose (rows x cols) -> (cols x rows), 2-D only
inline ValueP transpose2d(ValueP A) {
    const int R = A->shape()[0], C = A->shape()[1];
    Tensor o = Tensor::uninit({C, R});
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            o[c * R + r] = (*A->t)[r * C + c];
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A, R, C](const Tensor& g) {
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c)
                    (*A->grad)[r * C + c] += g[c * R + r];
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// reshape (same numel)
inline ValueP reshape(ValueP A, std::vector<int> shape) {
    Tensor o = Tensor::uninit(std::move(shape));
    for (int i = 0; i < o.numel(); ++i) o[i] = (*A->t)[i];
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A](const Tensor& g) {
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            for (int i = 0; i < g.numel(); ++i) (*A->grad)[i] += g[i];
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Batch-3 ops: mul, scale, concat, slice, cross_entropy, gather (embedding).
// Conventions: 2-D tensors are row-major (rows, cols); softmax/CE are
// row-wise (rows = samples, cols = classes); gather treats rows of E as
// embedding vectors.
// ---------------------------------------------------------------------------

// elementwise product, same shapes
inline ValueP mul(ValueP A, ValueP B) {
    if (A->shape() != B->shape())
        throw std::invalid_argument("mul: shape mismatch");
    Tensor out = Tensor::uninit(A->shape());
    {
        const float* a = A->t->data(); const float* b = B->t->data(); float* o = out.data();
        tm_elem_stream(out.numel(), [=](int i0, int i1) {
            vDSP_vmul(a + i0, 1, b + i0, 1, o + i0, 1, vDSP_Length(i1 - i0));
        }, [=](int i0, int i1) { tm_loop_mul(a + i0, b + i0, o + i0, i1 - i0); });
    }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && (A->requires_grad || B->requires_grad)) {
        Node n; n.inputs = {A, B}; n.output = v;
        n.backward = [A, B](const Tensor& g) {
            // d(a*b)/da = g*b, d(a*b)/db = g*a; only inputs that want a grad.
            for (auto [p, other] : {std::pair{A, B}, std::pair{B, A}}) {
                if (!p->requires_grad) continue;
                float* pd = tm_grad(p).data();
                const float* od = other->t->data();
                const float* gd = g.data();
                tm_elem_stream(g.numel(), [=](int i0, int i1) {
                    vDSP_vma(gd + i0, 1, od + i0, 1, pd + i0, 1, pd + i0, 1, vDSP_Length(i1 - i0));
                }, [=](int i0, int i1) { tm_loop_fma_acc(gd + i0, od + i0, pd + i0, i1 - i0); });
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// multiply by a scalar constant
inline ValueP scale(ValueP A, float s) {
    Tensor out = Tensor::uninit(A->shape());
    {
        const float* a = A->t->data(); float* o = out.data();
        tm_elem_stream(out.numel(), [=](int i0, int i1) {
            vDSP_vsmul(a + i0, 1, &s, o + i0, 1, vDSP_Length(i1 - i0));
        }, [=](int i0, int i1) { tm_loop_scale(a + i0, s, o + i0, i1 - i0); });
    }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A, s](const Tensor& g) {
            float* ad = tm_grad(A).data();
            const float* gd = g.data();
            tm_elem_stream(A->numel(), [=](int i0, int i1) {
                vDSP_vsma(gd + i0, 1, &s, ad + i0, 1, ad + i0, 1, vDSP_Length(i1 - i0));
            }, [=](int i0, int i1) { tm_loop_axpy_acc(gd + i0, s, ad + i0, i1 - i0); });
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// concat along columns (last dim), 2-D. Rows must match.
inline ValueP concat(ValueP A, ValueP B) {
    int rows = A->shape()[0], ca = A->shape()[1], cb = B->shape()[1];
    Tensor out = Tensor::uninit({rows, ca + cb});
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < ca; ++c) out[r * (ca + cb) + c] = (*A->t)[r * ca + c];
        for (int c = 0; c < cb; ++c) out[r * (ca + cb) + ca + c] = (*B->t)[r * cb + c];
    }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled) {
        Node n; n.inputs = {A, B}; n.output = v;
        n.backward = [A, B, rows, ca, cb](const Tensor& g) {
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            if (!B->grad) B->grad = std::make_shared<Tensor>(Tensor::zeros(B->shape()));
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < ca; ++c) (*A->grad)[r * ca + c] += g[r * (ca + cb) + c];
                for (int c = 0; c < cb; ++c) (*B->grad)[r * cb + c] += g[r * (ca + cb) + ca + c];
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// 2-D slice: rows [r0,r1), cols [c0,c1)
inline ValueP slice(ValueP A, int r0, int r1, int c0, int c1) {
    const int rows = r1 - r0, cols = c1 - c0, W = A->shape()[1];
    Tensor out = Tensor::uninit({rows, cols});
    {
        const float* a = A->t->data(); float* o = out.data();
        tm_par_rows(rows, cols, [=](int i0, int i1) {
            for (int r = i0; r < i1; ++r)
                std::memcpy(o + std::size_t(r) * cols,
                            a + std::size_t(r0 + r) * W + c0,
                            std::size_t(cols) * sizeof(float));
        });
    }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A, r0, r1, c0, c1](const Tensor& g) {
            const int rows = r1 - r0, cols = c1 - c0, W = A->shape()[1];
            float* ad = tm_grad(A).data();
            const float* gd = g.data();
            tm_par_rows(rows, cols, [=](int i0, int i1) {
                for (int r = i0; r < i1; ++r) {
                    float* dst = ad + std::size_t(r0 + r) * W + c0;
                    vDSP_vadd(dst, 1, gd + std::size_t(r) * cols, 1, dst, 1,
                              vDSP_Length(cols));
                }
            });
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// cross-entropy with fused softmax, row-wise: logits (B, C), labels int per row.
// loss = -mean_valid log p[b, y[b]]. Rows whose label equals ignore_index
// (SFT loss masks) contribute nothing to the loss or the gradient; the
// mean divides by the VALID row count. Backward: dlogits = (p - onehot)/B.
inline ValueP cross_entropy(ValueP logits, const std::vector<int>& labels,
                            int ignore_index = -1) {
    int B = logits->shape()[0], C = logits->shape()[1];
    if (int(labels.size()) != B)
        throw std::invalid_argument("cross_entropy: one label per row required");
    auto p = std::make_shared<Tensor>(Tensor::uninit(logits->shape()));
    {
        const float* x = logits->t->data(); float* pd = p->data();
        tm_par_rows(B, C, [=](int r0, int r1) {
            tm_softmax_rows(x + std::size_t(r0) * C, pd + std::size_t(r0) * C, r1 - r0, C);
        });
    }
    int valid = 0;
    float loss = 0.f;
    for (int b = 0; b < B; ++b) {
        if (labels[b] == ignore_index) continue;
        ++valid;
        loss -= std::log(std::max(1e-30f, (*p)[b * C + labels[b]]));
    }
    Tensor out = Tensor::zeros({1});
    out[0] = valid ? loss / float(valid) : 0.f;
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && logits->requires_grad) {
        Node n; n.inputs = {logits}; n.output = v;
        n.backward = [logits, p, labels, B, C, valid, ignore_index](const Tensor& g) {
            if (!valid) return;
            float* ld = tm_grad(logits).data();
            const float* pd = p->data();
            const float s = g[0] / float(valid);
            const int* lab = labels.data();
            tm_par_rows(B, C, [=](int r0, int r1) {
                for (int b = r0; b < r1; ++b) {
                    if (lab[b] == ignore_index) continue;
                    const float* pr = pd + std::size_t(b) * C;
                    float* lr = ld + std::size_t(b) * C;
                    for (int c = 0; c < C; ++c) lr[c] += s * pr[c];
                    lr[lab[b]] -= s;
                }
            });
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// embedding lookup: E (V, D), idx int per row -> out (B, D) rows of E
inline ValueP gather(ValueP E, const std::vector<int>& idx) {
    const int D = E->shape()[1], B = int(idx.size());
    Tensor out = Tensor::uninit({B, D});
    {
        const float* e = E->t->data(); float* o = out.data(); const int* ix = idx.data();
        tm_par_rows_bw(B, D, [=](int r0, int r1) {
            for (int b = r0; b < r1; ++b)
                std::memcpy(o + std::size_t(b) * D, e + std::size_t(ix[b]) * D,
                            std::size_t(D) * sizeof(float));
        });
    }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && E->requires_grad) {
        Node n; n.inputs = {E}; n.output = v;
        // Rows repeat (that is the point of an embedding table), so the
        // scatter-add stays serial: rows of E are the contended resource.
        n.backward = [E, idx, B, D](const Tensor& g) {
            float* ed = tm_grad(E).data();
            const float* gd = g.data();
            for (int b = 0; b < B; ++b) {
                float* dst = ed + std::size_t(idx[b]) * D;
                vDSP_vadd(dst, 1, gd + std::size_t(b) * D, 1, dst, 1, vDSP_Length(D));
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// ---------------------------------------------------------------------------
// conv ops (NCHW, OIHW weights; input pre-padded by the caller).
// tm_im2col/tm_col2im: stride-aware (engine im2col is stride-1 only).
// ---------------------------------------------------------------------------
inline void tm_im2col(const float* x, float* col, int B, int C, int H, int W,
                      int kH, int kW, int oH, int oW, int s) {
    const int K = C * kH * kW;
    // Each output row of `col` is written by exactly one (n,oy,ox) triple,
    // so we can parallelize over rows, not just samples — this matters at
    // inference where B == 1 and R = oH*oW is the only parallel axis.
    const int R = B * oH * oW;
    const int chunk = std::max(64, R / (4 * std::max(1, traink::nchunk())));
    traink::par_chunks(R, chunk, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const int n = r / (oH * oW), rem = r % (oH * oW);
            const int oy = rem / oW, ox = rem % oW;
            float* row = col + size_t(r) * K;
            int k = 0;
            for (int c = 0; c < C; ++c)
                for (int ky = 0; ky < kH; ++ky)
                    for (int kx = 0; kx < kW; ++kx)
                        row[k++] = x[((n * C + c) * H + oy * s + ky) * W + ox * s + kx];
        }
    });
}
inline void tm_col2im(const float* col, float* x, int B, int C, int H, int W,
                      int kH, int kW, int oH, int oW, int s) {
    const int K = C * kH * kW;
    traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            for (int oy = 0; oy < oH; ++oy)
                for (int ox = 0; ox < oW; ++ox) {
                    const float* row = col + ((n * oH + oy) * oW + ox) * K;
                    int k = 0;
                    for (int c = 0; c < C; ++c)
                        for (int ky = 0; ky < kH; ++ky)
                            for (int kx = 0; kx < kW; ++kx)
                                x[((n * C + c) * H + oy * s + ky) * W + ox * s + kx] += row[k++];
                }
    });
}
inline void _tm_pack_w(const Tensor& Wt, Tensor& Wm, int O, int C, int kH, int kW, int K) {
    for (int o = 0; o < O; ++o)
        for (int c = 0; c < C; ++c)
            for (int ky = 0; ky < kH; ++ky)
                for (int kx = 0; kx < kW; ++kx)
                    Wm[o * K + c * kH * kW + ky * kW + kx] = Wt[((o * C + c) * kH + ky) * kW + kx];
}
// Packed-weight cache: packing W (O,C,kH,kW) -> (O,K) every call wastes
// bandwidth at inference where weights never change. Keyed by weight
// pointer + shape; training code MUST call tm_clear_wcache() after
// mutating a weight's data (no tape-side optimizer mutates weights today).
struct _tm_WKey {
    const void* p; int O, C, kH, kW;
    bool operator==(const _tm_WKey& o) const {
        return p == o.p && O == o.O && C == o.C && kH == o.kH && kW == o.kW;
    }
};
struct _tm_WKeyHash {
    size_t operator()(const _tm_WKey& k) const {
        size_t h = std::hash<const void*>()(k.p);
        h ^= std::hash<int>()(k.O) << 1; h ^= std::hash<int>()(k.C) << 2;
        h ^= std::hash<int>()(k.kH) << 3; h ^= std::hash<int>()(k.kW) << 4;
        return h;
    }
};
inline std::unordered_map<_tm_WKey, Tensor, _tm_WKeyHash>& _tm_wcache() {
    static thread_local std::unordered_map<_tm_WKey, Tensor, _tm_WKeyHash> cache;
    return cache;
}
inline void tm_clear_wcache() { _tm_wcache().clear(); }
// ---- tunable kernel knobs (autotune "wisdom" targets) ---------------------
// tm_set_param adjusts execution thresholds. Changes to reduction partitioning
// can change floating-point accumulation order; validate numerical equivalence
// with workload-appropriate tolerances rather than requiring bit identity.
inline std::uint64_t& tm_gemm_par_threshold() {  // FLOPs below this run single-threaded
    static std::uint64_t v = 1u << 24; return v;
}
inline int& tm_dw_chunks_mult() {  // depthwise plane task multiplier (chunks = B*C*mult)
    static int v = 4; return v;
}
// Knob storage lives here (autograd.h is included by attention.h and
// kda.h, never the reverse), so the accessors are always defined where
// they are used — no undefined-inline hazards.
inline int& tm_attn_tile() { static int v = 64; return v; }   // attention tile edge
inline int& tmg_kda_chunk() { static int v = 64; return v; }  // KDA recurrent chunk
inline void tm_set_param(const std::string& name, double val) {
    if (name == "gemm_par_threshold") tm_gemm_par_threshold() = std::uint64_t(val);
    else if (name == "dw_chunks_mult") tm_dw_chunks_mult() = std::max(1, int(val));
    else if (name == "elem_par_threshold") tm_elem_par_threshold() = std::max(1, int(val));
    else if (name == "bw_par_threshold") tm_bw_par_threshold() = std::max(1, int(val));
    else if (name == "stream_threshold") tm_stream_threshold() = std::max(1, int(val));
    else if (name == "pool_spin") traink::pool().spin_budget.store(std::clamp(int(val), 0, 1 << 24));
    else if (name == "gemm_split_m") traink::gemm_split_threshold_m() = std::uint64_t(val);
    else if (name == "gemm_split_n") traink::gemm_split_threshold_n() = std::uint64_t(val);
    else if (name == "attn_tile") tm_attn_tile() = std::max(1, int(val));
    else if (name == "kda_chunk") tmg_kda_chunk() = std::max(1, int(val));
    else if (name == "buffer_pool") {
        buffer_pool().enabled = val != 0.0;
        if (val == 0.0) buffer_pool().clear();
    }
    else throw std::invalid_argument("unknown tensormark param: " + name);
}

// GEMM NT parallelized over the larger of M/N: traink::psgemm_nt only
// splits N, which serializes conv1-style layers (O=32, R=16k rows).
inline void tm_gemm_nt(int M, int N, int K, const float* A, const float* Bp,
                       float* C, int ldc) {
    if (std::uint64_t(M) * N * K < tm_gemm_par_threshold()) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f,
                    A, K, Bp, K, 0.0f, C, ldc);
        return;
    }
    int nc = traink::nchunk();
    if (N >= M) { traink::psgemm_nt(M, N, K, A, Bp, C, 1.0f, 0.0f, ldc); return; }
    if (M < nc * 2) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f,
                    A, K, Bp, K, 0.0f, C, ldc);
        return;
    }
    traink::pool().run([&](int b) {
        int m0 = M * b / nc, m1 = M * (b + 1) / nc;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m1 - m0, N, K,
                    1.0f, A + std::size_t(m0) * K, K, Bp, K, 0.0f,
                    C + std::size_t(m0) * ldc, ldc);
    }, nc);
}
inline ValueP conv2d(ValueP A, ValueP W, int stride = 1) {
    int B = A->shape()[0], C = A->shape()[1], H = A->shape()[2], Wd = A->shape()[3];
    // 1x1 stride-1 fast path: NCHW (C, HW) IS the im2col matrix, and the GEMM
    // output (O, HW) IS NCHW — no im2col, no permute. One cblas_sgemm.
    if (W->shape()[2] == 1 && W->shape()[3] == 1 && stride == 1 && B == 1) {
        int O = W->shape()[0], HW = H * Wd;
        _tm_WKey key{W->t.get(), O, C, 1, 1};
        auto it = _tm_wcache().find(key);
        if (it == _tm_wcache().end())
            it = _tm_wcache().emplace(key, Tensor::zeros({O, C})).first;
        Tensor& Wm = it->second;
        _tm_pack_w(*W->t, Wm, O, C, 1, 1, C);
        Tensor out = Tensor::zeros({1, O, H, Wd});
        const float* X = A->t->data();
        float* Y = out.data();
        auto gemm_range = [&](int n0, int n1) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, O, n1 - n0, C,
                        1.0f, Wm.data(), C, X + n0, HW, 0.0f, Y + n0, HW);
        };
        if (std::uint64_t(O) * HW * C < tm_gemm_par_threshold()) {
            gemm_range(0, HW);
        } else {
            int nc = std::min(traink::nchunk(), std::max(1, HW / 256));
            traink::pool().run([&](int b) {
                gemm_range(HW * b / nc, HW * (b + 1) / nc);
            }, nc);
        }
        auto v = std::make_shared<Value>(std::move(out), true);
        if (current_tape().enabled && (A->requires_grad || W->requires_grad)) {
            Node n; n.inputs = {A, W}; n.output = v;
            n.backward = [A, W, O, C, HW, Wm](const Tensor& g) {
                if (W->requires_grad) {
                    if (!W->grad) W->grad = std::make_shared<Tensor>(Tensor::zeros(W->shape()));
                    // dW(O,C) = g(O,HW) @ X(C,HW)^T
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, O, C, HW,
                                1.0f, g.data(), HW, A->t->data(), HW, 1.0f, W->grad->data(), C);
                }
                if (A->requires_grad) {
                    if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
                    // dX(C,HW) = Wm(O,C)^T @ g(O,HW)
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, C, HW, O,
                                1.0f, Wm.data(), C, g.data(), HW, 1.0f, A->grad->data(), HW);
                }
            };
            v->node = current_tape().push(std::move(n));
        }
        return v;
    }
    int O = W->shape()[0], kH = W->shape()[2], kW = W->shape()[3];
    int oH = (H - kH) / stride + 1, oW = (Wd - kW) / stride + 1, K = C * kH * kW;
    int R = B * oH * oW;
    // Pack once, reuse across calls (see _tm_wcache note above).
    _tm_WKey key{W->t.get(), O, C, kH, kW};
    auto it = _tm_wcache().find(key);
    if (it == _tm_wcache().end())
        it = _tm_wcache().emplace(key, Tensor::zeros({O, K})).first;
    Tensor& Wm = it->second;
    _tm_pack_w(*W->t, Wm, O, C, kH, kW, K);  // cheap; cache saves the alloc
    Tensor col = Tensor::zeros({R, K});
    tm_im2col(A->t->data(), col.data(), B, C, H, Wd, kH, kW, oH, oW, stride);
    // psgemm emits (R, O) rows = (n,oy,ox) — NOT NCHW; permute into NCHW.
    Tensor outRO = Tensor::zeros({R, O});
    tm_gemm_nt(R, O, K, col.data(), Wm.data(), outRO.data(), O);
    Tensor out4 = Tensor::zeros({B, O, oH, oW});
    traink::par_chunks(B * O, std::max(1, B * O / (2 * std::max(1, traink::nchunk()))),
                       [&](int p0, int p1) {
        for (int p = p0; p < p1; ++p) {
            const int n = p / O, o = p % O;
            for (int y = 0; y < oH; ++y)
                for (int x = 0; x < oW; ++x)
                    out4[((n * O + o) * oH + y) * oW + x] =
                        outRO[((n * oH + y) * oW + x) * O + o];
        }
    });
    auto v = std::make_shared<Value>(std::move(out4), true);
    if (current_tape().enabled && (A->requires_grad || W->requires_grad)) {
        Node n; n.inputs = {A, W}; n.output = v;
        n.backward = [A, W, B, C, H, Wd, O, kH, kW, oH, oW, stride, K, R, Wm](const Tensor& g) {
            Tensor col = Tensor::zeros({R, K});
            tm_im2col(A->t->data(), col.data(), B, C, H, Wd, kH, kW, oH, oW, stride);
            Tensor g2 = Tensor::zeros({R, O});
            std::memcpy(g2.data(), g.data(), size_t(R) * O * sizeof(float));
            if (W->requires_grad) {
                if (!W->grad) W->grad = std::make_shared<Tensor>(Tensor::zeros(W->shape()));
                Tensor dWm = Tensor::zeros({O, K});
                traink::psgemm_tn(O, K, R, g2.data(), col.data(), dWm.data(), 1.0f);
                for (int o = 0; o < O; ++o)
                    for (int k = 0; k < K; ++k) (*W->grad)[o * K + k] = dWm[o * K + k];
            }
            if (A->requires_grad) {
                if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
                Tensor dcol = Tensor::zeros({R, K});
                traink::psgemm_nn_m(R, K, O, g2.data(), Wm.data(), dcol.data(), 1.0f);
                tm_col2im(dcol.data(), A->grad->data(), B, C, H, Wd, kH, kW, oH, oW, stride);
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}
inline ValueP depthwise_conv2d(ValueP A, ValueP W, int stride = 1) {
    int B = A->shape()[0], C = A->shape()[1], H = A->shape()[2], Wd = A->shape()[3];
    int kH = W->shape()[1], kW = W->shape()[2];
    int oH = (H - kH) / stride + 1, oW = (Wd - kW) / stride + 1;
    Tensor out = Tensor::zeros({B, C, oH, oW});
    const Tensor& At = *A->t; const Tensor& Wt = *W->t;
    // Parallelize over (n,c) planes, not just samples: at B==1 this is the
    // only parallel axis (16 BlazeFace dw layers were fully serial before).
    // The 3x3 s==1 fast path is a contiguous dot product per output row,
    // which clang vectorizes; the generic path keeps the ky/kx gather.
    traink::par_chunks(B * C, std::max(1, B * C / std::max(1, traink::nchunk() * tm_dw_chunks_mult())),
                       [&](int p0, int p1) {
        for (int p = p0; p < p1; ++p) {
            const int n = p / C, c = p % C;
            const float* xc = At.data() + (size_t(n) * C + c) * H * Wd;
            float* oc = out.data() + (size_t(n) * C + c) * oH * oW;
            const float* wc = Wt.data() + (size_t)c * kH * kW;
            if (kH == 3 && kW == 3 && stride == 1) {
                #ifdef TM_HAVE_NEON
                    // 1-row NEON: shifted unaligned loads, FMA chains. A 2-row
                    // variant sharing input-row loads measured SLOWER (dw
                    // 2.95-3.23 ms vs 2.62 over two runs): register pressure.
                    const float32x4_t w0 = vdupq_n_f32(wc[0]), w1 = vdupq_n_f32(wc[1]),
                                      w2 = vdupq_n_f32(wc[2]), w3 = vdupq_n_f32(wc[3]),
                                      w4 = vdupq_n_f32(wc[4]), w5 = vdupq_n_f32(wc[5]),
                                      w6 = vdupq_n_f32(wc[6]), w7 = vdupq_n_f32(wc[7]),
                                      w8 = vdupq_n_f32(wc[8]);
                    for (int oy = 0; oy < oH; ++oy) {
                        const float* x0 = xc + (size_t)oy * Wd;
                        const float* x1 = x0 + Wd;
                        const float* x2 = x1 + Wd;
                        float* orow = oc + (size_t)oy * oW;
                        int ox = 0;
                        for (; ox + 4 <= oW; ox += 4) {
                            float32x4_t s = vmulq_f32(vld1q_f32(x0 + ox), w0);
                            s = vmlaq_f32(s, vld1q_f32(x0 + ox + 1), w1);
                            s = vmlaq_f32(s, vld1q_f32(x0 + ox + 2), w2);
                            s = vmlaq_f32(s, vld1q_f32(x1 + ox), w3);
                            s = vmlaq_f32(s, vld1q_f32(x1 + ox + 1), w4);
                            s = vmlaq_f32(s, vld1q_f32(x1 + ox + 2), w5);
                            s = vmlaq_f32(s, vld1q_f32(x2 + ox), w6);
                            s = vmlaq_f32(s, vld1q_f32(x2 + ox + 1), w7);
                            s = vmlaq_f32(s, vld1q_f32(x2 + ox + 2), w8);
                            vst1q_f32(orow + ox, s);
                        }
                        for (; ox < oW; ++ox)
                            orow[ox] = x0[ox] * wc[0] + x0[ox + 1] * wc[1] + x0[ox + 2] * wc[2]
                                     + x1[ox] * wc[3] + x1[ox + 1] * wc[4] + x1[ox + 2] * wc[5]
                                     + x2[ox] * wc[6] + x2[ox + 1] * wc[7] + x2[ox + 2] * wc[8];
                    }
#else
                    for (int oy = 0; oy < oH; ++oy) {
                        const float* x0 = xc + (size_t)oy * Wd;
                        const float* x1 = x0 + Wd;
                        const float* x2 = x1 + Wd;
                        float* orow = oc + (size_t)oy * oW;
                        for (int ox = 0; ox < oW; ++ox)
                            orow[ox] = x0[ox] * wc[0] + x0[ox + 1] * wc[1] + x0[ox + 2] * wc[2]
                                     + x1[ox] * wc[3] + x1[ox + 1] * wc[4] + x1[ox + 2] * wc[5]
                                     + x2[ox] * wc[6] + x2[ox + 1] * wc[7] + x2[ox + 2] * wc[8];
                    }
#endif
            } else {
                for (int oy = 0; oy < oH; ++oy)
                    for (int ox = 0; ox < oW; ++ox) {
                        float acc = 0.f;
                        for (int ky = 0; ky < kH; ++ky)
                            for (int kx = 0; kx < kW; ++kx)
                                acc += xc[((size_t)(oy * stride + ky)) * Wd + ox * stride + kx]
                                     * wc[ky * kW + kx];
                        oc[oy * oW + ox] = acc;
                    }
            }
        }
    });
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && (A->requires_grad || W->requires_grad)) {
        Node n; n.inputs = {A, W}; n.output = v;
        n.backward = [A, W, B, C, H, Wd, kH, kW, oH, oW, stride](const Tensor& g) {
            if (W->requires_grad) {
                if (!W->grad) W->grad = std::make_shared<Tensor>(Tensor::zeros(W->shape()));
                for (int n = 0; n < B; ++n)
                    for (int c = 0; c < C; ++c)
                        for (int oy = 0; oy < oH; ++oy)
                            for (int ox = 0; ox < oW; ++ox) {
                                float gv = g[((n * C + c) * oH + oy) * oW + ox];
                                for (int ky = 0; ky < kH; ++ky)
                                    for (int kx = 0; kx < kW; ++kx)
                                        (*W->grad)[(c * kH + ky) * kW + kx] +=
                                            gv * (*A->t)[((n * C + c) * H + oy * stride + ky) * Wd + ox * stride + kx];
                            }
            }
            if (A->requires_grad) {
                if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
                for (int n = 0; n < B; ++n)
                    for (int c = 0; c < C; ++c)
                        for (int oy = 0; oy < oH; ++oy)
                            for (int ox = 0; ox < oW; ++ox) {
                                float gv = g[((n * C + c) * oH + oy) * oW + ox];
                                for (int ky = 0; ky < kH; ++ky)
                                    for (int kx = 0; kx < kW; ++kx)
                                        (*A->grad)[((n * C + c) * H + oy * stride + ky) * Wd + ox * stride + kx] +=
                                            gv * (*W->t)[(c * kH + ky) * kW + kx];
                            }
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}
// ---------------------------------------------------------------------------
// Fused inference conv ops: symmetric pad + conv + bias + ReLU in one kernel.
// pad ∈ {0,1} (BlazeFace's MIRROR_PAD 1,1); bias may be null; relu optional.
// Inference only: when the tape is enabled, inputs must not require gradients.
// For training, explicitly compose padding, convolution, bias and ReLU ops.
// ---------------------------------------------------------------------------
// Forward declaration (defined below).
inline ValueP pad_symmetric(ValueP A);
// zero-pad NCHW (A) into a pre-zeroed `out` (pt,pb / pl,pr on H,W)
    inline void _tm_pad_zero_into(const float* x, float* o, int C, int H, int W,
                              int Hp, int Wp, int pt, int pl) {
        for (int c = 0; c < C; ++c) {
        const float* xc = x + size_t(c) * H * W;
        float* oc = o + size_t(c) * Hp * Wp;
                       for (int y = 0; y < H; ++y)
            std::memcpy(oc + size_t(y + pt) * Wp + pl, xc + size_t(y) * W,
                        sizeof(float) * W);
            }
            }
inline void _tm_bias_relu_epilogue(float* out, int total, int oHW, int O,
                                   const float* bias, bool use_relu) {
    if (!bias && !use_relu) return;
    // per-plane bias (no per-element integer division — that alone was
    // measurable at BlazeFace sizes); vectorizable inner loop
    const int planes = total / oHW;
    auto body = [&](int pl0, int pl1) {
        for (int pl = pl0; pl < pl1; ++pl) {
            const float bv = bias ? bias[pl % O] : 0.f;
            float* q = out + size_t(pl) * oHW;
            #pragma clang loop vectorize(enable)
            for (int i = 0; i < oHW; ++i) {
                float v = q[i] + bv;
                q[i] = (use_relu && v < 0.f) ? 0.f : v;
            }
        }
    };
    if (planes < 8) { body(0, planes); return; }
    traink::par_chunks(planes, std::max(1, planes / std::max(1, traink::nchunk())),
                       body);
}
inline ValueP _tm_zeropad_ex(ValueP A, int pt, int pb, int pl, int pr) {
    int B = A->shape()[0], C = A->shape()[1], H = A->shape()[2], W = A->shape()[3];
    int Hp = H + pt + pb, Wp = W + pl + pr;
    Tensor out = Tensor::zeros({B, C, Hp, Wp});
    traink::par_chunks(B * C, std::max(1, B * C / std::max(1, traink::nchunk() * tm_dw_chunks_mult())),
                       [&](int c0, int c1) {
        for (int pc = c0; pc < c1; ++pc) {
            const int n = pc / C;
            _tm_pad_zero_into(A->t->data() + size_t(n) * C * H * W,
                              out.data() + size_t(n) * C * Hp * Wp,
                              C, H, W, Hp, Wp, pt, pl);
        }
    });
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A, B, C, H, W, Hp, Wp, pt, pl](const Tensor& g) {
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            for (int n = 0; n < B; ++n)
                for (int c = 0; c < C; ++c)
                    for (int y = 0; y < H; ++y)
                        std::memcpy(&(*A->grad)[((n * C + c) * H + y) * W],
                                    g.data() + ((n * C + c) * Hp + y + pt) * Wp + pl,
                                    sizeof(float) * W);
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}
// broadcast channel bias (+ optional relu) as a tape op
inline ValueP _tm_bias_ex(ValueP y, int O, const float* bias) {
    int oHW = y->numel() / (y->shape()[0] * O);
    Tensor out = Tensor::uninit(y->t->shape());
    for (int i = 0; i < out.numel(); ++i)
        out[i] = (*y->t)[i] + bias[(i / oHW) % O];
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled) {
        Node n; n.inputs = {y}; n.output = v;
        n.backward = [y](const Tensor& g) {
            if (!y->requires_grad) return;
            if (!y->grad) y->grad = std::make_shared<Tensor>(Tensor::zeros(y->shape()));
            for (int i = 0; i < g.numel(); ++i) (*y->grad)[i] += g[i];
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}
inline ValueP conv2d_ex(ValueP A, ValueP W, int stride,
                        int pt, int pb, int pl, int pr,
                        const float* bias, bool use_relu) {
    const bool anypad = pt || pb || pl || pr;
    // inference-only: the fused epilogue writes the output in place, so
        // there is no backward. Training uses conv2d + explicit pad/bias/relu.
        assert(!(current_tape().enabled && (A->requires_grad || W->requires_grad)) &&
           "conv2d_ex is inference-only; use pad/conv2d/bias/relu for training");
    // 1x1 stride-1 B=1: delegate to conv2d's no-im2col GEMM fast path,
    // then apply the bias/relu epilogue in place.
    if (W->shape()[2] == 1 && W->shape()[3] == 1 && stride == 1 &&
        A->shape()[0] == 1 && !anypad) {
        ValueP y = conv2d(A, W, 1);
        int O1 = y->shape()[1], oHW1 = y->numel() / O1;
        _tm_bias_relu_epilogue(y->t->data(), y->numel(), oHW1, O1, bias, use_relu);
        return y;
    }
    int B = A->shape()[0], C = A->shape()[1], H = A->shape()[2], Wd = A->shape()[3];
    int O = W->shape()[0], kH = W->shape()[2], kW = W->shape()[3];
    int Hp = H + pt + pb, Wp = Wd + pl + pr;
    int oH = (Hp - kH) / stride + 1, oW = (Wp - kW) / stride + 1, K = C * kH * kW;
    int R = B * oH * oW;
    _tm_WKey key{W->t.get(), O, C, kH, kW};
    auto it = _tm_wcache().find(key);
    if (it == _tm_wcache().end())
        it = _tm_wcache().emplace(key, Tensor::zeros({O, K})).first;
    Tensor& Wm = it->second;
    _tm_pack_w(*W->t, Wm, O, C, kH, kW, K);
    Tensor Xin; const float* xp = A->t->data();
    int Hs = H, Ws = Wd;
    if (anypad) {
        Xin = Tensor::zeros({B, C, Hp, Wp});
        for (int n = 0; n < B; ++n)
            _tm_pad_zero_into(A->t->data() + size_t(n) * C * H * Wd,
                              Xin.data() + size_t(n) * C * Hp * Wp,
                              C, H, Wd, Hp, Wp, pt, pl);
        xp = Xin.data(); Hs = Hp; Ws = Wp;
    }
    Tensor col = Tensor::zeros({R, K});
    tm_im2col(xp, col.data(), B, C, Hs, Ws, kH, kW, oH, oW, stride);
    Tensor out = Tensor::zeros({B, O, oH, oW});
    float* optr = out.data();
    if (B == 1) {
        // (O, oH*oW) IS NCHW for B==1: GEMM writes the output layout
        // directly, no (R,O)->NCHW permute pass.
        tm_gemm_nt(O, R, K, Wm.data(), col.data(), optr, R);
    } else {
        Tensor outRO = Tensor::zeros({R, O});
        tm_gemm_nt(R, O, K, col.data(), Wm.data(), outRO.data(), O);
        traink::par_chunks(B * O, std::max(1, B * O / (2 * std::max(1, traink::nchunk()))),
                           [&](int p0, int p1) {
            for (int p = p0; p < p1; ++p) {
                const int n = p / O, o = p % O;
                for (int y = 0; y < oH; ++y)
                    for (int x = 0; x < oW; ++x)
                        optr[((n * O + o) * oH + y) * oW + x] =
                            outRO[((n * oH + y) * oW + x) * O + o];
            }
        });
    }
    _tm_bias_relu_epilogue(optr, B * O * oH * oW, oH * oW, O, bias, use_relu);
    return std::make_shared<Value>(std::move(out), false);
}
inline ValueP depthwise_conv2d_ex(ValueP A, ValueP W, int stride,
                                  int pt, int pb, int pl, int pr,
                                  const float* bias, bool use_relu) {
    // inference-only (see conv2d_ex note)
    assert(!(current_tape().enabled && (A->requires_grad || W->requires_grad)) &&
           "depthwise_conv2d_ex is inference-only; use pad/depthwise_conv2d/bias/relu for training");
    Tensor Xin;
    ValueP x = A;
    if (pt || pb || pl || pr) {
        int B = A->shape()[0], C = A->shape()[1], H = A->shape()[2], Wd = A->shape()[3];
        int Hp = H + pt + pb, Wp = Wd + pl + pr;
        Xin = Tensor::zeros({B, C, Hp, Wp});
        for (int n = 0; n < B; ++n)
            _tm_pad_zero_into(A->t->data() + size_t(n) * C * H * Wd,
                              Xin.data() + size_t(n) * C * Hp * Wp,
                              C, H, Wd, Hp, Wp, pt, pl);
        x = std::make_shared<Value>(std::move(Xin), false);
    }
    ValueP y = depthwise_conv2d(x, W, stride);
    int C = y->shape()[1], oHW = y->numel() / y->shape()[0] / C;
    _tm_bias_relu_epilogue(y->t->data(), y->numel(), oHW, C, bias, use_relu);
    return y;
}
// symmetric-replicate the channel dim (NCHW) — BlazeFace pads channels to
// multiples of 8 via tflite MIRROR_PAD ops before some convs
inline ValueP chpad(ValueP A, int pre, int post) {
    int B = A->shape()[0], C = A->shape()[1], H = A->shape()[2], W = A->shape()[3];
    int C2 = C + pre + post;
    Tensor out = Tensor::zeros({B, C2, H, W});
    for (int n = 0; n < B; ++n)
        for (int c2 = 0; c2 < C2; ++c2) {
            int c = c2 - pre;
            if (c < 0) c = -c - 1;              // symmetric: -1 -> 0, -2 -> 1, ...
            if (c >= C) c = 2 * C - 1 - c;      // C -> C-1, C+1 -> C-2, ...
            std::memcpy(out.data() + (size_t(n) * C2 + c2) * H * W,
                        A->t->data() + (size_t(n) * C + c) * H * W,
                        sizeof(float) * H * W);
        }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A, B, C2, C, H, W, pre](const Tensor& g) {
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            // scatter from EVERY padded position (mirrored edges feed back)
            for (int n2 = 0; n2 < B; ++n2)
                for (int c2 = 0; c2 < C2; ++c2) {
                    int c = c2 - pre;
                    if (c < 0) c = -c - 1;
                    if (c >= C) c = 2 * C - 1 - c;
                    for (int i = 0; i < H * W; ++i)
                        (*A->grad)[(size_t(n2) * C + c) * H * W + i] +=
                            g[(size_t(n2) * C2 + c2) * H * W + i];
                }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}
// symmetric pad (TFLite MIRROR_PAD) on H,W: pads=1
inline ValueP pad_symmetric(ValueP A) {
    int B = A->shape()[0], C = A->shape()[1], H = A->shape()[2], W = A->shape()[3];
    Tensor out = Tensor::zeros({B, C, H + 2, W + 2});
    for (int n = 0; n < B; ++n)
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < H + 2; ++y) {
                int sy = y < 1 ? 0 : (y >= H + 1 ? H - 1 : y - 1);
                for (int x = 0; x < W + 2; ++x) {
                    int sx = x < 1 ? 0 : (x >= W + 1 ? W - 1 : x - 1);
                    out[((n * C + c) * (H + 2) + y) * (W + 2) + x] =
                        (*A->t)[((n * C + c) * H + sy) * W + sx];
                }
            }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A, B, C, H, W](const Tensor& g) {
            // scatter from EVERY padded position (mirrored edges feed back)
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            auto map_ = [](int i, int n) { return i < 1 ? 0 : (i >= n + 1 ? n - 1 : i - 1); };
            for (int n = 0; n < B; ++n)
                for (int c = 0; c < C; ++c)
                    for (int y = 0; y < H + 2; ++y)
                        for (int x = 0; x < W + 2; ++x)
                            (*A->grad)[((n * C + c) * H + map_(y, H)) * W + map_(x, W)] +=
                                g[((n * C + c) * (H + 2) + y) * (W + 2) + x];
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}
inline ValueP sigmoid(ValueP A) {
    Tensor out = Tensor::uninit(A->shape());
    for (int i = 0; i < A->numel(); ++i) out[i] = 1.f / (1.f + std::exp(-(*A->t)[i]));
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A](const Tensor& g) {
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            for (int i = 0; i < g.numel(); ++i) {
                float p = 1.f / (1.f + std::exp(-(*A->t)[i]));
                (*A->grad)[i] += g[i] * p * (1.f - p);
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// maxpool 2x2 stride 2 (the only pool BlazeFace needs); valid windows.
inline ValueP maxpool2x2(ValueP A) {
    int B = A->shape()[0], C = A->shape()[1], H = A->shape()[2], W = A->shape()[3];
    int oH = H / 2, oW = W / 2;
    Tensor out = Tensor::zeros({B, C, oH, oW});
    Tensor arg = Tensor::zeros({B, C, oH, oW});
    const Tensor& At = *A->t;
    for (int n = 0; n < B; ++n)
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < oH; ++y)
                for (int x = 0; x < oW; ++x) {
                    int best = 0; float bv = -1e30f;
                    for (int e = 0; e < 4; ++e) {
                        int dy = e / 2, dx = e % 2;
                        float v = At[((n * C + c) * H + y * 2 + dy) * W + x * 2 + dx];
                        if (v > bv) { bv = v; best = e; }
                    }
                    out[((n * C + c) * oH + y) * oW + x] = bv;
                    arg[((n * C + c) * oH + y) * oW + x] = float(best);
                }
    auto v = std::make_shared<Value>(std::move(out), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n; n.inputs = {A}; n.output = v;
        n.backward = [A, arg, B, C, H, W, oH, oW](const Tensor& g) {
            if (!A->grad) A->grad = std::make_shared<Tensor>(Tensor::zeros(A->shape()));
            for (int n = 0; n < B; ++n)
                for (int c = 0; c < C; ++c)
                    for (int y = 0; y < oH; ++y)
                        for (int x = 0; x < oW; ++x) {
                            int e = int(arg[((n * C + c) * oH + y) * oW + x]);
                            (*A->grad)[((n * C + c) * H + y * 2 + e / 2) * W + x * 2 + e % 2] +=
                                g[((n * C + c) * oH + y) * oW + x];
                        }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// ---------------------------------------------------------------------------
// GPT block ops: broadcast bias, transposed-weight matmul (weight tying),
// dropout. Everything else the GPT-2 block needs — layernorm, gelu, matmul,
// softmax, gather, cross_entropy — already exists above; the fused causal
// attention lives in attention.h.
// ---------------------------------------------------------------------------

// y(R,C) = X(R,C) + b(C), broadcast down the rows.
inline ValueP add_rowvec(ValueP X, ValueP b) {
    const int C = X->shape().back(), R = X->numel() / C;
    Tensor o = Tensor::uninit(X->shape());
    {
        // One fused, auto-vectorized loop: a vDSP call per 128-float row
        // cost more in call overhead than the bytes it moved (4,096 calls
        // per (4096,128) bias add measured at ~0.2 ms; the loop is
        // bandwidth-bound).
        const float* x = X->t->data(); const float* bd = b->t->data(); float* od = o.data();
        tm_par_rows_bw(R, C, [=](int r0, int r1) {
            tm_loop_add_rowvec(x + std::size_t(r0) * C, bd, od + std::size_t(r0) * C, r1 - r0, C);
        });
    }
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled && (X->requires_grad || b->requires_grad)) {
        Node n; n.inputs = {X, b}; n.output = v;
        std::weak_ptr<Value> wout = v;
        n.backward = [X, b, R, C, wout](const Tensor& g) {
            const float* gd = g.data();
            if (X->requires_grad && !tm_grad_alias(X, wout, g)) {
                bool fresh; float* xd = tm_grad_fresh(X, fresh).data();
                if (fresh) std::memcpy(xd, gd, sizeof(float) * (std::size_t)R * C);
                else tm_par(R * C, [=](int i0, int i1) {
                    vDSP_vadd(xd + i0, 1, gd + i0, 1, xd + i0, 1, vDSP_Length(i1 - i0));
                });
            }
            if (b->requires_grad) {
                // db = g^T * 1: one BLAS call, rather than R short vector adds
                // whose per-call overhead dominates at C = 128.
                cblas_sgemv(CblasRowMajor, CblasTrans, R, C, 1.0f, gd, C,
                            tm_ones(R), 1, 1.0f, tm_grad(b).data(), 1);
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// y(M,N) = A(M,K) @ W(N,K)^T — the weight stored row-per-output. This is the
// layout a tied lm_head wants: the same (vocab, dim) table used by gather.
inline ValueP matmul_nt(ValueP A, ValueP W) {
    const int M = A->shape()[0], K = A->shape()[1], N = W->shape()[0];
    Tensor o = Tensor::uninit({M, N});
    // tm_gemm_nt splits the LONGER of M/N. For a tied lm_head N is the vocab
    // (65 here) and M is every token in the batch, so splitting N would hand
    // each core an 8-column sliver.
    tm_gemm_nt(M, N, K, A->t->data(), W->t->data(), o.data(), N);
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled && (A->requires_grad || W->requires_grad)) {
        Node n; n.inputs = {A, W}; n.output = v;
        n.backward = [A, W, M, N, K](const Tensor& g) {
            if (A->requires_grad) {  // dA = g(M,N) @ W(N,K)
                bool fresh; float* ad = tm_grad_fresh(A, fresh).data();
                traink::psgemm_nn_m(M, K, N, g.data(), W->t->data(), ad, 1.0f, fresh ? 0.0f : 1.0f);
            }
            if (W->requires_grad) {  // dW = g(M,N)^T @ A(M,K)
                bool fresh; float* wd = tm_grad_fresh(W, fresh).data();
                traink::psgemm_tn(N, K, M, g.data(), A->t->data(), wd, 1.0f, fresh ? 0.0f : 1.0f);
            }
        };
        v->node = current_tape().push(std::move(n));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Dropout. Inverted (train-time scaling), so inference is the identity and
// needs no branch. The mask is generated from a splitmix64 stream keyed by
// (op counter, chunk offset), which makes it reproducible from tm_manual_seed
// independently of how the work was split across threads.
// ---------------------------------------------------------------------------
inline std::uint64_t& tm_seed() { static std::uint64_t s = 0x9E3779B97F4A7C15ull; return s; }
inline std::uint64_t& tm_op_counter() { static std::uint64_t c = 0; return c; }
inline void tm_manual_seed(std::uint64_t s) { tm_seed() = s ? s : 1; tm_op_counter() = 0; }

inline std::uint64_t tm_splitmix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline ValueP dropout(ValueP A, float p, bool training = true) {
    if (!training || p <= 0.f) return A;
    const int n = A->numel();
    const float keep = 1.f - p, inv = 1.f / keep;
    const std::uint64_t key = tm_splitmix(tm_seed() + (++tm_op_counter()));
    Tensor o = Tensor::uninit(A->shape());
    auto mask = pooled(std::size_t(n), /*zero=*/false);
    {
        const float* a = A->t->data(); float* od = o.data(); float* md = mask->data();
        // Fixed-size blocks, each seeded from (key, block index): the mask is
        // the same however the range gets split across threads.
        constexpr int BLK = 1024;
        const int nblk = (n + BLK - 1) / BLK;
        tm_par_rows(nblk, BLK, [=](int b0, int b1) {
            for (int b = b0; b < b1; ++b) {
                const int i0 = b * BLK, i1 = std::min(n, i0 + BLK);
                const std::uint64_t s = tm_splitmix(key + std::uint64_t(b));
                std::uint32_t l[4] = {std::uint32_t(s), std::uint32_t(s >> 16),
                                      std::uint32_t(s >> 32), std::uint32_t(s >> 48)};
                int i = i0;
#if defined(TM_HAVE_NEON)
                // four independent LCG lanes; the top 24 bits are the sample
                uint32x4_t st = vld1q_u32(l);
                const uint32x4_t ma = vdupq_n_u32(1664525u), ad = vdupq_n_u32(1013904223u);
                const float32x4_t keepv = vdupq_n_f32(keep), invv = vdupq_n_f32(inv);
                const float32x4_t zero = vdupq_n_f32(0.f), sc = vdupq_n_f32(1.f / 16777216.f);
                for (; i + 4 <= i1; i += 4) {
                    st = vmlaq_u32(ad, st, ma);
                    float32x4_t u = vmulq_f32(vcvtq_f32_u32(vshrq_n_u32(st, 8)), sc);
                    float32x4_t mk = vbslq_f32(vcltq_f32(u, keepv), invv, zero);
                    vst1q_f32(md + i, mk);
                    vst1q_f32(od + i, vmulq_f32(vld1q_f32(a + i), mk));
                }
                vst1q_u32(l, st);
#endif
                for (int k = 0; i < i1; ++i, ++k) {
                    l[k & 3] = l[k & 3] * 1664525u + 1013904223u;
                    const float u = float(l[k & 3] >> 8) * (1.f / 16777216.f);
                    md[i] = (u < keep) ? inv : 0.f;
                    od[i] = a[i] * md[i];
                }
            }
        });
    }
    auto v = std::make_shared<Value>(std::move(o), true);
    if (current_tape().enabled && A->requires_grad) {
        Node n_; n_.inputs = {A}; n_.output = v;
        n_.backward = [A, mask, n](const Tensor& g) {
            bool fresh; float* ad = tm_grad_fresh(A, fresh).data();
            const float* gd = g.data();
            const float* md = mask->data();
            tm_par(n, [=](int i0, int i1) {
                if (fresh)
                    vDSP_vmul(gd + i0, 1, md + i0, 1, ad + i0, 1, vDSP_Length(i1 - i0));
                else
                    vDSP_vma(gd + i0, 1, md + i0, 1, ad + i0, 1, ad + i0, 1,
                             vDSP_Length(i1 - i0));
            });
        };
        v->node = current_tape().push(std::move(n_));
    }
    return v;
}

}  // namespace tmg
