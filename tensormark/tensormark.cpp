// tensormark/tensormark.cpp — TensorMark Python bindings.
//
// Python API over the TensorMark C++23 numerical engine (neural_demo.cpp).
// Zero-copy numpy views over engine Tensors; GIL released around training
// steps (the engine's own thread pool does the parallelism).
#define TM_MAIN main
#include "engine.h"
#undef TM_MAIN
#include "autograd.h"
#include "attention.h"
#include "kda.h"
#include "optim.h"
#include "version.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace py = pybind11;
using ::Tensor;
using ::Rng;


static std::vector<int> to_ints(const std::vector<int64_t>& v) {
    return std::vector<int>(v.begin(), v.end());
}
static std::vector<int64_t> to_i64(const std::vector<int>& v) {
    return std::vector<int64_t>(v.begin(), v.end());
}

// ---- Tensor <-> numpy ----------------------------------------------------

// Write-through view over a LIVE engine tensor: the capsule keeps the
// shared_ptr alive, so numpy can read and write the buffer in place (used by
// the gradcheck harness, which perturbs parameters directly).
static py::array_t<float> tm_numpy_view(const std::shared_ptr<Tensor>& t) {
    std::vector<py::ssize_t> shape, strides;
    const auto& s = t->shape();
    std::vector<py::ssize_t> stride(s.size(), 1);
    for (int k = int(s.size()) - 2; k >= 0; --k) stride[k] = stride[k + 1] * s[k + 1];
    for (size_t k = 0; k < s.size(); ++k) {
        shape.push_back(s[k]);
        strides.push_back(stride[k] * py::ssize_t(sizeof(float)));
    }
    auto* holder = new std::shared_ptr<Tensor>(t);
    py::capsule cap(holder, [](void* q) {
        delete reinterpret_cast<std::shared_ptr<Tensor>*>(q);
    });
    return py::array_t<float>(shape, strides, t->data(), cap);
}

// Owned-copy view: the array carries a capsule holding a clone of the
// buffer, so it never aliases a dead temporary. (Zero-copy write-through
// views of STORED engine tensors land in tier-2 with a Tensor holder.)
static py::array_t<float> tm_numpy(const Tensor& t) {
    std::vector<py::ssize_t> shape, strides;
    const auto& s = t.shape();
    std::vector<int> stride(s.size(), 1);
    for (int k = int(s.size()) - 2; k >= 0; --k) stride[k] = stride[k + 1] * s[k + 1];
    for (size_t k = 0; k < s.size(); ++k) {
        shape.push_back(s[k]);
        strides.push_back(py::ssize_t(stride[k]) * sizeof(float));
    }
    auto* vec = new std::vector<float>(t.raw());  // capsule owns this
    py::capsule cap(vec, [](void* p) {
        delete reinterpret_cast<std::vector<float>*>(p);
    });
    return py::array_t<float>(std::move(shape), std::move(strides),
                              vec->data(), cap);
}

// Copy-in construction (documented: numpy owns its memory; we take a c-contig
// float32 snapshot). Zero-copy reads use .numpy() on engine tensors instead.
static Tensor tm_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> a) {
    py::buffer_info info = a.request();
    Tensor t(std::vector<int>(info.shape.begin(), info.shape.end()));
    std::memcpy(t.raw().data(), info.ptr, size_t(info.size) * sizeof(float));
    return t;
}

// ---- Sequential wrapper ---------------------------------------------------
// Engine Sequential needs an Rng& for forward(); we own one, seeded via
// manual_seed. Training/backward uses the engine's fused lr-per-step path.
struct TMModel {
    Sequential net;
    std::shared_ptr<Rng> rng = std::make_shared<Rng>(0);
    float lr = 1e-3f;          // current lr (set by scheduler / fit loop)
    float weight_decay = 5e-4f;
    int total_steps = 1, warmup_steps = 0, step = 0;

    Tensor forward(const Tensor& x, bool training) {
        py::gil_scoped_release release;
        return net.forward(x, *rng, training);
    }
    void backward(const Tensor& grad) {
        py::gil_scoped_release release;
        net.backward(grad, lr);
    }
    void set_lr(float v) { lr = v; }
    void cosine_step() {  // one scheduler step: warmup + cosine (oracle formula)
        if (step < warmup_steps)
            lr = lr_max_ * float(step + 1) / float(std::max(1, warmup_steps));
        else {
            float t = float(step - warmup_steps) /
                      float(std::max(1, total_steps - warmup_steps));
            lr = lr_max_ * 0.5f * (1.f + std::cos(3.14159265f * t));
        }
        ++step;
    }
    float lr_max_ = 1e-3f;
    bool is_image_net() const {
        return net.size() > 0 && dynamic_cast<Conv2D*>(net.layers()[0].get()) != nullptr;
    }
};

// fit(): Keras-flavored training loop. X (N, features...) float32,
// y (N,) int64 labels. CE loss + onehot via engine SoftmaxCrossEntropy.
struct FitOpts { int batch=32, epochs=1; float lr=1e-3f, label_smooth=0.f;
                 int h=0, w=0; bool augment=false; };
static py::dict tm_fit(TMModel& m, py::array_t<float> X,
                       py::array_t<int64_t> y, FitOpts o) {
    const int batch = o.batch, epochs = o.epochs;
    const float lr_max = o.lr, label_smooth = o.label_smooth;
    py::buffer_info xi = X.request(), yi = y.request();
    const int N = int(xi.shape[0]);

    std::vector<int> xshape(xi.shape.begin(), xi.shape.end());
    // Image mode (h>0): rows are H*W flattened images -> engine
    // (B,1,H,W) NCHW per sample, optional ±2px shift augmentation.
    const bool image_mode = o.h > 0;

    int steps = (N + batch - 1) / batch;
    m.lr_max_ = lr_max;
    m.total_steps = steps * epochs;
    m.warmup_steps = std::max(1, m.total_steps / 20);
    m.step = 0;
    float last_loss = 0.f, last_acc = 0.f;
    int n_classes = 0;
    for (int i = 0; i < N; ++i)
        n_classes = std::max(n_classes, int(static_cast<int64_t*>(yi.ptr)[i]) + 1);

    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;

    for (int ep = 0; ep < epochs; ++ep) {
        std::shuffle(idx.begin(), idx.end(), m.rng->engine());
        float ep_loss = 0.f; long ep_correct = 0; int nb = 0;
        for (int off = 0; off < N; off += batch) {
            int bs = std::min(batch, N - off);
            Tensor xb;
            std::vector<int> yb_int(bs);
            if (image_mode) {
                // (B,1,H,W) NCHW, optional ±2px zero-padded shift augmentation.
                const int H = o.h, W = o.w, HW = H * W;
                xb = Tensor({bs, 1, H, W});
                for (int n = 0; n < bs; ++n) {
                    int i = idx[off + n];
                    const float* row = &static_cast<float*>(xi.ptr)[size_t(i) * HW];
                    int sx = 0, sy = 0;
                    if (o.augment) {
                        sx = int(m.rng->uniform() * 5.0f) - 2;
                        sy = int(m.rng->uniform() * 5.0f) - 2;
                    }
                    for (int r = 0; r < H; ++r) {
                        int sr = r - sy;
                        for (int c = 0; c < W; ++c) {
                            int sc = c - sx;
                            float v = 0.f;
                            if (sr >= 0 && sr < H && sc >= 0 && sc < W)
                                v = row[sr * W + sc];
                            xb[((size_t(n) * 1) * H + r) * W + c] = v;
                        }
                    }
                    yb_int[n] = int(static_cast<int64_t*>(yi.ptr)[i]);
                }
            } else {
                // Dense mode: feature-major columns — x is (feat, B),
                // one column per sample (Dense reads x.dim(1) as batch).
                const int feat = int(xi.size / N);
                xb = Tensor({feat, bs});
                for (int n = 0; n < bs; ++n) {
                    int i = idx[off + n];
                    const float* row = &static_cast<float*>(xi.ptr)[size_t(i) * feat];
                    for (int f = 0; f < feat; ++f) xb.raw()[size_t(f) * bs + n] = row[f];
                    yb_int[n] = int(static_cast<int64_t*>(yi.ptr)[i]);
                }
            }
            Tensor yoh = make_onehot(yb_int, bs, n_classes, label_smooth);
            Tensor logits, probs;
            {
                py::gil_scoped_release release;
                logits = m.net.forward(xb, *m.rng, true);
            }
            SoftmaxCrossEntropy ce;
            probs = ce.forward(logits, yoh);
            // accuracy
            const int C = probs.dim(0);
            for (int n = 0; n < bs; ++n) {
                int pc = 0; float pv = probs[n];
                for (int c = 1; c < C; ++c) if (probs[c * bs + n] > pv) { pv = probs[c * bs + n]; pc = c; }
                if (pc == yb_int[n]) ++ep_correct;
            }
            Tensor grad = ce.backward(yoh);
            m.backward(grad);   // GIL released inside
            m.cosine_step();
            ep_loss += ce.loss(); ++nb;
        }
        last_loss = ep_loss / std::max(1, nb);
        last_acc = float(ep_correct) / float(N);
    }
    py::dict out;
    out["loss"] = last_loss; out["accuracy"] = last_acc;
    out["epochs"] = epochs;
    return out;
}

// ---- module ---------------------------------------------------------------
// ---- BFGraph: whole-network single-call inference executor -----------------
// The full op plan (weights prepacked NCHW at build time) runs inside ONE C++
// call: no per-op Python glue, no per-op tape interaction, one GIL release.
// Ops address tensors by slot index. conv/dw use the fused *_ex kernels
// (zero-pad + conv + bias + optional ReLU); mirror = TFLite MIRROR_PAD
// (symmetric on H/W and/or channels); 'to_nhwc' emits the head conv outputs
// in anchor-major order so downstream reshape/concat match TFLite exactly.
struct BFOp {
    int kind;                       // see enum below
    std::vector<int> ins;           // input slots
    int out = -1;
    // conv/dw
    tmg::ValueP wv, bv; int stride = 1, pt = 0, pb = 0, pl = 0, pr = 0; bool relu = false;
    // mirror: h/w pad count, channel pre/post
    int hw = 0, cpre = 0, cpost = 0;
    // reshape / concat
    std::vector<int> shape; int axis = 0;
};
struct BFGraph {
    enum { CONV, DW, ADD, RELU, MAXPOOL, MIRROR, TO_NHWC, RESHAPE, CONCAT, CHSTORE };
    std::vector<BFOp> ops;
    void add(BFOp o) { ops.push_back(std::move(o)); }

    py::list run(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                 std::vector<int> out_slots) {
        py::buffer_info info = x.request();
        int maxslot = 0;
        for (const BFOp& o : ops) {
            maxslot = std::max(maxslot, o.out);
        }
        std::vector<tmg::ValueP> env(size_t(maxslot) + 1);
        tmg::Tape& tape = tmg::current_tape();
        bool saved = tape.enabled;
        tape.enabled = false;
        {
        py::gil_scoped_release gil;
        env[0] = std::make_shared<tmg::Value>(
            Tensor(std::vector<int>(info.shape.begin(), info.shape.end())), false);
        std::memcpy(env[0]->t->raw().data(), info.ptr, size_t(info.size) * sizeof(float));
        const bool dbg = std::getenv("TM_BFDEBUG") != nullptr;
        auto dbg_all0 = std::chrono::steady_clock::now();
        for (const BFOp& o : ops) {
            auto dbg_t0 = std::chrono::steady_clock::now();
            const tmg::ValueP a = o.ins.empty() ? nullptr : env[o.ins[0]];
            tmg::ValueP y;
            switch (o.kind) {
            case CONV:
                y = tmg::conv2d_ex(a, o.wv, o.stride,
                                   o.pt, o.pb, o.pl, o.pr,
                                   o.bv->t->numel() ? o.bv->t->data() : nullptr, o.relu);
                break;
            case DW:
                y = tmg::depthwise_conv2d_ex(a, o.wv, o.stride,
                                             o.pt, o.pb, o.pl, o.pr,
                                             o.bv->t->numel() ? o.bv->t->data() : nullptr, o.relu);
                break;
            case ADD: {
                const Tensor& p = *env[o.ins[0]]->t; const Tensor& q = *env[o.ins[1]]->t;
                Tensor r = Tensor::zeros(p.shape());
                const float* pp = p.data(); const float* qq = q.data(); float* rr = r.data();
                for (size_t i = 0; i < r.numel(); ++i) rr[i] = pp[i] + qq[i];
                env[o.out] = std::make_shared<tmg::Value>(std::move(r), false);
                break;
            }
            case RELU: {
                Tensor r(env[o.ins[0]]->t->shape());
                const float* pp = env[o.ins[0]]->t->data(); float* rr = r.data();
                for (size_t i = 0; i < r.numel(); ++i) rr[i] = pp[i] > 0.f ? pp[i] : 0.f;
                env[o.out] = std::make_shared<tmg::Value>(std::move(r), false);
                break;
            }
            case MAXPOOL:
                y = tmg::maxpool2x2(a);
                break;
            case MIRROR: {
                tmg::ValueP t = a;
                if (o.hw) t = tmg::pad_symmetric(t);
                if (o.cpre || o.cpost) t = tmg::chpad(t, o.cpre, o.cpost);
                y = t;
                break;
            }
            case TO_NHWC: {
                const Tensor& t = *env[o.ins[0]]->t;
                int B = t.shape()[0], C = t.shape()[1], H = t.shape()[2], W = t.shape()[3];
                Tensor r({B, H, W, C});
                const float* src = t.data(); float* dst = r.data();
                for (int n = 0; n < B; ++n)
                    for (int c = 0; c < C; ++c)
                        for (int i = 0; i < H * W; ++i)
                            dst[(size_t(n) * H * W + i) * C + c] = src[(size_t(n) * C + c) * H * W + i];
                env[o.out] = std::make_shared<tmg::Value>(std::move(r), false);
                break;
            }
            case RESHAPE: {
                const Tensor& s = *env[o.ins[0]]->t;
                Tensor r(o.shape);
                std::memcpy(r.raw().data(), s.raw().data(), s.raw().size() * sizeof(float));
                env[o.out] = std::make_shared<tmg::Value>(std::move(r), false);
                break;
            }
            case CONCAT: {
                const Tensor& p = *env[o.ins[0]]->t; const Tensor& q = *env[o.ins[1]]->t;
                int ax = o.axis;
                std::vector<int> sh = p.shape();
                sh[ax] += q.shape()[ax];
                Tensor r(sh);
                size_t outer = 1, inner = 1;
                for (int k = 0; k < ax; ++k) outer *= p.shape()[k];
                for (int k = ax + 1; k < int(p.shape().size()); ++k) inner *= p.shape()[k];
                size_t np1 = p.shape()[ax] * inner, np2 = q.shape()[ax] * inner;
                for (size_t oi = 0; oi < outer; ++oi) {
                    std::memcpy(r.raw().data() + oi * (np1 + np2), p.raw().data() + oi * np1, np1 * sizeof(float));
                    std::memcpy(r.raw().data() + oi * (np1 + np2) + np1, q.raw().data() + oi * np2, np2 * sizeof(float));
                }
                env[o.out] = std::make_shared<tmg::Value>(std::move(r), false);
                break;
            }
            }
            if (y) env[o.out] = y;
            if (dbg) std::fprintf(stderr, "kind=%d out=%d %.3f ms\n", o.kind, o.out,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dbg_t0).count());
        }
        if (dbg) std::fprintf(stderr, "TOTAL %.3f ms\n",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dbg_all0).count());
        tape.enabled = saved;
        }
        py::list out;
        for (int s : out_slots) out.append(tm_numpy(*env[s]->t));
        return out;
    }
};

PYBIND11_MODULE(tensormark, mod) {
    mod.doc() = "TensorMark: PyTorch-style API over the WeOptimize C++ engine";

    // Zero-copy Tensor holder: the python object owns shared_ptr<Tensor>;
    // numpy() returns a view over THAT buffer with the holder as base, so
    // writes flow through in BOTH directions and lifetimes are safe.
    py::class_<Tensor, std::shared_ptr<Tensor>>(mod, "Tensor")
        .def("numpy", [](Tensor& t) {
            std::vector<py::ssize_t> shape, strides;
            const auto& s = t.shape();
            std::vector<int> stride(s.size(), 1);
            for (int k = int(s.size()) - 2; k >= 0; --k)
                stride[k] = stride[k + 1] * s[k + 1];
            for (size_t k = 0; k < s.size(); ++k) {
                shape.push_back(s[k]);
                strides.push_back(py::ssize_t(stride[k]) * sizeof(float));
            }
            // py::cast(t) inside a shared_ptr holder class returns the
            // existing python object; the view aliases the engine buffer.
            return py::array_t<float>(std::move(shape), std::move(strides),
                                      t.raw().data(), py::cast(t));
        })
        .def("tolist", [](Tensor& t) {
            py::list l;
            for (float v : t.raw()) l.append(v);
            return l;
        })
        .def("__len__", [](const Tensor& t) { return (py::ssize_t)t.numel(); })
        .def_property_readonly("shape", [](const Tensor& t) { return to_i64(t.shape()); })
        .def_property_readonly("data", [](Tensor& t) {
            // same zero-copy view as .numpy()
            std::vector<py::ssize_t> shape, strides;
            const auto& s = t.shape();
            std::vector<int> stride(s.size(), 1);
            for (int k = int(s.size()) - 2; k >= 0; --k)
                stride[k] = stride[k + 1] * s[k + 1];
            for (size_t k = 0; k < s.size(); ++k) {
                shape.push_back(s[k]);
                strides.push_back(py::ssize_t(stride[k]) * sizeof(float));
            }
            return py::array_t<float>(std::move(shape), std::move(strides),
                                      t.raw().data(), py::cast(t));
        });
    mod.def("tensor", [](py::array_t<float, py::array::c_style | py::array::forcecast> a) {
        return std::make_shared<Tensor>(tm_from_numpy(a));
    });

    py::class_<TMModel, std::shared_ptr<TMModel>>(mod, "Sequential")
        .def(py::init<>())
        .def("forward", [](TMModel& m, py::array_t<float, py::array::c_style | py::array::forcecast> a, bool training) {
                // Accept (B, feat) row-major; image rows (feat=H*W) go to
                // engine (B,1,H,W) NCHW, dense rows go to (feat, B).
                // Output: conv nets give (B, Cout, oH, oW) -> argmax over
                // channels; dense nets give (C, B) — normalize both to
                // a (B, C) row-per-sample result.
                py::buffer_info info = a.request();
                int nb = int(info.shape[0]), nf = int(info.size / std::max(1, nb));
                Tensor y;
                if (info.shape.size() == 2 && nf == 784 && m.is_image_net()) {
                    Tensor x({nb, 1, 28, 28});
                    std::memcpy(x.raw().data(), info.ptr, size_t(info.size) * sizeof(float));
                    y = m.forward(x, training);
                } else {
                    Tensor x({nf, nb});
                    const float* src = static_cast<float*>(info.ptr);
                    for (int n = 0; n < nb; ++n)
                        for (int f = 0; f < nf; ++f)
                            x.raw()[size_t(f) * nb + n] = src[size_t(n) * nf + f];
                    y = m.forward(x, training);
                }
                // Normalize output to (B, C): engine conv output is
                // (B, Cout, oH, oW) — global-avg? No: Flatten nets end in
                // Dense, so final layer output is (C, B) dense-style.
                return tm_numpy(y);
            }, py::arg("x"), py::arg("training") = false)
        .def("fit", [](TMModel& m, py::array_t<float> x, py::array_t<int64_t> y, py::kwargs kw) {
                FitOpts o;
                if (kw.contains("batch")) o.batch = kw["batch"].cast<int>();
                if (kw.contains("epochs")) o.epochs = kw["epochs"].cast<int>();
                if (kw.contains("lr")) o.lr = kw["lr"].cast<float>();
                if (kw.contains("label_smooth")) o.label_smooth = kw["label_smooth"].cast<float>();
                if (kw.contains("h")) o.h = kw["h"].cast<int>();
                if (kw.contains("w")) o.w = kw["w"].cast<int>();
                if (kw.contains("augment")) o.augment = kw["augment"].cast<bool>();
                return tm_fit(m, x, y, o);
            }, py::arg("x"), py::arg("y"))
        .def("set_lr", &TMModel::set_lr)
        .def("num_params", [](TMModel& m) { return m.net.num_params(); })
        .def("manual_seed", [](TMModel& m, int s) { m.rng = std::make_shared<Rng>(s); })
        .def("set_weight_decay", [](TMModel& m, float wd) {
            m.weight_decay = wd; m.net.set_weight_decay(wd);
        })
        .def("init", [](TMModel& m) { Rng rr(42); m.net.init_all(rr); })
        // tier-1 layer builders (PyTorch-style ctor args, engine layers)
        .def("dense", [](TMModel& m, int in_dim, int out_dim) {
            m.net.add(Dense(in_dim, out_dim));
            return py::cast(m, py::return_value_policy::reference_internal);
        }, py::arg("in_features"), py::arg("out_features"))
        .def("conv2d", [](TMModel& m, int cin, int cout, int kh, int kw) {
            m.net.add(Conv2D(cin, cout, kh, kw));
            return py::cast(m, py::return_value_policy::reference_internal);
        }, py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel"), py::arg("kernel_w") = -1)
        .def("batchnorm2d", [](TMModel& m, int c) {
            m.net.add(BatchNorm2D(c, 0.1f, 1e-5f));
            return py::cast(m, py::return_value_policy::reference_internal);
        }, py::arg("channels"))
        .def("batchnorm1d", [](TMModel& m, int c) {
            m.net.add(BatchNorm1D(c, 0.1f, 1e-5f));
            return py::cast(m, py::return_value_policy::reference_internal);
        }, py::arg("features"))
        .def("relu", [](TMModel& m) { m.net.add(ReLU()); return py::cast(m, py::return_value_policy::reference_internal); })
        .def("maxpool2d", [](TMModel& m, int k) { m.net.add(MaxPool2D(k)); return py::cast(m, py::return_value_policy::reference_internal); },
             py::arg("kernel") = 2)
        .def("flatten", [](TMModel& m) { m.net.add(Flatten()); return py::cast(m, py::return_value_policy::reference_internal); })
        .def("dropout", [](TMModel& m, float p) { m.net.add(Dropout(p)); return py::cast(m, py::return_value_policy::reference_internal); },
             py::arg("p") = 0.5f);

    // expose engine scalar
    mod.attr("__version__") = TENSORMARK_VERSION;

// ---- tape (autograd) bindings: Value + ops + control ----
py::class_<tmg::Value, tmg::ValueP>(mod, "Value")
    .def("numpy", [](tmg::ValueP v) { return tm_numpy(*v->t); })
    .def_property_readonly("shape", [](const tmg::ValueP& v) {
        return to_i64(v->shape());
    })
    .def_property_readonly("requires_grad", [](const tmg::ValueP& v) {
        return v->requires_grad;
    })
    // write-through view of the parameter buffer itself
    .def_property_readonly("view", [](const tmg::ValueP& v) { return tm_numpy_view(v->t); })
    .def_property_readonly("grad", [](const tmg::ValueP& v) -> py::object {
        if (!v->grad) return py::none();
        return tm_numpy_view(v->grad);
    })
    .def("zero_grad", [](const tmg::ValueP& v) {
        if (v->grad) std::fill(v->grad->raw().begin(), v->grad->raw().end(), 0.f);
    })
    .def_property_readonly("numel", [](const tmg::ValueP& v) { return v->numel(); });
py::class_<BFGraph>(mod, "BFGraph")
    .def(py::init<>())
    .def("conv", [](BFGraph& g, int in, int out,
                    py::array_t<float, py::array::c_style | py::array::forcecast> w,
                    py::array_t<float, py::array::c_style | py::array::forcecast> b,
                    int stride, int pt, int pb, int pl, int pr, bool relu) {
        BFOp o; o.kind = BFGraph::CONV; o.ins = {in}; o.out = out;
        o.wv = std::make_shared<tmg::Value>(tm_from_numpy(w), false);
            o.bv = std::make_shared<tmg::Value>(tm_from_numpy(b), false);
        o.stride = stride; o.pt = pt; o.pb = pb; o.pl = pl; o.pr = pr; o.relu = relu;
        g.add(std::move(o));
    }, py::arg("in"), py::arg("out"), py::arg("w"), py::arg("b"),
       py::arg("stride") = 1, py::arg("pt") = 0, py::arg("pb") = 0,
       py::arg("pl") = 0, py::arg("pr") = 0, py::arg("relu") = false)
    .def("dw", [](BFGraph& g, int in, int out,
                  py::array_t<float, py::array::c_style | py::array::forcecast> w,
                  py::array_t<float, py::array::c_style | py::array::forcecast> b,
                  int stride, int pt, int pb, int pl, int pr, bool relu) {
        BFOp o; o.kind = BFGraph::DW; o.ins = {in}; o.out = out;
        o.wv = std::make_shared<tmg::Value>(tm_from_numpy(w), false);
            o.bv = std::make_shared<tmg::Value>(tm_from_numpy(b), false);
        o.stride = stride; o.pt = pt; o.pb = pb; o.pl = pl; o.pr = pr; o.relu = relu;
        g.add(std::move(o));
    }, py::arg("in"), py::arg("out"), py::arg("w"), py::arg("b"),
       py::arg("stride") = 1, py::arg("pt") = 0, py::arg("pb") = 0,
       py::arg("pl") = 0, py::arg("pr") = 0, py::arg("relu") = false)
    .def("add2", [](BFGraph& g, int a, int b, int out) {
        BFOp o; o.kind = BFGraph::ADD; o.ins = {a, b}; o.out = out; g.add(std::move(o));
    })
    .def("relu", [](BFGraph& g, int in, int out) {
        BFOp o; o.kind = BFGraph::RELU; o.ins = {in}; o.out = out; g.add(std::move(o));
    })
    .def("maxpool", [](BFGraph& g, int in, int out) {
        BFOp o; o.kind = BFGraph::MAXPOOL; o.ins = {in}; o.out = out; g.add(std::move(o));
    })
    .def("mirror", [](BFGraph& g, int in, int out, int hw, int cpre, int cpost) {
        BFOp o; o.kind = BFGraph::MIRROR; o.ins = {in}; o.out = out;
        o.hw = hw; o.cpre = cpre; o.cpost = cpost; g.add(std::move(o));
    }, py::arg("in"), py::arg("out"), py::arg("hw"), py::arg("cpre"), py::arg("cpost"))
    .def("to_nhwc", [](BFGraph& g, int in, int out) {
        BFOp o; o.kind = BFGraph::TO_NHWC; o.ins = {in}; o.out = out; g.add(std::move(o));
    })
    .def("reshape", [](BFGraph& g, int in, int out, std::vector<int64_t> sh) {
        BFOp o; o.kind = BFGraph::RESHAPE; o.ins = {in}; o.out = out;
        o.shape = to_ints(sh); g.add(std::move(o));
    })
    .def("concat", [](BFGraph& g, int a, int b, int out, int axis) {
        BFOp o; o.kind = BFGraph::CONCAT; o.ins = {a, b}; o.out = out;
        o.axis = axis; g.add(std::move(o));
    })
    .def("run", [](BFGraph& g, py::array_t<float, py::array::c_style | py::array::forcecast> x,
                   std::vector<int> out_slots) {
        return g.run(x, out_slots);
    }, py::arg("x"), py::arg("out_slots"));
mod.def("set_param", [](const std::string& name, double val) { tmg::tm_set_param(name, val); },
        py::arg("name"), py::arg("value"));
mod.def("tape_clear", []() { tmg::current_tape().clear(); });
mod.def("clear_weight_cache", []() { tmg::tm_clear_wcache(); });
mod.def("tape_backward", [](tmg::ValueP loss) {
    if (!loss) throw std::invalid_argument("tape_backward: loss is None");
    tmg::current_tape().backward(loss);
});
mod.def("param", [](py::array_t<float, py::array::c_style | py::array::forcecast> a) {
    return std::make_shared<tmg::Value>(tm_from_numpy(a), true);
});
mod.def("const_", [](py::array_t<float, py::array::c_style | py::array::forcecast> a) {
    return std::make_shared<tmg::Value>(tm_from_numpy(a), false);
});
mod.def("matmul", [](tmg::ValueP a, tmg::ValueP b) { return tmg::matmul(a, b); });
mod.def("add", [](tmg::ValueP a, tmg::ValueP b) { return tmg::add(a, b); });
mod.def("relu", [](tmg::ValueP a) { return tmg::relu(a); });
mod.def("gelu", [](tmg::ValueP a) { return tmg::gelu(a); });
mod.def("sigmoid", [](tmg::ValueP a) { return tmg::sigmoid(a); });
mod.def("scale", [](tmg::ValueP a, float s) { return tmg::scale(a, s); });
mod.def("softmax", [](tmg::ValueP a) { return tmg::softmax(a); });
mod.def("layernorm", [](tmg::ValueP a, tmg::ValueP g, tmg::ValueP b) {
    return tmg::layernorm(a, g, b);
});
mod.def("reshape", [](tmg::ValueP a, std::vector<int64_t> sh) {
    return tmg::reshape(a, std::vector<int>(sh.begin(), sh.end()));
});
mod.def("transpose2d", [](tmg::ValueP a) { return tmg::transpose2d(a); });
mod.def("concat", [](tmg::ValueP a, tmg::ValueP b) { return tmg::concat(a, b); });
mod.def("conv2d", [](tmg::ValueP a, tmg::ValueP w, int stride) {
    return tmg::conv2d(a, w, stride);
}, py::arg("x"), py::arg("w"), py::arg("stride") = 1);
mod.def("depthwise_conv2d", [](tmg::ValueP a, tmg::ValueP w, int stride) {
    return tmg::depthwise_conv2d(a, w, stride);
}, py::arg("x"), py::arg("w"), py::arg("stride") = 1);
mod.def("pad_symmetric", [](tmg::ValueP a) { return tmg::pad_symmetric(a); });
mod.def("chpad", [](tmg::ValueP a, int pre, int post) {
    return tmg::chpad(a, pre, post);
}, py::arg("x"), py::arg("pre") = 0, py::arg("post") = 0);
mod.def("conv2d_ex", [](tmg::ValueP a, tmg::ValueP w, int stride,
                        int pt, int pb, int pl, int pr,
                        py::array_t<float, py::array::c_style | py::array::forcecast> bias,
                        bool use_relu) {
    const float* bp = bias.size() ? bias.data() : nullptr;
    return tmg::conv2d_ex(a, w, stride, pt, pb, pl, pr, bp, use_relu);
}, py::arg("x"), py::arg("w"), py::arg("stride") = 1,
   py::arg("pt") = 0, py::arg("pb") = 0, py::arg("pl") = 0, py::arg("pr") = 0,
   py::arg("bias") = py::array_t<float>(0), py::arg("relu") = false);
mod.def("depthwise_conv2d_ex", [](tmg::ValueP a, tmg::ValueP w, int stride,
                                  int pt, int pb, int pl, int pr,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> bias,
                                  bool use_relu) {
    const float* bp = bias.size() ? bias.data() : nullptr;
    return tmg::depthwise_conv2d_ex(a, w, stride, pt, pb, pl, pr, bp, use_relu);
}, py::arg("x"), py::arg("w"), py::arg("stride") = 1,
   py::arg("pt") = 0, py::arg("pb") = 0, py::arg("pl") = 0, py::arg("pr") = 0,
   py::arg("bias") = py::array_t<float>(0), py::arg("relu") = false);
mod.def("maxpool2x2", [](tmg::ValueP a) { return tmg::maxpool2x2(a); });
// Zero-conversion overload first: pybind dispatches in registration order,
// and the vector<int> overload would also accept an array via the (slow)
// per-element sequence walk (measured ~0.09 ms per 4096-row call).
mod.def("gather_rows", [](tmg::ValueP e,
                          py::array_t<std::int32_t, py::array::c_style> idx) {
    if (idx.ndim() != 1) throw std::invalid_argument("gather_rows: indices must be 1-D");
    const auto* p = idx.data();
    return tmg::gather(e, std::vector<int>(p, p + idx.shape(0)));
});
mod.def("gather_rows", [](tmg::ValueP e, std::vector<int> idx) {
    return tmg::gather(e, idx);
});
mod.def("cross_entropy_rows", [](tmg::ValueP logits, std::vector<int> labels,
                                 int ignore_index) {
    return tmg::cross_entropy(logits, labels, ignore_index);
}, py::arg("logits"), py::arg("labels"), py::arg("ignore_index") = -1);
mod.def("mse_vals", [](tmg::ValueP a, tmg::ValueP b) { return tmg::mse(a, b); });
mod.def("sum_all", [](tmg::ValueP a) { return tmg::sum(a); });
mod.def("mean_all", [](tmg::ValueP a) { return tmg::mean(a); });
mod.def("mul_vals", [](tmg::ValueP a, tmg::ValueP b) { return tmg::mul(a, b); });
mod.def("scale_val", [](tmg::ValueP a, float s) { return tmg::scale(a, s); });
mod.def("slice2d", [](tmg::ValueP a, int r0, int r1, int c0, int c1) {
    return tmg::slice(a, r0, r1, c0, c1);
});
// ---- GPT block ops -------------------------------------------------------
mod.def("add_rowvec", [](tmg::ValueP x, tmg::ValueP b) { return tmg::add_rowvec(x, b); },
        py::arg("x"), py::arg("b"),
        "y[r,c] = x[r,c] + b[c] (bias broadcast down rows)");
mod.def("matmul_nt", [](tmg::ValueP a, tmg::ValueP w) { return tmg::matmul_nt(a, w); },
        py::arg("a"), py::arg("w"),
        "y(M,N) = a(M,K) @ w(N,K)^T -- weight stored row-per-output (tied lm_head)");
mod.def("dropout", [](tmg::ValueP x, float p, bool training) {
    return tmg::dropout(x, p, training);
}, py::arg("x"), py::arg("p"), py::arg("training") = true);
mod.def("mha_causal", [](tmg::ValueP q, tmg::ValueP k, tmg::ValueP v,
                         int batch, int seq, int heads) {
    return tmg::mha_causal(q, k, v, batch, seq, heads);
}, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("batch"), py::arg("seq"),
   py::arg("heads"),
   "Fused causal multi-head attention over (batch*seq, dim) head-major tensors");
mod.def("kda_causal", [](tmg::ValueP q, tmg::ValueP k, tmg::ValueP v,
                         tmg::ValueP g, tmg::ValueP beta,
                         int batch, int seq, int heads) {
    return tmg::kda_causal(q, k, v, g, beta, batch, seq, heads);
}, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
   py::arg("batch"), py::arg("seq"), py::arg("heads"),
   "Gated delta-rule linear attention (KDA) over (batch*seq, dim); g is the "
   "per-channel log-decay, beta the per-head write gate");
mod.def("manual_seed", [](std::uint64_t s) { tmg::tm_manual_seed(s); }, py::arg("seed"));
mod.def("set_grad_enabled", [](bool on) { tmg::current_tape().enabled = on; }, py::arg("on"));
mod.def("is_grad_enabled", []() { return tmg::current_tape().enabled; });
mod.def("tape_len", []() { return py::ssize_t(tmg::current_tape().nodes.size()); });
mod.def("lr_at", [](long step, long warmup, long total, float lr_max, float lr_min) {
    return tmg::lr_at(step, warmup, total, lr_max, lr_min);
}, py::arg("step"), py::arg("warmup"), py::arg("total"), py::arg("lr_max"),
   py::arg("lr_min"));

// ---- AdamW ---------------------------------------------------------------
py::class_<tmg::AdamW>(mod, "AdamW")
    .def(py::init([](std::vector<tmg::ValueP> params, std::vector<bool> decay,
                     float beta1, float beta2, float eps, float weight_decay) {
        if (params.size() != decay.size())
            throw std::invalid_argument("params and decay must have equal length");
        auto o = std::make_unique<tmg::AdamW>();
        o->beta1 = beta1; o->beta2 = beta2; o->eps = eps; o->weight_decay = weight_decay;
        for (size_t i = 0; i < params.size(); ++i) o->add(params[i], decay[i]);
        return o;
    }), py::arg("params"), py::arg("decay"), py::arg("beta1") = 0.9f,
        py::arg("beta2") = 0.95f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.1f)
    .def("step", [](tmg::AdamW& o, float lr) {
        py::gil_scoped_release nogil; o.step(lr);
    }, py::arg("lr"))
    .def("zero_grad", [](tmg::AdamW& o) { py::gil_scoped_release nogil; o.zero_grad(); })
    .def("clip_grad_norm", [](tmg::AdamW& o, float max_norm) {
        py::gil_scoped_release nogil; return o.clip_grad_norm(max_norm);
    }, py::arg("max_norm"))
    .def_property_readonly("num_params", [](tmg::AdamW& o) { return o.num_params(); })
    .def_property_readonly("t", [](tmg::AdamW& o) { return o.t; });

mod.def("buffer_pool_stats", []() {
    auto& bp = buffer_pool();
    return py::dict(py::arg("bytes") = bp.bytes(), py::arg("hits") = bp.hits(),
                    py::arg("misses") = bp.misses(), py::arg("enabled") = bp.enabled);
});
mod.def("set_lr_unused", []() {});
mod.attr("__version_tape__") = "1";
}
