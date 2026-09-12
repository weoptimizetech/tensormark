"""A limited PyTorch-style compatibility layer backed by TensorMark.

Use explicitly with ``import tensormark.torch as torch``. TensorMark complements
PyTorch for a narrower set of workloads; arbitrary PyTorch programs may need
changes or unsupported operations and are not guaranteed to run.

The supported surface includes from_numpy/rand/randn/zeros, Linear, ReLU,
Sequential, MSELoss, SGD, Adam, parameters/zero_grad, loss.backward/item, and
no_grad. Tensor values use TensorMark's autograd tape. This is not the full
PyTorch API: ones, general tensor addition/subtraction and the @ operator are
not implemented. Unsupported attributes use Python's normal AttributeError.
"""
import numpy as np
import tensormark as _tm

_GRAD = {"on": True}


def _val(t):
    return t._v if isinstance(t, Tensor) else t


def _np(t):
    if isinstance(t, Tensor):
        return t._v.view
    return np.asarray(t, dtype=np.float32)


class Tensor:
    """Wraps one tensormark Value (leaf param, const, or op result)."""

    def __init__(self, v):
        self._v = v

    # ---- shapes / conversion ------------------------------------------------
    @property
    def shape(self):
        return tuple(self._v.shape)

    def numpy(self):
        return self._v.view

    def item(self):
        return float(self._v.numpy()[0])

    def backward(self):
        # torch autograd call -> tensormark tape backward on this Value.
        _tm.tape_backward(self._v)

    @property
    def data(self):
        return Tensor(self._v)

    @data.setter
    def data(self, v):
        # `param.data -= x` mutates the view in place via Tensor.__isub__ and
        # then reassigns the (identical) wrapper; the assignment is a no-op.
        pass

    @property
    def grad(self):
        # tensormark exposes the gradient as a numpy array view.
        return self._v.grad

    # ---- arithmetic (no_grad style, through writable numpy views) ----------
    def __mul__(self, other):
        return Tensor(_tm.scale_val(self._v, float(other))) if not isinstance(
            other, Tensor) else NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __isub__(self, other):
        v = self._v.view          # property has no setter; mutate in place
        v -= _np(other)
        return self

    def __iadd__(self, other):
        v = self._v.view
        v += _np(other)
        return self

    def __repr__(self):
        return f"Tensor(tm.Value{self._v.shape})"


# ---- factory functions -------------------------------------------------------

def randn(*shape, device=None, requires_grad=False):
    a = np.random.randn(*shape).astype(np.float32)
    return Tensor(_tm.param(a) if requires_grad else _tm.const_(a))


def rand(*shape, device=None, requires_grad=False):
    a = np.random.random_sample(shape).astype(np.float32)
    return Tensor(_tm.param(a) if requires_grad else _tm.const_(a))


def zeros(*shape, device=None, requires_grad=False):
    a = np.zeros(shape, dtype=np.float32)
    return Tensor(_tm.param(a) if requires_grad else _tm.const_(a))


def device(spec):
    """torch.device shim: tensormark is CPU-only; 'cpu' passes, else error."""
    s = getattr(spec, "type", str(spec))
    if s != "cpu":
        raise ValueError(f"tensormark.torch is CPU-only; got device {spec!r}")
    return spec


def manual_seed(seed):
    np.random.seed(int(seed))


def from_numpy(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return Tensor(_tm.const_(a))


# ---- autograd ----------------------------------------------------------------

class no_grad:
    def __enter__(self):
        _GRAD["prev"] = _tm.is_grad_enabled()
        _tm.set_grad_enabled(False)

    def __exit__(self, *exc):
        _tm.set_grad_enabled(_GRAD.get("prev", True))


# ---- nn ----------------------------------------------------------------------

class nn:
    class Module:
        def parameters(self):
            return [p for _, p in self._named()]

        def named_parameters(self):
            return self._named()

        def zero_grad(self):
            for p in self.parameters():
                p._v.zero_grad()

        def __call__(self, *args, **kw):
            return self.forward(*args, **kw)

        def to(self, device=None, *a, **kw):
            # CPU-only shim: .to() is a no-op (validates the device string).
            if device is not None:
                globals()["device"](device)
            return self

        def _named(self, prefix=""):
            out = []
            for k, v in vars(self).items():
                items = v if isinstance(v, (list, tuple)) else [v]
                for i, item in enumerate(items):
                    name = k if len(items) == 1 else f"{k}{i}"
                    if isinstance(item, Tensor) and item._v.requires_grad:
                        out.append((prefix + name, item))
                    elif isinstance(item, nn.Module):
                        out.extend(
                            (f"{prefix}{name}.{n}", p)
                            for n, p in item.named_parameters())
            return out

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            # torch: weight (out,in), y = x @ W.T + b
            lim = 1.0 / in_features ** 0.5
            w = np.random.uniform(-lim, lim, (out_features, in_features))
            w = w.astype(np.float32)
            self.weight = Tensor(_tm.param(w))
            self.bias = Tensor(_tm.param(np.zeros(out_features, np.float32))) \
                if bias else None

        def forward(self, x):
            y = _tm.matmul_nt(_val(x), self.weight._v)
            if self.bias is not None:
                y = _tm.add_rowvec(y, self.bias._v)
            return Tensor(y)

    class ReLU(Module):
        def forward(self, x):
            return Tensor(_tm.relu(_val(x)))

    class Sequential(Module):
        def __init__(self, *mods):
            self.mods = list(mods)

        def forward(self, x):
            if _GRAD["on"]:
                _tm.tape_clear()          # one pass == one tape
            for m in self.mods:
                x = m(x)
            return x

    class MSELoss(Module):
        def __init__(self, reduction="mean"):
            assert reduction in ("mean", "sum"), \
                "shim MSELoss supports mean/sum"
            self.sum = reduction == "sum"

        def forward(self, pred, target):
            n = 1
            for d in pred.shape:
                n *= d
            mse = _tm.mse_vals(_val(pred), _val(target))
            # mse_vals is mean((a-b)^2); torch sum-reduction scales by numel.
            return Tensor(_tm.scale_val(mse, float(n) if self.sum else 1.0))


# ---- optim -------------------------------------------------------------------

class optim:
    class Optimizer:
        def __init__(self, params, **kw):
            self.params = [p for p in params]
            self.state = {}

        def zero_grad(self):
            for p in self.params:
                p._v.zero_grad()

        def step(self):
            for p in self.params:
                self._update(p)

    class SGD(Optimizer):
        def __init__(self, params, lr, momentum=0.0):
            super().__init__(params)
            self.lr, self.momentum = lr, momentum

        def _update(self, p):
            g = _np(p.grad)
            if self.momentum:
                buf = self.state.setdefault(id(p), np.zeros_like(g))
                buf *= self.momentum
                buf += g
                g = buf
            v = p._v.view          # property has no setter; mutate in place
            v -= self.lr * g

    class Adam(Optimizer):
        def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
            super().__init__(params)
            self.lr, self.b1, self.b2, self.eps = lr, *betas, eps

        def _update(self, p):
            g = _np(p.grad)  # grad is a numpy view; step runs outside the tape
            st = self.state.setdefault(
                id(p), {"m": np.zeros_like(g), "v": np.zeros_like(g), "t": 0})
            st["t"] += 1
            t = st["t"]
            st["m"] = self.b1 * st["m"] + (1 - self.b1) * g
            st["v"] = self.b2 * st["v"] + (1 - self.b2) * g * g
            mh = st["m"] / (1 - self.b1 ** t)
            vh = st["v"] / (1 - self.b2 ** t)
            v = p._v.view          # property has no setter; mutate in place
            v -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---- module-level aliases ------------------------------------------------------

randn.__doc__ = "torch.randn subset"
