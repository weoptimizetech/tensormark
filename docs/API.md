# TensorMark API reference

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with: python3 tensormark/gen_api_docs.py
     Checked in CI:   python3 tensormark/gen_api_docs.py --check -->

This is the reference for the Python layer's **public surface only**. It is
generated from the source, so a signature here is the signature the code has.

Two things it deliberately does not claim:

* **It is not the full PyTorch API.** `tensormark.torch` implements a subset,
  chosen so that a training loop written against PyTorch runs unchanged. Names it
  does not implement raise the ordinary `AttributeError`.
* **It is not the engine.** None of the throughput in the README travels through
  this layer; that is the C++ engine's job, and its entry points are the binaries
  `tensormark/build_llama.sh` produces.

Every symbol below is listed in `PUBLIC` in `tensormark/gen_api_docs.py`.
Anything not listed there is an implementation detail and may change without
notice.

## What this layer promises

A limited PyTorch-style compatibility layer backed by TensorMark.

Use explicitly with ``import tensormark.torch as torch``. TensorMark complements
PyTorch for a narrower set of workloads; arbitrary PyTorch programs may need
changes or unsupported operations and are not guaranteed to run.

The supported surface includes from_numpy/rand/randn/zeros, Linear, ReLU,
Sequential, MSELoss, SGD, Adam, parameters/zero_grad, loss.backward/item, and
no_grad. Tensor values use TensorMark's autograd tape. This is not the full
PyTorch API: ones, general tensor addition/subtraction and the @ operator are
not implemented. Unsupported attributes use Python's normal AttributeError.

## Tensors

### `Tensor(v)`

*class*

Wraps one tensormark Value (leaf param, const, or op result).

### `Tensor.shape`

*property*

The value's shape, as a tuple.

### `Tensor.numpy()`

*method*

The data as a NumPy array view — not a copy.

This is the engine's buffer, so writing to it mutates the tensor. That is
what makes in-place updates (``param.data -= lr * g``) work here.

### `Tensor.item()`

*method*

The first element as a Python float — the loss-scalar case.

### `Tensor.data`

*property*

A Tensor sharing this tensor's storage.

Assigning to it is accepted and ignored: ``param.data -= x`` mutates in
place through ``__isub__`` and then rebinds the identical wrapper.

### `Tensor.grad`

*property*

The gradient as a NumPy array (not a Tensor), set by ``backward()``.

### `Tensor.backward()`

*method*

Run the autograd tape backward from this tensor.

The tape covers one forward pass: ``nn.Sequential.forward`` clears it on
entry, so gradients are read before the next forward.

### `from_numpy(a)`

*function*

Wrap a NumPy array as a constant. Copied to contiguous float32;
gradients do not flow back into the original array.

### `zeros(*shape, device=None, requires_grad=False)`

*function*

A zero-filled tensor of the given shape.

### `rand(*shape, device=None, requires_grad=False)`

*function*

Uniform samples in [0, 1) of the given shape, ``torch.rand`` subset.

### `randn(*shape, device=None, requires_grad=False)`

*function*

Standard-normal samples of the given shape, ``torch.randn`` subset.

``requires_grad=True`` makes a parameter (a training target); otherwise the
result is a constant. ``device`` is accepted and ignored.

## Reproducibility and devices

### `manual_seed(seed)`

*function*

Seed the RNG this layer draws from: NumPy's.

Layer initialisation (``nn.Linear``) draws from the same generator, so this
seeds a whole model, not just the factory functions.

### `device(spec)`

*function*

torch.device shim: tensormark is CPU-only; 'cpu' passes, else error.

### `no_grad`

*class*

Context manager that turns taping off for the block (``torch.no_grad``
subset), restoring the previous setting on exit.

## Neural network layers

### `nn.Module`

*class*

Base class for layers: subclasses define ``forward`` and hold their
parameters as ``Tensor`` attributes.

### `nn.Module.parameters()`

*method*

Every parameter, recursively — ``Tensor`` attributes whose value
has ``requires_grad``, plus those of nested modules.

### `nn.Module.zero_grad()`

*method*

Clear every parameter's gradient.

### `nn.Module.to(device=None, *a, **kw)`

*method*

No-op: this layer is CPU-only. Still validates, so
``model.to("cuda")`` raises rather than appearing to succeed.

### `nn.Linear(in_features, out_features, bias=True)`

*class*

Fully connected layer: ``y = x @ W.T + b``.

``weight`` is ``(out_features, in_features)``, initialised uniformly in
``±1/sqrt(in_features)`` (torch's default); ``bias`` is zeros, or omitted
with ``bias=False``.

### `nn.Linear.forward(x)`

*method*

Apply the projection through the engine's ``matmul_nt``.

### `nn.ReLU`

*class*

Elementwise ``max(0, x)``.

### `nn.Sequential(*mods)`

*class*

Applies the given modules in order.

Clears the autograd tape at the start of each forward: one pass is one
tape, so a second forward overwrites the first's gradients rather than
accumulating into them.

### `nn.MSELoss(reduction='mean')`

*class*

Mean squared error. ``reduction`` is ``"mean"`` (default) or
``"sum"``; torch's other reductions are not implemented.

## Optimizers

### `optim.SGD(params, lr, momentum=0.0)`

*class*

SGD with optional momentum. ``lr`` is required; ``momentum`` defaults
to 0.0 (plain SGD). Updates are applied to parameter storage outside the
tape, so call ``zero_grad()`` each step.

### `optim.Adam(params, lr, betas=(0.9, 0.999), eps=1e-08)`

*class*

Adam with bias correction. ``betas`` and ``eps`` are configurable;
weight decay and amsgrad are not implemented.

---

0 public symbols are still undocumented: none.
