# TensorMark

A focused C++23 neural-network engine with Python bindings and a small,
PyTorch-compatible API, built by [WeOptimize](https://weoptimizetech.com) for
Apple Silicon.

TensorMark **complements** PyTorch: it explores a narrower Apple Silicon niche
while keeping familiar interfaces for the operations it supports. PyTorch
remains the broader ecosystem and a valued numerical reference — not something
this project replaces. Compatibility is a documented subset, not the full
PyTorch API, and the benchmark numbers describe specific measured workloads
only.

**The repository root [`README.md`](../README.md) is the main document.** It
carries the Performance section first — the measured numbers with their machine
and protocol — then the Quick start, the PyTorch/llama.cpp/Ollama migration
guides, and the LLM inference engine. This file covers only installing and using
the Python package.

## Install

Public PyPI distribution is not available as part of this source launch; install
from the repository checkout.

Installing the package requires the **complete repository checkout** — the
native extension compiles `../neural_demo.cpp` alongside the package headers, so
a copied package subdirectory alone is insufficient:

```bash
git clone https://github.com/weoptimizetech/tensormark.git
cd tensormark
python3 -m venv .venv && source .venv/bin/activate
python -m pip install ./tensormark
```

pip builds the extension with pybind11 and links Apple's Accelerate framework;
no separate build step is required. The public CI installs exactly this way.

To build a distribution instead — the sdist stages the full native dependency
tree, so it can be built into a wheel outside the checkout. Both commands run
from the package directory, where `pyproject.toml` lives:

```bash
cd tensormark
python -m pip install -U build wheel 'setuptools>=77' pybind11
python -m build --sdist --no-isolation
```

The build targets the build host's CPU. For a wheel that must run on a different
M-series machine, set the architecture explicitly:

```bash
TENSORMARK_ARCH=apple-m1 python -m pip wheel . --no-deps -w dist
```

The extension is built against the interpreter's own ABI, so use the artifact
produced by the Python version you are targeting. The current build uses
Darwin-specific flags and Accelerate: Linux and Windows need porting, not just a
rebuild.

## Python surface

```python
import tensormark
```

The package re-exports the full native API — tensor ops, layers, optimizers, and
autograd entry points — so `import tensormark` exposes everything the extension
defines, including `tensor`, `param`, `matmul`, `Sequential`, `relu`, and
`softmax`. `tensormark.__version__` tracks the native engine version.

For the PyTorch-shaped entry point (`import tensormark.torch as torch`) and its
measured fine-tuning numbers, see the root [`README.md`](../README.md). The
GPT-2 fine-tuning utilities ship in the same package as `tensormark.gpt2_sft`
and `tensormark.train_sft`.

## Scope and license

Validated on Apple Silicon macOS with CPython 3.10+. Model weights and datasets
are not bundled.

WeOptimize-owned code is licensed under [Apache-2.0](LICENSE); see
[NOTICE](NOTICE). Separately identified third-party materials retain their
applicable terms.
