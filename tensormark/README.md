# TensorMark

A focused C++23 neural-network engine with Python bindings and a small,
PyTorch-compatible API, developed by [WeOptimize](https://weoptimizetech.com).

TensorMark complements PyTorch: it explores a narrower Apple Silicon niche
while retaining familiar interfaces for supported operations. PyTorch remains
the broader ecosystem and a valued numerical reference, not something this
project aims to replace. Compatibility is a documented subset, not the full
PyTorch API; benchmark results below describe specific experiments only.

## Install (any machine with AWS CLI credentials to the WeOptimize account)

    aws codeartifact login --tool pip \
        --domain wmark --domain-owner 767397993071 --repository wmark-pypi
    pip install tensormark

## Reuse in other repos

The installed wheel contains the compiled extension. Building from source
requires the **complete repository checkout**: the extension includes
`../neural_demo.cpp` as well as local headers. From the repository root, run
`python -m pip install ./tensormark`; pip builds with pybind11 and links Apple's
Accelerate framework. No separate `build.sh` step is required. A copied package
subdirectory alone is insufficient. A source archive created with
`python -m build --sdist tensormark` stages the full native dependency tree and
has been tested by building a wheel outside the checkout.

From source (wheel lands in `dist/`, install with `pip install dist/*.whl`):

    pip3 wheel . --no-deps -w dist

The build targets the build host's CPU (`-march=native`). For a wheel that
must run on a different M-series machine, set the architecture:

    TENSORMARK_ARCH=apple-m1 pip3 wheel . --no-deps -w dist

Current publishing channel is internal CodeArtifact (`wmark/wmark-pypi`):
twine upload --repository codeartifact dist/tensormark-*.whl
(after `aws codeartifact login --tool twine ...`).

The owner approved an open-source company GitHub launch on 2026-09-08;
[publication gates](../specs/TENSORMARK_OPEN_SOURCE_SPEC.md) are in preparation.
Public PyPI distribution is not yet available or promised by these commands.

Consumers use the stable Python surface `import tensormark as tm` — every
symbol the native extension defines is re-exported at package level
(`tensor`, `param`, `matmul`, `tape_*`, `Sequential`, ...), and the pure
Python GPT-2 runtime ships as `tensormark.gpt2_runtime` (weights are NOT
in the wheel; point `TENSORMARK_DATA` at a directory containing `gpt2/`).
`tm.__version__` tracks the native engine version. The wheel is ABI-tagged
per Python version; each interpreter needs its own wheel.

Canonical artifact (no public PyPI index): the wheel lives in this repo at
`tensormark/dist/tensormark-<ver>-cp314-cp314-macosx_<ver>_arm64.whl` and on
the internal CodeArtifact index (`wmark/wmark-pypi`) once released. It is
built for CPython 3.14 / macOS arm64 (Apple Silicon) only. Other Python versions
need their own build and validation; the declared Python range is not a tested
matrix. The current build uses Darwin-specific flags and Accelerate, so
Linux/Windows require porting, not just rebuilding a wheel.

## Quick start

```python
import tensormark as tm
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([0,1,1,0], dtype=np.int64)

model = tm.Sequential()
model.dense(2, 16).relu().dense(16, 2)
model.init()
model.manual_seed(7)
result = model.fit(X, y, batch=4, epochs=400, lr=0.08)
print(result)   # {'loss': 0.0007, 'accuracy': 1.0, ...}

logits = model.forward(X)          # (C, B) column-per-sample, engine convention
pred = logits.T.argmax(axis=1)     # [0 1 1 0]
```

## Status (tier 1)

- `Sequential` builder: `dense`, `conv2d`, `batchnorm2d`, `batchnorm1d`,
  `relu`, `maxpool2d`, `flatten`, `dropout`
- `fit()` with warmup+cosine LR schedule, label smoothing, engine-side
  Adam + weight decay, GIL released during train/eval steps
- `forward()` inference; `manual_seed`; `num_params`
- numpy interop: owned-copy views (zero-copy write-through lands in tier 2)

## BlazeFace inference benchmark (Apple M-series, fp32)

Running the hollance/BlazeFace-PyTorch MediaPipe front model (37 conv/dw
layers, 128×128 input, weights ported from the official TFLite checkpoint):

| Stack | forward latency (median) |
|---|---|
| TensorMark (1x1-conv GEMM fast path, NEON depthwise) | ~5.0–5.7 ms |
| PyTorch eager CPU, same machine | ~4.6–6.4 ms |

Day-to-day machine state shifts both numbers (TensorMark measured 8.25 ms,
PyTorch 5.97 ms in an earlier session; 10.8 vs 4.94 ms in a later one) —
the ratio is the stable part. The 2026-09-02 kernel session took
TensorMark from 7.79 ms to ~5.0–5.7 ms (PyTorch 4.59 ms same-session
start-of-session): the gap is closed to parity/within noise. Outputs are
verified against the PyTorch reference every run (1 detection on Lena,
score 0.923, matching box and keypoints); gradchecks pass (incl. a
dedicated 1x1-path gradcheck, `test_conv1x1_gradcheck.cpp`).

What produced the gain (all algorithmic, not parameter tuning):
- **1x1 stride-1 conv fast path**: the entire BlazeFace backbone is 1x1
  convs + 3x3 depthwise (weight-shape audit). In NCHW B=1, the input
  (C,HW) IS the im2col matrix and the GEMM output (O,HW) IS NCHW — one
  cblas_sgemm, no im2col copy, no (R,O)->NCHW permute. conv 4.09 -> 1.4 ms.
- **NEON depthwise** (4-wide shifted loads + FMA per row): dw 2.99 -> 2.6 ms.
- Measured dead ends: 2-row NEON dw sharing input-row loads (slower:
  register pressure, dw 2.95–3.23 ms — reverted); fusing pad/bias/relu
  into the Python interpreter loop (numpy elementwise ops are
  multithreaded-SIMD); fused `conv2d_ex`/`depthwise_conv2d_ex` measured
  only break-even vs the plain kernel + numpy-glue pipeline.

### Autotune ("wisdom") pass — 2026-09-03

The engine exposes two tunable knobs (`tm.set_param`):
`gemm_par_threshold` (FLOPs below which a GEMM runs single-threaded) and
`dw_chunks_mult` (depthwise plane-task multiplier). `bench_autotune.py`
sweeps them against the real BlazeFace op mix, verifies outputs are
bit-identical, and the winners are now the compiled defaults
(1<<24 / 4). Paired same-session medians: **TensorMark 5.47-5.66 ms vs
PyTorch eager 5.85-5.90 ms** — TensorMark is ahead on medians; PyTorch
keeps a best-case edge (min 4.05 vs our 4.92 across sessions). Within
machine noise; call it parity-to-slight-lead, not a blowout.

Remaining headroom (measured, not guessed): batched pool dispatch +
buffer arena (see BFGraph profiling below), and extending the autotune
sweep to more knobs.

### The single-call graph executor: built, measured, slower (kept as an
### experiment)

`BFGraph` (C++ side of the extension, `bench_bfgraph.py` builds the full
94-op BlazeFace plan into it) runs the entire network in ONE C++ call:
no Python glue, weights prepacked once, NCHW end-to-end, one GIL release.
Result: **9.4 ms vs 5.4 ms for the op-by-op pipeline** — the Python glue
it eliminates was nearly free, and the engine's per-op fork-join pool
syncs are not.

`sample` profiling of the executor shows the main thread spending >50%
of samples in pool wait/sync (`__psynch_cvwait` ~20k, `__psynch_mutexwait`
~4k, `traink::Pool::worker` 8k samples per 6 s), plus `Tensor::zeros`
allocation churn per op. 94 ops × 1 fork-join each, back-to-back with
nothing between them, is strictly worse than the pipeline where cheap
numpy ops fill the gaps. The fix is engine-level: batch whole op chains
into single pool dispatches and reuse output buffers (arena) — a larger
change, deliberately not half-done here. The executor stays in the tree,
correct and env-gated-instrumented (`TM_BFDEBUG=1` emits per-op timings).

## M1 — a char-level GPT trains inside the engine

`minigpt.py` trains a GPT-2-shaped character model on tiny-Shakespeare
entirely on the tape: embeddings are gathers, attention is one fused causal
kernel, the lm_head is a transposed matmul against the tied token table, and
AdamW runs in C++. numpy shuffles indices and holds the checkpoint; the engine
does every FLOP. Spec: [TENSORMARK_MINIGPT_SPEC](../specs/TENSORMARK_MINIGPT_SPEC.md).

```
tensormark/run_m1_acceptance.sh 30      # numerics + benches + a 30-minute run
python3 tensormark/minigpt.py train --minutes 30
python3 tensormark/minigpt.py sample --prompt $'ROMEO:\n' --tokens 400
```

### What the milestone added to the engine

| Piece | Why it exists |
|---|---|
| `mha_causal` (`attention.h`) | scores → mask → softmax → AV as one op. Scores are produced in row tiles, so a tile of rows [i0,i1) only needs columns [0,i1) and the masked triangle is never multiplied (0.63x the dense FLOPs at ctx 256). Heads are strided GEMM views of the (B·T, D) tensor — no packing, no permute, no per-head tensor. |
| `add_rowvec`, `matmul_nt`, `dropout` | the rest of the GPT-2 block. `matmul_nt` is what makes a tied lm_head one GEMM against the embedding table; it splits the long axis, because the vocab axis is 65 wide and splitting it hands each core a sliver. |
| `AdamW` (`optim.h`) | decoupled decay, bias correction, global-norm clip, engine-side: the update touches 19 MB per step and has no business crossing into Python. |
| Accelerate vector math in the elementwise ops | gelu / layernorm / softmax / cross-entropy / add / scale / gather / slice were scalar single-threaded loops. They now run vDSP and vForce over pool chunks, with the staging buffer blocked to stay in L1. |
| `BufferPool` under `Tensor` | freeing a large vector hands its pages back to the OS, so the next step re-faults every one of them. Buffers are now recycled by exact element count; a training loop cycles a handful of shapes, so the hit rate is ~85%. |

### Numerics

Every op and the whole model are checked against PyTorch, and the composition
is checked against finite differences — never against "looks right":

| Check | Result |
|---|---|
| `test_attention_gradcheck.cpp` — fused attention vs a naive loop reference at every tile height, plus central differences on Q/K/V. No PyTorch, no Python | pass, forward maxerr 6e-8, gradients at 0.22x of tolerance |
| `check_ops_torch.py` — attention, block ops, vectorised elementwise ops, forward **and** backward vs PyTorch | all pass, worst maxerr 3.1e-6 |
| `check_minigpt_torch.py` — whole model vs an identical PyTorch GPT: logits, loss, all 52 parameter gradients | pass, logits maxerr 1.5e-7 |
| `test_minigpt_gradcheck.py` — central differences over all 1,896 parameters of a 2-layer config, through the real training path | pass |
| attention tile height 1 / 7 / 16 / 64 / 256 | same output to 5e-7, i.e. float reassociation only — the tile is a pure performance knob |

### The acceptance run

5,915 steps — the budget a 30-minute run was estimated to allow, sized from a
step time measured under load — at batch 16 x ctx 256, lr 2e-3 cosine to 2e-4
after 100 warmup steps, dropout 0.1, AdamW with decoupled decay 0.1 and
global-norm clip 1.0:

| | |
|---|---|
| Model | 8 layers, 4 heads, d=128, ctx 256, tied lm_head — 1,627,520 parameters |
| Tokens seen | 24,227,840 (5,915 steps x 4,096) |
| Validation loss | **1.4852** (spec gate: < 1.6), perplexity 4.42 on 20 held-out batches |
| Training loss | 1.1792 |
| Wall clock | 44.0 min, against a 5,915-step budget sized for 30 min — the laptop was running other projects throughout (mean 5.27, peak 7.68 foreign cores of 8), and the step budget itself was under-sized by that contention (see below) |

Validation loss by step: 2.44 (250), 1.98 (750), 1.72 (1500), 1.59 (2500),
1.52 (3750), 1.4852 (5915). It crossed the gate at step 2500 and was still
falling when the schedule ended.

### What it writes

Sampled at temperature 0.8, top-k 40, from the prompt `ROMEO:`:

```
ROMEO:
And I let me spirit 'tis not so much pretty.

ROMEO:
If this some through I am the sword,
Or required that their fields with great point;
And let me foul roze the rosal thou remember thy
Of thy life love to meet to be deliver
Than thou till find the noble charm of no thy time
```

Speaker tags, line breaks and near-words in the right places at 1.6M
parameters. The spec asks for "no degenerate repetition", so measure it
rather than admire it (`minigpt.py sample --stats`):

| Decoding | char-distribution distance | distinct 4-grams | longest block repeat |
|---|---|---|---|
| greedy (argmax) | 0.408 | 0.182 | 3x |
| temperature 0.8, top-k 40 | 0.164 | 0.874 | 2x |

**Greedy decoding does collapse** — "The state of the state of the state" —
and the numbers say so: 18% of its 4-grams are distinct. That is a property
of argmax decoding at this scale, not of the trained model: the same
checkpoint sampled with temperature stays at 87% distinct 4-grams and lands
within 0.164 of the corpus's character distribution. Reported as measured
rather than by picking the flattering decoder.

Generation runs at 154 tokens/s at ctx 256 with no KV cache (the whole window
is recomputed per token — that is exactly what M2 fixes).

### Measuring on a shared laptop

This machine also builds other projects. `benchguard.py` samples system-wide
CPU ticks, subtracts this process's own usage, and labels any run that saw
more than 0.75 foreign cores as CONTENDED — the benchmarks wait for an idle
machine before they start, and say so when they could not get one.

That mattered here: for the whole of this session the laptop sat at two to
seven cores of other work (another project's vitest suite, then clang, then
system daemons), confirmed against `top` and the load average. The acceptance
run's wall clock is therefore inflated, and its step budget was sized from a
step time measured during that contention — which, as the last note below
records, under-counted what 30 minutes actually buys.

Everything below is labelled with the foreign load it saw. The lowest this
laptop reached was ~1.7 cores of system background, so nothing here clears the
0.75-core bar the harness calls clean, and none of it is presented as a
settled claim. `run_m1_acceptance.sh` re-runs the lot on an idle machine.

**Paired training step vs PyTorch eager CPU.** Identical model and shapes,
interleaved step by step so drift hits both stacks equally. Measured twice, at
two very different load levels, and the pair is more interesting than either
number:

| foreign cores | tensormark median | PyTorch median | tensormark min | PyTorch min |
|---|---|---|---|---|
| 1.71 | **188.0 ms** | 206.6 ms | 183.5 ms | 203.5 ms |
| 5.91 | 681.4 ms | 519.4 ms | 423.6 ms | 372.4 ms |

On a near-quiet machine tensormark is **1.10x ahead** on a full transformer
training step — forward, backward and optimiser, 4,096 tokens per step. Under
six cores of someone else's work it is 1.31x behind. That inversion is a real
property worth recording rather than a measurement artifact: PyTorch ran on 4
threads and the engine's pool on 8, so when six of eight cores belong to
someone else the wider fork-join loses more. It is the same fork-join cost the
BFGraph profiling found, showing up from a different direction.

**Per-op rates** at the acceptance shapes (4,096 rows, d=128, 4 heads, 2.2
foreign cores):

| op | median ms | rate |
|---|---|---|
| matmul (4096,512)x(512,128) | 0.751 | 715 GFLOP/s |
| matmul (4096,128)x(128,512) | 0.840 | 639 GFLOP/s |
| matmul (4096,128)x(128,128) | 0.367 | 366 GFLOP/s |
| matmul_nt tied lm_head | 0.198 | 344 GFLOP/s |
| mha_causal B16 T256 H4 | 1.171 | 230 GFLOP/s |
| add | 0.091 | 69 GB/s |
| dropout | 0.124 | 51 GB/s |
| add_rowvec | 0.086 | 49 GB/s |
| gelu | 0.653 | 26 GB/s |
| gather rows | 0.185 | 23 GB/s |
| layernorm | 0.216 | 19 GB/s |

The GEMMs sit at Accelerate's ceiling. The headroom is in the elementwise
ops, and it is written down rather than guessed at: `gelu` moves 26 GB/s where
`add` gets 69, because it makes three passes and a vForce call per block
instead of one fused NEON pass. `layernorm` and `gather` are the same shape of
problem. Together that is roughly 8% of a training step, waiting for M3.

**Knob sweep** (`autotune_gpt.py`, 24 combinations of attention tile height,
elementwise parallel threshold and buffer pool). The load-independent result
is the one that matters: **every combination produced bit-identical logits**
(max difference 0.00e+00), which is exactly what a performance knob is
supposed to do. Masked score entries are exactly zero, so changing the tile
height only changes how many exact zeros a GEMM accumulates. Timings spanned
9% and did not rank consistently under load, so the shipped defaults stand
until the sweep can be re-run idle.

**One thing the load cost us.** The acceptance run's schedule was sized from a
step time of 283 ms measured during the worst contention. On a near-quiet
machine the same step takes ~188 ms, so 30 minutes is about 9,400 steps and
38M tokens, not the 5,915 steps and 24.2M tokens the run actually took. The
1.6 gate was cleared with roughly two thirds of the budget the machine
actually allows.

## Roadmap

Where the engine goes next — from parity with PyTorch to running open-source
LLMs on the M1 8 GB: see [docs/ROADMAP.md](../docs/ROADMAP.md) (M1 mini-GPT
-> M2 GPT-2 runtime -> M3 perf floor -> M4 quantization -> M5 Llama family ->
M6 optional Metal), with a dedicated SPEC per milestone in `specs/`.

### Metal (optional, M6) and the device policy

With the Metal backend built (`bash build_metal.sh`; requires Xcode/CLT) the
runtime has GPU prefill (fused Q4_0-dequant GEMM, weights kept as raw `.tmq`
blocks) and GPU decode (single command buffer per token, NoCopy weights over
the mmap). Without `TM_HAVE_METAL` the engine is unchanged: every call falls
back to the CPU path.

**Defaults since 2026-09-06 (measured, TinyLlama Q4, same-window pairs):**

| path | quiet | under 4 foreign compute threads |
|---|---:|---:|
| CPU AMX prefill, 512@512 | 321-333 t/s | 112 t/s |
| GPU prefill | 246 t/s | 205 t/s |
| CPU decode (SDOT, spin, batched attention) | 67-70 t/s | 25 t/s |
| GPU decode (SIMD16) | 58 t/s | 53 t/s |

The CPU paths win quiet; the GPU paths keep 83-92% of their speed under load
where the CPU paths keep 35-37%. So with `TM_PREFILL_GPU` / `TM_DECODE_GPU`
unset, a Metal build resolves at model load: **GPU for both when the host is
already busy** (ambient >= `TM_DECODE_GPU_AUTO_BUSY` cores, default 1.0) and
the Q4 weights fit `TM_DECODE_GPU_AUTO_MAX_GB` (default 2.0 — 7B stays on the
CPU until its guarded GPU gate closes), **CPU otherwise**. Explicit `1`/`0`
force a path (explicit `TM_PREFILL_GPU=1` with `TM_DECODE_GPU=1` still fails
fast: copied prefill blobs plus the NoCopy decode mapping double the 7B's
footprint). `TM_DEBUG_POOL=1` logs the decision. Gates (`test_llama_quant`,
`check_prompt_amx`) pin the CPU paths when the variables are unset.

### AMX prefill (CPU, default)

Quantized prefill projections run through `cblas_sgemm` over weights
dequantized on the fly into a reused fp32 scratch — Accelerate dispatches to
Apple AMX — by default (`TM_PREFILL_AMX=0` or `set_prefill_amx(false)` restores
the per-token qgemv loop, 84 t/s). 320-333 t/s at 512@512 on TinyLlama-1.1B
Q4, +60 MB RSS, max |dlogit| 3.2e-05 vs the qgemv loop with identical argmax
(`./build/check_prompt_amx`, which pins `TM_SDOT=0` for its fp32 reference).
Attention scores/V-weighting are batched into sgemms per GQA group for
prefill and decode alike (`TM_ATTN_NAIVE=1` restores the per-head loop), with
the softmax rows on vDSP/vForce (`TM_ATTN_VSOFTMAX=0` for the scalar loop).
See the M5 spec's dated sections for every measurement.

### 7B class (OpenHermes-2.5-Mistral-7B Q4)

The pipeline now runs 7B-class checkpoints on the 8 GB target: config.json
drives the architecture (layers/heads/KV heads/intermediate/vocab — no
per-model C++), and the converter reads bf16 safetensors shard-by-shard
(one tensor in RAM; the old fp32-re-save step needed 26 GB). Convert and
generate:

    ./build/convert_tmq "$SNAP/model-00001-of-00002.safetensors,$SNAP/model-00002-of-00002.safetensors" \
        data/openhermes7b/openhermes7b_q40.tmq q40 asis
    ./build/gen7b data/openhermes7b/openhermes7b_q40.tmq $SNAP/tokenizer.model "The capital of France is" 12

Honest 8 GB caveat: the 7B working set is ~4.7 GB (3.8 GB Q4 weights +
0.54 GB GQA KV + scratch). It RUNS (gates: dlogit 5.9e-05, argmax
identical; coherent greedy text), but on a machine whose other apps force
swap, macOS evicts the mmap'd weight pages and decode re-faults ~3.8 GB
from disk per token (measured 40 s/token with 2.4 GB swap in use — see
`benchguard.swap_used_mb()` and the M5 spec's memory-cliff section).
Resident decode (~expected near the >=8 t/s stretch bar) needs a quiet
machine; the load guard now flags both foreign CPU and swap.

**llama.cpp baseline — never run `llama-cli`.** Homebrew llama.cpp 0.4.0's
`llama-cli` is the interactive chat REPL: detached with stdin at
`/dev/null` it busy-loops on EOF forever without loading the model (a
`nohup llama-cli ...` burned a core for 18 h with an empty log on
2026-09-05). One-shot generation goes through `./llamacpp_completion.sh`
(wraps `llama-completion -no-cnv`, closes stdin, hard watchdog, non-zero
exit on timeout); throughput goes through `llama-bench` as in
`bench_llamacpp_realistic.py`. An empty log after the expected runtime
(minutes, not hours) is a bug to diagnose with `sample <pid>`, not a
reason to retry; reap every process you start before finishing.

## Engine

C++23, zero third-party dependencies, Accelerate/BLAS-backed GEMMs,
persistent fork-join thread pool. See `REPORT.md` in the engine repo.
