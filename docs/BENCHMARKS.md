# tensormark vs PyTorch on this M1 / 8 GB — measured comparison

**How to read this document.** There are exactly two comparisons here, and
every row states its **weight precision** and **device**:

1. **Design-point comparison** — each framework at its own production
   configuration. Precisions differ *by design* (tensormark's product is Q4_0
   serving; PyTorch's is fp16/fp32 on MPS). This answers "which framework
   serves this model faster as shipped."
2. **Precision-matched control** — the *same F32 weights* in both engines,
   *same device* (CPU), the only configuration both can serve at equal
   precision here. This answers "is the engine itself faster, given identical
   inputs" — and it cuts both ways.

All numbers were read programmatically from the `framework_compare` probe JSON
produced by `tools/bench_framework_compare.py`, with
`tools/build_benchmarks_table.py` doing the extraction — nothing is
hand-entered. The probe JSON itself is not bundled with this source edition, but
the harness is: re-running it regenerates the inputs. The raw sweep outputs that
*are* shipped sit alongside this file (`prefill_sweep_results.txt`,
`ane_*_results.txt`). Both sides receive identical fixed-seed token IDs.
Numerical equivalence is covered separately by the `test_llama_quant` oracle.

## 1. Design-point comparison

| Task | Framework | Precision | Device | Result |
|---|---|---|---|---|
| prefill 1,024 tok | tensormark Metal+CPU hybrid | **Q4_0** weights, fp16 compute | GPU+CPU | 1.131 s = **913.8 tok/s** (5 runs) — [fwc_pre.json] |
| decode 64 tok | tensormark GPU greedy chain | **Q4_0** weights, fp16 compute | GPU | 0.843 s = **75.9 tok/s** — [fwc_dec_full.json] |
| prefill 1,024 tok | PyTorch | **fp32** | MPS | **OOM**: MPS allocated 8.84 GiB of 9.07 GiB — [harness stderr] (dtype artifact of loading the BF16 checkpoint as fp32) |
| prefill 1,024 tok | PyTorch | **fp16** | MPS | **815.5 tok/s** (torch 2.14/py3.12; the torch 2.11/py3.14 stack crashes at the embedding op) |
| decode 64 tok | PyTorch cache loop | **fp16** | MPS | **10.65 tok/s**; per-step flat 96 ms, 0.2 GiB swap — clean run — [decode_attribution.json] |
| decode 64 tok | PyTorch cache loop | **fp32** | MPS | **0.05 tok/s**, 178.7 GiB swap-in in one leg — thrash collapse, not compute — [decode_attribution.json] |
| train 1.1 B | both | fp32 | — | **infeasible**: 16.4 GiB computed weights+grads+AdamW — [fwc_train.json] |

Verdict: tensormark wins both serving phases at design points — and the decode
gap decomposes exactly: 3.56× weight bytes (Q4_0 0.62 GB vs fp16 2.20 GB per
token) × 2.00× engine efficiency (tensormark 47.0 GB/s = 92% of the ~51 GB/s
achievable DRAM ceiling measured in that window; PyTorch 23.4 GB/s = 46%) =
7.13× observed. The cross-check tables quote a different decode window
(~53 GB/s), so the efficiency denominators are not shared across tables —
compare within a table.

## Precision-matched control — fp32 weights, CPU device

The only configuration both engines can serve at equal precision on this 8 GB
host (Metal path is Q4-only — see limitations; PyTorch fp32 on MPS thrashes).

| Task | tensormark (fp32/CPU) | PyTorch (fp32/CPU) | Verdict |
|---|---:|---:|---|
| prefill 256 tok | 3.41 s = 78.0 tok/s | 2.21 s = **117.3 tok/s** | **PyTorch 1.50× faster** |
| prefill 256 tok (2026-09-12, post gemm_w fp32 fix) | 0.560–1.484 s = 172–356 tok/s (best 309–356) | 0.828 s = 309.4 best-of | **parity at best-of; no stable winner — see protocol note** |
| decode 64 tok | **1.240 tok/s** (555 ms/step) | 0.90 tok/s (74 s/64) | tensormark ~1.4×; both memory-bound, noisy |

Source: [fair_cpu.json], [fair_cpu_pt.json]. **Measurement protocol note
(2026-09-12):** point comparisons on this host swing up to 3× with machine
state (thermal/ambient) — only back-to-back interleaved pairs are valid
(the re-measured row above is such a pair; an unpaired run earlier the same
day showed tensormark at 295-356 tok/s, which proved to be a variance
artifact, not a win). **The control cuts both ways:**
at equal precision on CPU, tensormark's prefill is 1.5-1.8× *slower* than
PyTorch's (oneDNN-class CPU GEMMs beat the Accelerate path; the sgemms measured
~2-5 effective GFLOPS at the time — the 2026-09-12 correction below shows that
reading was thermal, not architectural).
**Do not chase "the ~11% spent splitting the fused QKV output"** (earlier text
here): it was a misattribution. The `qkv_split` profile mark brackets the q/k/v
*projections*, not the unpack — on the 0.8B decode profile it reads 12.1%, and
the fused-output unpack itself is T contiguous copies (microseconds at any T).
The profiler's marks also overlap, so their percentages sum to the whole pass
and no single mark is a cost centre on its own.
**The CPU decode path is bandwidth-saturated, and that is the real limit:**
a 0.8B `.tmq` is 651 MB, decode streams essentially all of it per token, and the
measured 27.8 ms/pass is ≈23 GB/s effective — at the ~19 GB/s qgemv ceiling this
file documents elsewhere. Consistent with that: `TM_SDOT=0` (dequant + fp32 FMA)
*doubles* the pass (1191 vs 583 ms accounted), so the int8-dot path is already
the fast one, and vectorising the scalar fp16 GEMV changed nothing — an A/B
against the old scalar kernel in one binary, two interleaved rounds, identical
within noise and identical argmax, so the change was reverted rather than shipped.
Further instruction-level work on CPU decode has nothing to
win; the levers are the GPU lane, fewer weight bytes, or MTP speculation.
tensormark's
wins are where its engineering is: Q4 GPU chain decode (7×), hybrid 3-way
prefill, ANE integration, fixed memory footprint. There is no universal win.

**Correction (2026-09-12 evening):** the "~2-5 GFLOPS" reading above was
thermal-throttled data. A standalone microbench
(`bench_sgemm_shapes.cpp`) shows raw `cblas_sgemm` at the
exact prefill shapes running 870–1380 GFLOPS — AMX fully engaged; layout,
threading and QoS irrelevant. After the fp32 direct-sgemm fix
(`TM_GEMM_F32`), tensormark's best-of matches
PyTorch's best-of (~309 tok/s both; tensormark range 172–356 across
machine states, PyTorch 34–309). The fp32/CPU prefill cell is **parity at
best-of with no stable winner on this host**; the fixed 1.50×/1.8× rows
above are historical readings under unmatched machine states. Both engines
must be re-benchmarked on a quiesced host for a publishable verdict.

## Engine limitations found (documented, not hidden)

- **No PyTorch-on-Q4 row exists, and it cannot exist today** (verified
  2026-09-12, torchao 0.18 / torch 2.11 / transformers 4.57.6): the packed
  int4 weights refuse the MPS device move ("weight is on cpu but expected on
  mps", both packing formats), no fused int4 kernel ships for MPS — an
  unfused dequantize-then-GEMM is *slower* than fp16 (packed 0.62 GB + fp16
  2.2 GB read/write per token exceeds fp16's 2.2 GB) — and the CPU fused
  tinygemm path crashes at forward on this stack (five configurations
  attempted). So fp32/CPU really is the only equal-precision cell above;
  a hypothetical PyTorch with fused int4 would erase the 3.56× weight-byte
  term of the decode gap (7.13× → ~2×) and turn prefill into an open race.
- **Metal GPU path is Q4-only, with one exception.** Prefill and the resident
  greedy chain refuse non-Q4 weights ("requested backend did not execute
  exactly once; refusing fallback"). Decode now admits a non-Q4 **lm_head** by
  running that single projection on the CPU and leaving the rest of the model
  on the GPU (`gpu_head_cpu_`): on TinyLlama-1.1B Q4_0 with a
  Q8_0 lm_head — a model that previously stranded all 586 MB of Q4 weights to
  save 65 MB — decode went 33.0-49.9 t/s → 65.9-78.6 t/s (1.58-2.21x) with a
  bit-identical greedy sequence. Everything else still needs Q4. A
  same-precision fp16 GPU comparison — the cleanest possible engine-vs-engine
  measurement — requires fp16/F32 weight support in the Metal path. Engine
  work; requested.
- **PyTorch fp32 on MPS cannot run this model**: prefill OOMs (dtype
  artifact of OUR fp32 choice — the BF16 checkpoint loaded as fp32 doubles
  the weights; bf16/MPS is the natural config); decode thrashes (178.7 GiB
  swap-in). Only fp16 was viable on MPS here.
- **MPS bf16/fp16 prefill is currently broken on this machine's stack**
  (2026-09-12): "Placeholder storage has not been allocated on MPS device"
  at the embedding op — reproducible 6 ways (fp16 and bf16, our deq dir and
  the original HF snapshot, `low_cpu_mem_usage` both ways, `device_map='mps'`
  + accelerate, PYTORCH_MPS watermark workarounds). This is a stack-version
  regression: PyTorch fp16/MPS decode was successfully measured at 10.65
  tok/s on this same host earlier — torch 2.11 / python 3.14 /
  transformers 4.57.6 broke it since. Not a PyTorch limitation; re-bench
  when the fix lands.
- **PyTorch `from_pretrained` default on 8 GB stalls the host** (7 GB CPU-side
  RSS alongside the MPS copy): `low_cpu_mem_usage=True` + dropping the CPU
  copy is mandatory; tensormark's fixed-footprint workers lack this failure
  mode.
- **MPS fp16 prefill at 1,024-token contexts ran minutes-per-forward** even
  after the RSS fix (sampled: Metal command queue in flight, ~5 s CPU progress
  per wall-minute); a corrected fp16 prefill number is still pending.
- CPU fp32 decode is memory-bound (~4.4 GB/token) and noisy on this host
  (PyTorch leg variance 96→52 s); treat its decimals as indicative only.

Quality is covered separately by the `test_llama_quant` oracle — these tables
measure serving speed, not numerical equivalence.

Harness: `tools/bench_framework_compare.py` (fixed-seed IDs drive both sides;
per-task JSON with `--output`, refuses to overwrite). The figures above were
extracted from that JSON by `tools/build_benchmarks_table.py`; neither the probe
JSON nor a byte-exact regenerator of this file is bundled.