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
  bit-identical greedy sequence. Everything else still needed Q4 when this was
  written. The same-precision fp16 GPU comparison it asked for — the cleanest
  possible engine-vs-engine measurement — SHIPPED on 2026-09-12: the fp16
  decode lane (`TM_DECODE_F16=1`, weights from `model.f16.safetensors`) ran
  TinyLlama-1.1B at 14-17 t/s against PyTorch fp16/MPS at 10.65 t/s, 1.6x at
  equal precision. See docs/F16_METAL_PLAN.md for the lane and its Phase 2.
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
## K-quant weights in the Metal token lane — measured, refused (2026-09-15)

Question: the K-quant GEMV kernels (Q4_1 / Q5_K / Q6_K, per-row gated by
`test_kquant_gemv.cpp`) exist and nothing ships through them, because
`gpu_decode_usable()` admits only f16/Q4_0. Does admitting K-quant pay off?

Instrument and model: `tools/tmq_census.cpp` on `qwen38_4b_q6k.tmq` — 442
tensors, **3,887.7 MB/token**, of which **99.8% of bytes are Q6_K** (210
tensors); the remaining 232 tensors are F16/BF16 norms and biases, 9.72 MB.
`bench_llama_decode`, 8 tokens, ctx 512, warmup 2, one window:

| config | lane | t/s | ms/step | greedy hash | last id |
|---|---|---:|---:|---|---:|
| default | refused -> CPU | 5.46 (10.35 warm, see below) | 185.4 | 2cf8717f69eac084 | 7509 |
| K-quant admitted | GDN GPU lane | 4.07 | 244.7 | 3b5ffdb82ac2d8a3 | 98395 |

Load 4.60/4.66 (CPU leg) and 4.96/5.59 (GPU leg) — other agents were running on
this host throughout, which is why both legs carry their load and why the two
CPU readings of this model (5.46 and 10.35 t/s) are quoted as a range with the
warm one marked.

Verdict: **the K-quant token lane is wrong as well as slower** — a second
checkpoint with a different failure mode from the first (`qwen35_0.8b_q40`,
which produced a two-token attractor). Here it decodes fluently but its greedy
trajectory diverges from the CPU's inside the same 8 tokens, at ~2.5x the
per-step cost. Divergence at that rate is not what a reassociation difference
looks like on this harness: with this section's two knob settings
(`TM_DECODE_GPU=0` and `=1`) the two lanes share the first 64 tokens
(`201c33ba174bbc60`, ctx 128 and 2000) and the per-step traces agree through
TOK 73 before parting at TOK 74 (ctx 2000).
**Correction, 2026-09-17:** the `1a0888ba122b61a4` cited here before is the
*pinned pure-CPU* leg's 64-token hash (`TM_DECODE_GPU=0 TM_PREFILL_GPU=0`), not
either lane's — and `TM_DECODE_GPU=0` alone does not name a CPU leg, because the
prefill lane is auto-selected there (GPU on this box). Hashes have to be quoted
with both knobs. With the CPU pinned, t=128 is `862c6832e2a87873` on BOTH the
pre-campaign binary and mainline (the CPU decode's own numerics never moved in
the idiom campaign); what moved is the GPU-prefill path — `43eeef6935cdff91`
pre-campaign vs `bf6f1126cdf381f5` on mainline, the change entering at
`7334cd1` (the host-built metal rope table). So the 64-token agreement is a
window, not an invariant — the 32-token goldens in `data/tinyllama` are why no
gate saw it, and `test_llama_quant`'s 128-token CPU trajectory lock (both lanes
pinned) is what now pins the CPU side of that horizon. So fail-closed
stays the default, and `TM_DECODE_KQ=1`
re-admits K-quant for measurement only, until a numerics gate
covers a checkpoint whose projection weights are K-quant.

**Update, 2026-09-17:** the Gated-DeltaNet GPU decode lane this table
measured was **RETIRED**. Its other opt-in (`TM_METAL_GDN_KQ=1`) is gone with
it: it was wrong here and 0.6–0.9× the CPU on the Q4_0 hybrid, so there was
no checkpoint on which it won. The five `gdn_*` kernels stay in `metal.h` as
library code with their own fp64 gate; `TM_DECODE_KQ=1` (the plain lane's
measurement escape hatch) is unaffected.

Correction to a first reading of this model: its CPU decode measured **3.09
t/s** (297 ms/step, RSS 3.97 GB) on the first run and **10.35 t/s** once the
3.9 GB file was warm in the page cache. The 3.09 figure is the cost of a cold
read of this model, not a lane comparison — and the warm number settles the
question that motivated the experiment: at 10.35 t/s the CPU path moves ~40 GB/s
of the ~61 GB/s ceiling the C stream probe measures, i.e. it is bandwidth-bound,
so a GPU lane has little room to add even if it were correct.

Two latent bugs were found and fixed on the way, both in the token encode path
and both reachable only by a checkpoint that mixes dtypes inside a layer: the
fused q|k|v branch fetched its handles (`gw()`, which refuses K-quant, and an
unconditional `q4_.at()`) *before* testing whether the branch was admissible, so
a K-quant layer threw mid-encode and left a half-encoded command buffer — the
same failure species the gate|up site documents. Both sites now decide first,
then fetch. Dead locals (`qkv_w`, `qkv_n`, `o_w`, `o_n`, `gu_w`, `gu_n`) are
gone; each was an unguarded `q4_.at()` on a tensor the lane may not hold.

## Do the two engines share one bus, or add? (2026-09-15)

The question behind every GPU/CPU composition idea on this machine: if the CPU
decode already walks ~40 GB/s, is the GPU idle capacity or a competitor for the
same envelope? `tensormark/overlap_probe.sh` runs three legs in one window on
`tinyllama_q40.tmq` (618.8 MB/token) — the GPU greedy chain alone, the C stream
probe alone, then both at once, at load 2.0-2.7:

| leg | GPU chain | CPU stream (1 / 2 / 4 / 8 threads) |
|---|---:|---|
| GPU chain alone | 69.4 t/s | - |
| CPU stream alone | - | 50.2 / 52.8 / 55.1 / **56.1** GB/s |
| both at once | **73.4 t/s** (hash 5d1f5028e5a7eee1, unchanged) | 58.9 / **61.0** / 58.2 / 50.8 GB/s |

**Neither engine lost anything**: the GPU chain carried ~45 GB/s of traffic while
the CPU streamed 61 GB/s, and its greedy trajectory is byte-identical with and
without the concurrent load. The two paths do not serialize on a shared envelope
— the GPU is real independent capacity while the CPU decodes.

CAVEAT, and it is the load-bearing half: this probe re-reads a file the previous
legs just streamed, so a hot 619 MB file can be partly served from cache. 56-61
GB/s is the achievable rate for a *hot* file, a floor for the ceiling, not
necessarily DRAM. The 3.09 t/s cold-read figure in the section above is the same
effect in the other direction.

Why it still does not make decode composition pay: a decode step must read each
weight block *once*, whichever engine reads it, so two engines add throughput
only if each is faster per byte — and none is. The CPU stream reaches 56-61 GB/s,
the CPU Q6_K decode sustains ~40 GB/s (10.35 t/s on 3887.7 MB/token), the Q4_0
GPU chain ~45 GB/s at 73 t/s. Both engines sit at ~70-75% of the stream rate, so
the remaining headroom is per-byte kernel efficiency, not bus contention. The
GPU's spare capacity converts into throughput where the work is compute-bound
rather than byte-bound — the long-context prefill case the hybrid split already
exploits, which is why that split wins and a decode split would only move the
same bytes between engines.

## Correction: decode composition DOES pay, as a parallel split (2026-09-15)

The section above reasoned from per-byte rates to "a decode split would only move
the same bytes between engines". That is true of a *sequential* layer split — the
llama.cpp `-ngl` shape, where for one token layer k+1 waits on layer k, so the
step costs the sum of the per-layer times and each engine idles while the other
works. It is NOT true of a parallel split: two engines reading *different* rows,
or two streams in flight at once. `tensormark/overlap_decode.sh` measures real
decode on both engines simultaneously, one window, load 2.7-3.2:

| leg | alone | both at once | retained |
|---|---:|---:|---:|
| GPU greedy chain | 76.97 t/s | 54.53 t/s | 71% |
| CPU decode | 62.07 t/s | 41.22 t/s | 66% |
| **aggregate** | **76.97 t/s** | **95.75 t/s** | **1.24x the best single engine; 1.54x the CPU alone** |

Greedy hashes are unchanged in every leg (GPU `5d1f5028e5a7eee1`, CPU
`e116121008a9dd12`), so both engines kept decoding correct streams.

- **"Decode composition cannot pay" was too strong, and this corrects it.** It
  holds for sequential layer assignment, which is a pipeline and cannot add here:
  the two engines' per-byte rates are within ~15% of each other (45 vs 40 GB/s),
  so taking turns just concatenates the same total. It does not hold for a split
  that keeps both engines busy on different bytes — that adds.
- **The contention is not only the bus.** Two pure streams were fully additive
  (CPU 61 GB/s alongside GPU ~45 GB/s, nothing lost), but two *decodes* each lose
  ~30%. So what they share is CPU compute and driver time, not a saturated memory
  pipe, which is why the loss is proportional rather than a collapse.
- **Ceiling for a single-stream split:** ~1.2-1.5x, less a per-layer reduction.
  The row-split machinery already exists and is proven for T=rows (`HybridSplit`
  + `block_linear`, the ESC prefill split); decode is T=1, so the same shape is an
  output-channel or head split with one reduction per layer.
- **Best fit is serving rather than one stream:** cross-token pipelining (two
  requests, one per engine) needs no barrier and no new kernel — the KV caches are
  already separate, and both processes can share the one read-only `.tmq` mapping.

## What an agent-shaped conversation costs — and two ways it was broken (2026-09-15)

A coding agent is not one request: it re-sends the whole conversation every turn. Driven
through the shipped HTTP shim (`/api/chat`, stateless, full history each turn), six turns,
budget 64, TinyLlama Q4_0, M1/8 GB, `--ctx 4096`:

| turn | chars re-sent | wall | reply chars |
|---:|---:|---:|---:|
| 1 | 7 788 | 8.78 s | 245 |
| 2 | 8 061 | 4.71 s | 4 |
| 3 | 8 093 | 5.11 s | 36 |
| 4 | 8 157 | 9.46 s | 640 |
| 5 | 8 825 | 10.04 s | 712 |
| 6 | 9 565 | 11.10 s | 708 |

49.2 s for roughly 500 tokens of output, and the wall-clock is the transcript being read
back: each turn re-prefills conversation the engine already had, so the cost grows with the
history while the generated part stays roughly flat. Prefill measures 915–1000 tok/s against
decode at ~12 ms/token (77 t/s), so at these lengths the re-prefill dominates by an order of
magnitude — and it is the term an agent actually feels, as seconds before its first token.

Two defects surfaced before the numbers could be trusted, both invisible to the existing
gates because every serving gate sent requests a real client does not:

- **A system message was a hard 500.** Ollama clients put the system prompt in the message
  list; the shim replays that history turn by turn, and the front-end's `/prefill` role
  check accepted only `user`/`assistant`. `[system, user]` → HTTP 500 "would not load the
  history", while `[user]` and `[user, assistant, user]` answered normally. Fixed; gate 9b.
- **A context overflow was a 500 that named nothing.** A 2 849-token turn against the
  default 2048 context returned the same opaque 500. Now HTTP 400 naming the window and the
  knob, plus `--ctx N` on the shim (the engine already accepted `TM_CTX`; nothing exposed
  it). Gate 9c.

Both were live on the public mirror. The script that found them is the instrument the next
change has to beat: reusing the KV cache across turns should collapse most of that 49.2 s,
and it must not move a single reply byte — the reply hashes in the run above are the
invariant.

### Prefix reuse across turns: fast, and wrong — measured, not shipped (2026-09-15)

The baseline says most of an agent's wall-clock is re-prefill, so the obvious fix is to
keep the cache and load only the new turns. Implemented shim-side (a `sync()` that
compares the client's normalized `(role, text)` history against a record of what the
engine holds, with `TM_SHIM_NO_REUSE=1` to switch it off), it is fast and it is wrong —
and the reply hashes caught it, which is the whole reason the benchmark records them.

Paired, one window, six turns, budget 64, `--ctx 4096`, only the switch differing:

| turn | no reuse | reuse | reply sha1 (no reuse → reuse) |
|---:|---:|---:|---|
| 1 | 8.66 s | 8.74 s | 3fb84b0719b6 → 3fb84b0719b6 |
| 2 | 4.77 s | **0.45 s** | b0d891163365 → **9e5b4562e9b2** |
| 3 | 5.19 s | 4.85 s | 0f5401d3a0a3 → **d1a1ab5bb3f4** |
| 4 | 9.85 s | 4.85 s | 1a707308586a → **bffeec05bc3d** |
| 5 | 10.79 s | 5.12 s | bffeec05bc3d → **c21cf6f91d3a** |
| 6 | 10.65 s | 10.43 s | 7247ddbc0153 → **cb30ea49f798** |
| **total** | **49.92 s** | **34.45 s** | — |

Turn 1 agrees (there is nothing to reuse), and every turn after it differs: 31 % off the
total, single turns up to 10× faster, and a different conversation.

The cache the engine builds by **generating** a turn is not the cache it builds by
**prefilling** the same text. The turn loop closes a reply by feeding the end-of-turn
marker plus `\n` (`close` in `llama_chat.cpp`), while `prefill_turn` loads
`{text}<eos>\n` — and the tokens that reach the closing marker are the *sampled* ones,
not the tokenizer's re-encoding of the decoded text. One token of difference is enough to
move the continuation, and here it did, on the very first reuse.

So the shippable form is not a shim-side trick: it needs an engine-side rewind — keep the
cache, expose its length, continue from a known position instead of reconstructing the
prefix by re-tokenizing it. That rewind is the same primitive the MTP verifier needs to
undo a rejected draft. Until it landed, the shim kept today's behaviour, and this table is
why: the fast version was 31 % cheaper and quietly answering different questions.

### Finishing it: the shim now reuses the cache, and answers what a full replay answers (2026-09-15)

The primitive exists and is gated: `rewind_to(n)` in `llama.h`, `/pos`, `/rewind <n>` and
`/turnpos` on the batch protocol, and `tensormark/test_chat_rewind.py` proves the sound form
— rewind to the boundary the engine reports, re-prefill the generated turn from text, and the
cache is byte-identical to a replay (A == B == `449726839d40`; leg D, the engine's own
boundary, answers `e6abd6283b9a` against its own reference replay, while keeping the sampled
reply answers `d4c3e1cfad2d` — the defect this primitive exists for). 2 123 of the 2 185
cached rows are reused untouched; only the 62-token reply is rebuilt.

The boundary is a token count — `m.seen` minus the header — so a rewind point no longer
depends on splitting the forward. It does depend on the two paths agreeing where a turn
ENDS, which is a one-token matter and the subject of the section below.

The shim's flow follows from the same fact that broke the first attempt: every assistant
turn in the cache is *sampled*, so a conversation is only replay-equal if each turn is
canonicalized as it is produced — generate it, take `/turnpos`, `/rewind` there, re-prefill
that reply from text. The cache then holds exactly what a full replay would hold, and the
next request loads only what it has not seen. Paired, one window, three turns, budget 64,
`--ctx 4096`, only the switch differing — and the reference leg (`TM_SHIM_NO_REUSE=1`) *is*
the behaviour before any of this existed:

| turn | no reuse | reuse | reply sha1 (both legs) |
|---:|---:|---:|---|
| 1 | 6.83 s | 6.78 s | d2ec1f3ef3cc |
| 2 | 5.63 s | **1.49 s** | b8f68f6a69cc |
| 3 | 5.35 s | **0.90 s** | ce42599574e1 |
| **total** | **17.81 s** | **9.17 s** (1.94×) | — |

A second run of the same pairing — its own window, its own machine state — gave 17.36 s →
9.04 s (1.92×) with the identical three hashes, and a third, run as the last gate of the
full serving suite on an already-loaded machine, gave 24.24 s → 13.44 s (1.80×) — all three
hashes still identical. The conversation is deterministic run to run; the times are not,
which is why the hashes are the invariant this ships on, and why the gate asserts a floor
(1.25×) rather than the factor.

Not one reply moved, and the shim says so itself rather than leaving it to be inferred:
`[shim] reuse: 5/5 turns already cached, loading 0`. What the fast path pays is re-prefilling
each reply — 62 to 107 tokens here — plus a KV re-upload, against skipping the whole history
(~2 000 tokens), which is where the factor comes from. It is a property of this conversation
shape, not of the engine: short turns with little history to skip would not pay for the
re-prefill, and the shim's own guard rejects reuse when the cache is near-full rather than
discovering the edge with a wrong answer.

`tensormark/test_shim_reuse.py` is this pairing as a gate — both legs, all three turns, hash
equality asserted, reuse-firing asserted in the log, and (on a real model) a floor on the
ratio. The reference leg is a full replay per request, so the gate cannot pass by measuring
the same code twice.

**Re-measured 2026-09-17 (full gate suite, ambient load ≈ 15): the ratio moved to 2.69x, and
the movement is entirely the reference leg.** Same three turns, same reply hashes
(`d2ec1f3ef3cc b8f68f6a69cc ce42599574e1`): no reuse 24.62 s → reuse 9.16 s. Against the 1.95x
recorded above, the reuse leg moved by **0.2%** (9.14 → 9.16 s) while the full replay moved by
**38%** (17.78 → 24.62 s). That direction matters, because a single ratio implies the reuse
path is the variable quantity and it is not: the replay re-prefills every turn, so it absorbs
ambient load that the reuse leg skips by not doing that work. Read the band as the reference
leg's load exposure, not as reuse getting faster.

Two caveats, both measured rather than assumed. First, the *timings* above are from `--ctx
4096`; an earlier pairing at `--ctx 8192` showed the same six-for-six hash equality but one
leg whose wall-clock (1 051 s for a turn whose code path is identical in both legs) could not
be explained and is therefore not quoted. Second, the answers themselves depend on the
configured context size: the same conversation at `--ctx 4096` and `--ctx 8192` produced a
different reply hash on turn 6, at identical prompts. Both are reasons to quote a figure with
its window and its configuration attached, and neither is understood yet.

### The turn-boundary token: why a rewound cache must end where a replay ends it

Getting leg D to pass came down to one token, and the way it failed is worth recording
because a token *count* cannot see it.

`encode()` is not associative across call boundaries. On TinyLlama's SentencePiece:

```
encode("\n<|assistant|>\n")                = [29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13]
encode("\n") ++ encode("<|assistant|>\n")  = [29871, 13,  529, 29989, 465, 22137, 29989, 29958, 13]
```

Nine tokens either way; position 3 is `29966` in one and `529` in the other. Both decode to
the same text, the model sees a different embedding, and it answers differently — at an
identical cache length, which is why every count-based check said the two builds agreed and
only the reply hash disagreed.

That is why the rewind has to land where `prefill_turn` ends a turn rather than wherever the
turn happens to finish. `prefill_turn` builds one as `append("<|role|>\n"+text); eos;
append("\n")`, so its boundary is after that separate `"\n"`; `run_turn` now builds the user
half the same way, which changed the generated replies by one junction token
(`2e0728258358` → `77587afbbb3c`) — stated here rather than discovered later.

The general lesson is sharper than the incident: **any template assembled with several
`append()` calls is not guaranteed to produce the token stream it looks like**, and count
equality proves nothing. Compare token ids, or compare reply hashes.

## The MTP head on a real checkpoint: correct, well-matched, and still a slowdown (2026-09-16)

The engine has carried MTP (multi-token-prediction) speculative decoding since the qwen35
port, but its gate has always been synthetic, and `tensormark/test_mtp.cpp` says why in its own
header: no released qwen35 checkpoint shipped the head. Qwen3.8-4B-Distill Q6_K does — it
carries `mtp.eh_proj`, `mtp.enorm`, `mtp.hnorm`, a full attention block and a SwiGLU FFN — so
this is the first measurement of the path on real weights, and the first thing it did was
throw.

**The bug it found.** `alias_mtp_weights()` copies the head's tensors onto a free layer slot so
the ordinary dense layer path can run them. It copied from `qshape_`/`q4_`/`q8_`/`f16_`/`w` and
never from `qk_`. A checkpoint that quantizes its head as a K-quant keeps those rows in `qk_`,
so the slot arrived empty and `block()` threw `map::at` on the first draft. One line to fix,
and it stayed hidden for exactly the reason the synthetic fixture exists: a head that only ever
runs as F16/Q8_0 cannot miss a K-quant alias.

**The measurement** (`tensormark/spec_bench.cpp`, 64 greedy tokens, `TM_CTX=4096`, M1 8 GB):

| | plain | spec |
|---|---:|---:|
| wall | 5.97 s | 7.24 s |
| rate | 10.6 tok/s | 8.7 tok/s → **0.82×** |

33 tokens drafted, 30 accepted, 3 rejected — **90.9 % acceptance** — and the emitted sequence is
**bit-identical** to the plain loop, which is the assertion that separates an optimization from
a different model. The head itself is therefore in good shape: correct, and agreeing with the
trunk nine rounds in ten.

A second run reproduced the counts exactly — 33/30/3, identical output — while a `clang++` gate
build shared the machine, and reported 0.73×. Its *rates* are not quotable: the contention shows
up in the plain leg itself, whose two runs took 10.88 s and 25.93 s. The acceptance and the
identity are quotable from it, because neither depends on machine state — which is the whole
reason the two are reported separately.

The time does not go to the head — and it does not go where this section first said it did. What
was written here was an *estimate* of a bookkeeping cost; it has now been decomposed instead
(`TM_SPEC_TIMING=1`, same bench, 64 tokens, 33 rounds, 90.9 % acceptance):

| term | ms | share |
|---|---:|---:|
| verify `forward({t,d})`, T=2 | 6824.7 | **82.1 %** |
| draft (`mtp_step`) | 617.2 | 7.4 % |
| replay, 3 rejections | 384.0 | 4.6 % |
| `snap_states` | 345.5 | **4.2 %** |
| other (argmax, dispatch) | 140.4 | 1.7 % |

The snapshot is `linear_layers_ * (lv_heads_*lk_dim_*lv_dim_ + (conv_kernel_-1)*conv_channels)`
floats — ~50 MB on this model, ~10 ms a round — so eliminating it outright would move 0.74× to
roughly **0.78×**. It was never the reason to stay opt-in, and the earlier attribution is
retracted. The reason is the next section.

**What is not covered.** `convert_tmq` quantizes a whole checkpoint at one precision
(`q80`/`q40`), so CI's fixture head is F16/Q8_0 and cannot reach the `qk_` alias. A converter
option to quantize the `mtp.*` namespace on its own would let `gen_mtp_fixture.py` build a
K-quant head and put this one line under regression; until then the fix is verified on a real
checkpoint (this section) and not by CI.

### Why the loop cannot be rescued by drafting harder (2026-09-16)

The obvious reading of "0.82× at 91 % acceptance" is that the loop advances too few tokens per
verify pass, and the fix is to draft more of them. Before believing that, measure what a
k-position pass actually costs (same bench, quiet host, load 2.17→2.73, both models):

| k | dense Q4_0 (Qwen2.5-3B) | hybrid K-quant (Qwen3.8-4B) |
|---:|---:|---:|
| 1 | 34.6 ms/position | 83.6 ms/position |
| 2 | **0.90×** | **0.90×** |
| 4 | **0.84×** | **0.86×** |
| 8 | 3.71× | 4.84× |

Two things fall out, and the first refutes a hypothesis I had already written down. Batching
positions saves only ~10–15 % at k=2–4 — and it does so **identically on a dense Q4_0 model**,
which has the grouped-GEMV fast path, and on the hybrid K-quant one, which falls to a per-row
GEMV. So the cost is not the K-quant row loop and not the linear-attention layers; it is the
engine's per-position path generally. Concretely: a two-position verify pass costs ~1.8 plain
steps, and a round also pays a full draft step, so the loop spends ~2.05 units of work to emit
~1.9 tokens. That is the 0.82×, and no drafting depth fixes it.

**The k=8 cliff is gone (2026-09-17), and `kq_deq_amx_ok` is why.** That function now returns
false below `TM_KQ_AMX_MIN` (default **64**), so the whole-model-dequant-then-AMX path this entry
blamed no longer fires at k=5..32 — the threshold the table described as `prefill_loop_t() == 5`
was moved to 64 by the AMX work that landed *after* it. Three re-runs, k extended to 32 (legs
within one window each; load printed per leg in the harness that produced them):

| k | run A | run B1 | run B2 |
|---:|---:|---:|---:|
| 1 | 83.9 ms/pos | 93.4 ms/pos | 88.4 ms/pos |
| 2 | 0.84× | 0.74× | 1.01× |
| 4 | 0.75× | 0.68× | 0.78× |
| 8 | **0.72×** | 1.36× | **0.75×** |
| 16 | 0.71× | 0.64× | 0.76× |
| 32 | 0.73× | 0.72× | 0.79× |

So "drafting depth ≥5 is off the table" is **retracted as stated**: there is no cliff, and the
paragraph above it is now the one that pins the depth. What replaces it is not better news — the
ratio is **flat at ~0.7× from k=4 to k=32**, i.e. a k-position pass never amortizes a fixed
per-pass cost, so batching positions is worth about 1.4× **once** and drafting deeper cannot
rescue the loop: at k=16 a round would still spend ~11 plain-steps of verify to emit ~11 tokens,
and the head must itself run 16 times to produce 16 drafts. The conclusion survives; its stated
reason does not.

Two caveats on those three runs, because they are not the quiet-host window the table above came
from. They were taken with a browser and a chat client resident, which is why the absolute
milliseconds drift (k=1: 83.9 / 93.4 / 88.4) and why k=2 ranges 0.74×..1.01×. Every ratio is
*within-run*, which is what makes it usable — and B1's k=8 = 1.36× is a single outlier rather
than a cliff, since that same run put k=16 (0.64×) *below* k=8, which no fixed per-call cost can
do. An earlier run of the same probe with a clean baseline is what produced the 83.6 ms/position
figure in the table above, and its k=1 still reproduces to 0.4 %.

One number from an earlier run of this same bench is retracted: it showed 0.55×/position at k=4
(and 0.80× at k=2), which would have made "draft four tokens" look promising. That run was
contended — its own k=1 baseline was 234 ms/pass against 85.6 ms quiet — and a 2.7×-inflated
baseline makes every larger k look good in comparison. The curve above is the quiet one.

**Not the fix either.** The switches guarding these paths were being read with `getenv()` on
every call, including in `gemv_w` (once per projection per layer per token) — a debug flag
paying work on the path it exists to observe. Hoisted to `static const` (and verified
behaviour-neutral: greedy `--oneshot` output byte-identical, `sha1 3fd77a40552c`, before and
after). Measured cost of the old form: `getenv` at 52.4 ns/call × ~224 calls/token ≈ 11.7 µs per
token, **0.013 %** of a 90 ms token. Worth fixing on shape; not worth claiming as a win.

### The loop exchange, and the variant it refuted (2026-09-16)

The curve above says a k-position pass does not get cheaper per position, and that it does so
identically on a dense Q4_0 model and on this hybrid K-quant one. It is the loop order, and the
kernel shows why: `kqgemv_q6k_rows` spends ~8 instructions per 16 values rebuilding a Q6_K code
(nibble | two `qh` bits, minus 32) before one `vdotq_s32` on it — once per position, over weight
bytes it re-reads each time. `kq6k_multi_rows<T>` exchanges the two loops: each 16-value segment
is rebuilt **once** and dotted against every position's quantized activation.

The ceiling is the unpack, and it is honest to state it: at T > 1 the 16 sub-block factors of a
super-block (16 vectors) no longer stay in registers, so they are re-derived per position. This
buys the rebuild amortized over T, not a free 1/T.

**A refuted variant, committed as a refutation.** The first attempt kept the fp32 dequantizer and
handed a band of rows to `cblas_sgemm` at M ≤ 4: it measured **0.17×** the plain decode against
**0.73×** for the per-row loop it was meant to replace, with the token stream diverging. It traded
the int8 sdot path for 4 bytes per weight instead of 0.75 and a tiny-M GEMM that Accelerate packs
internally. Reuse the fast path; do not substitute for it.

| gate | what it covers | result |
|---|---|---|
| `test_quant_gemv` (`check_kquant_multi`) | synthetic Q6_K, per position vs `kqgemv_q6k_rows` on the same quantized activations, T = 1/2/3/4/8, under ASan/UBSan/TSan in `qa.sh` | **PASS**, maxrel < 1e-6 |
| `check_kquant_kernel` | every Q6_K tensor of Qwen3.8-4B against the independent dequantizer, T = 2/3/4/8/9/16/17 (the last three are partial chunk tails: 9 = 8+1, 17 = 8+8+1) | **PASS**, 1519 calls, worst 1.26e-06 |

**What the exchange is worth at the kernel** (`kq_multi_ab`, single-threaded, 53.8 MB of Q6_K
rows — past the 12 MB L2 — arms interleaved, same quantized activations, same arithmetic):

| T | per-row ms/position | multi ms/position | vs T=1 | maxrel |
|---:|---:|---:|---:|---:|
| 1 | 4.412 | — | 1.00× | — |
| 2 | 4.366 | **2.982** | 0.68× | 2.0e-07 |
| 4 | 4.459 | **2.190** | 0.50× | 2.3e-07 |
| 8 | 4.455 | **1.873** | 0.42× | 2.6e-07 |

The shape is the model: per 16 values the per-row kernel spends 8 instructions rebuilding the
code plus 4 of dot and scale *per position*; the multi kernel spends 8 plus 4T. Predicted T=8
ratio 10/24 = 0.42, measured 0.42.

**What it is worth in the engine: not measurable in the windows this box offered.** The paired
engine runs (load 4.3 → 9.9, arms A B B A) put the multi arm at 0.82/0.83/0.74× per position at
k=2 and 0.92/0.87× at k=4 against the per-row arm's 0.87/0.93 and 0.83/0.87 — overlapping, and
the same-window k=1 baselines differ by up to 11 %, which is the same order as the expected
effect. The projection is not the whole pass: at T=2..4 the kernel is 1.5–2× faster but the
per-position pass also carries attention, the DeltaNet recurrence, the norms and the pool
barriers. So the honest engine-level statement is **no regression, benefit below this box's
measurement noise at k ≤ 4**; the kernel-level table above is where the effect is legible.

**At k ≥ 8 the engine does show it, once the path chunks.** Per-row (`TM_KQ_MULTI=0`) against
chunked, legs A B B A in one window (load 4.2 → 5.7), both legs reporting an identical token
stream, per position:

| k | row loop | chunked multi | ratio |
|---:|---:|---:|---:|
| 8 | 75.9 / 75.3 ms | 65.9 / 63.2 ms | **1.17×** |
| 16 | 78.4 / 76.9 ms | 66.0 / 63.7 ms | **1.20×** |
| 32 | 83.4 / 77.7 ms | 64.9 / 63.9 ms | **1.25×** |

The chunked column is flat (63–66) while the row column climbs with k — the signature of a kernel
that reads a weight chunk once for eight positions instead of re-reading every row for each one. The
2.38× kernel factor does not survive to this level because the projection is only part of a pass;
1.17–1.25× is what the pass gets. This is also what moved the AMX crossover in the next section.

The chunk width is not arbitrary either. 16 positions per chunk — the default, and what the register
file holds — against two 8-wide chunks on the same weights and activations measures **1.529 / 1.527
ms per position at T=16 versus 1.709 / 1.706**, with **bit-identical output**: the accumulators are
per position, so widening a chunk only halves how often the weight stream is read. It also makes the
amortization curve monotone, which is the two-term model confirming itself rather than being fitted:
2.982 ms per position at T=2, 2.190 at T=4, 1.873 at T=8, 1.529 at T=16 — a fixed ~1.18 ms of dot
plus ~5.5/T ms of unpack. `TM_KQ_CHUNK=8` restores the narrow width.

### The K-quant AMX threshold was 5, and it should be ~60 (2026-09-16)

Found while chasing the divergence above, and it is a bigger user-visible number than the kernel
that led to it. A K-quant tensor's AMX prefill lane dequantizes the **whole tensor to fp32 on
every call** before the sgemm, so it carries a fixed cost that only amortizes at large M.
Measured in-engine, paired and interleaved, on Qwen3.8-4B Q6_K:

I ran both admissions interleaved (40, 5, 40, 5) in one quiet window (load 2.2 → 4.9) on a 6-token
prompt, so the two columns are paired and the second pair repeats the first:

| k | AMX lane admitted | row / chunked-multi path | ratio |
|---:|---:|---:|---:|
| 8 | 404.7 / 411.1 ms per position | 58.5 / 62.6 | **6.9×** |
| 16 | 206.4 / 204.9 | 59.3 / 63.3 | **3.3×** |
| 32 | 108.3 / 106.0 | 61.9 / 64.3 | **1.7×** |

The fixed term is visible directly: **3238 ms per pass at k=8, 3302 at k=16 and 3464 at k=32** —
`3.18 s + 7.5 ms × k` to within 2 %, i.e. ~3.2 s of scalar unpack (210 bytes of Q6_K → 1024 bytes,
~7 GB/s) plus a sgemm that is not the cost. The row path is ~90 ms per position under load, and
the chunked multi path is **flat at ~61 ms from k=8 to k=32**. So the crossover is
`3.18/(0.061−0.0075) ≈ 59`, and `TM_KQ_AMX_MIN` defaults to 64. Admitting this lane from
`prefill_loop_t() == 5`, which is what shipped, made every prompt shorter than that 2.5–6.9× slower
than the path it displaced — and chat prompts on this box are in that range. The lane is kept
because a long prompt still wants it: at M=256 it is 12 + 7 = 19 ms per position against 61.

And that threshold is verified at the threshold, not only inferred from the fit: legs A B B A
(64, 1000, 1000, 64) at k=64 and k=128, `TM_CTX=8192`, all four reporting an identical token
stream —

| k | AMX admitted (`=64`) | chunked path only (`=1000`) | AMX advantage |
|---:|---:|---:|---:|
| 64 | 58.5 / 58.9 ms per position | 63.8 / 69.5 | **1.14×** |
| 128 | 34.9 / 35.7 | 66.1 / 67.3 | **1.89×** |

At exactly the threshold the lane is already the faster one, so 64 is not eager; and the fitted
`3.18/T + 7.5` predicts 57.2 ms at k=64 against 58.5 measured, so the two-term model holds at the
point it is used.

**The disease remains the per-call serial unpack.** Row-parallelising it would divide the fixed term
by the pool width and move the crossover to ~M=10, and would speed up long prefills too (a
512-token prefill spends 3.2 s of its ~6.8 s in that unpack) — so it was measured before it was
written. It is refuted; see immediately below.

### The per-call unpack cannot be row-parallelised — measured, refused (2026-09-16)

The proposed change, measured in a single-tensor A/B rather than after a rewrite. `deq_scale`
dequantizes the **largest K-quant tensor** of the 4B checkpoint — 2542.8 MB of fp32 written per
pass — three ways, interleaved A B C in one window, best of 3, with the system load printed before
every arm:

| window | serial | pool (`GemvPool`) | plain threads, default QoS |
|---|---:|---:|---:|
| `TM_THREADS=4`, load 3.99 → 4.87 | **2.951 s** (215 Mvalues/s) | 9.333 s (0.32×) | 10.466 s (0.28×) |
| `TM_THREADS=2`, load 6.2 → 7.2 | **5.123 s** (124 Mvalues/s) | 13.989 s (0.37×) | 11.882 s (0.43×) |

One sequential stream beats N streams, and the third arm exists to kill the alternative
explanations before they cost a rewrite: plain `std::thread` workers at **default QoS** — no pool,
no `GemvJob::chunk` granularity, none of the `QOS_CLASS_USER_INTERACTIVE` the pool requests by
default — are just as slow. So this is not the launcher, not the chunk size, and not the QoS band.
Splitting a 2.5 GB destination into four streams 640 MB apart, each writing a different row range
of one buffer, is simply the slower shape here. A whole-model pass had shown the pool *winning*
1.22× (38.5 s → 31.7 s over 217 tensors, 5.06 Gvalues), which is exactly why the conclusion rests
on paired single-tensor arms and a printed load rather than on one favourable window.

The earlier q4/q8 note reads consistently too, and part of it was already settled: a pool-parallel
dequant measured no faster there either (271 vs 257 t/s, `TM_POOL_DEQ`), which is this same result
at a different scale — and the heap corruption once blamed on the pool was root-caused to an
lda/ldc out-of-bounds in the batched-attention sgemm and fixed (`repro_poolbug.cpp`), so it was
never evidence against the pool at all. Serial stayed that lane's default on throughput alone.

What is refuted is this fix, not the problem. Writing the unpack into one huge shared destination is
the design that fails; an unpack folded into the AMX lane's own per-call tile buffer is a different
question, and this measurement does not answer it.

Instrument: `deq_scale <model.tmq> [reps] [all|one]`, built by `build_llama.sh` alongside the other
diagnostics. Two traps are
documented in it, both of which produced wrong answers first: the pool raises the **calling thread**
into `QOS_CLASS_USER_INTERACTIVE` (`qos 33`, `TM_QOS` default `interactive`), so a min-over-reps
silently compares different bands; and a rate in MB/s is meaningless here, which is why the arms
report Mvalues/s.

### What a lower precision would buy, priced at the kernel (2026-09-16)

Decode is 88 % weight-streaming GEMV and that GEMV is instruction-bound: the same 53.8 MB of
weights costs the same at T=1 whether it sits in L2 or streams from DRAM (`kq_multi_ab` at 5.4 /
21.5 / 53.8 MB: 16.5 / 16.2 / 16.0 MB per ms). So "would a cheaper precision decode faster?" is a
question about per-value cost, and it is answerable with the kernels already in the tree — no
checkpoint required. `kq_multi_ab <in> <out> [q41|q40|q80]` builds the same values at each width and
times the lane that would serve them, single-threaded, one window:

| lane | bytes/value | ms per 2560x25600 values | effective GB/s | vs Q6_K |
|---|---:|---:|---:|---:|
| **Q6_K** (shipped) | 0.820 | 3.99 – 4.11 | 13.4 | — |
| Q4_1 | 0.625 | 5.64 | 7.3 | **0.71× (slower)** |
| **Q4_0** | 0.5625 | **1.386** | 26.6 | **2.9×** |
| **Q8_0** | 1.0625 | **1.654** | 22.3 | **2.4×** |

Two things fall out, and the first corrects the reasoning this started from. **The bits are not the
lever, the unpack is, and precision is not monotonic in speed**: Q8_0 carries 29 % MORE bytes per
weight than Q6_K and is still 2.4× faster per value, while Q4_1 carries 24 % fewer and is 29 %
slower. Q4_0/Q8_0 ride `qgemv_sdot_rows` (~8 ops per 32 values, x-side loads amortized over a row
quad); Q6_K pays ~11 per 16 values; Q4_1 is affine and pays two horizontal reductions per 32 values
plus a scalar FMA chain. The second: a T=1 GEMV is not where bytes are saved, which is why the
multi-position kernel — the one change here that removes work instead of moving it — was worth
building.

Projected end to end for decode (88 % of it is this kernel): a Q4_0 Qwen3.8-4B would run ~1.9–2.4×
the token rate of the Q6_K one and occupy ~2.3 GB against 4.07 GB. Q8_0 is faster per value too and
close to lossless, but at ~4.8 GB of weights it does not fit this box beside the KV cache and the
runtime. Neither is a change made here: the engine converts `q80|q40` from safetensors, no source
checkpoint is on disk, and the quality side of the trade is unpriced until `eval_ppl_llama` runs on
a converted model. That download is the owner's call, not a conclusion this bench can reach.

**And the divergence that started this, explained rather than waved away.** `spec_bench`'s
"identical token stream" bit compares two plain runs against the speculative one, so it is a
*cross-path* assertion — and in this case it was telling the truth about a real inconsistency
between lanes, not just about a knife-edge argmax. In ONE window, same prompt (6 tokens), same
model:

| window | lane held out (`TM_KQ_AMX_MIN=64`) | lane admitted (`TM_KQ_AMX_MIN=5`) |
|---|---|---|
| A (load 8 → 22) | 76.9 % acceptance, **YES** | 46.7 % acceptance, **NO** |
| B (load 2.2 → 4.9, two legs each, interleaved) | 0.81× and 0.80× spec, **YES** | 0.68× and 0.62× spec, **NO** |

Four legs, two windows, same direction each time: it is not noise.

The K-quant AMX lane computes its prefill from **fp32** activations while every decode step
quantizes them to int8 first. Mixing the two across a verify round changes the state by ~1/127 per
element — not a rounding difference — which is enough to halve acceptance and flip tokens. The
multi-position kernel is *not* the cause: it uses the same int8 dots and the same sub-block factors
as the decode kernel, differing only in fp32 accumulation order (1e-6, per the gates), and with the
AMX lane out of the picture the assertion passes again in the same binary. Two lessons, both
narrower than "the bit is noise": a cross-path assertion needs both paths pinned, and an engine
that mixes activation precisions across stages pays for it in speculative acceptance.

## What "the best Qwen 3.8 on this box" actually costs (2026-09-16)

Qwen3.8-4B-Distill **Q6_K** (3.8 GB) is the best-fitting member of the family here: the official
smallest is the 27B (not conceivably runnable on 8 GB), and the community 9B distills need about
5.46 GB of weights at the only precision they ship that this engine can convert — past the
5.03 GB of weights already measured into swap-thrash on this machine, though that measurement was
at a 16k context, so treat it as a bound rather than an equivalence.

| | |
|---|---|
| peak footprint | 4.71 GB at `TM_CTX=4096` (max RSS 4.35 GB) |
| decode, healthy box | 11 tok/s (32-token reply), 6 tok/s (9-token reply) |
| decode, machine at 11 % free memory | 6 tok/s — same weights, prompt and budget |
| prompt processing | ~5 tok/s |
| chat | two-turn exchange verified; turn 2 answered "Thaylo" |

The memory-pressure row is the one worth keeping. The same weights, prompt and budget measured
6 tok/s when the machine had 11 % free memory and 4.7 GB of the model in play, and 10.6–11 tok/s
after a reboot cleared it — so a figure for this model quoted without the machine's state is a
figure the next run will contradict. Also worth knowing before treating Q8_0 (4.61 GB of
weights) as an upgrade: it is the same architecture and buys well under a percent of quality for
roughly a quarter less throughput, which is why Q6_K is the point on the curve this machine
should hold.

## Q4_0 for this checkpoint — priced at last, and the instrument that had to be fixed first (2026-09-16)

The note above closes with Q6_K as the point on the curve this machine should hold, on the grounds
that Q8_0 bought under a percent for a quarter less throughput. That is one direction of the curve.
The other direction was unmeasurable for a mundane reason: the checkpoint ecosystem ships BF16 /
Q4_K_M / Q5_K_M / Q6_K / Q8_0 for this distill and **no Q4_0 anywhere**, so comparing meant
downloading the model again in another precision. `tmq_requant <src.tmq> <dst.tmq> q40` removes that
blocker: it dequantizes the K-quant tensors already on disk and re-quantizes them, which took **24 s**
and no bandwidth. It converts only 2-D K-quant tensors whose rows are a whole number of Q4_0 blocks
and copies everything else block for block, so the diff between the two models is exactly the
projections: **210 tensors converted, 232 kept** (every RMS/LayerNorm is 1-D and never touched — the
converter's own comment records a 40x cancellation amplification from requantizing norms), 442
records in both files, payload length equal to file length in both, **4.08 GB -> 2.80 GB of weights
(1.46x smaller)**.

### The instrument had to be fixed before it could price anything

The first paired run reported `ppl 108 028 / top-1 0.88%` for Q6_K and `81 434 / 0.98%` for Q4_0 —
bit-identical per model across legs, and nonsense: uniform over this vocabulary is 248 320, so both
models were scoring in near-random territory while one of them writes a clean haiku. `align_probe`
pinned the cause in six steps: feeding token *t*, the argmax is token *t+1* on 3 of 6 steps of a
7-token sentence and token *t* on **0 of 6**. `forward()` returns next-token logits, and
`eval_ppl_llama` paired the logits of step *t* with the target of step *t* — so every token was
scored against the distribution the model produced *after* being shown it, which reports ~log(vocab)
for any model, including a perfect one. **The tool certified quant-vs-precision deltas while
measuring noise, and looked credible doing it because it was equally wrong on both sides of every
comparison** — the same failure shape as a gate that only ever sends the easy request. Fixed, with
the probe kept as the regression. The older ppl figures in `ROADMAP.md` / `DISPATCH_M1.md` (23.94,
26.78) come from the GPT-2-era tools (`eval_ppl_metal`), not this one, and stand.

### The paired result, legs A B B A, load recorded per leg

| leg | weights | load | ppl | top-1 | decode |
|---|---|---:|---:|---:|---:|
| A | Q6_K 4.08 GB | 1.80 | **5.7667** | **59.98%** | 11.1 t/s |
| B | Q4_0 2.80 GB | 4.55 | **6.1940** | **57.63%** | 17.1 t/s |
| B | Q4_0 | 4.61 | 6.1940 | 57.63% | 17.3 t/s |
| A | Q6_K | 4.95 | 5.7667 | 59.98% | 10.5 t/s |

Both quantities are deterministic per model across legs (ppl identical to four decimals), and Q4_0
is faster in both pairs **while running at the higher load**: 17.1/11.1 = 1.54x and 17.3/10.5 = 1.65x,
against the kernel price table's 2.9x for the sdot lane — the rest is the layers that are not GEMV
and the GDN machinery that charges to `attention`. Quality costs **+7.41% ppl** (nll 1.75209 ->
1.82357, i.e. +0.071 nats = +0.103 bits/token) and **-2.35 points of top-1**. Peak footprint at
`TM_CTX=4096` measured with `/usr/bin/time -l`: **4 420 272 128 B (4.42 GB) -> 3 377 037 312 B
(3.38 GB)**, which on an 8 GB box shared with a browser is the part a chat user actually feels.

### What validates the fixed instrument

An instrument repaired after reporting 108 028 must be checked against something outside itself.
On the qwen35-0.8B Q4_0 checkpoint the corrected tool reports **ppl 11.4279, top-1 49.22%, 57.6
tok/s**; the commit that added this tool records llama.cpp at **PPL 15.38 on the same .gguf**. Same
order, correct model ordering (the 4B at 5.77 beating the 0.8B at 11.43, as it should), and a
residual 26% that is NOT explained away here: the corpus file, the chunk length and llama.cpp's own
scoring conventions are all candidates and none has been isolated. Treat the ratio between two
configurations on the same corpus as sound; treat the absolute number as corpus-specific.

### The trade, stated plainly

Prompt processing went the same direction, by less. `hybrid_bench <model.tmq> 512 3`, legs A B B A,
load recorded per leg:

| leg | weights | load | rounds (tok/s) | mean |
|---|---|---:|---|---:|
| A | Q6_K | 3.00 | 56.4 / 57.3 / 55.6 | 56.4 |
| B | Q4_0 | 2.63 | 74.6 / 73.6 / 74.8 | 74.3 |
| B | Q4_0 | 2.48 | 74.2 / 73.2 / 73.3 | 73.6 |
| A | Q6_K | 2.23 | 57.6 / 57.3 / 56.6 | 57.2 |

**1.30x** (73.95 / 56.80) — a 512-token prompt costs 6.9 s instead of 9.0 s — and that is much
better than the ~5 tok/s this document records elsewhere for prompt processing on this model, so
the expected prefill catastrophe did not exist: the chunked multi-position path amortizes its
unpack across the chunk, which is the whole reason it exists. Two notes for whoever repeats this:
the chat front-end's per-turn stat line is gated behind `TM_CHAT_DEBUG`, so grepping for it returns
nothing and a leg looks like it ran fine while producing no data (the first attempt at this
measurement did exactly that, four legs in a row), and `hybrid_bench` rejects T < 512.

### The trade, stated plainly

Q4_0 derived this way is **1.5-1.65x faster, 1.46x smaller and ~1.0 GB lighter for +7.4% ppl**, and
those are the numbers that should decide it — not the fact that a 4-bit format sounds cheap. Two
caveats belong with them: the derivation re-quantizes Q6_K (no imatrix, no calibration corpus), so
this is the cost of THIS derivation and not of the ecosystem's Q4_K_M; and the two models are
different models, so their greedy streams differ — every figure above is a comparison, not an
equivalence. The 0.8B oracle line is the one to remember about the total: both of these models are
cheap to run because they are small, and the 8 GB machine is the binding constraint, not the kernel.

## What a chat actually feels like: the two settings that decide it (2026-09-17)

The tables above price the engine. A chat REPL is priced differently — what a user feels is the gap
between pressing enter and reading words, the seconds per reply, and whether the machine stays
usable while it answers. Measured through `llama_chat_metal` on the same 8 GB M1: the **interactive**
path, 120-token replies to one fixed prompt, legs paired A B B A, free+inactive memory read before
and after the window.

| model | `TM_CTX` | decode | peak footprint |
|---|---:|---:|---:|
| Qwen3.8-4B Q4_0 | 16384 | 17 / 17 t/s | 4.31 GB |
| Qwen3.8-4B Q4_0 | 4096 | 18 / 17 t/s | **3.40 GB** |
| Qwen3.8-4B Q6_K | 16384 | 11 / 11 t/s | 5.41 GB |
| Qwen3.8-4B Q6_K | 4096 | 11 / 11 t/s | 4.50 GB |

**The context setting is free.** `TM_CTX=4096` buys back **0.91 GB on both checkpoints** — 4.31 ->
3.40 GB and 5.41 -> 4.50 GB — with every leg inside 1 t/s of its pair. Auto-sizing (`llama.h`
`auto_size_ctx`: KV <= 12.5% of RAM) is a ceiling that stops an OOM, not a chat default. It picks
16384 here, which is far more conversation than a REPL turn needs and costs a gigabyte to hold.

**Only the memory row is load-dependent, and this box sits on the edge of it.** Free+inactive read
3936 MB before the window and 4450 MB after, so the 16384-context Q6_K leg was running with a
5.41 GB peak past available memory — the same regime `examples/models.json` records for Q8_0, where
5.03 GB of weights drove 2.99 GB of swap and left the engine in uninterruptible I/O. That is why
the chat default is now the Q4_0 derivation: **17-18 t/s against 11**, **3.40 GB against 5.41**,
for the +7.4% ppl priced above. Q6_K stays one `-m` away.

**Neither hybrid checkpoint can reach the GPU, and that is policy, not breakage.** `TM_DEBUG_CHAIN=1`
refuses both with `usable=0`: `resolve_decode_gpu_auto()` forces every hybrid model onto the CPU,
because the only end-to-end K-quant evidence came from the retired Gated-DeltaNet lane
(`docs/QWEN35_PORT.md`, M3). So the 1.6x above is a CPU-vs-CPU ratio, not a lane that failed to
engage.

### And a reasoning trace nobody asked for

Qwen3.8-4B carries a `<think>` / `</think>` token pair (248068 / 248069) and uses it. On a 200-token
one-shot the plain `<|im_start|>assistant\n` header produced **281 characters of reasoning before
456 characters of answer**; at 17 t/s that is several seconds of silence at the head of every reply,
and the trace is charged to the reply's own budget. The template's `enable_thinking=false` works by
pre-closing an EMPTY block rather than by asking the model to skip the step, and that reproduces
here: the same prompt with `assistant\n<think>\n\n</think>\n\n` produced **zero** reasoning
characters and 326 characters of answer, immediately. Appending `/no_think` to the user turn did
**not** work on this checkpoint — 380 characters of reasoning, more than the plain arm — so the
soft switch is not a substitute for the template. The chat front-end now opens the assistant turn
with the pre-closed block, with `/think` (interactive and `--batch`) or `--think` to restore the
trace.

## The Gated-DeltaNet share, and why it is smaller than it looks (2026-09-17)

Asked "how much of a decode is GDN", the engine's own trace could not answer: a GDN layer (24 of
this model's 32) charged its four projections to `qkv_gemm` and its conv + recurrence to
`attention`, indistinguishable from full attention's rows. The profiler now separates
`gdn_proj` / `gdn_conv` / `gdn_recur` / `gdn_out`.

**Read the profiler's own percentages with care.** `gemv_core` is a labelled *subset* that
double-counts against the projection rows, so the real section time is `total - gemv_core`. On
qwen38_4b_q6k, 32 decode tokens: accounted 3106.7 ms, `gemv_core` 1312.85, so **1793.9 ms is the
truth**. Rebasing on that:

| | ms | share |
|---|---:|---:|
| MLP (`gate_up` + `down`) | 955.4 | 53.3 % |
| **GDN total** | **594.2** | **33.1 %** |
| lm_head | 119.0 | 6.6 % |
| full attention (`qkv_split` + `attention` + `o_proj`) | 104.0 | 5.8 % |
| *of which `gemv_core` (subset)* | *1312.9* | ***73 %*** |

Within GDN: **projections 471.5 ms = 79 %**, recurrence 79.6, conv 43.1. So the honest statement is
not "GDN is a third of decode" but **"a third of decode is GDN, and 79 % of that is
`linear_proj` calling the same `gemv_w` kernel as everything else"**. GDN's *own* code — its
convolution and its recurrence — is **122.7 ms, 6.8 % of a decode**. That bounds GDN-specific
optimisation here at a few percent, and it is why the projection work and the GEMV work are the
same work.

### One win taken, two refused

**Taken.** `conv1d_silu` ran channel-major with the taps innermost — a `K=4` loop carrying a branch
on `tt < 0`, so the innermost loop could not be vectorised and every channel walked four rows at
column `c`. Inverted to tap-major (contiguous, branch-free inner loop over `channels`). The
accumulation order over `j` is unchanged, so it is **bit-identical**, which was checked rather than
asserted — a 64-token greedy oneshot hashes `259ebd3210e4245151549d0caf31d757d5a95cda` before and
after. Paired, interleaved old/new in ONE window, `gdn_conv` ms:

| | old | new |
|---|---:|---:|
| run 1 | 47.73 | 30.25 |
| run 2 | 46.22 | 29.23 |

**1.58×**, with `gdn_recur` — untouched by the change — reading 106.60 / 86.93 old against
104.13 / 90.97 new, i.e. no systematic shift. The two *new* runs had the **larger** totals
(4222.0 / 4035.4 against 4080.6 / 3686.8), so they ran in the busier half and still won. Worth
0.72 ms of an ~84 ms token, about **0.9 %** — kept at that size only because it is measured and
bit-identical.

**Refused.** Fusing `S *= decay` into `S = dec*S + k⊗d` would drop one 64 KB read-modify-write per
value head per layer, but it moves the `dec` factor across a summation and so **breaks
bit-exactness** for roughly 1 %. Parallelising `step()` across its 32 independent value heads is
bit-exact but worth ~2 % against per-layer dispatch overhead that eats a third of it. Neither was
shipped.

The remaining lever is not in GDN at all: it is the shared T=1 K-quant GEMV, which runs at
**48.6 GB/s against 68 GB/s peak** — and see the note below on what a source-level attempt at it
actually produced.

### A source-level swipe at that GEMV, and what it taught (2026-09-17)

The arithmetic says why 48.6 GB/s is where it sits: ~238 instructions per 210-byte super-block is
**~1.13 instructions per weight byte**, and saturating 68 GB/s across four P-cores at 3.2 GHz
would need ~6 IPC. Reaching the bandwidth wall therefore means cutting about a third of the
instructions, and the unpack is ~60 % of them (~6 ALU ops per 16 values against one `vdotq_s32`).

Exactly one op is removable by inspection: `nib | (hi << 4)` is `vsliq_n_u8(nib, hi, 4)`, because
VSLI inserts the low 4 bits of `hi << 4` into `nib` and `nib`'s bits 4..7 are already zero. The
edit is bit-identical (`259ebd32…` unchanged) and it did change codegen — so it was not a no-op.
Disassembling both binaries shows what the backend did with it:

| | `shl` | `orr` | `and.16b` | `sli` | `sdot` |
|---|---:|---:|---:|---:|---:|
| old | 80 | 996 | 263 | 0 | 264 |
| new | 6 | 922 | 226 | 148 | 264 |

74 `shl` and 74 `orr` became **148 `sli`** — the same instruction count, net −37. The "one op
instead of two" reasoning did not survive codegen, so there was nothing for the benchmark to
detect. **Reverted.**

The accompanying measurement was unusable in any case: the four profile windows spanned
**4773–9860 ms** of accounted time, so a 7 % kernel delta could not be read through them. The one
load-invariant signal was `gemv_core`'s *share*, which stayed flat (42.8 % → 43.4 %); if the kernel
had got 7 % faster the share would have fallen by about that much. Both lines agree, and neither is
a win.

**The lesson, which generalises past this kernel:** two independent attempts to speed the T=1
GEMV from C++ have now produced nothing, one because it was reverted as unprovable and one because
the backend handed the instruction straight back. Further work belongs at the assembly level —
read what the compiler emits, then restructure *that* — not in more source-level rewriting of a
loop whose operation count is already near its floor (`sdot` is fixed at one per 16 values).
