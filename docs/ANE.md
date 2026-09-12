# ANE prefill: current operating guide

TensorMark can run an initial, fixed-shape prefill prefix on CoreML configured
for CPU + ANE, export its real K/V cache, and finish the remaining layers with
the Metal layer stack. This remains opt-in (`TM_ANE=1`). It is fp16 inference,
not bit-exact CPU inference, and does not yet implement a three-unit serving
controller or establish aggregate capacity under concurrent load.

## Build and run

On Apple Silicon with the CoreML/torch conversion environment:

```sh
python tools/anepack.py tensormark/data/tinyllama/tinyllama_q40.tmq --layers 16 --tokens 1024
# Use the printed package path. Compile once to a stable cache directory:
xcrun coremlcompiler compile /absolute/path/ane_seg.mlpackage /absolute/path/cache-directory

cd tensormark
./build_llama.sh
TM_ANE=1 TM_ANE_LAYERS=16 \
  TM_ANE_PATH=/absolute/path/cache-directory/ane_seg.mlmodelc \
  TM_PREFILL_GPU=1 TM_DECODE_GPU=0 TM_DEBUG_ANE=1 \
  ./build/bench_ane_split data/tinyllama/tinyllama_q40.tmq 1024 3 7 32
```

The positional benchmark arguments are model, token count, rounds, optional
random seed, and optional teacher-forced decode steps. `TM_BENCH_TOKENS` can
name a whitespace-separated file containing exactly T valid token IDs instead
of a synthetic random prompt. Reference computation and decode diagnostics are
outside prefill timing. Setup refusal, engine fallback, and nonfinite output
fail the execution gate; printed quality metrics are not a corpus-quality gate.

The default builder exports actual per-layer K/V. Do **not** use `--no-kv` for
normal continuation: reconstructing every earlier cache from the final hidden
is mathematically different. That mode requires `TM_ANE_APPROX_KV=1` and exists
only for explicit approximate experiments. The earlier claim that K/V export
fails at T=1024 was caused by a different, broken float32 mask graph; corrected
N16/T1024 export works.

Packages have positions [0,T), no past-cache input, and a fixed T. They only run
at cache base zero. Other shapes and later chunks use the ordinary engine.
CoreML configuration and successful shim execution do not establish the
hardware residency of every individual operation.

## Measured improvement on this M1 / 8 GB

TinyLlama Q4_0 source weights, N16/T1024, remaining six layers in the engine:

| Stage | Warm prefill tok/s | Initial logit relative L2, seed 7 |
|---|---:|---:|
| Broken double-execution integration | 207 | 46.0% |
| Corrected ANE + CPU tail, hidden-only | 739–763 | 6.27% |
| Actual K/V + CPU tail | 769–772 | 6.27% |
| Actual K/V + per-layer GPU GEMMs | 958–979 | 6.27% |
| Actual K/V + fused Metal suffix | 1,230–1,267 | 6.29% |

The double-execution fix prevents the complete CPU stack from running after
ANE + tail have already completed. The suffix API reuses the original Metal
weight mapping and cache indices; it adds neither another model copy nor a
separate tail package. Warm ANE segment execution is about 460–490 ms, K/V
handoff about 5 ms, and the fused GPU tail about 330 ms. Cold first predictions
can be much slower; these warm rates are not cold-TTFT claims.

Three distinct 1024-token documentation-text fixtures (README, contributing
guide, and numerical-test notes, using TinyLlama's tokenizer) gave:

| Fixture | Warm tok/s | Prefill relative L2 | Decode relative L2 | Teacher-forced top-1 |
|---|---:|---:|---:|---:|
| README | 1,272–1,275 | 2.01% | 2.66% | 32/32 |
| Contributing guide | 1,270–1,296 | 1.71% | 2.58% | 32/32 |
| Numerical notes | 1,272–1,287 | 3.15% | 2.19% | 32/32 |

That is **96/96 on these three short continuation checks**, not a general
quality guarantee or freely generated transcript comparison. A random-token
prompt gave 15/16 matches with the half tail; the fp32-tail diagnostic gave
31/32 over a longer continuation and 1,139–1,193 tok/s. Those unequal-length
checks do not establish a quality advantage for fp32, so the existing Metal
precision default is unchanged (`TM_LLAMA_GPU_HALF=0` remains a diagnostic).
Full traces and historical corrections: [ane_layer_split_results.txt](ane_layer_split_results.txt).
Local fixture hashes and raw probe results are kept out of the repository.

## Preserve inactive models in compressed storage

Do not discard inactive weights or repeatedly rebuild/download packages. The
stdlib-only archive tool preserves file bytes, permissions and modification
times (important for model cache identity), with SHA-256 checksums per file:

```sh
# Source stays present by default:
python tools/model_archive.py pack /path/model.mlpackage /path/model.mlpackage.tar.gz

# Only for an INACTIVE asset: verify the archive, then remove the expanded copy:
python tools/model_archive.py pack /path/model.mlpackage /path/model.mlpackage.tar.gz --remove-source

python tools/model_archive.py verify /path/model.mlpackage.tar.gz
python tools/model_archive.py restore /path/model.mlpackage.tar.gz /path/model.mlpackage
```

Restore refuses to overwrite an existing destination, rejects unsafe archive
paths/links, verifies checksums and gzip integrity, and checks available space.
It keeps the archive. Restore an archived source package before passing that
source path to anepack/CoreML; no reconversion or download is required. Keep the
active compiled `.mlmodelc` expanded. Logical compression savings can differ
from physical APFS free-space changes because of clones/snapshots and other
concurrent disk activity.

The generated N16/T1024 source package (cache key `20e967ad3c0492dd74c0`) was
archived and verified: **1,411,939,989 bytes → 844,456,421 bytes**, about 40%
smaller. The working `ane_seg.mlmodelc` remains expanded. The older hidden-only
package was removed before the archive-preservation policy was adopted; its
historical measurements remain recorded.

For home-box overflow storage, transfer the compressed archive to a confirmed
`user@host:/absolute/path`, verify its SHA-256 remotely and locally, then remove
only the authorized local redundant copy. Do not move active model files or
assume a remote transfer succeeded. The N16/T1024 source archive was successfully
offloaded to an authorized home box: the remote SHA-256 matched a fresh local
checksum before the redundant local archive was removed. The active compiled
package was retained and its presence checked afterward.

The full archive SHA-256 is
`de18b9ac741d548432364c48c288dfcec987f90de7e61d557cb3a066986b4ca0`.
The local transfer record at
`~/.cache/tensormark/ane/20e967ad3c0492dd74c0/ane_seg.mlpackage.remote.json`
contains the exact remote `user@host:/absolute/path`, checksum and byte count.
To recover the source package, retrieve that archive, confirm its SHA-256
matches the record, then use the `verify` and `restore` commands above. Remote
retrieval is explicit, not automatic; the active compiled package needs no
retrieval or rebuild.

## Concurrent-lane capacity probe

`tools/bench_ane_capacity.py` drives two persistent, process-isolated workers:
CPU/Accelerate or Metal for the ordinary engine lane, and actual-K/V ANE plus
the Metal suffix for the second lane. Separate processes are intentional:
mutable session state and process-global Metal/hybrid state are not a safe
in-process multi-session serving interface. This is a benchmark/controller
foundation, **not a request-serving scheduler**.

```sh
# build_llama.sh also builds bench_prefill_lane; use the existing compiled package.
python3 tools/bench_ane_capacity.py tensormark/data/tinyllama/tinyllama_q40.tmq \
  --ane-path /absolute/path/cache-directory/ane_seg.mlmodelc \
  --tokens 1024 --rounds 3 --seed 7 --exact-lane metal --half 1 \
  --token-file /path/1024-token-prompt.txt --output /path/new-capacity-report.json
```

Omit `--token-file` to use the deterministic random prompt; `--exact-lane cpu`
selects the CPU/Accelerate control instead. The historical “exact lane” name
means the ordinary non-ANE engine, not bitwise CPU equality for half Metal.
`--half 0` selects fp32 Metal glue; ANE itself remains fp16. `--exact-tokens`
with `--exact-token-file` lets the ordinary lane serve a different prompt
length than the fixed ANE package shape — the heterogeneous serving mix.

Workers initialize sequentially and remain resident. Before/after baselines
run one lane while its peer is idle; these are **co-resident idle-peer
baselines**, not standalone-memory-footprint measurements. Each concurrent
batch dispatches one prefill to both workers before waiting for either.
Aggregate throughput is completed tokens divided by the common controller
wall time, never the sum of separate token rates. Reports include per-lane
latency slowdowns, bracketed baseline drift, CPU-reference differences and
maximum deviation from each lane's warmed logits. Every request resets its
cache; warmup and CPU-reference computation are excluded from token accounting.

The worker rejects ANE or Metal fallback (including a CPU fallback for the ANE
suffix), malformed prompts and nonfinite logits. The controller rejects
protocol errors, timeouts, nonzero exits and replay drift above
`--replay-tolerance` (default `1e-4` absolute logit difference). It cleans up
only its own children and refuses to overwrite an existing report. Inherited
`TM_*` overrides are cleared; supported configuration is explicit in the
report. Model weights and source packages are neither generated nor moved.

These are **finite paired bursts of mixed numerical quality**, not sustained
serving capacity, a quality-equivalence result, or proof that every operation
ran on ANE. The report preserves raw observations for those distinctions;
continuation quality remains covered by the separate teacher-forced tests.

Measured on this M1/8 GB (TinyLlama Q4_0, T=1024, half Metal glue, exact lane
Metal, actual-K/V ANE+suffix lane; documentation-text fixtures, 3 then 8
concurrent pairs):

| Fixture (pairs) | Aggregate tok/s | Metal paired latency | ANE paired latency | Replay drift |
|---|---:|---:|---:|---:|
| README (3) | 1,233 | ×1.347 vs post-run baseline | ×1.539 | 0 both lanes |
| Contributing (8) | 1,294 | ×1.293 | ×1.214 | 0 both lanes |

The 8-pair run processed 16,384 tokens in 12.657 s with batch rates 1,284–1,305
tok/s and 50/50 top-1 observations. Idle-peer Metal baselines warmed
653→831 (README) and stayed ~830–837 (contributing) tok/s; paired aggregate is
+48–55% over the faster post-run Metal baseline. Both lanes kept zero deviation
from their warmed logits and every request passed its CPU-reference/top-1
observation. **Pairing is not latency-neutral**: each lane's own request got
21–35% slower under contention. This demonstrates additional finite-burst
throughput, not unrestricted concurrent admission, an SLA policy, or sustained
independent serving.

A heterogeneous mix (`--exact-tokens 256 --exact-token-file` with the same
1,024-token ANE fixture, 8 pairs) behaves differently: the shorter exact-lane
requests interleave with the ANE request instead of piling up behind it. The
Metal lane ran ×1.018 and the ANE lane ×0.990 versus idle-peer baselines —
contention within noise — while processing 10,240 tokens in 6.740 s for
**1,519 tok/s aggregate** (batch rates 1,471–1,551, zero replay drift, 50/50
top-1). Same-quality mixed sizes are not equivalent work, so cross-run
aggregate comparisons across different `--exact-tokens` values are indicative,
not matched.

## Admission-share controllers under stochastic contention

`tools/bench_ane_controller.py` extends the capacity probe with a routing-share
study: batches of identical 1,024-token requests are dispatched to the ANE and
Metal study lanes with ANE share ρ, and controllers choose ρ batch by batch.
A seeded Bernoulli metal co-tenant (`--competitor-probability`) emulates
stochastic contention; every controller sees the **same** schedule. Reports
carry a measurement-scope block (fixed-size cache-reset prefills, controller-clock
queueing, fp16 ANE numerics) so the accounting cannot be mistaken for a serving SLA.

```sh
python3 tools/bench_ane_controller.py tensormark/data/tinyllama/tinyllama_q40.tmq \
  --ane-path "$HOME/.cache/tensormark/ane/20e967ad3c0492dd74c0/ane_seg.mlmodelc" \
  --tokens 1024 --requests-per-batch 4 --rounds 2 --ratios 0,0.25,0.5,0.75,1 \
  --pi-rounds 12 --competitor-probability 0.5 --competitor-requests 1 --seed 7 \
  --token-file <local prompt fixture> --timeout 240 --output report.json
```

Quiet-regime static sweep (this M1/8 GB, TinyLlama Q4_0, 4×1024-token batches):

| ρ (ANE share) | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| aggregate tok/s | 833 | 1,022 | 1,299 | **1,378** | 1,267 |

Balance-PI on per-lane rate slowdowns scored 0.66× the best static share in the
quiet regime — the same degenerate-signal disease that killed the GPU/CPU
balance-PI: each lane's realized rate is deflated by the whole-batch wall time,
which is dominated by its peer, so the error signal is confounded by batch
composition and the controller limit-cycles instead of climbing the gradient.

Stochastic regime (competitor p=0.5, 12 batches per controller, one seeded
schedule shared by all four controllers):

| Controller | Aggregate tok/s | vs static | ρ behaviour |
|---|---:|---:|---|
| static best-quiet (ρ=0.75 fixed) | 1,143 | 1.00 | — |
| omniscient regime switcher | 1,185 | ×1.037 | ρ=0.75 every batch |
| balance-PI | 838 | ×0.733 | limit-cycled 0.0–0.5 |
| wall-time extremum seeker | 1,016 | ×0.888 | stable band 0.375–0.625 |

The decisive number is not ES's 0.89× but the omniscient controller's ρ
trajectory: it **chose ρ=0.75 in every batch, loaded and quiet alike**, because
the measured best share was 0.75 in both regimes. This competitor loads the
Metal lane; it does not move the ANE/Metal routing optimum of the study lanes.
With the optimum regime-invariant, the dynamic opportunity itself was ≈ zero —
the blind-adaptivity upper bound (omniscient − static, +3.7%) is within
run-to-run noise — so no blind controller could have won, and both adaptive
controllers only added variance. PI reproduced the confounded-signal disease;
wall-time ES was better behaved (no limit cycle, no divergence) but its ±0.125
dither around a 0.5 center plus a 12-batch horizon never explored ρ≥0.75.

Fine sweep at 0.05 granularity (both sweeps persisted in the report since the
sweep-persistence fix; 2 batches per ρ, so per-ρ noise is ±5% and run-to-run
drift is 2.5–3.4% — the 0.70 point dips below 0.65, which is variance, not
structure):

| ρ | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| quiet tok/s | 1,286 | 1,303 | 1,356 | 1,340 | **1,392** | 1,326 | 1,381 | 1,301 | 1,264 |
| loaded tok/s | 935 | 938 | 1,034 | 1,014 | 1,148 | **1,154** | 1,151 | 1,125 | 1,120 |

The peak is a **regime-invariant plateau at ρ ≈ 0.75–0.85** (loaded plateau
spread 0.5%), not a sharp optimum: the quiet argmax moved between 0.75 and 0.85
across runs, and the loaded argmax (0.80) is one grid step from quiet's — inside
noise. Contention lowers the whole curve (−17.6% at ρ=0.75) but does not move
its shape. Confirming that, the rerun's omniscient switcher — given the
fine-grained per-regime optima (0.80 loaded / 0.75 quiet) *and* the schedule —
still did not beat the fixed ρ=0.75 policy (×0.997). A practical controller
should park at ~0.80 with small dither; the earlier ES band (0.375–0.625)
probed the wrong region entirely.

Honest scope: one fixed-size workload, 12 batches per controller, competitor on
the Metal lane only, 2 batches per fine-sweep ρ. The untested regimes are the
ones where the optimum actually moves — a co-tenant that shifts the ANE/Metal
balance itself, heterogeneous request sizes inside the study lanes, or thermal
throttling of the ANE tier. Within this regime the answer is: adaptivity paid
nothing because there was nothing to adapt to; the static share was already
optimal in both regimes, and the fine sweep did not change that — it only
localized the plateau.

### Three-way surface: ANE share × GPU/CPU row split

The study lanes cover two of the three compute units; the third (CPU/AMX)
enters through the ordinary lane's hybrid row split. `--metal-hybrid` pins
`TM_LLAMA_HYBRID` (fraction of prefill rows given to the CPU); the worker
honors it only with the explicit `TM_BENCH_HYBRID=1` opt-in, so ambient
inheritance still cannot silently change a lane's numerics. Solo Metal at
T=1024: 1,250 → 1,097 ms peak at frac 0.25 (+15%).

Joint static surface (aggregate tok/s, quiet, 4×1024-token batches; 2–3
batches per point, cross-run solo-reference drift up to ~10% — thermal):

| frac \ ρ | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 |
|---|---:|---:|---:|---:|---:|
| 0 (GPU only) | — | 1,331–1,352 | — | 1,372–1,406 | 1,348 |
| 0.25 | 1,458 | 1,444–1,451 | 1,404 | 1,398 | — |
| 0.375 | — | 1,403–1,406 | 1,403–1,432 | 1,446 | 1,072 (outlier) |

Two honest readings:

- The **interaction is real and reproducible** (seed 7 and seed 11 agree):
  with a fast Metal lane (frac 0.25, service ~1,360 ms) the ρ argmax moves to
  0.65–0.7 — give the faster lane more requests; with a slower split (frac
  0.375, service ~1,530 ms) it moves back to 0.8. The 2-way "regime-invariant
  plateau" conclusion was specific to a fixed lane speed.
- The **payoff is small and the top is flat**: the two best corners
  (ρ≈0.65–0.7, frac 0.25) and (ρ≈0.8, frac 0.375) tie at ~1,446–1,458, about
  1% apart, and ~3–5% over the best hybrid-off points (1,372–1,416). Caveats:
  frac 0.125 scored below frac 0 despite a faster solo reference, and the
  ρ=0.85/frac=0.375 point collapsed (1,072) — both are variance, not structure.

Verdict for a 3-way blending PI: the surface's peak is a flat ridge, so
balancing control has almost nothing to harvest even where the optimum moves.
The measured answer is a **parked composition**: ρ≈0.70 with frac≈0.25 (or
ρ≈0.80 with frac≈0.375), letting the in-engine hybrid ESC own the frac axis.
A 3-way balance-PI remains contraindicated — the confounded per-lane signal
that killed it twice now has three peers polluting each lane's wall.

Sustained-soak check of the parked composition (30 batches of the joined
3-way system, ρ=0.70 + frac 0.25, trajectory retained since the batch-trajectory
patch): batch wall 3,001/3,133/3,036/2,970 ms by quartile, ANE service
1,119/1,099/1,130/1,048 ms, Metal service 1,466/1,565/1,463/1,396 ms — all
three units flat; aggregate 1,353.9 tok/s with no trend. A per-unit soak
(Metal/ANE alternating solo, 31 runs each) agrees: Metal flat 1,234→1,203 ms,
ANE only a warmup transient (one cold 2,644 ms outlier, then flat ~800 ms).
No thermal mover at these timescales; the parked verdict has no known regime
that moves its optimum on this chip.

### Admission depth and memory pressure

The routing studies fix admission (4 requests/batch, 2–3 persistent workers);
real serving must decide how much work the chip can hold. Measured on 8 GB
(join ρ=0.70 + frac 0.25, equal total work per depth): aggregate stays flat
across 4/8/16-wide admission — 1,392/1,325/1,378 tok/s, queue ≤1.3 ms, batch
wall scales linearly with the ANE queue without degrading per-token cost. A
16-wide run under otherwise quiet conditions held 1,344 tok/s with wall
quartiles 11,670/12,063/12,307/12,712 ms. One earlier 16-wide run showed a
+55% final-quartile wall spike, but it did not reproduce in clean reruns — it
tracked unrelated machine activity, not admission depth.

System-wide vm_stat during a clean 16-wide run: Swapins delta ~8,062 pages
(≈131 MB) and Pageins delta ~1.0M pages (≈16.6 GB — dominated by the expected
mmap re-reads of the mapped weights, not clearly anonymous thrash); free-page
minimum 0.05 GiB (normal macOS compressor behavior). No throughput collapse at
the tested depth. The engine's per-prefill pressure sample (page faults +
context switches, `TM_HYBRID_P_CSW`/`TM_HYBRID_P_FAULT`) already exists for
the hybrid row split; a production admission guardrail should gate on that
same per-process signal (refuse or queue before thrash), not on throughput
balancing. Untested beyond depth 16, longer contexts, and decode-stream
coexistence — those are the regimes where paging pressure could still bind.

## Regression gates

```sh
python -m unittest discover -s tools/tests -p test_model_archive.py -v
python -m unittest discover -s tools/tests -p test_ane_benchmark.py -v
# CoreML/torch environment:
python -m unittest discover -s tools/tests -p 'test_ane*.py' -v
cd tensormark
TM_LLAMA_GPU_HALF=0 ./build/test_llama_gpu_suffix data/tinyllama/tinyllama_q40.tmq
TM_LLAMA_GPU_HALF=1 ./build/test_llama_gpu_suffix data/tinyllama/tinyllama_q40.tmq
./build/test_ane_dispatch data/tinyllama/tinyllama_q40.tmq
```

The suffix tests cover first layers 0/16/21, all tail-cache heads/rows,
unchanged prefix caches and invalid arguments, in fp32 and half modes. The
exact fake-ANE oracle catches double execution and validates actual-K/V decode.
The full GPU-stack, chunked continuation, greedy-chain and hybrid gates pass
in both modes. The exact CPU Q4 oracle remains 32/32 with max error 1.1063e-4.
The earlier sanitizer diagnostic was user-cancelled; no sanitizer pass is claimed.
