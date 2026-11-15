// ANE offload probe (2026-09-09): measured go/no-go for a CPU/GPU/ANE
// three-way split. Converted TinyLlama's largest GEMM (MLP up-projection
// 2048 -> 5632) to CoreML fp16/fp32, ran on ComputeUnit.ALL (ANE-preferred),
// coremltools 9 (needs Python <=3.11 for native bindings), M1 8 GB:
//
//   decode GEMV  (batch 1):   fp16 0.73 ms = 31.7 GB/s eff | fp32 1.25 ms = 36.9 GB/s
//   prefill GEMM (batch 256): fp16 2.26 ms = 2617 GFLOP/s  | fp32 5.9 ms = 1001 GFLOP/s
//
// Our shipped paths for the same GEMM: GPU Q4 decode ~53 GB/s effective;
// GPU prefill ~2.15 TFLOP/s effective (640 MB Q4 weights vs 2.2 GB fp16).
//
//   22-layer GEMM stack (real per-layer sizes, 0.97B params fp16):
//   batch 256: SUSTAINED 278 ms/forward = 921 tok/s; batch 1024: 1458-1600 ms
//   = 627-702 tok/s (CORRECTION: the earlier 951 t/s at 1077 ms was measured
//   under ComputeUnit.ALL — MPSGraph assigns ops to GPU too, so that number
//   was GPU-assisted. Pinned ComputeUnit.CPU_AND_NE gives the true ANE-only
//   rate). The ANE stays weight-feed-bound, not compute-bound, at these
//   shapes.
//
// Concurrent addition probe (CPU/GPU hybrid bench + ANE stack running
// SIMULTANEOUSLY, each side measured under the other's load): hybrid holds
// 827-851 t/s (97% of its 858 solo), ANE holds 702 t/s (~112% of its 627
// pinned solo) — i.e. the units barely contend. ANE's 1024-token forward
// (1458 ms) fits entirely inside the hybrid's 2048-token forward (2448 ms),
// so the addition arithmetic nets ~1250 t/s effective prefill (+46%) IF a
// real integration existed. What remains between probe and product is
// engineering, not physics: static-shape CoreML rewrite, fp16 2.2 GB
// duplicate weights on an 8 GB machine, output sync per forward.
//
// Verdict: decode offload loses (ANE effective bandwidth ~60% of ours at 2x
// the bytes; weight palettization unsupported on M1-class ANE). Prefill has
// a +21%/GEMM ceiling but requires a full CoreML re-implementation (static
// shapes, KV-cache management, 3.4x weight footprint on an 8 GB machine) —
// revisit only if prefill becomes the strategic bottleneck.
//
// Run: uv venv --python 3.11 /tmp/ane2 && uv pip install --python
// /tmp/ane2/bin/python coremltools numpy torch --index-url ... (torch cpu)
// then: /tmp/ane2/bin/python docs/ane_offload_probe.py
import numpy as np, time, torch, torch.nn as nn
import coremltools as ct

D, F = 2048, 5632   # TinyLlama MLP up-projection: the largest GEMM per layer

def build(batch, fp16):
    W = (np.random.default_rng(7).standard_normal((F, D)) * 0.02).astype(np.float16 if fp16 else np.float32)
    mod = nn.Linear(D, F, bias=False)
    with torch.no_grad():
        mod.weight.copy_(torch.from_numpy(W.astype(np.float32)).view_as(mod.weight))
    if fp16: mod = mod.half()
    traced = torch.jit.trace(mod.eval(), torch.zeros(batch, D, dtype=torch.float16 if fp16 else torch.float32))
    mlmodel = ct.convert(traced,
                         inputs=[ct.TensorType(name="x", shape=(batch, D))],
                         compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
                         compute_units=ct.ComputeUnit.ALL)
    path = f"/tmp/mlp_b{batch}_{'fp16' if fp16 else 'fp32'}.mlpackage"
    mlmodel.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)

for batch, label in ((1, "decode GEMV"), (256, "prefill GEMM")):
    for fp16 in (True, False):
        try:
            m = build(batch, fp16)
            x = (np.zeros((batch, D), np.float16 if fp16 else np.float32))
            for _ in range(20): y = m.predict({"x": x})
            ts = []
            for _ in range(60):
                t0 = time.perf_counter(); y = m.predict({"x": x}); ts.append(time.perf_counter() - t0)
            ms = float(np.median(ts) * 1e3)
            bytes_w = F * D * (2 if fp16 else 4) / 1e6
            flops = 2 * batch * D * F / 1e9
            print(f"{label:14s} {'fp16' if fp16 else 'fp32'}: {ms:8.3f} ms | {flops/ms*1e3:7.1f} GFLOP/s | weights {bytes_w:5.0f} MB -> {bytes_w/ms:6.1f} GB/s")
        except Exception as e:
            print(f"{label:14s} {'fp16' if fp16 else 'fp32'}: FAILED {type(e).__name__}: {str(e)[:120]}")
