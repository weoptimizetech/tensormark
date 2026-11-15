#!/usr/bin/env python3
"""Generate the shared input vectors for the paired microbenchmarks.

Both microbench.cpp (engine) and microbench_torch.py (PyTorch) consume
build/bench_inputs.bin so both sides compute on BYTE-IDENTICAL inputs —
the joiner can then validate output checksums, proving the speed-ups
compare the same math. Layout (float32 little-endian, in order):

    for n in (64, 256, 1024, 4096):   a1[n], w[n*n], a256[256*n]
    for S in (128, 512, 2048):        q[768], K[S*768], V[S*768]
    gelu:                             a[196608], b[196608]
    conv1x1:                          x[96*1024], w[96*96]
    dw3x3:                            x[96*34*34], w[96*9] (engine hardcodes w)

Deterministic seed 7 — regenerate any time, same bytes.
"""
import numpy as np

rng = np.random.default_rng(7)
parts = []
for n in (64, 256, 1024, 4096):
    parts += [rng.uniform(-1, 1, n), rng.uniform(-1, 1, n * n),
              rng.uniform(-1, 1, 256 * n)]
for S in (128, 512, 2048):
    parts += [rng.uniform(-1, 1, 768), rng.uniform(-1, 1, S * 768),
              rng.uniform(-1, 1, S * 768)]
parts += [rng.uniform(-1, 1, 196608), rng.uniform(-1, 1, 196608)]
parts += [rng.uniform(-1, 1, 96 * 1024), rng.uniform(-1, 1, 96 * 96)]
parts += [rng.uniform(-1, 1, 96 * 34 * 34), rng.uniform(-1, 1, 96 * 9)]
blob = np.concatenate([p.astype(np.float32) for p in parts])
out = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "build/bench_inputs.bin"
import os
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
blob.tofile(out)
print(f"wrote {out}: {blob.size} floats, {blob.nbytes} bytes")
