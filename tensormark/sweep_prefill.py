#!/usr/bin/env python3
"""Serial, solo prefill sweep — nothing else may run on the GPU meanwhile."""
import itertools
import os
import subprocess
import sys

grid = {
    "TM_LLAMA_Q4MM": ["0", "1", "2"],
    "TM_LLAMA_FLASH": ["1", "0"],
    "TM_LLAMA_GPU_HALF": ["1", "0"],
    "TM_ATTN_PREFILL_BLOCK": ["128", "256"],
}
rows = []
for combo in itertools.product(*grid.values()):
    env = dict(os.environ, TM_DECODE_GPU="1", **dict(zip(grid, combo)))
    out = subprocess.run(["./build/hybrid_bench", "data/tinyllama/tinyllama_q40.tmq", "2000", "3"],
                         env=env, capture_output=True, text=True).stdout
    vals = [float(l.split('"tok_per_s":')[1].rstrip("}")) for l in out.splitlines() if "tok_per_s" in l]
    best = max(vals) if vals else 0.0
    rows.append((best, dict(zip(grid, combo))))
    print(f"{best:8.1f} t/s  {dict(zip(grid, combo))}", flush=True)

rows.sort(key=lambda r: r[0], reverse=True)
print("\n== ranking ==")
for best, env in rows:
    print(f"{best:8.1f} t/s  {env}")
