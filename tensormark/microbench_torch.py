# tensormark/microbench_torch.py — paired PyTorch side of the microbench.
# Runs the same shapes on byte-identical inputs as microbench.cpp (see
# gen_bench_inputs.py) and prints JSON: {name: {ms, chk}} — the joiner
# (join_perf.py) validates checksums before trusting any speed-up.
#
#   python3 microbench_torch.py            # eager
#   python3 microbench_torch.py --compile  # torch.compile / inductor
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

# PyTorch's default here is 4 (half the P-cores); honor TM_TORCH_THREADS to
# experiment, else match the default so the pairing stays apples-to-apples.
torch.set_num_threads(int(__import__("os").environ.get("TM_TORCH_THREADS", "4")))


def bench(fn, warmup=3, reps=20):
    last = None
    for _ in range(warmup):
        last = fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        last = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), last


def load_inputs():
    """Byte-identical inputs to microbench.cpp (gen_bench_inputs.py layout)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "build", "bench_inputs.bin")
    blob = np.fromfile(path, dtype=np.float32)
    cur = [0]

    def take(*shape):
        n = int(np.prod(shape))
        v = blob[cur[0]:cur[0] + n].reshape(shape)
        cur[0] += n
        return torch.from_numpy(v.copy())

    return take


def main():
    compile_ops = "--compile" in sys.argv

    def maybe(fn):
        return torch.compile(fn) if compile_ops else fn

    take = load_inputs()
    out = {}

    def emit(name, fn):
        ms, o = bench(maybe(fn))
        # torch.compile may return a lazily-evaluated graph tensor; force it
        # so `o` holds final values before checksumming.
        if hasattr(o, "force_load"):
            o = o.force_load()
        out[name] = {"ms": round(ms, 4),
                     "chk": round(float(o.abs().sum()), 6)}

    for n in (64, 256, 1024, 4096):
        a1 = take(1, n)
        w = take(n, n)
        a256 = take(256, n)          # layout matches gen_bench_inputs.py
        emit(f"gemm_T1_K{n}_N{n}", lambda: a1 @ w)
        emit(f"gemm_T256_K{n}_N{n}", lambda: a256 @ w)
    for S in (128, 512, 2048):
        H, dh = 12, 64
        D = H * dh
        q = take(D)
        K = take(S, D)
        V = take(S, D)
        scale = dh ** -0.5

        def attn():
            # Per-head loop, matching microbench.cpp's per-head structure
            # (same math per head; softmax over the FULL context — a
            # decode query attends everything present, no causal mask).
            o = torch.empty(H, dh)
            for hh in range(H):
                sc = (K[:, hh * dh:(hh + 1) * dh] @ q[hh * dh:(hh + 1) * dh]) * scale
                p = torch.softmax(sc, dim=0)
                o[hh] = p @ V[:, hh * dh:(hh + 1) * dh]
            return o

        ms, o = bench(maybe(attn))
        out[f"attn_decode_H12_dh64_S{S}"] = {
            "ms": round(ms, 4), "chk": round(float(o.abs().sum()), 6)}
    a = take(256 * 768)
    b = take(256 * 768)

    def gelu_fused():
        return 0.5 * (a + b) * (1 + torch.tanh(0.7978845608 * ((a + b) + 0.044715 * (a + b) ** 3)))

    emit("gelu_fused_N196608", gelu_fused)
    x = take(96, 32, 32)
    x2 = x.permute(1, 2, 0).reshape(-1, 96)
    w2 = take(96, 96)

    def conv1x1():
        return (x2 @ w2).reshape(32, 32, 96).permute(2, 0, 1)

    emit("conv1x1_C96_32x32", conv1x1)
    x3 = take(1, 96, 34, 34)
    dw = take(96, 1, 3, 3)
    emit("dw3x3_C96_32x32", lambda: torch.nn.functional.conv2d(x3, dw, groups=96))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
