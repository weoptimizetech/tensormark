#!/usr/bin/env python3
"""render_readme_perf.py — the README's performance tables, as markdown.

Prints the two tables the README carries, generated from
docs/perf_evidence.json so the numbers are never typed into the prose by hand.
The README is still hand-composed around them, but a table update is a re-run of
this and a paste, not a transcription.

Usage:
    python3 tensormark/render_readme_perf.py            # both tables
    python3 tensormark/render_readme_perf.py --kernels  # kernel table only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "perf_evidence.json"

KERNEL_ORDER = [
    ("gemm_T256_K4096_N4096", "gemm T256 K4096 N4096"),
    ("gemm_T256_K1024_N1024", "gemm T256 K1024 N1024"),
    ("gemm_T256_K256_N256", "gemm T256 K256 N256"),
    ("gemm_T256_K64_N64", "gemm T256 K64 N64"),
    ("gemm_T1_K4096_N4096", "gemm T1 K4096 N4096 (decode GEMV)"),
    ("gemm_T1_K1024_N1024", "gemm T1 K1024 N1024"),
    ("gemm_T1_K256_N256", "gemm T1 K256 N256"),
    ("gemm_T1_K64_N64", "gemm T1 K64 N64"),
    ("attn_decode_H12_dh64_S2048", "attn decode ctx 2048 (12 heads × 64 dim)"),
    ("attn_decode_H12_dh64_S512", "attn decode ctx 512 (12 heads × 64 dim)"),
    ("attn_decode_H12_dh64_S128", "attn decode ctx 128 (12 heads × 64 dim)"),
    ("dw3x3_C96_32x32", "depthwise 3×3, C96 32×32"),
    ("conv1x1_C96_32x32", "conv 1×1, C96 32×32"),
    ("gelu_fused_N196608", "gelu fused, 196,608 elems"),
]

# NOTE: the README prints exactly this legend, then adds one hand-written
# paragraph explaining the † marks. Keep the two in step — the wording here was
# once "both sides provably compute the same math", which the README replaced
# with the accurate "same arithmetic summed in a different order (fp32
# reassociation)", so a re-render used to silently revert a correction.
LEGEND = """
Shape legend: `T` = rows (tokens), `K` = contraction dim, `N` = output columns;
`ctx` = cached sequence length in attention decode, attention shape as
`heads × head_dim`; `C96 32×32` = 96 channels over a 32×32 spatial map. All fp32.
Both sides run on byte-identical shared inputs and pass a checksum gate at max rel
deviation 6.9e-04 — the same arithmetic summed in a different order (fp32
reassociation), not a different computation.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--evidence", default=str(EVIDENCE))
    a = ap.parse_args()
    ev = json.loads(Path(a.evidence).read_text())

    if not a.kernels:
        tr = ev.get("training", {})
        ib = ev.get("inference_bench", {})
        dp = ev.get("design_point", {})
        print("| what | baseline | TensorMark | speed-up |")
        print("|---|---|---:|---|")
        if tr:
            print(f"| MNIST CNN, full 60k epoch | {tr['mnist_torch_compile']:,.0f} "
                  f"samples/s (`torch.compile`) | {tr['mnist_ours']:,.0f} samples/s "
                  f"| **{tr['speedup']:.2f}×** |")
        if ib:
            print(f"| TinyLlama prefill, 2000 tok | {ib['prefill_llama']['median']:.1f} "
                  f"tok/s (`llama-bench`) | {ib['prefill_ours']['median']:.1f} tok/s "
                  f"| **{ib['prefill_speedup']:.2f}×** |")
            if {"prefill_mlx", "prefill_speedup_mlx"} <= ib.keys():
                print(f"| TinyLlama prefill, 2000 tok | {ib['prefill_mlx']['median']:.1f} "
                      f"tok/s (`mlx-lm`) | {ib['prefill_ours']['median']:.1f} tok/s "
                      f"| **{ib['prefill_speedup_mlx']:.2f}×** |")
            print(f"| TinyLlama decode, 128 tok | {ib['decode_llama']['median']:.1f} "
                  f"tok/s (`llama-bench`) | {ib['decode_ours']['median']:.1f} tok/s "
                  f"| **{ib['decode_speedup']:.2f}×** |")
            if {"decode_mlx", "decode_speedup_mlx"} <= ib.keys():
                print(f"| TinyLlama decode, 128 tok | {ib['decode_mlx']['median']:.1f} "
                      f"tok/s (`mlx-lm`) | {ib['decode_ours']['median']:.1f} tok/s "
                      f"| **{ib['decode_speedup_mlx']:.2f}×** |")
        if dp.get("decode_speedup"):
            # The ‡ marks this row as a second campaign. The README's hand-written
            # paragraph after the table explains it and says to read the ratio
            # rather than either absolute rate; a re-render must not drop the mark.
            print(f"| TinyLlama decode, 64 tok | {dp['decode_pytorch']:.2f} tok/s "
                  f"(PyTorch fp16 on MPS) | {dp['decode_tensormark']:.1f} tok/s "
                  f"| **{dp['decode_speedup']:.2f}×** ‡ |")
        print()

    kern = ev.get("kernels", {})
    print("| op (shape) | torch.compile ms | TensorMark ms | speed-up (range over rounds) |")
    print("|---|---|---|---|")
    unstable = 0
    for key, label in KERNEL_ORDER:
        v = kern.get(key)
        if not v:
            continue
        rng = f"{v['ratio_min']:.2f}–{v['ratio_max']:.2f}"
        if not v["sign_stable"]:
            rng += " †"
            unstable += 1
        print(f"| {label} | {v['torch_compile_ms']:.4f} | {v['ours_ms']:.4f} "
              f"| {v['speedup']:.2f}× ({rng}) |")
    print(LEGEND.strip())
    print(f"\n({unstable} of {len(KERNEL_ORDER)} rows change sign between rounds.)")


if __name__ == "__main__":
    main()
