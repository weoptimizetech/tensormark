#!/usr/bin/env python3
"""gen_benchmark_figs.py — the README's two charts, drawn from the evidence.

Every bar is read from docs/perf_evidence.json, which
tensormark/collect_perf_evidence.py derives from the raw measurement files.
Nothing is typed in here: a series the evidence does not hold is reported and
skipped, never guessed at.

That rule exists because of a specific defect. This script used to carry its
numbers as literals under a docstring claiming it sourced them from the measured
tables, and one bar had drifted away from every measurement: the llama.cpp
baseline in decode_llamacpp.png was drawn at 74.0 tok/s while the three recorded
llama.cpp decode rounds were 79.40 / 78.08 / 72.36 — median 78.08, min 72.36. No
recorded statistic is 74.0, and the bar beside it, 84.0, was TensorMark's *best*
round rather than its median. The figure showed 1.14x above a table reporting
parity. Now a bar without evidence does not get drawn.

Outputs:
    docs/figs/kernel_speedups.png    per-shape kernel speed-up vs torch.compile
    docs/figs/endtoend_speedups.png   training and serving vs their baselines
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figs"
EVIDENCE = ROOT / "docs" / "perf_evidence.json"

TM, BASE, GRID = "#2E7D32", "#9E9E9E", "#D0D0D0"

# Display order and names, matching join_perf.py's KERNEL_ORDER.
KERNELS = [
    ("gemm_T1_K64_N64", "gemm T1 K64 N64"),
    ("gemm_T1_K256_N256", "gemm T1 K256 N256"),
    ("gemm_T1_K1024_N1024", "gemm T1 K1024 N1024"),
    ("gemm_T1_K4096_N4096", "gemm T1 K4096 N4096"),
    ("gemm_T256_K64_N64", "gemm T256 K64 N64"),
    ("gemm_T256_K256_N256", "gemm T256 K256 N256"),
    ("gemm_T256_K1024_N1024", "gemm T256 K1024 N1024"),
    ("gemm_T256_K4096_N4096", "gemm T256 K4096 N4096"),
    ("attn_decode_H12_dh64_S128", "attn decode ctx 128"),
    ("attn_decode_H12_dh64_S512", "attn decode ctx 512"),
    ("attn_decode_H12_dh64_S2048", "attn decode ctx 2048"),
    ("conv1x1_C96_32x32", "conv 1x1, C96 32x32"),
    ("dw3x3_C96_32x32", "depthwise 3x3, C96 32x32"),
    ("gelu_fused_N196608", "gelu fused, 196608"),
]


def load_evidence(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"no evidence at {path}\n"
                         f"run: python3 tensormark/collect_perf_evidence.py")
    return json.loads(path.read_text())


def speedup_chart(plt, path, rows, title, xlabel):
    """Horizontal log-scale bars of speed-up, with the note at 1.00x."""
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    lows = [r[2] if len(r) > 2 else r[1] for r in rows]
    highs = [r[3] if len(r) > 3 else r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(rows) + 1.8))
    y = range(len(rows))
    ax.barh(list(y), [v - 1.0 for v in vals], left=1.0, height=0.55,
            color=[TM if v >= 1.0 else "#B71C1C" for v in vals])
    for i, (v, lo, hi) in enumerate(zip(vals, lows, highs)):
        if hi > lo:
            ax.plot([lo, hi], [i, i], color="#333333", lw=1.0, zorder=3)
            ax.plot([lo, hi], [i, i], "|", color="#333333", ms=6, zorder=3)
        ax.annotate(f"{v:.2f}x" if v < 10 else f"{v:.0f}x",
                    (v, i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold")
    ax.axvline(1.0, color=BASE, lw=1.4, zorder=4)
    ax.annotate("parity", (1.0, len(rows) - 0.35), color=BASE, fontsize=8,
                ha="center", va="bottom")
    ax.set_yticks(list(y), labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.5, max(highs) * 2.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", color=GRID, ls=":", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify each chart has evidence; render nothing")
    ap.add_argument("--evidence", default=str(EVIDENCE))
    a = ap.parse_args()

    ev = load_evidence(Path(a.evidence))
    plans: list[tuple[str, list | None, str]] = []

    # ---- chart 1: kernel speed-up per shape --------------------------------
    kern = ev.get("kernels", {})
    krows = []
    for key, label in KERNELS:
        v = kern.get(key)
        if not v:
            continue
        krows.append((label, v["speedup"], v["ratio_min"], v["ratio_max"]))
    plans.append(("kernel_speedups.png", krows or None,
                  "docs/perf_evidence.json:kernels"))

    # ---- chart 2: end-to-end vs each baseline ------------------------------
    e2e = []
    tr = ev.get("training", {})
    if tr.get("speedup"):
        e2e.append(("MNIST training\nvs torch.compile", tr["speedup"]))
    ib = ev.get("inference_bench", {})
    if ib.get("prefill_speedup"):
        e2e.append(("TinyLlama prefill\nvs llama.cpp", ib["prefill_speedup"]))
    if ib.get("prefill_speedup_mlx"):
        e2e.append(("TinyLlama prefill\nvs mlx-lm", ib["prefill_speedup_mlx"]))
    if ib.get("decode_speedup"):
        e2e.append(("TinyLlama decode\nvs llama.cpp", ib["decode_speedup"]))
    if ib.get("decode_speedup_mlx"):
        e2e.append(("TinyLlama decode\nvs mlx-lm", ib["decode_speedup_mlx"]))
    dp = ev.get("design_point", {})
    if dp.get("decode_speedup"):
        e2e.append(("TinyLlama decode\nvs PyTorch fp16", dp["decode_speedup"]))
    plans.append(("endtoend_speedups.png", e2e or None,
                  "docs/perf_evidence.json:training, inference_bench, design_point"))

    ok = [(n, d) for n, d, _ in plans if d]
    missing = [(n, src) for n, d, src in plans if not d]
    for n, d in ok:
        print(f"  ok    {n} ({len(d)} bars)")
    for n, src in missing:
        print(f"  SKIP  {n} (no evidence for {src})")

    if a.check:
        return 1 if missing else 0
    if missing:
        print("\nrefusing to draw the skipped charts: their numbers are not in "
              "the evidence file", file=sys.stderr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in ok:
        if name == "kernel_speedups.png":
            speedup_chart(plt, OUT / name, data,
                          "TensorMark vs torch.compile (inductor) — "
                          "per-shape speed-up, fp32",
                          "speed-up vs torch.compile  (baseline / TensorMark, "
                          "log scale; bar = median over 3 rounds, ticks = range)")
        else:
            speedup_chart(plt, OUT / name, data,
                          "TensorMark vs each baseline — end-to-end speed-up",
                          "speed-up  (baseline / TensorMark, log scale)")
        print(f"  wrote docs/figs/{name}")

    print("\ncharts come from docs/perf_evidence.json — to change a bar, "
          "regenerate the evidence, not this script")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
