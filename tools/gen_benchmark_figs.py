#!/usr/bin/env python3
"""gen_benchmark_figs.py — README bar plots from the measured tables.
Sources: docs/BENCHMARKS.md + README inference tables (local probe-result JSONs).
Writes docs/figs/*.png. Clean brand style, annotation-first."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figs")
os.makedirs(OUT, exist_ok=True)
TM, OTHER = "#2E7D32", "#9E9E9E"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

def bar(path, title, ylabel, labels, vals, color=TM, fmt="{:.1f}", note=None):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    b = ax.bar(labels, vals, 0.5, color=[TM if i == 0 else OTHER for i in range(len(labels))])
    for rect in b:
        ax.annotate(fmt.format(rect.get_height()),
                    (rect.get_x() + rect.get_width()/2, rect.get_height()),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    if note:
        ax.axhline(note[0], ls="--", lw=1.2, color="#B71C1C")
        ax.annotate(note[1], (len(labels)-0.45, note[0]), color="#B71C1C",
                    fontsize=9, ha="right", va="bottom")
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

# All numbers from the measured tables (docs/BENCHMARKS.md + README).
bar(f"{OUT}/prefill_vs_rivals.png",
    "Prefill 1,024 tok — TinyLlama 1.1B, GPU",
    "tok/s", ["TensorMark\nQ4_0 hybrid", "Ollama\nQ4_0", "PyTorch\nfp16 MPS"],
    [913.8, 830.0, 815.5], fmt="{:.0f}")
bar(f"{OUT}/decode_llamacpp.png",
    "Decode 64 tok — TinyLlama Q4_0 GPU chain",
    "tok/s", ["TensorMark", "llama.cpp\n(Metal, 4 thr)"], [84.0, 74.0], fmt="{:.1f}")
bar(f"{OUT}/decode_pytorch_fp16.png",
    "Decode 64 tok — equal precision (fp16/MPS)",
    "tok/s", ["TensorMark\nfp16 GPU", "PyTorch\nfp16 MPS"], [17.0, 10.65], fmt="{:.1f}")
fig, ax = plt.subplots(figsize=(6.8, 4.2))
x = [1, 2, 3]
ax.bar(x, [537, 362, 362], 0.5, color=TM, label="with KV reuse")
ax.plot(x, [537, 1410, 1730], "o--", color="#B71C1C", lw=1.5,
        label="if history were re-prefilled")
for xi, v in zip(x, [537, 362, 362]):
    ax.annotate(f"{v}", (xi, v), ha="center", va="bottom", fontweight="bold")
ax.set_xticks(x, ["turn 1", "turn 2", "turn 3"])
ax.set_ylabel("prefill ms / turn"); ax.set_ylim(0, 2300)
ax.set_title("Warm KV cache — per-turn prefill cost stays flat", fontsize=12)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/kv_warm_turns.png", bbox_inches="tight")
print("4 figs in docs/figs/")