#!/usr/bin/env python3
"""Join paired benchmark JSON, validate checksums, emit tables + pgfplots.

Inputs: medians-per-round JSON files from microbench.cpp (list of
{name, ms, chk}) and microbench_torch.py ({name: {ms, chk}}), plus
end-to-end section evidence produced by bench_all.sh.

Outputs: markdown tables, a pgfplots bar-chart document in the
weoptimizetech.com visual identity, and a JSON evidence file.
"""
from __future__ import annotations

import argparse
import json
import dataclasses
import statistics
from pathlib import Path

KERNEL_ORDER = [
    "gemm_T256_K4096_N4096", "gemm_T256_K1024_N1024", "gemm_T256_K256_N256",
    "gemm_T256_K64_N64", "gemm_T1_K4096_N4096", "gemm_T1_K1024_N1024",
    "gemm_T1_K256_N256", "gemm_T1_K64_N64",
    "attn_decode_H12_dh64_S2048", "attn_decode_H12_dh64_S512",
    "attn_decode_H12_dh64_S128",
    "dw3x3_C96_32x32", "conv1x1_C96_32x32", "gelu_fused_N196608",
]
KERNEL_LABEL = {
    "gemm_T256_K4096_N4096": "gemm T256 K4096 N4096",
    "gemm_T256_K1024_N1024": "gemm T256 K1024 N1024",
    "gemm_T256_K256_N256": "gemm T256 K256 N256",
    "gemm_T256_K64_N64": "gemm T256 K64 N64",
    "gemm_T1_K4096_N4096": "gemm T1 K4096 N4096 (decode GEMV)",
    "gemm_T1_K1024_N1024": "gemm T1 K1024 N1024",
    "gemm_T1_K256_N256": "gemm T1 K256 N256",
    "gemm_T1_K64_N64": "gemm T1 K64 N64",
    "attn_decode_H12_dh64_S2048": "attn decode ctx 2048 (12×64 heads)",
    "attn_decode_H12_dh64_S512": "attn decode ctx 512",
    "attn_decode_H12_dh64_S128": "attn decode ctx 128",
    "dw3x3_C96_32x32": "depthwise 3×3, C96 32×32",
    "conv1x1_C96_32x32": "conv 1×1, C96 32×32",
    "gelu_fused_N196608": "gelu fused, 196 608 elems",
}


def median_ms(files: list[Path]) -> dict[str, float]:
    per_name: dict[str, list[float]] = {}
    for f in files:
        d = json.loads(f.read_text())
        entries = d if isinstance(d, list) else [
            {"name": k, "ms": v["ms"]} for k, v in d.items()]
        for e in entries:
            per_name.setdefault(e["name"], []).append(e["ms"])
    return {k: statistics.median(v) for k, v in per_name.items()}


def median_chk(files: list[Path]) -> dict[str, float]:
    per_name: dict[str, list[float]] = {}
    for f in files:
        d = json.loads(f.read_text())
        entries = d if isinstance(d, list) else [
            {"name": k, "chk": v["chk"]} for k, v in d.items()]
        for e in entries:
            if "chk" in e:
                per_name.setdefault(e["name"], []).append(e["chk"])
    return {k: statistics.median(v) for k, v in per_name.items()}


def speedup(baseline: float, ours: float) -> float:
    return baseline / ours


@dataclasses.dataclass
class KernelRow:
    op: str
    ours: float
    compiled: float
    speedup: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="+", required=True)
    ap.add_argument("--torch-eager", nargs="*", default=[])
    ap.add_argument("--torch-compiled", nargs="+", required=True)
    ap.add_argument("--training", default=None, help="e2e training evidence json")
    ap.add_argument("--finetuning", default=None, help="e2e SFT evidence json")
    ap.add_argument("--inference", default=None, help="llama pair evidence json")
    ap.add_argument("--out-md", default="build/PERF_TABLES.md")
    ap.add_argument("--out-tex", default="build/perf_charts.tex")
    ap.add_argument("--out-json", default="build/perf_evidence.json")
    args = ap.parse_args()

    ours = median_ms([Path(p) for p in args.engine])
    chk_ours = median_chk([Path(p) for p in args.engine])
    comp = median_ms([Path(p) for p in args.torch_compiled])
    chk_comp = median_chk([Path(p) for p in args.torch_compiled])
    eager = median_ms([Path(p) for p in args.torch_eager]) if args.torch_eager else {}

    # Checksum gate: both sides must have computed the same math on the same
    # byte-identical inputs before any speed-up is quoted.
    checks = []
    for k in ours:
        if k in chk_ours and k in chk_comp:
            rel = abs(chk_ours[k] - chk_comp[k]) / max(chk_ours[k], 1e-30)
            checks.append((k, rel))
    bad = [(k, r) for k, r in checks if r > 1e-3]
    if bad:
        raise SystemExit(f"CHECKSUM GATE FAILED (math differs across sides): {bad}")

    rows: list[KernelRow] = []
    for k in KERNEL_ORDER:
        if k not in ours or k not in comp:
            continue
        su = speedup(comp[k], ours[k])
        rows.append(KernelRow(op=k, ours=ours[k], compiled=comp[k],
                              speedup=round(su, 2)))
    evidence: dict[str, object] = {
        "kernels": [dataclasses.asdict(r) for r in rows],
        "checksum_max_rel": max((r for _, r in checks), default=None),
    }
    if eager:
        evidence["kernels_eager_ms"] = eager
    if args.training:
        evidence["training"] = json.loads(Path(args.training).read_text())
    if args.finetuning:
        evidence["finetuning"] = json.loads(Path(args.finetuning).read_text())
    if args.inference:
        evidence["inference"] = json.loads(Path(args.inference).read_text())

    # ---- markdown ----
    md = ["## Kernels vs torch.compile", "",
          "| op (shape) | torch.compile ms | TensorMark ms | speed-up |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {KERNEL_LABEL[r.op]} | {r.compiled:.4f} | "
                  f"{r.ours:.4f} | {r.speedup:.2f}× |")
    md += ["", "Speed-up = baseline ÷ TensorMark (>1 = TensorMark faster).",
           f"Checksum gate (same math both sides): max rel diff "
           f"{evidence['checksum_max_rel']:.2e}."]
    Path(args.out_md).write_text("\n".join(md) + "\n")

    # ---- pgfplots (weoptimizetech.com identity: dark, steel-blue/coral) ----
    labels = ";".join(KERNEL_LABEL[r.op] for r in rows)
    base = ";".join(f"{r.compiled:.4f}" for r in rows)
    tmv = ";".join(f"{r.ours:.4f}" for r in rows)
    tex = r"""\documentclass[border=6pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{wmBG}{HTML}{0A0B0F}
\definecolor{wmPanel}{HTML}{111318}
\definecolor{wmText}{HTML}{D1D5DB}
\definecolor{wmGrid}{HTML}{2A2E36}
\definecolor{wmBase}{HTML}{565D66}
\definecolor{wmTM}{HTML}{8FA5B8}
\definecolor{wmAccent}{HTML}{D45D6B}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    xbar, width=19cm, height=24cm,
    title={\color{wmText}\bfseries TensorMark vs torch.compile (inductor) --- kernel time, lower is better},
    xmode=log, log origin=infty,
    xlabel={\color{wmText}ms per call (log scale)},
    xmin=0.0015, xmax=30,
    symbolic y coords={%%LABELS%%},
    ytick=data, yticklabel style={color=wmText, font=\footnotesize},
    axis background/.style={fill=wmBG},
    axis x line*=bottom, axis y line*=left,
    tick align=outside, tick color=wmGrid,
    xmajorgrids, grid style={color=wmGrid, dashed, line width=0.3pt},
    bar width=5.2pt, bar shift=0pt,
    every axis plot/.append style={draw=none},
    legend style={at={(1,1.02)}, anchor=south east, draw=wmGrid,
                  fill=wmPanel, text=wmText, font=\footnotesize,
                  /tikz/every even column/.append style={column sep=0.35cm}},
    enlarge y limits=0.035,
]
\addplot+[xbar, fill=wmBase] coordinates {%%BASE%%};
\addlegendentry{\color{wmText}torch.compile (inductor)}
\addplot+[xbar, fill=wmTM] coordinates {%%TM%%};
\addlegendentry{\color{wmText}TensorMark}
\end{axis}
\end{tikzpicture}
\end{document}
"""
    tex = tex.replace("%%LABELS%%", labels).replace("%%BASE%%", base).replace("%%TM%%", tmv)
    Path(args.out_tex).write_text(tex)

    Path(args.out_json).write_text(json.dumps(evidence, indent=1))
    n_better = sum(1 for r in rows if r.speedup >= 1)
    print(f"kernels: TensorMark leads {n_better}/{len(rows)} shapes vs torch.compile")
    print(f"wrote {args.out_md}, {args.out_tex}, {args.out_json}")


if __name__ == "__main__":
    main()
