#!/usr/bin/env python3
"""Assemble the Training / Fine-tuning / Inference section tables from the
raw measurement files bench_all.sh produces. All speed-ups are computed
here — nothing is hand-transcribed. Writes build/SECTIONS.md and merges
the numbers into build/perf_evidence.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def med(vals):
    return statistics.median(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="build")
    a = ap.parse_args()
    d = Path(a.dir)
    ev_path = d / "perf_evidence.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}

    sections: dict[str, list[str]] = {"training": [], "finetuning": [], "inference": []}

    # ---- training ----
    ours = [float(m.group(1)) for f in sorted(d.glob("mnist_ours_*.txt"))
            if (m := re.search(r"samples_s=(\d+)", f.read_text()))]
    torch_s = [float(m.group(1)) for f in sorted(d.glob("mnist_torch_*.txt"))
               if (m := re.search(r"torch_compile_samples_s=(\d+)", f.read_text()))]
    if ours and torch_s:
        o, t = med(ours), med(torch_s)
        # Throughput rows: speed-up = ours ÷ baseline (>1 = TensorMark
        # faster), same direction convention as the time-based tables.
        ev["training"] = {"mnist_ours": o, "mnist_torch_compile": t,
                          "speedup": round(o / t, 2),
                          "ours_rounds": ours, "torch_rounds": torch_s}
        sections["training"] = [
            "| workload | torch.compile | TensorMark | speed-up |",
            "|---|---|---|---|",
            f"| MNIST CNN, full training epoch | {t:.0f} samples/s "
            f"| {o:.0f} samples/s | **{o / t:.2f}×** |",
            "",
            f"Rounds: ours {ours}, torch.compile {torch_s} (medians shown).",
        ]

    # ---- fine-tuning ----
    sft_txt = ""
    sf = d / "sft_steps.txt"
    if sf.exists():
        sft_txt = sf.read_text()
    times = [float(m.group(1)) for m in re.finditer(r"(\d+\.\d\d)s\s*$", sft_txt, re.M)]
    if times:
        steady = times[4:] or times  # drop warmup/compile steps  # first step includes warmup effects
        med_step = med(steady)
        eff_batch = 8
        ev["finetuning"] = {"step_s_median": med_step, "steps": times,
                            "effective_batch": eff_batch,
                            "samples_s": round(eff_batch / med_step, 2)}
        sections["finetuning"] = [
            "| workload | value |",
            "|---|---|",
            f"| GPT-2 124M SFT step (effective batch 8, steady state) "
            f"| {med_step:.2f} s/step |",
            f"| throughput at that batch | {eff_batch / med_step:.2f} samples/s |",
            "",
            f"Steps measured: {times}. No torch pair: the SFT path is "
            "correctness-focused; the baseline for its kernels is above.",
        ]

    # ---- inference ----
    ours_pre = [float(m.group(1)) for f in sorted(d.glob("llm_ours_prefill.txt"))
                for m in re.finditer(r'"tok_per_s":\s*(\d+\.?\d*)', f.read_text())]
    llama_pre = [float(m.group(1)) for f in sorted(d.glob("llm_llama_prefill.txt"))
                 for m in re.finditer(r"pp2000\s*\|\s*(\d+\.?\d*)", f.read_text())]
    ours_dec = [float(m.group(1)) for f in sorted(d.glob("llm_ours_decode.txt"))
                for m in re.finditer(r"= (\d+\.?\d*) t/s", f.read_text())]
    llama_dec = [float(m.group(1)) for f in sorted(d.glob("llm_llama_decode.txt"))
                 for m in re.finditer(r"tg128\s*\|\s*(\d+\.?\d*)", f.read_text())]
    mlx_pre = [float(m.group(1)) for f in sorted(d.glob("llm_mlx_prefill.txt"))
               for m in re.finditer(r'"tok_per_s":\s*(\d+\.?\d*)', f.read_text())]
    mlx_dec = [float(m.group(1)) for f in sorted(d.glob("llm_mlx_decode.txt"))
               for m in re.finditer(r'"tok_per_s":\s*(\d+\.?\d*)', f.read_text())]
    if ours_pre and llama_pre and ours_dec and llama_dec:
        po, pl = med(ours_pre), med(llama_pre)
        do, dl = med(ours_dec), med(llama_dec)
        ev["inference"] = {"prefill_ours": po, "prefill_llama": pl,
                           "prefill_speedup": round(po / pl, 2),
                           "decode_ours": do, "decode_llama": dl,
                           "decode_speedup": round(do / dl, 2)}
        header = "| workload | llama.cpp | TensorMark | speed-up |"
        rule = "|---|---|---|---|"
        pre_row = (f"| prefill, 2000 tok (median) | {pl:.1f} tok/s | {po:.1f} tok/s "
                   f"| **{po / pl:.2f}×** |")
        dec_row = (f"| decode, 128 tok (median) | {dl:.1f} tok/s | {do:.1f} tok/s "
                   f"| {do / dl:.2f}× |")
        note = (f"Rounds: prefill ours {ours_pre} vs llama {llama_pre}; "
                f"decode ours {ours_dec} vs llama {llama_dec}.")
        if mlx_pre and mlx_dec:
            pm, dm = med(mlx_pre), med(mlx_dec)
            ev["inference"].update({"prefill_mlx": pm, "decode_mlx": dm,
                                    "prefill_speedup_mlx": round(po / pm, 2),
                                    "decode_speedup_mlx": round(do / dm, 2)})
            header = "| workload | llama.cpp | MLX | TensorMark | vs llama.cpp | vs MLX |"
            rule = "|---|---|---|---|---|---|"
            pre_row = (f"| prefill, 2000 tok (median) | {pl:.1f} | {pm:.1f} | {po:.1f} "
                       f"| **{po / pl:.2f}×** | **{po / pm:.2f}×** |")
            dec_row = (f"| decode, 128 tok (median) | {dl:.1f} | {dm:.1f} | {do:.1f} "
                       f"| {do / dl:.2f}× | {do / dm:.2f}× |")
            note += (f" MLX prefill {mlx_pre}, decode {mlx_dec} "
                     f"(affine 4-bit g32, 5.0 bpw; TensorMark Q4_0, 4.5 bpw).")
        sections["inference"] = [header, rule, pre_row, dec_row, "", note]

    md = []
    for name in ("training", "finetuning", "inference"):
        if sections[name]:
            md.append(f"### {name.capitalize()}\n")
            md += sections[name] + ["\n"]
    (d / "SECTIONS.md").write_text("\n".join(md))
    ev_path.write_text(json.dumps(ev, indent=1))
    print(f"wrote {d / 'SECTIONS.md'} ({sum(len(v) for v in sections.values())} lines)")


if __name__ == "__main__":
    main()
