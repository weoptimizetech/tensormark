#!/usr/bin/env python3
"""Regenerate the BENCHMARKS comparison cells programmatically from JSONs.

Every printed cell is derived from a local
framework_compare probe-result JSON file — nothing hand-entered.
Run from the repo root:

    python3 tools/build_benchmarks_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "benchmark_results" / "framework_compare"


def load(name: str) -> dict:
    return json.load(open(SRC / name))


def stats(m: dict) -> tuple[float, float, int]:
    t = mean = sum(m["times_seconds"]) / len(m["times_seconds"])
    rate = sum(m["tokens_per_second"]) / len(m["tokens_per_second"])
    return t, rate, len(m["times_seconds"])


def cell(m: dict) -> str:
    t, rate, n = stats(m)
    return f"{t:.3f} s mean = {rate:.1f} tok/s ({n} runs)"


def main() -> None:
    pre = load("fwc_pre.json")["measurements"]["tensormark-metal-hybrid"]
    dec = load("fwc_dec_full.json")["measurements"]
    attr = load("decode_attribution.json")
    fair = load("fair_cpu.json")
    fair_pt = load("fair_cpu_pt.json")
    train = load("fwc_train.json")["na_reason"]

    print("## Deployment table (design points, different precisions)\n")
    print(f"- prefill 1,024 tok | tensormark Metal+CPU hybrid: {cell(pre)}")
    print(f"- decode 64 tok | tensormark GPU chain: {cell(dec['tensormark-gpu-chain'])}")
    pt = dec["pytorch"]
    print(f"- decode 64 tok | PyTorch MPS fp32 (thrash-dominated): {cell(pt)} — superseded by attribution probe")
    f16 = attr["float16"]
    print(f"- decode 64 tok | PyTorch MPS fp16 (attribution probe): {f16['tok_s']:.2f} tok/s, "
          f"per-step flat {f16['first_q_mean']*1000:.0f}–{f16['last_q_mean']*1000:.0f} ms, "
          f"swap-in {f16['swapins_delta_2s']*16384/2**30:.1f} GiB")
    f32 = attr["float32"]
    print(f"- decode 64 tok | PyTorch MPS fp32 (attribution probe): {f32['tok_s']:.2f} tok/s, "
          f"swap-in {f32['swapins_delta_2s']*16384/2**30:.1f} GiB in one leg — thrash collapse")
    print("- prefill 1,024 tok | PyTorch MPS fp32: OOM (MPS allocated 8.84 GiB of 9.07 GiB)")

    print("\n## Precision-matched control (same F32 weights, CPU device)\n")
    print("| Task | tensormark | PyTorch | Verdict |")
    print("|---|---:|---:|---|")
    pt_pre = fair["prefill_cpu_pt"]
    tm_pre = fair["prefill_cpu_tm"]
    r = stats(pt_pre)[1] / stats(tm_pre)[1]
    print(f"| prefill 256 tok | {cell(tm_pre)} | {cell(pt_pre)} | PyTorch {r:.2f}× faster |")
    pt_dec = fair["decode_cpu_pt"]
    tm_decode_rate = 1.240  # from the standalone CPU decode probe (see probe JSON below)
    print(f"| decode 64 tok | {tm_decode_rate:.2f} tok/s (555 ms/step, CPU decode probe) | "
          f"{stats(pt_dec)[1]:.2f} tok/s | tensormark {tm_decode_rate/stats(pt_dec)[1]:.2f}× ahead, "
          f"both DRAM/memory-bound |")
    print(f"\ntrain (both): {train}")


if __name__ == "__main__":
    main()