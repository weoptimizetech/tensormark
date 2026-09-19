#!/usr/bin/env python3
"""Collect the raw measurement files into the versioned evidence document.

The README's Performance tables claim to be extracted from raw measurement files
rather than transcribed. That is only checkable if the extracted values live
somewhere a reader can see them, so this writes docs/perf_evidence.json — a
versioned file, not build/perf_evidence.json, which .gitignore hides from
everyone except the machine that produced it.

Every value here is DERIVED from a raw file: medians over the recorded rounds,
computed the same way the tables compute them. Nothing is typed in. When an
input is absent the section is omitted and named under _meta.missing, so
tensormark/check_readme_perf.py reports the table rows that depend on it as
unevidenced rather than silently passing them.

Usage:
    python3 tensormark/collect_perf_evidence.py            # writes the evidence
    python3 tensormark/collect_perf_evidence.py --dir build --out docs/perf_evidence.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
from pathlib import Path

MACHINE = ("Apple M1 MacBook, 8 GB unified memory, macOS, CPU path at 4 threads")

# Two series belong to the same paired window only if they were written within
# this of each other. Measured windows are minutes apart; anything older is a
# carried-over file from a previous run.
WINDOW_TOLERANCE_S = 1800.0


def rounds(d: Path, pattern: str, rx: str, group: int = 1) -> list[float]:
    """Every recorded round across the files matching pattern."""
    out: list[float] = []
    for f in sorted(d.glob(pattern)):
        for m in re.finditer(rx, f.read_text(), re.M):
            out.append(float(m.group(group)))
    return out


def series(d: Path, pattern: str, rx: str, unit: str) -> dict | None:
    vals = rounds(d, pattern, rx)
    if not vals:
        return None
    files = sorted(d.glob(pattern))
    # Which run produced these rounds. A re-measurement rewrites the files it
    # covers and silently leaves the rest — bench_all.sh skips the mlx evidence
    # when the mlx venv or model dir is absent, so a later run divides a fresh
    # TensorMark number by a stale MLX baseline and the table still reads as one
    # same-window comparison. Carrying the age makes that visible instead.
    newest = max((f.stat().st_mtime for f in files), default=0.0)
    oldest = min((f.stat().st_mtime for f in files), default=0.0)
    return {"rounds": vals, "median": statistics.median(vals), "unit": unit,
            "files": [f.name for f in files],
            "age_s": round(newest - oldest, 1), "mtime": newest}


PROBE_DIRS = (
    # published first: it survives a build reset, and a reader can audit the
    # design-point derivation against the same file the script used.
    "docs/probes/framework_compare",
    "tensormark/build/framework_compare",
)


def find_probes() -> str:
    for p in PROBE_DIRS:
        if Path(p).is_dir():
            return p
    return PROBE_DIRS[0]


def load_probes(dirpath: str) -> dict[str, dict]:
    """The framework-compare probe documents, keyed by filename.

    These are written by tools/bench_framework_compare.py. During the public
    repo recreation they were recovered from a clone, so the default argument
    accepts any directory holding them.
    """
    d = Path(dirpath)
    if not d.is_dir():
        return {}
    out: dict[str, dict] = {}
    for f in sorted(d.glob("*.json")):
        try:
            out[f.name] = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tensormark/build",
                    help="directory holding the raw measurement files")
    ap.add_argument("--out", default="docs/perf_evidence.json")
    ap.add_argument("--probes", default=None,
                    help="directory of framework-compare probe JSON; searched "
                         "in docs/probes/framework_compare (published, survives "
                         "a build reset) then tensormark/build/framework_compare")
    a = ap.parse_args()
    d = Path(a.dir)

    ev: dict = {
        "_meta": {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generated_by": "tensormark/collect_perf_evidence.py",
            "machine": MACHINE,
            "method": ("medians over the recorded rounds of each interleaved "
                       "same-window pair; no value is typed in"),
            "missing": [],
        }
    }

    # ---- inference: llama.cpp / MLX cross-check (bench_all.sh section 4) ----
    spec = {
        "prefill_ours":   ("llm_ours_prefill.txt",  r'"tok_per_s":\s*(\d+\.?\d*)'),
        "prefill_llama":  ("llm_llama_prefill.txt", r"pp2000\s*\|\s*(\d+\.?\d*)"),
        "prefill_mlx":    ("llm_mlx_prefill.txt",   r'"tok_per_s":\s*(\d+\.?\d*)'),
        "decode_ours":    ("llm_ours_decode.txt",   r"= (\d+\.?\d*) t/s"),
        "decode_llama":   ("llm_llama_decode.txt",  r"tg128\s*\|\s*(\d+\.?\d*)"),
        "decode_mlx":     ("llm_mlx_decode.txt",    r'"tok_per_s":\s*(\d+\.?\d*)'),
    }
    inf: dict = {}
    for key, (pat, rx) in spec.items():
        s = series(d, pat, rx, "tok/s")
        if s is None:
            ev["_meta"]["missing"].append(f"{key} ({pat})")
        else:
            inf[key] = s
    # Same-window check. Every series in one table must come from the same run.
    # A re-measurement rewrites the files it covers and leaves the rest: when
    # the mlx venv or model dir is absent the pipeline skips that section, so a
    # later run would divide a fresh TensorMark number by a stale rival baseline
    # and the table would still read as one paired comparison.
    stamps = [s["mtime"] for s in inf.values() if isinstance(s, dict) and "mtime" in s]
    if stamps and (max(stamps) - min(stamps)) > WINDOW_TOLERANCE_S:
        inf["window_mixed"] = True
        inf["window_gap_s"] = round(max(stamps) - min(stamps), 1)
        stale = sorted(k for k, s in inf.items()
                       if isinstance(s, dict) and "mtime" in s
                       and (max(stamps) - s["mtime"]) > WINDOW_TOLERANCE_S)
        inf["window_stale_series"] = stale
    if {"prefill_ours", "prefill_llama"} <= inf.keys():
        po, pl = inf["prefill_ours"]["median"], inf["prefill_llama"]["median"]
        inf["prefill_speedup"] = round(po / pl, 2)
    if {"decode_ours", "decode_llama"} <= inf.keys():
        do, dl = inf["decode_ours"]["median"], inf["decode_llama"]["median"]
        inf["decode_speedup"] = round(do / dl, 2)
    if {"prefill_ours", "prefill_mlx"} <= inf.keys() and not inf.get("window_mixed"):
        inf["prefill_speedup_mlx"] = round(
            inf["prefill_ours"]["median"] / inf["prefill_mlx"]["median"], 2)
    if {"decode_ours", "decode_mlx"} <= inf.keys() and not inf.get("window_mixed"):
        inf["decode_speedup_mlx"] = round(
            inf["decode_ours"]["median"] / inf["decode_mlx"]["median"], 2)
    if inf:
        ev["inference_bench"] = inf

    # ---- design point: each framework at its own production configuration ----
    # Source: the framework-compare probes (tools/bench_framework_compare.py),
    # staged in tensormark/build/framework_compare/.
    pr = load_probes(a.probes or find_probes())
    if pr:
        sec: dict = {"_probe": "framework_compare", "aggregation": "mean of runs"}
        pre = pr.get("fwc_pre.json", {}).get("measurements", {}).get(
            "tensormark-metal-hybrid", {})
        if pre.get("tokens_per_second"):
            r = pre["tokens_per_second"]
            sec["prefill_tensormark"] = round(statistics.mean(r), 1)
            sec["prefill_rounds"] = len(r)
            sec["prefill_tensormark_median"] = round(statistics.median(r), 1)
        # The prefill probe carries no PyTorch leg; the ff16/MPS number is
        # recorded, not derived, so it is named rather than silently assumed.
        sec["prefill_pytorch"] = None
        sec["_prefill_pytorch_note"] = (
            "not in the retained probes: fwc_pre.json measured the tensormark "
            "leg only")
        dec = pr.get("fwc_dec_full.json", {}).get("measurements", {})
        chain = dec.get("tensormark-gpu-chain", {})
        if chain.get("times_seconds"):
            med = statistics.median(chain["times_seconds"])
            tpr = chain.get("tokens_per_run", 64)
            sec["decode_tensormark"] = round(tpr / med, 1)
            sec["decode_rounds"] = len(chain["times_seconds"])
            sec["decode_aggregation"] = "median of runs"
        attr = pr.get("decode_attribution.json", {}).get("float16", {})
        if attr.get("tok_s"):
            sec["decode_pytorch"] = round(attr["tok_s"], 2)
        train = pr.get("fwc_train.json", {})
        if train.get("na_reason"):
            sec["train_na_reason"] = train["na_reason"]
        if {"prefill_tensormark", "prefill_pytorch"} <= sec.keys() \
                and sec["prefill_pytorch"]:
            sec["prefill_speedup"] = round(
                sec["prefill_tensormark"] / sec["prefill_pytorch"], 2)
        if {"decode_tensormark", "decode_pytorch"} <= sec.keys():
            sec["decode_speedup"] = round(
                sec["decode_tensormark"] / sec["decode_pytorch"], 2)
        ev["design_point"] = sec
    else:
        ev["_meta"]["missing"].append(f"design_point (no probes in {a.probes})")

    # ---- training: MNIST full epoch (bench_all.sh section 2) ----
    ours = rounds(d, "mnist_ours_*.txt", r"samples_s=(\d+)")
    torc = rounds(d, "mnist_torch_*.txt", r"torch_compile_samples_s=(\d+)")
    if ours and torc:
        o, t = statistics.median(ours), statistics.median(torc)
        ev["training"] = {"mnist_ours": o, "mnist_torch_compile": t,
                          "speedup": round(o / t, 2),
                          "ours_rounds": ours, "torch_rounds": torc}
    else:
        ev["_meta"]["missing"].append("training (mnist_ours_*.txt, mnist_torch_*.txt)")

    # ---- fine-tuning: GPT-2 SFT step cost (bench_all.sh section 3) ----
    sf = d / "sft_steps.txt"
    if sf.exists():
        times = [float(m.group(1))
                 for m in re.finditer(r"(\d+\.\d\d)s\s*$", sf.read_text(), re.M)]
        if times:
            steady = times[4:] or times          # drop warmup / first-compile steps
            med = statistics.median(steady)
            ev["finetuning"] = {"step_s_median": med, "steps": times,
                                "effective_batch": 8,
                                "samples_s": round(8 / med, 2)}
    if "finetuning" not in ev:
        ev["_meta"]["missing"].append("finetuning (sft_steps.txt)")

    # ---- kernels vs torch.compile (bench_all.sh section 1) ----
    def per_round_ms(pattern: str) -> dict[str, list[float]]:
        per: dict[str, list[float]] = {}
        for f in sorted(d.glob(pattern)):
            data = json.loads(f.read_text())
            entries = data if isinstance(data, list) else [
                {"name": k, "ms": v["ms"]} for k, v in data.items()]
            for e in entries:
                per.setdefault(e["name"], []).append(e["ms"])
        return per

    ours_k, torc_k = per_round_ms("ours_*.json"), per_round_ms("torchc_*.json")
    common = sorted(set(ours_k) & set(torc_k))
    if common:
        kern: dict = {}
        for k in common:
            o, t = ours_k[k], torc_k[k]
            # Ratio per round, not only ratio of medians. A row can read as a
            # 1.04x win while its rounds are 1.04 / 0.96 / 1.07 — the sign
            # changes inside one measurement, and a single factor hides that.
            ratios = [tv / ov for ov, tv in zip(o, t)]
            kern[k] = {
                "ours_ms": statistics.median(o),
                "torch_compile_ms": statistics.median(t),
                "speedup": round(statistics.median(t) / statistics.median(o), 2),
                "ours_rounds": o,
                "torch_rounds": t,
                "ratio_rounds": [round(r, 4) for r in ratios],
                "ratio_min": round(min(ratios), 2),
                "ratio_max": round(max(ratios), 2),
                "sign_stable": all(r > 1.0 for r in ratios)
                               or all(r < 1.0 for r in ratios),
            }
        ev["kernels"] = kern
    else:
        ev["_meta"]["missing"].append("kernels (ours_*.json, torchc_*.json)")

    Path(a.out).write_text(json.dumps(ev, indent=1, sort_keys=True) + "\n")
    n = sum(len(v) for k, v in ev.items() if k != "_meta")
    print(f"wrote {a.out}: {n} derived values, "
          f"{len(ev['_meta']['missing'])} input group(s) missing")
    for m in ev["_meta"]["missing"]:
        print(f"  missing: {m}")


if __name__ == "__main__":
    main()
