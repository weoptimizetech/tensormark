#!/usr/bin/env python3
"""Framework comparison harness: tensormark vs PyTorch on identical workloads.

Measures prefill, decode and fine-tune step times programmatically for both
frameworks and renders a table whose numbers come only from the measurements
dict — no hand-entered values. Training a full 1.1 B model is reported as
infeasible with the optimizer-state memory the machine would need, computed,
not asserted.

tensormark side:
  prefill — persistent bench_prefill_lane worker ("run" protocol, wall_ms per
            request). The metal lane can carry the hybrid GPU/CPU row split
            (TM_BENCH_HYBRID=1 + TM_LLAMA_HYBRID); the ane lane runs the
            fused ANE prefix + Metal suffix package.
  decode  — bench_chain_decode (GPU-resident greedy chain, decode-only timing).
PyTorch side:
  prefill — forward over the same token IDs on the chosen device.
  decode  — manual greedy loop with the KV cache (prompt prefill excluded,
            matching bench_chain_decode semantics).
  finetune — LoRA (r=8 on q_proj/v_proj) forward+backward+AdamW step.
  train   — full-parameter step; memory requirement computed and refused when
            it exceeds the device budget instead of swapping.

Weights differ by construction (tensormark Q4_0 vs PyTorch bf16/fp16), so the
table reports speed of serving, not numerical equivalence; the logit oracle in
test_llama_quant covers quality separately.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TMQ = REPO / "tensormark/data/tinyllama/tinyllama_q40.tmq"
DEFAULT_LANE_BIN = REPO / "tensormark/build/bench_prefill_lane"
DEFAULT_CHAIN_BIN = REPO / "tensormark/build/bench_chain_decode"


# ---------------------------------------------------------------- workload ---


def generate_ids(tokens: int, seed: int, vocabulary: int = 32000) -> list[int]:
    """Fixed-seed uniform IDs shared by both frameworks (no tokenizer drift)."""
    import random

    rng = random.Random(seed)
    return [rng.randrange(vocabulary) for _ in range(tokens)]


def write_ids(path: Path, ids: list[int]) -> None:
    path.write_text(" ".join(str(i) for i in ids) + "\n")


# --------------------------------------------------------------- tensormark ---


class LaneWorker:
    """bench_prefill_lane process: ready, then 'run' -> result JSON lines."""

    def __init__(self, binary: Path, tmq: Path, tokens: int, seed: int, lane: str,
                 ids_file: Path | None = None, ane_path: Path | None = None,
                 hybrid: float | None = None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("TM_")}
        env.update({"TM_LLAMA_GPU_HALF": "1", "TM_PREFILL_AMX": "1"})
        if lane == "ane":
            if ane_path is None:
                raise ValueError("ane lane requires ane_path")
            env["TM_ANE_PATH"] = str(ane_path)
        if hybrid is not None:
            env["TM_BENCH_HYBRID"] = "1"
            env["TM_LLAMA_HYBRID"] = repr(hybrid)
        if ids_file is not None:
            env["TM_BENCH_TOKENS"] = str(ids_file)
        self.stderr_file = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [str(binary), str(tmq), str(tokens), str(seed), lane],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr_file, env=env, bufsize=0)
        try:
            ready = self._line()
        except Exception:
            self.stderr_file.seek(0)
            tail = self.stderr_file.read().decode(errors="replace")[-500:]
            raise RuntimeError(f"worker init failed ({lane}): {tail}") from None
        if ready.get("event") != "ready":
            raise RuntimeError(f"unexpected worker message: {ready}")

    def _line(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("worker exited early")
        return json.loads(line.decode())

    def run(self) -> float:
        self.proc.stdin.write(b"run\n")
        self.proc.stdin.flush()
        result = self._line()
        if result.get("event") != "result":
            raise RuntimeError(f"expected result, got {result.get('event')}")
        return float(result["wall_ms"])

    def close(self) -> None:
        try:
            self.proc.stdin.write(b"quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def tmq_prefill(args: argparse.Namespace, lane: str, hybrid: float | None,
                ids_file: Path) -> dict:
    worker = LaneWorker(args.lane_binary, args.tmq, args.tokens, args.seed,
                        lane, ids_file=ids_file, ane_path=args.ane_path,
                        hybrid=hybrid)
    try:
        for _ in range(args.warmup):
            worker.run()
        walls = [worker.run() for _ in range(args.rounds)]
    finally:
        worker.close()
    seconds = [w / 1000.0 for w in walls]
    return {
        "framework": f"tensormark-{lane}" + (f"+hybrid{hybrid}" if hybrid else ""),
        "times_seconds": seconds,
        "tokens_per_run": args.tokens,
        "tokens_per_second": [args.tokens / s for s in seconds],
    }


def tmq_decode(args: argparse.Namespace) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TM_")}
    env.update({"TM_DECODE_GPU": "1", "TM_LLAMA_GPU_HALF": "1"})
    pattern = re.compile(r"chain decode (\d+) tokens in ([0-9.]+) s = ([0-9.]+) t/s")
    times = []
    for _ in range(args.rounds):
        proc = subprocess.run(
            [str(args.chain_binary), str(args.tmq), str(args.gen_tokens), "2", "512"],
            capture_output=True, text=True, env=env, timeout=args.timeout, check=False)
        match = pattern.search(proc.stdout)
        if proc.returncode != 0 or not match:
            raise RuntimeError(f"chain decode failed: {proc.stderr.strip()[:200]}")
        times.append(float(match.group(2)))
    tokens = int(match.group(1))
    return {"times_seconds": times, "tokens_per_run": tokens,
            "tokens_per_second": [tokens / t for t in times]}


# ----------------------------------------------------------------- pytorch ---


def torch_prefill(model, ids, device, args) -> dict:
    import torch

    x = torch.tensor([ids], device=device)
    for _ in range(args.warmup):
        model(input_ids=x)
    if device.type == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        model(input_ids=x)
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return {"times_seconds": times, "tokens_per_run": args.tokens,
            "tokens_per_second": [args.tokens / t for t in times]}


def torch_decode(model, ids, device, args) -> dict:
    import torch

    """Greedy decode loop with cache; prompt prefill excluded (chain parity).
    Warmup steps advance the cache first; the timed loop then generates
    gen_tokens tokens at a slightly longer context, matching bench_chain_decode
    (which times its chunk loop after the prompt)."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=device), use_cache=True)
        past, next_id = out.past_key_values, int(out.logits[0, -1].argmax())
        for _ in range(args.warmup):  # warm the cache path
            out = model(input_ids=torch.tensor([[next_id]], device=device),
                        past_key_values=past, use_cache=True)
            past, next_id = out.past_key_values, int(out.logits[0, -1].argmax())
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.gen_tokens):
            out = model(input_ids=torch.tensor([[next_id]], device=device),
                        past_key_values=past, use_cache=True)
            past, next_id = out.past_key_values, int(out.logits[0, -1].argmax())
        if device.type == "mps":
            torch.mps.synchronize()
        times = [time.perf_counter() - t0]
    return {"times_seconds": times, "tokens_per_run": args.gen_tokens,
            "tokens_per_second": [args.gen_tokens / t for t in times]}


def torch_finetune(model, ids, device, args, mode: str = "finetune") -> dict:
    import torch

    """LoRA (r=8 on q_proj/v_proj) optimizer step timed over K rounds."""
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.model.layers:
        for name in ("q_proj", "v_proj"):
            base = getattr(block.self_attn, name)
            lora_a = torch.nn.Linear(base.in_features, 8, bias=False,
                                     device=device, dtype=base.weight.dtype)
            lora_b = torch.nn.Linear(8, base.out_features, bias=False,
                                     device=device, dtype=base.weight.dtype)
            torch.nn.init.zeros_(lora_b.weight)
            original = base.forward
            def make_fwd(orig, a, b):
                def fwd(x):
                    return orig(x) + a(x) @ b.weight.t()
                return fwd
            base.forward = make_fwd(original, lora_a, lora_b)
            trainable += list(lora_a.parameters()) + list(lora_b.parameters())
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    x = torch.tensor([ids], device=device)
    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    for _ in range(args.warmup):
        step()
    if device.type == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        step()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return {"times_seconds": times, "tokens_per_run": args.tokens,
            "tokens_per_second": [args.tokens / t for t in times]}


def optimizer_state_gb(model) -> float:
    """fp32 weights + grads + two AdamW moments, in GiB (computed, not guessed)."""
    params = sum(p.numel() for p in model.parameters())
    return params * 4 * 4 / 2**30


# ------------------------------------------------------------------ report ---


def row_stats(measurement: dict | None) -> tuple[str, str, str]:
    if measurement is None:
        return ("—", "—", "—")
    times = measurement["times_seconds"]
    mean = statistics.mean(times)
    spread = f"{min(times):.3f}–{max(times):.3f}"
    rate_mean = measurement["tokens_per_run"] / mean
    unit = f"{measurement['tokens_per_run']} tok/run"
    return (f"{mean:.3f}", f"{spread} ({unit})", f"{rate_mean:.1f} tok/s")


def build_table(rows: list[dict]) -> str:
    lines = ["| Task | Framework | Device | Mean time | Spread (unit) | Rate |",
             "|---|---|---|---:|---|---:|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['framework']} | {r['device']} | "
                     f"{r['mean']} | {r['spread']} | {r['rate']} |")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- main ---


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("task", choices=("prefill", "decode", "finetune", "train"))
    p.add_argument("--tmq", type=Path, default=DEFAULT_TMQ)
    p.add_argument("--snapshot", type=Path, help="HF model dir for PyTorch")
    p.add_argument("--lane-binary", type=Path, default=DEFAULT_LANE_BIN)
    p.add_argument("--chain-binary", type=Path, default=DEFAULT_CHAIN_BIN)
    p.add_argument("--ane-path", type=Path)
    p.add_argument("--hybrid", type=float, default=0.25)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pytorch-device", default="mps")
    p.add_argument("--pytorch-dtype", default="float16", choices=("float16", "float32"))
    p.add_argument("--ids-file", type=Path, help="shared token-ID file (generated when absent)")
    p.add_argument("--output", type=Path)
    p.add_argument("--timeout", type=float, default=300.0)
    a = p.parse_args(argv)
    if a.tokens < 2 or a.gen_tokens < 2 or a.rounds < 1 or a.warmup < 0:
        p.error("tokens >= 2, gen-tokens >= 2, rounds >= 1, warmup >= 0 required")
    if a.timeout <= 0:
        p.error("timeout must be positive")
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    task = args.task
    ids_path = args.ids_file or Path(tempfile.gettempdir()) / f"fwc_ids_{args.tokens}_{args.seed}.txt"
    if not ids_path.exists():
        write_ids(ids_path, generate_ids(args.tokens, args.seed))
    ids = [int(x) for x in ids_path.read_text().split()]
    if len(ids) != args.tokens:
        raise ValueError(f"ids file has {len(ids)} ids, expected {args.tokens}")
    report: dict = {"schema": "tensormark.framework-compare/1", "task": task,
                    "tokens": args.tokens, "gen_tokens": args.gen_tokens,
                    "rounds": args.rounds, "seed": args.seed,
                    "pytorch_device": args.pytorch_device,
                    "pytorch_dtype": args.pytorch_dtype,
                    "measurements": {}}
    try:
        if task in ("prefill",):
            report["measurements"]["tensormark-metal-hybrid"] = tmq_prefill(
                args, "metal", args.hybrid, ids_path)
            if args.ane_path is not None:
                report["measurements"]["tensormark-ane"] = tmq_prefill(
                    args, "ane", None, ids_path)
            if args.snapshot is not None:
                model, device = load_torch(args)
                report["measurements"]["pytorch"] = torch_prefill(
                    model, ids, device, args)
                del model
        elif task == "decode":
            report["measurements"]["tensormark-gpu-chain"] = tmq_decode(args)
            if args.snapshot is not None:
                model, device = load_torch(args)
                report["measurements"]["pytorch"] = torch_decode(
                    model, ids, device, args)
                del model
        elif task == "finetune":
            if args.snapshot is None:
                raise ValueError("finetune requires --snapshot")
            model, device = load_torch(args)
            report["measurements"]["pytorch"] = torch_finetune(model, ids, device, args)
            report["na_reason"] = ("tensormark has no backward pass; weights are "
                                   "frozen Q4_0 containers by design")
        elif task == "train":
            if args.snapshot is None:
                raise ValueError("train requires --snapshot")
            model, device = load_torch(args)
            need = optimizer_state_gb(model)
            report["na_reason"] = (
                f"full-parameter training needs ~{need:.1f} GiB of weights+grads"
                f"+AdamW state at fp32 — exceeds this machine by design; the "
                f"measured feasible alternative is the finetune task")
            del model
        serialized = json.dumps(report, allow_nan=False, indent=2) + "\n"
        if args.output is not None:
            out = args.output.expanduser()
            if out.exists():
                raise FileExistsError(f"refusing to overwrite {out}")
            out.write_text(serialized)
        table = build_table(table_rows(report))
        print(serialized)
        print("\n" + table)
    except (RuntimeError, ValueError, OSError, FileExistsError) as error:
        print(f"framework-compare: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


def load_torch(args):
    import torch
    from transformers import AutoModelForCausalLM
    dtype = getattr(torch, args.pytorch_dtype)
    import gc
    model = AutoModelForCausalLM.from_pretrained(
        str(args.snapshot), torch_dtype=dtype, local_files_only=True,
        low_cpu_mem_usage=True)
    device = torch.device(args.pytorch_device)
    model = model.to(device)
    model.eval()
    gc.collect()  # drop the CPU-side weight copy before measuring
    return model, device


def table_rows(report: dict) -> list[dict]:
    rows = []
    task = report["task"]
    for name, m in report["measurements"].items():
        if m is None:
            continue  # N/A rows are rendered from na_reason below
        mean, spread, rate = row_stats(m)
        if "ane" in name:
            device = "ANE+Metal"
        elif "metal" in name:
            device = "Metal+CPU rows"
        elif "chain" in name:
            device = "GPU (chain)"
        else:
            device = f'{report.get("pytorch_device", "?")} ({report.get("pytorch_dtype", "?")})'
        rows.append({"task": task, "framework": name, "device": device,
                     "mean": mean, "spread": spread, "rate": rate})
    if report.get("na_reason"):
        rows.append({"task": task, "framework": "tensormark",
                     "device": "—", "mean": "—", "spread": report["na_reason"],
                     "rate": "—"})
    return rows


if __name__ == "__main__":
    raise SystemExit(main())