#!/usr/bin/env python3
"""bench_mlx.py — MLX-LM prefill/decode evidence for the benchmark pipeline.

Same measurement shape as hybrid_bench + bench_llama_decode_gpu: a single
2000-token prefill forward, then greedy decode of 128 tokens with the KV
cache filled outside the timed window. Emits the same plain-text evidence
files emit_sections.py parses, so the README numbers stay programmatic.

    ./build/bench_mlx.py --model data/tinyllama/mlx-4bit-g32 \
        --out-dir build --prefill-tokens 2000 --decode-tokens 128 --rounds 3

Requires the `mlx-lm` package (run with the venv interpreter that has it).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from mlx_lm.models import cache as mc
from mlx_lm.utils import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MLX model directory")
    ap.add_argument("--out-dir", default="build")
    ap.add_argument("--prefill-tokens", type=int, default=2000)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    model, tok = load(args.model)
    # repeated filler text; throughput is content-independent, and the
    # tokenizer truncation keeps the prefill length exact
    text = ("The quick brown fox jumps over the lazy dog while the sun sets "
            "behind the mountains and the river flows steadily toward the "
            "distant sea. ") * 400
    ids = tok.encode(text)[: args.prefill_tokens]
    n = len(ids)

    prefill: list[float] = []
    decode: list[float] = []
    for _ in range(args.rounds):
        # prefill: one full forward, timed
        t0 = time.perf_counter()
        lg = model(mx.array([ids]))
        mx.eval(lg)
        prefill.append(n / (time.perf_counter() - t0))

        # decode: fill the KV cache outside the timed window (that call IS a
        # prefill), then time single-token steps only
        cache = mc.make_prompt_cache(model)
        lg = model(mx.array([ids]), cache=cache)
        mx.eval(lg)
        buf = [int(mx.argmax(lg[:, -1, :], axis=-1))]
        t0 = time.perf_counter()
        for _ in range(args.decode_tokens):
            lg = model(mx.array([[buf[-1]]]), cache=cache)
            buf.append(int(mx.argmax(lg[:, -1, :], axis=-1)))
        mx.eval(lg)
        decode.append(args.decode_tokens / (time.perf_counter() - t0))

    out = argparse.Namespace(out_dir=args.out_dir)
    import pathlib
    d = pathlib.Path(out.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "llm_mlx_prefill.txt").write_text(
        "\n".join(json.dumps({"round": i, "tok_per_s": round(v, 1)})
                  for i, v in enumerate(prefill)) + "\n")
    (d / "llm_mlx_decode.txt").write_text(
        "\n".join(json.dumps({"round": i, "tok_per_s": round(v, 1)})
                  for i, v in enumerate(decode)) + "\n")
    print(f"mlx prefill median {statistics.median(prefill):.1f} tok/s "
          f"({prefill}), decode median {statistics.median(decode):.1f} tok/s "
          f"({decode})")


if __name__ == "__main__":
    main()
