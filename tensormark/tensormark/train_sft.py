"""SFT training loop for pretrained GPT-2 124M on prompt/target JSONL pairs.

Everything runs on tensormark tape ops — numpy only shuffles indices and holds
checkpoints, matching the minigpt.py discipline. No PyTorch anywhere.

    python3 train_sft.py --data pairs.jsonl --steps 200 --export
    python3 train_sft.py --data pairs.jsonl --resume out/sft/ckpt.npz --steps 100

JSONL rows: {"prompt": "...", "target": "..."}; loss is computed on target
tokens only (prompt and pad rows carry ignore_index -100).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../tensormark/tensormark
sys.path.insert(0, os.path.dirname(_HERE))                  # native build dir
import tensormark as tm  # noqa: E402
from tensormark.gpt2_sft import (  # noqa: E402
    GPT2BPE, GPT2SFT, SFTBatcher, export_safetensors)


def load_weights(model_dir: str) -> dict[str, np.ndarray]:
    from tensormark.gpt2_runtime import load_safetensors
    return load_safetensors(os.path.join(model_dir, "model.safetensors"))


def lr_at(step: int, args) -> float:
    """Linear warmup to peak, then cosine decay to min-lr-ratio * peak."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_ratio
                      + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path: str, model: GPT2SFT, opt, step: int, args) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, step=step, lr=args.lr,
             **{f"p:{k}": v for k, v in model.state_dict().items()})


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--data", required=True, help="JSONL with prompt/target rows")
    a.add_argument("--model-dir", default="data/gpt2")
    a.add_argument("--out", default="out/sft")
    a.add_argument("--steps", type=int, default=200)
    a.add_argument("--batch", type=int, default=2)
    a.add_argument("--grad-accum", type=int, default=4)
    a.add_argument("--lr", type=float, default=1e-5)
    a.add_argument("--min-lr-ratio", type=float, default=0.1)
    a.add_argument("--warmup", type=int, default=10)
    a.add_argument("--clip", type=float, default=1.0)
    a.add_argument("--weight-decay", type=float, default=0.01)
    a.add_argument("--max-len", type=int, default=512)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--train-embeddings", action="store_true",
                   help="also fine-tune wte/wpe (default: frozen, saves ~1.2 GB)")
    a.add_argument("--resume", default=None)
    a.add_argument("--save-every", type=int, default=50)
    a.add_argument("--export", action="store_true",
                   help="write <out>/model.safetensors for gpt2_runtime.GPT2")
    args = a.parse_args()

    with open(args.data, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"{len(pairs)} prompt/target pairs")

    bpe = GPT2BPE(args.model_dir)
    model = GPT2SFT(load_weights(args.model_dir), train_embeddings=args.train_embeddings)
    batcher = SFTBatcher(pairs, bpe, max_len=args.max_len, seed=args.seed)
    params, decay = model.param_lists()
    opt = tm.AdamW(params, decay, beta1=0.9, beta2=0.999, eps=1e-8,
                   weight_decay=args.weight_decay)
    print(f"{opt.num_params:,} trainable parameters "
          f"(embeddings {'trained' if args.train_embeddings else 'frozen'})")

    start = 0
    if args.resume and os.path.exists(args.resume):
        ck = np.load(args.resume)
        start = int(ck["step"])
        model.load_state_dict({k[2:]: ck[k] for k in ck.files if k.startswith("p:")})
        print(f"resumed from {args.resume} at step {start}")

    losses: list[float] = []
    for step in range(start, args.steps):
        t0 = time.time()
        opt.zero_grad()
        loss_val = 0.0
        for _ in range(args.grad_accum):
            # The engine's tape is one-shot per forward: clearing before each
            # micro-batch is REQUIRED. Reusing a tape across backwards walks
            # stale pooled buffers and produces garbage gradients (verified:
            # grad magnitudes explode x2000 per extra backward). Gradients
            # themselves live on the params, so accumulation is unaffected.
            tm.tape_clear()
            ids, labels = batcher.batch(args.batch)
            loss = model.loss(ids, labels)
            tm.tape_backward(loss)
            loss_val += float(loss.numpy()[0])
        loss_val /= args.grad_accum
        opt.clip_grad_norm(args.clip)
        lr = lr_at(step, args)
        opt.step(lr)
        losses.append(loss_val)
        if True:  # bench_all parses every step; print cost is negligible
            print(f"step {step + 1}/{args.steps}  loss {loss_val:.4f}  "
                  f"lr {lr:.2e}  {time.time() - t0:.2f}s")
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save_checkpoint(os.path.join(args.out, "ckpt.npz"), model, opt, step + 1, args)

    if args.export:
        os.makedirs(args.out, exist_ok=True)
        export_safetensors(model, os.path.join(args.out, "model.safetensors"))
        print(f"exported {os.path.join(args.out, 'model.safetensors')} "
              f"(gpt2_runtime.GPT2 consumes it unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
