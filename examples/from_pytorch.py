#!/usr/bin/env python3
"""from_pytorch.py — the same training loop in PyTorch and TensorMark.

TensorMark's compatibility layer is a PyTorch subset: you swap
`import torch` for `import tensormark.torch as torch` and the training
loop stays the same. This script runs BOTH, side by side, on the same
task (a small MLP on a synthetic regression problem), and prints the
loss curves and wall times so you can see what changes — the numbers,
not just the API.

    python3 examples/from_pytorch.py

PyTorch is only needed for the comparison column; TensorMark itself
depends on NumPy alone (install it with `python -m pip install ./tensormark`).
"""
from __future__ import annotations

import importlib
import time

import numpy as np


def make_task(n: int = 4096, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 8)).astype(np.float32)
    w = np.array([2.0, -1.5, 1.0, 0.5, -2.0, 1.5, -0.5, 1.0], dtype=np.float32)
    y = (x @ w + 0.3 * np.sin(x[:, 0]))[:, None]
    return x, y


def train(torch, x, y, epochs: int = 60):
    """The identical loop for both frameworks — only the import differs."""
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))  # .item() is the portable scalar escape
    return losses, time.perf_counter() - t0


def main() -> None:
    x_np, y_np = make_task()
    epochs = 60
    results = {}

    try:
        torch_real = importlib.import_module("torch")
        results["pytorch (CPU)"] = train(torch_real, torch_real.from_numpy(x_np),
                                         torch_real.from_numpy(y_np), epochs)
    except ImportError:
        print("PyTorch not installed — showing TensorMark only.\n")

    import tensormark.torch as tm

    # NumPy in, NumPy out; the graph, optimizer and autograd are TensorMark's.
    results["tensormark"] = train(tm, tm.from_numpy(x_np), tm.from_numpy(y_np), epochs)

    print(f"{'framework':<16}{'final loss':>14}{'epoch 1':>12}{'wall s':>10}")
    for name, (losses, wall) in results.items():
        print(f"{name:<16}{losses[-1]:>14.5f}{losses[0]:>12.5f}{wall:>10.2f}")

    if len(results) == 2:
        lp, lt = results["pytorch (CPU)"][0], results["tensormark"][0]
        print(f"\nfinal loss gap: {abs(lp[-1] - lt[-1]):.4f} (curves start from "
              f"different inits, so early losses diverge; both converge on the "
              f"same optimum).")
    print("\nThe loop above is UNCHANGED between frameworks — only the import "
          "differs. Note: gradients in the TensorMark layer are NumPy arrays, "
          "and the covered subset is documented in the README (Alongside PyTorch).")


if __name__ == "__main__":
    main()
