"""SFT gates: graph parity vs the numpy reference, padding exactness,
ignore_index loss/grad correctness, one-step effect, export roundtrip.

Run from the repo's tensormark/ directory (native module must be importable):
    python3 tests/test_sft.py
"""

from __future__ import annotations

import os
import sys
import math

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # native module dir
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root shims
import tensormark as tm
from tensormark.gpt2_runtime import gelu as ref_gelu, layernorm as ref_ln
from tensormark.gpt2_sft import (GPT2SFT, save_safetensors,
                                 IGNORE_INDEX)

CFG = dict(n_layer=2, n_head=4, ctx=32)             # D=32, V=64, tiny + fast
V, D = 64, 32


def tiny_weights(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w: dict[str, np.ndarray] = {
        "wte.weight": rng.normal(0, 0.02, (V, D)).astype(np.float32),
        "wpe.weight": rng.normal(0, 0.02, (CFG["ctx"], D)).astype(np.float32),
        "ln_f.weight": np.ones(D, dtype=np.float32),
        "ln_f.bias": np.zeros(D, dtype=np.float32),
    }
    for l in range(CFG["n_layer"]):
        pre = f"h.{l}."
        for k, shape in (("ln_1.weight", (D,)), ("ln_1.bias", (D,)),
                         ("attn.c_attn.weight", (D, 3 * D)), ("attn.c_attn.bias", (3 * D,)),
                         ("attn.c_proj.weight", (D, D)), ("attn.c_proj.bias", (D,)),
                         ("ln_2.weight", (D,)), ("ln_2.bias", (D,)),
                         ("mlp.c_fc.weight", (D, 4 * D)), ("mlp.c_fc.bias", (4 * D,)),
                         ("mlp.c_proj.weight", (4 * D, D)), ("mlp.c_proj.bias", (D,))):
            w[pre + k] = rng.normal(0, 0.02, shape).astype(np.float32)
        w[pre + "ln_1.weight"][:] = 1.0
        w[pre + "ln_2.weight"][:] = 1.0
    return w


def numpy_forward(w: dict[str, np.ndarray], ids: np.ndarray, n_head: int,
                  D: int = D) -> np.ndarray:
    """Reference GPT-2 forward, mirroring tensormark.gpt2_runtime.GPT2 math."""
    B, T = ids.shape
    dh = D // n_head
    x = w["wte.weight"][ids.reshape(-1)] + np.tile(w["wpe.weight"][:T], (B, 1))
    H = n_head
    for l in range(CFG["n_layer"]):
        pre = f"h.{l}."
        p = lambda k, pre=pre: w[pre + k]
        h = ref_ln(x, p("ln_1.weight"), p("ln_1.bias"))
        qkv = h @ p("attn.c_attn.weight") + p("attn.c_attn.bias")
        q, k, v = np.split(qkv, 3, axis=-1)
        qh = q.reshape(B * T, H, dh).transpose(1, 0, 2)
        kh = k.reshape(B * T, H, dh).transpose(1, 0, 2)
        vh = v.reshape(B * T, H, dh).transpose(1, 0, 2)
        scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(dh)
        # batched causal: row b*T+t attends to keys b*T+0 .. b*T+t
        r = np.arange(B * T)
        allow = (r[:, None] // T == r[None, :] // T) & (r[None, :] % T <= r[:, None] % T)
        scores = np.where(~allow, -1e10, scores)
        e = np.exp(scores - scores.max(-1, keepdims=True))
        a = (e / e.sum(-1, keepdims=True)) @ vh
        a = a.transpose(1, 0, 2).reshape(B * T, D)
        x = x + a @ p("attn.c_proj.weight") + p("attn.c_proj.bias")
        h2 = ref_ln(x, p("ln_2.weight"), p("ln_2.bias"))
        m = ref_gelu(h2 @ p("mlp.c_fc.weight") + p("mlp.c_fc.bias"))
        x = x + m @ p("mlp.c_proj.weight") + p("mlp.c_proj.bias")
    x = ref_ln(x, w["ln_f.weight"], w["ln_f.bias"])
    return x @ w["wte.weight"].T


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        global failures
        failures += 1


failures = 0
W = tiny_weights()
model = GPT2SFT(W, n_layer=CFG["n_layer"], n_head=CFG["n_head"])
tm.set_grad_enabled(False)

# 1) graph parity vs the numpy reference ------------------------------------
ids = np.array([[5, 9, 2, 8, 1, 3], [7, 4, 6, 0, 2, 5]], dtype=np.int32)
tm.tape_clear()
got = model.forward(ids).numpy()
want = numpy_forward(W, ids, CFG["n_head"])
err = float(np.abs(got - want).max())
check("graph parity vs numpy reference", err <= 1e-3, f"max|dlogit|={err:.2e}")

# 2) right-padded batching is exact ------------------------------------------
full, partial = ids[0], ids[1][:4]                      # row 1 padded to len 6
tm.tape_clear()
padded = model.forward(np.stack([full, np.concatenate([partial, [0, 0]])])).numpy()
single = model.forward(np.stack([full])) .numpy()
part_single = model.forward(np.stack([partial])).numpy()
d_full = float(np.abs(padded[:6] - single).max())
d_part = float(np.abs(padded[6:10] - part_single).max())
check("right-pad + causal needs no attention mask", max(d_full, d_part) <= 1e-4,
      f"full={d_full:.2e} partial={d_part:.2e}")

# 3) ignore_index loss and gradients -----------------------------------------
tm.set_grad_enabled(True)
opt = tm.AdamW(*model.param_lists(), beta1=0.9, beta2=0.999, eps=1e-8,
               weight_decay=0.01)
targets = ids.copy()
targets[0, 3:] = IGNORE_INDEX                            # mask part of row 0
tm.tape_clear()
loss = model.loss(ids, targets)
val = float(loss.numpy()[0])
logits = numpy_forward(W, ids, CFG["n_head"])
p = np.exp(logits - logits.max(-1, keepdims=True))
p /= p.sum(-1, keepdims=True)
ref = -np.mean([np.log(p[b * 6 + t, targets[b, t]])
                for b in range(2) for t in range(6) if targets[b, t] != IGNORE_INDEX])
check("ignore_index loss == numpy CE over valid rows", abs(val - ref) <= 1e-3,
      f"{val:.5f} vs {ref:.5f}")
tm.tape_backward(loss)
g = model.w["h.0.attn.c_attn.weight"].grad
check("gradient flows through masked loss",
      g is not None and np.isfinite(g).all() and np.abs(g).max() > 0,
      f"max|g|={np.abs(g).max():.2e}")
tm.tape_clear()
opt.zero_grad()
loss0 = model.loss(ids, np.full_like(ids, IGNORE_INDEX))
tm.tape_backward(loss0)
g0 = model.w["h.0.attn.c_attn.weight"].grad
check("all-ignored rows give zero loss and zero grads",
      float(loss0.numpy()[0]) == 0.0
      and (g0 is None or np.abs(g0).max() == 0.0))

# 3b) gradient accumulation: clear-per-micro-batch must equal the sum of
# individual gradients (the engine tape is one-shot per forward — reusing
# a tape across backwards walks stale pooled buffers and corrupts grads).
key = "h.0.attn.c_attn.weight"
ids1, ids2 = ids[:1], ids[1:2]
targets1, targets2 = targets[:1], targets[1:2]
tm.tape_clear(); opt.zero_grad()
tm.tape_backward(model.loss(ids1, targets1))
g1 = model.w[key].grad.copy()
tm.tape_clear(); opt.zero_grad()
tm.tape_backward(model.loss(ids2, targets2))
g2 = model.w[key].grad.copy()
tm.tape_clear(); opt.zero_grad()
tm.tape_clear(); tm.tape_backward(model.loss(ids1, targets1))
tm.tape_clear(); tm.tape_backward(model.loss(ids2, targets2))
gsum = model.w[key].grad
acc_err = float(np.abs(gsum - (g1 + g2)).max())
check("grad accumulation == sum of micro-batch grads",
      np.isfinite(acc_err) and acc_err <= 1e-5, f"maxerr={acc_err:.2e}")
tm.tape_clear(); opt.zero_grad()

# 4) one AdamW step changes logits by a small, finite amount -----------------
probe = ids[:1]
tm.tape_clear()
before = model.forward(probe).numpy()
loss = model.loss(ids, targets)
tm.tape_backward(loss)
opt.clip_grad_norm(1.0)
opt.step(1e-4)
tm.tape_clear()
after = model.forward(probe).numpy()
delta = float(np.abs(after - before).max())
check("one step moves logits finitely", np.isfinite(delta) and 0 < delta < 1.0,
      f"max|dlogit|={delta:.2e}")
tm.set_grad_enabled(False)

# 5) export roundtrip through the safetensors writer -------------------------
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model.safetensors")
    save_safetensors(path, model.state_dict())
    from tensormark.gpt2_runtime import load_safetensors
    back = load_safetensors(path)
    same = all(np.array_equal(back[k], model.state_dict()[k]) for k in back)
    check("safetensors writer roundtrip", same and len(back) == len(model.w))

# 6) real GPT-2 124M golden parity (skipped without the snapshot) ------------
real = os.path.join(os.path.dirname(_HERE), "data", "gpt2")
if os.path.exists(os.path.join(real, "model.safetensors")):
    from tensormark.gpt2_runtime import load_safetensors, GPT2
    big = GPT2SFT(load_safetensors(os.path.join(real, "model.safetensors")))
    ids_big = [464, 3139, 286, 4881, 318]               # "The theory of relativity is"
    ref_big = GPT2(real).forward(ids_big, 0)             # canonical numpy runtime
    tm.tape_clear()
    got_big = big.forward(np.array([ids_big], dtype=np.int32)).numpy()
    got_big = got_big.reshape(len(ids_big), -1)
    err_big = float(np.abs(got_big - ref_big).max())
    scale_big = float(np.abs(ref_big).max())
    check("124M golden parity vs gpt2_runtime", err_big / scale_big <= 1e-3,
          f"max|dlogit|={err_big:.2e} (logit scale {scale_big:.0f})")
else:
    print("[SKIP] 124M golden parity (data/gpt2/model.safetensors not present)")

print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
sys.exit(1 if failures else 0)
