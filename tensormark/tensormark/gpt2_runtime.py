"""GPT-2 124M fp32 inference — the numpy reference the SFT path mirrors.

Graph on existing engine ops (GPT-2 semantics per HF modeling_gpt2):
  x = wte[ids] + wpe[pos]; per block: h = LN1(x); qkv = h @ c_attn + b;
  causal attention over 12 heads (1/sqrt(dh) scaling); x += attn @ c_proj + b;
  h2 = LN2(x); x += gelu(h2 @ c_fc + b) @ c_proj + b; logits = LNf(x) @ wte^T.
HF Conv1D == x @ W + b with W stored (in, out) == our matmul.

KV cache: after a T-token prefill, each block's K/V tape values hold T rows.
Incremental decode appends one row per step and attends over all of it:
prefill computes the T-token logits once; steps 1..n each run a 1-token
forward whose K/V are the cached rows plus the new one, and only the last
logit row is read. All in numpy here (inference needs no tape/autograd);
the engine ops are exercised by minigpt's parity gates.
"""
from __future__ import annotations

import json
import math
import os
import struct
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def data_dir(name: str = "gpt2") -> str:
    """Locate model data: $TENSORMARK_DATA/<name> if set, else the repo's
    tensormark/data/<name> if present (model weights are not bundled)."""
    env = os.environ.get("TENSORMARK_DATA")
    if env:
        return os.path.join(env, name)
    return os.path.normpath(os.path.join(_HERE, os.pardir, "data", name))

# ---------------------------------------------------------------------------
# safetensors reader (python mirror of safetensors.h — same layout rules)
# ---------------------------------------------------------------------------
def load_safetensors(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        data_begin = 8 + hlen
    out = {}
    with open(path, "rb") as f:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if meta["dtype"] != "F32":
                raise ValueError(f"{name}: unsupported dtype {meta['dtype']}")
            b, e = meta["data_offsets"]
            f.seek(data_begin + b)
            arr = np.frombuffer(f.read(e - b), dtype="<f4")
            out[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class GPT2:
    def __init__(self, model_dir: str):
        self.w = load_safetensors(os.path.join(model_dir, "model.safetensors"))
        self.n_layer, self.D, self.H = 12, 768, 12
        self.dh = self.D // self.H
        self.ctx = 1024
        self.V = self.w["wte.weight"].shape[0]
        self.kv: list[tuple[np.ndarray, np.ndarray]] = []  # per layer (K, V)

    def reset_cache(self):
        self.kv = []

    def _attend(self, q, k, v, pos0: int):
        """(T, D) q; (S, D) k/v; causal attention over the full history."""
        T, D = q.shape
        H, dh = self.H, self.dh
        S = k.shape[0]
        qh = q.reshape(T, H, dh).transpose(1, 0, 2)       # (H, T, dh)
        kh = k.reshape(S, H, dh).transpose(1, 0, 2)       # (H, S, dh)
        vh = v.reshape(S, H, dh).transpose(1, 0, 2)
        scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(dh)  # (H, T, S)
        # query row t (global position pos0+t) may attend to keys 0..pos0+t
        allow = np.arange(S)[None, :] <= (pos0 + np.arange(T))[:, None]
        scores = np.where(~allow, -1e10, scores)
        out = softmax(scores) @ vh                        # (H, T, dh)
        return out.transpose(1, 0, 2).reshape(T, D)

    def forward(self, ids: list[int], pos0: int) -> np.ndarray:
        """Returns logits for every input position, (T, V). Appends K/V."""
        w = self.w
        T = len(ids)
        self._pos0 = pos0
        x = w["wte.weight"][ids] + w["wpe.weight"][pos0:pos0 + T]
        new_kv = []
        for l in range(self.n_layer):
            p = lambda k: w[f"h.{l}." + k]
            h = layernorm(x, p("ln_1.weight"), p("ln_1.bias"))
            qkv = h @ p("attn.c_attn.weight") + p("attn.c_attn.bias")
            q, k, v = np.split(qkv, 3, axis=-1)
            if l < len(self.kv):
                k = np.concatenate([self.kv[l][0], k], axis=0)
                v = np.concatenate([self.kv[l][1], v], axis=0)
            a = self._attend(q, k, v, pos0)
            x = x + a @ p("attn.c_proj.weight") + p("attn.c_proj.bias")
            h2 = layernorm(x, p("ln_2.weight"), p("ln_2.bias"))
            m = gelu(h2 @ p("mlp.c_fc.weight") + p("mlp.c_fc.bias"))
            x = x + m @ p("mlp.c_proj.weight") + p("mlp.c_proj.bias")
            new_kv.append((k, v))
        self.kv = new_kv
        return layernorm(x, w["ln_f.weight"], w["ln_f.bias"]) @ w["wte.weight"].T

    def generate(self, ids: list[int], n_tokens: int, greedy=True,
                 temperature=1.0, top_k=None, top_p=None):
        ids = list(ids)
        logits = self.forward(ids, 0)[-1]
        for _ in range(n_tokens):
            if greedy:
                nxt = int(np.argmax(logits))
            else:
                # Divide unconditionally: /1.0 is a no-op at temp=1.
                # (A walrus here once shadowed the parameter with the
                # literal 1.0, silently making temperature a no-op.)
                lg = logits.astype(np.float64) / max(temperature, 1e-6)
                if top_k:
                    keep = np.argpartition(-lg, top_k - 1)[:top_k]
                    m = np.full_like(lg, -np.inf)
                    m[keep] = lg[keep]
                    lg = m
                if top_p is not None:
                    srt = np.sort(lg)[::-1]
                    cum = np.cumsum(softmax(srt))
                    cut = srt[np.searchsorted(cum, top_p)]
                    lg = np.where(lg < cut, -np.inf, lg)
                pr = np.exp(lg - lg.max())
                pr /= pr.sum()
                nxt = int(np.random.choice(len(pr), p=pr))
            ids.append(nxt)
            logits = self.forward([nxt], len(ids) - 1)[-1]
        return ids


def main():
    d = data_dir("gpt2")
    model = GPT2(d)
    ref = json.load(open(os.path.join(d, "ref_greedy.json")))
    prompt_ids = ref["prompt_ids"]
    want = ref["tokens"][len(prompt_ids):]
    t0 = time.perf_counter()
    got = model.generate(prompt_ids, len(want), greedy=True)
    dt = time.perf_counter() - t0
    n_match = sum(1 for a, b in zip(got[len(prompt_ids):], want) if a == b)
    n = n_match
    print(f"greedy match: {n_match}/{len(want)} tokens ({n/len(want)*100:.0f}%)")
    print(f"decode: {len(want)/dt:.1f} tok/s (numpy reference path)")
    print("first 16:", got[:16])
    print("want  16:", want[:16])
    ok = n_match == len(want)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
