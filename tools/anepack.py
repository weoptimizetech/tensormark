#!/usr/bin/env python3
"""anepack — build the fused ANE prefill segment from a tensormark .tmq.

Production shape (docs/ane_layer_split_results.txt): layers [0, K) of the
model fuse into ONE CoreML mlpackage that runs on the Apple Neural Engine.
By default the package emits hidden state plus post-RoPE K and V for every
segment layer, so the engine (llama.h, TM_ANE=1) can ingest them into its KV
cache and decode normally through its own tail layers [K, L). Hidden-only
(--no-kv) is an experimental approximate lane: the final hidden state cannot
reconstruct exact per-layer K/V.

Usage:
  python3 tools/anepack.py <model.tmq> [--layers K] [--tokens T] [--force]

Build cache: ~/.cache/tensormark/ane/<key>/ane_seg.mlpackage with key derived
from model name/size/mtime, K, T, output mode, builder source digest and the
neighboring config.json contents (without hashing the large model contents).
A hit requires matching metadata and an existing package. Requires torch +
coremltools (validated with torch 2.7 / coremltools 9; run inside the build
venv, e.g. /tmp/ane2/bin/python).

Numerics profile: fp16 graph; measured fused-segment quality is top-1 94.9%
end-to-end vs fp32 (better than the 22-layer chained path's 92.5%).
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn


class TmqReader:
    """Minimal .tmq reader — format per tensormark/tmq.h (TMQ1 magic)."""

    def __init__(self, path):
        self.path = path
        self.m = np.memmap(path, np.uint8, "r")
        p = 0
        if self.m[:4].tobytes() != b"TMQ1":
            raise ValueError("not a TMQ1 container: " + path)
        (count,) = np.frombuffer(self.m[4:8].tobytes(), "<u4")
        off = 8
        self.info = {}
        for _ in range(count):
            (nl,) = np.frombuffer(self.m[off:off + 4].tobytes(), "<u4"); off += 4
            name = self.m[off:off + nl].tobytes().decode(); off += nl
            dt, nd = np.frombuffer(self.m[off:off + 8].tobytes(), "<u4"); off += 8
            shape = np.frombuffer(self.m[off:off + 4 * nd].tobytes(), "<u4"); off += 4 * nd
            (nb,) = np.frombuffer(self.m[off:off + 8].tobytes(), "<u8"); off += 8
            self.info[name] = (int(dt), tuple(int(s) for s in shape), off, int(nb))
            off += int(nb) * (34 if dt == 0 else 18)

    def dequant(self, name, dtype=np.float32):
        dt, shape, off, nb = self.info[name]
        raw = np.asarray(self.m[off:off + nb * (34 if dt == 0 else 18)])
        if dt == 0:  # Q8_0
            b = raw.reshape(nb, 34)
            d = b[:, :2].copy().view(np.float16).astype(np.float32)
            q = b[:, 2:].view(np.int8).astype(np.float32)
        else:        # Q4_0: nibble 2j = low lane, 2j+1 = high; value + 8
            b = raw.reshape(nb, 18)
            d = b[:, :2].copy().view(np.float16).astype(np.float32)
            q = b[:, 2:]
            lo = (q & 0x0F).astype(np.int8)
            hi = (q >> 4).astype(np.int8)
            qq = np.empty((nb, 32), np.int8)
            qq[:, 0::2] = lo
            qq[:, 1::2] = hi
            q = (qq.astype(np.int16) - 8).astype(np.float32)
        return (q * d.reshape(nb, 1)).reshape(shape).astype(dtype)


def build(tmq_path, nlayers, S, out_pkg, emit_kv=True):
    t = TmqReader(tmq_path)
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(tmq_path)), "config.json")
    if os.path.exists(cfg_path):
        c = json.load(open(cfg_path))
        D = c["hidden_size"]; H = c["num_attention_heads"]
        KVH = c["num_key_value_heads"]
        dh = c.get("head_dim") or D // H   # head_dim is implied when absent
        theta, eps = c["rope_theta"], c["rms_norm_eps"]
    else:
        D = t.info["model.embed_tokens.weight"][1][1]
        QD = t.info["model.layers.0.self_attn.q_proj.weight"][1][0]
        KVD = t.info["model.layers.0.self_attn.k_proj.weight"][1][0]
        dh, theta, eps = 64, 10000.0, 1e-5
        H, KVH = QD // dh, KVD // dh
    REP = H // KVH
    print(f"dims: D={D} H={H} KVH={KVH} dh={dh} layers={nlayers} S={S}", flush=True)

    half = dh // 2
    inv = theta ** (-np.arange(0, half) * 2 / dh)
    ang = np.outer(np.arange(S, dtype=np.float32), inv)
    COS = torch.from_numpy(np.cos(ang)).half()
    SIN = torch.from_numpy(np.sin(ang)).half()
    CAUSAL_BIAS = torch.triu(torch.full((S, S), -30000.0, dtype=torch.float16), 1)

    class Seg(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.ParameterDict()

        def rms(self, x, w):
            v = x.float()
            return (v / torch.sqrt((v * v).mean(-1, keepdim=True) + eps) * w.float()).half()

        def rope(self, a):
            a1, a2 = a[..., :half], a[..., half:]
            return torch.cat([a1 * COS - a2 * SIN, a2 * COS + a1 * SIN], -1)

        def layer(self, i, x):
            g = lambda nm: self.params[f"l{i}_{nm}"]
            h = self.rms(x, g("ln1"))
            q = (h @ g("q").t().half()).view(1, S, H, dh).transpose(1, 2)
            k = (h @ g("k").t().half()).view(1, S, KVH, dh).transpose(1, 2)
            v = (h @ g("v").t().half()).view(1, S, KVH, dh).transpose(1, 2)
            q, k = self.rope(q), self.rope(k)
            kr = k.repeat_interleave(REP, dim=1)
            vr = v.repeat_interleave(REP, dim=1)
            sc = (q @ kr.transpose(-1, -2)) / dh ** 0.5
            sc = sc + CAUSAL_BIAS                     # finite fp16 causal bias
            sc = torch.softmax(sc, -1).half()          # fp16 softmax
            o = sc @ vr
            o = o.transpose(1, 2).reshape(S, D)
            x1 = x + (o @ g("o").t().half())
            h2 = self.rms(x1, g("ln2"))
            a = h2 @ g("up").t().half()
            b = h2 @ g("gate").t().half()
            act = b * torch.sigmoid(b) * a
            return x1 + (act @ g("down").t().half()), k.squeeze(0), v.squeeze(0)

        def forward(self, x):
            if emit_kv:
                kvs = []
                for i in range(nlayers):
                    x, k, v = self.layer(i, x)
                    kvs += [k, v]
                return (x,) + tuple(kvs)
            # Hidden-only is an experimental approximate lane: final hidden
            # cannot reconstruct the exact per-layer K/V needed for decoding.
            # K/V stay internal; exact cache export requires emit_kv=True.
            for i in range(nlayers):
                x, _, _ = self.layer(i, x)
            return x

    import gc
    m = Seg().eval()
    for i in range(nlayers):
        p = f"model.layers.{i}."
        for nick, nm in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                         ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                         ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
                         ("down", "mlp.down_proj"),
                         ("ln1", "input_layernorm"), ("ln2", "post_attention_layernorm")):
            w = torch.from_numpy(t.dequant(p + nm + ".weight").astype(np.float32))
            m.params[f"l{i}_{nick}"] = nn.Parameter(w, requires_grad=False)
            del w
        gc.collect()
        print(f"layer {i} in", flush=True)

    ex = torch.zeros(S, D).half()
    with torch.no_grad():
        y = m(ex)
    print("torch fwd ok:", len(y), "outputs", flush=True)
    tm = torch.jit.trace(m, (ex,))
    del m
    gc.collect()
    import coremltools as ct
    # Preserve the single unnamed hidden output of the proven graph. KV mode
    # keeps the shim's explicit h/k_<i>/v_<i> output contract.
    output_options = {}
    if emit_kv:
        named = [ct.TensorType(name="h")]
        for i in range(nlayers):
            named += [ct.TensorType(name=f"k_{i}"), ct.TensorType(name=f"v_{i}")]
        output_options["outputs"] = named
    ml = ct.convert(tm,
                    inputs=[ct.TensorType(name="x", shape=(S, D), dtype=np.float16)],
                    compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.iOS16,
                    **output_options)
    ml.save(out_pkg)
    print("saved", out_pkg, flush=True)


def cache_key(tmq_path, nlayers, tokens, emit_kv=True):
    """Hash small build inputs; model identity uses metadata, not its contents."""
    st = os.stat(tmq_path)
    with open(__file__, "rb") as source:
        builder_digest = hashlib.sha256(source.read()).hexdigest()
    config_path = os.path.join(os.path.dirname(os.path.abspath(tmq_path)), "config.json")
    try:
        with open(config_path, "rb") as config:
            config_digest = hashlib.sha256(config.read()).hexdigest()
    except FileNotFoundError:
        config_digest = None
    identity = {
        "model": os.path.basename(tmq_path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "layers": nlayers,
        "tokens": tokens,
        "emit_kv": emit_kv,
        "builder": builder_digest,
        "config": config_digest,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tmq")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/tensormark/ane"))
    ap.add_argument("--out", help="write the package here instead of the cache")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-kv", action="store_true",
                    help="experimental approximate hidden-only lane; final hidden "
                         "cannot reconstruct exact per-layer K/V")
    a = ap.parse_args()

    key = cache_key(a.tmq, a.layers, a.tokens, emit_kv=not a.no_kv)
    if a.out:
        out_pkg, meta = a.out, a.out + ".meta.json"
    else:
        out_pkg = os.path.join(a.cache, key, "ane_seg.mlpackage")
        meta = os.path.join(a.cache, key, "meta.json")
    if os.path.isfile(meta) and os.path.isdir(out_pkg) and not a.force:
        try:
            with open(meta) as f:
                cached = json.load(f)
        except (OSError, ValueError):
            cached = None
        if (isinstance(cached, dict) and cached.get("cache_key") == key
                and cached.get("package") == out_pkg):
            print(out_pkg)
            return
    os.makedirs(os.path.dirname(out_pkg) or ".", exist_ok=True)
    build(a.tmq, a.layers, a.tokens, out_pkg, emit_kv=not a.no_kv)
    with open(meta, "w") as f:
        json.dump({"package": out_pkg, "layers": a.layers, "tokens": a.tokens,
                   "emit_kv": not a.no_kv, "cache_key": key,
                   "model": os.path.basename(a.tmq)}, f, indent=1)
    print(out_pkg)


if __name__ == "__main__":
    main()
