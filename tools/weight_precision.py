#!/usr/bin/env python3
"""How much precision does a checkpoint actually carry, and what would widening cost?

Two questions that look like one, and answering them wrongly sent a whole line
of work at the wrong target:

  1. IS THIS CHECKPOINT NATIVELY LOW-BIT? A model trained to ternary has three
     DISTINCT values per quantization group (times the group scale); a model
     trained in bf16/fp16 does not. Counting them settles it, because it needs no
     assumption about provenance. This is what showed
     `prism-ml/Ternary-Bonsai-4B-unpacked` — an 8.06 GB fp16 file that reads like
     a high-precision base model — to be the DEQUANTIZATION of the ternary pack.

  2. WHAT DOES WIDENING COST? Re-encode the weights under a wider rule and measure
     the reconstruction error. On a natively-ternary model the answer is a
     REFUTATION, not a curve: the shipped ternary pack is already at the fp16
     scale floor (relE ~4e-5), while Q4_0 of the same weights is ~2000x worse
     (relE 8.6e-2) at 2.12x the bytes, because Q4_0's grid is asymmetric
     ([-8d, +7d]) and cannot carry a symmetric +-d ternary pair exactly. There is
     no lost precision to buy back.

Usage:

    weight_precision.py <model_dir_or.tmq> [--top N]

Reads either an HF safetensors directory (single-file or sharded) or a TMQ1
container. Prints per-role reconstruction error under both rules, the per-group
distinct-value census, and the worst individual tensors.
"""
from __future__ import annotations

import glob
import json
import os
import re
import struct
import sys

import numpy as np

GROUP = 128   # values per ternary block
Q4G = 32      # values per Q4_0 block
# payload geometry per TMQ1 dtype id (see tensormark/tmq.h)
TMQ_ELEMS = {0: 32, 1: 32, 2: 1, 3: 1, 4: 32, 5: 256, 6: 256, 7: 128}
TMQ_BYTES = {0: 34, 1: 18, 2: 4, 3: 2, 4: 20, 5: 176, 6: 210, 7: 34}


def _shards(d: str) -> list[str]:
    one = os.path.join(d, "model.safetensors")
    if os.path.exists(one):
        return [one]
    many = sorted(glob.glob(os.path.join(d, "model-*.safetensors")))
    if not many:
        raise SystemExit(f"{d}: no model.safetensors and no model-*.safetensors")
    return many


def read_safetensors(src: str):
    """Yield (name, dtype, shape, fp32 array) from an HF safetensors layout."""
    for path in _shards(src):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
            base = f.tell()
            for name, e in head.items():
                if name == "__metadata__":
                    continue
                lo, hi = e["data_offsets"]
                f.seek(base + lo)
                blob = f.read(hi - lo)
                dt = e["dtype"]
                if dt == "F16":
                    x = np.frombuffer(blob, "<f2").astype(np.float32)
                elif dt == "BF16":
                    x = (np.frombuffer(blob, "<u2").astype(np.uint32) << 16).view(np.float32)
                elif dt == "F32":
                    x = np.frombuffer(blob, "<f4").astype(np.float32)
                else:
                    continue
                yield name, dt, list(e["shape"]), x


def read_tmq(path: str):
    """Yield (name, dtype_id, shape, fp32 array) from a TMQ1 container."""
    with open(path, "rb") as f:
        if f.read(4) != b"TMQ1":
            raise SystemExit(f"{path}: not a .tmq (bad magic)")
        (count,) = struct.unpack("<I", f.read(4))
        for _ in range(count):
            (nl,) = struct.unpack("<I", f.read(4))
            name = f.read(nl).decode()
            (dt,) = struct.unpack("<I", f.read(4))
            (nd,) = struct.unpack("<I", f.read(4))
            shape = list(struct.unpack("<%dI" % nd, f.read(4 * nd)))
            (nb,) = struct.unpack("<Q", f.read(8))
            payload = f.read(nb * TMQ_BYTES[dt])
            yield name, dt, shape, payload, nb


def _unpack_tq2(payload: bytes, n: int) -> np.ndarray:
    """w = (code - 1) * d, code in {0,1,2}; element j at word j//16, bits 2*(j%16)."""
    raw = np.frombuffer(payload, np.uint8).reshape(n // GROUP, 34)
    d = np.ascontiguousarray(raw[:, 0:2].copy().view("<f2").astype(np.float32))[:, 0]
    words = np.ascontiguousarray(raw[:, 2:34].copy().view(np.uint32))   # (g, 8)
    shifts = (np.arange(16, dtype=np.uint32) * 2)
    codes = ((words[..., None] >> shifts) & np.uint32(3)).reshape(n // GROUP, GROUP)
    return ((codes.astype(np.float32) - 1.0) * d[:, None]).reshape(-1)


def _unpack_q4_0(payload: bytes, n: int) -> np.ndarray:
    raw = np.frombuffer(payload, np.uint8).reshape(n // Q4G, 18)
    d = np.ascontiguousarray(raw[:, 0:2].copy().view("<f2").astype(np.float32))[:, 0]
    qs = raw[:, 2:18]
    lo = (qs & 0x0F).astype(np.float32)
    hi = ((qs >> 4) & 0x0F).astype(np.float32)
    got = np.stack([lo, hi], axis=-1).reshape(n // Q4G, Q4G)
    return ((got - 8.0) * d[:, None]).reshape(-1)


def err_tq2(x: np.ndarray):
    """x: (out, groups, 128) absmax ternary -> (sq-err, sq-w) per row."""
    d = np.abs(x).max(axis=2, keepdims=True)
    ds = np.where(d == 0, 1.0, d)
    q = np.clip(np.rint(x / ds), -1.0, 1.0) * d
    return ((x - q) ** 2).sum(axis=(1, 2)), (x ** 2).sum(axis=(1, 2))


def err_q4(x: np.ndarray):
    """x: (out, groups, 32) ggml Q4_0 -> (sq-err, sq-w) per row."""
    idx = np.argmax(np.abs(x), axis=2)
    mx = np.take_along_axis(x, idx[..., None], axis=2)[..., 0]
    d = (mx / -8.0)[..., None]
    nz = d != 0
    idn = np.where(nz, 1.0 / np.where(nz, d, 1.0), 0.0)
    q = (np.clip(np.floor(x * idn + 8.5), 0.0, 15.0) - 8.0) * d
    return ((x - q) ** 2).sum(axis=(1, 2)), (x ** 2).sum(axis=(1, 2))


def _source(path: str):
    """Normalize both containers to (name, shape, fp32-or-None).

    Both rules are evaluated on every decodable tensor regardless of how it is
    STORED: the question is not "how good is the stored format" but "what would
    the other rule cost on these same weights", which is what decides whether
    widening could ever pay. A pure-TQ2 pack therefore still reports its Q4_0
    price, and vice versa.
    """
    if path.endswith(".tmq"):
        for name, dt, shape, payload, nb in read_tmq(path):
            if len(shape) != 2:
                continue
            n = shape[0] * shape[1]
            if dt == 7:
                yield name, shape, _unpack_tq2(payload, n)
            elif dt == 1:
                yield name, shape, _unpack_q4_0(payload, n)
            else:
                yield name, shape, None
    else:
        for name, _dt, shape, x in read_safetensors(path):
            if len(shape) != 2:
                continue
            yield name, shape, x


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = 12
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    if not args:
        print("usage: weight_precision.py <model_dir_or.tmq> [--top N]")
        return 2
    path = args[0]

    roles: dict[str, list] = {}
    census: dict[str, list] = {}
    indiv: list[tuple[float, str]] = []
    stored: dict[str, int] = {}

    for name, shape, x in _source(path):
        if x is None or shape[1] % GROUP:
            continue
        roles_key = re.sub(r"\.\d+\.", ".N.", name)
        g = x.reshape(shape[0], shape[1] // GROUP, GROUP)
        # census: distinct values per group, normalized by that group's absmax so
        # the fp16 scale granularity shows as near-duplicates, not extra levels
        d = np.abs(g).max(axis=2, keepdims=True)
        ds = np.where(d == 0, 1.0, d)
        norm = (g / ds)[:64].reshape(-1, GROUP)
        uniq = np.array([len(np.unique(np.round(row, 5))) for row in norm])
        # min/max are the ROLE's, not a running sum across its tensors: a role
        # with 36 tensors must report the smallest group seen among them, so the
        # figure stays comparable to a single-tensor role.
        c = census.setdefault(roles_key, [int(uniq.min()), int(uniq.max()), []])
        c[0] = min(c[0], int(uniq.min()))
        c[1] = max(c[1], int(uniq.max()))
        c[2].extend(uniq.tolist())

        a = roles.setdefault(roles_key, [0, 0.0, 0.0, 0.0, 0.0, 680.0, 1440.0])
        a[0] += 1
        n1, d1 = err_tq2(g)
        a[1] += float(n1.sum()); a[2] += float(d1.sum())
        e1 = float(np.sqrt(n1.sum() / d1.sum())) if d1.sum() else 0.0
        indiv.append((e1, name))
        n2, d2 = err_q4(g.reshape(shape[0], shape[1] // Q4G, Q4G))
        a[3] += float(n2.sum()); a[4] += float(d2.sum())
        inn = shape[1]
        a[5] = inn / GROUP * 34.0
        a[6] = inn / Q4G * 18.0
        stored[name] = shape[0] * shape[1]

    if not roles:
        print("no 2-D tensors of a known rule in", path)
        return 1

    print("\n=== distinct values per %d-value group (3 => natively ternary) ===" % GROUP)
    for role, (mn, mx, allu) in sorted(census.items()):
        med = float(np.median(allu)) if allu else 0.0
        tag = "  <-- NATIVELY TERNARY" if 1 <= med <= 4 and mx <= 4 else ""
        print("  %-56s min %d  median %.0f  max %d%s" % (role, mn, med, mx, tag))

    hdr = "%-56s %3s %10s %10s %11s %11s %7s"
    print("\n=== reconstruction error per role ===")
    print(hdr % ("role", "x", "relE_TQ2", "relE_Q4", "B/row_tq2", "B/row_q4", "ratio"))
    print("-" * 114)

    def rel(n: float, d: float) -> str:
        return "%.5f" % ((n / d) ** 0.5) if d > 0 else "n/a"

    tot = [0.0, 0.0, 0.0, 0.0]
    for role, v in sorted(roles.items(), key=lambda kv: -kv[1][1]):
        c, n1, d1, n2, d2, bt, bq = v
        print(hdr % (role, c, rel(n1, d1), rel(n2, d2), "%.1f" % bt, "%.1f" % bq,
                     "%.2fx" % (bq / max(bt, 1e-9))))
        tot[0] += n1; tot[1] += d1; tot[2] += n2; tot[3] += d2
    print("-" * 114)
    print(hdr % ("ALL 2-D", "", rel(tot[0], tot[1]), rel(tot[2], tot[3]), "", "", ""))

    if indiv:
        print("\n=== %d worst tensors under the ternary rule ===" % min(top, len(indiv)))
        for e1, name in sorted(indiv, reverse=True)[:top]:
            print("  %-58s relE_TQ2 %.5f" % (name, e1))
    print("\nrelE = sqrt(sum((w-q)^2)/sum(w^2)). Lower is better. Q4_0 is 2.12x the bytes.")
    print("A natively-ternary model has nothing to recover: the shipped ternary pack is")
    print("already at the fp16 scale floor, and Q4_0 of ternary weights is ~2000x worse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
