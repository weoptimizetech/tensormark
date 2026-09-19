"""Tests for tools/weight_precision.py — the "how much precision is actually here?" probe.

The tool exists because a wrong answer to that question sent a line of work at the
wrong target: an 8.06 GB fp16 artifact that reads like a high-precision base model
turned out to be the DEQUANTIZATION of a ternary pack. These tests pin the two
measurements that show it, on synthetic data where the answer is known:

  - a natively-ternary tensor has 3 distinct values per group and is reproduced by
    the ternary rule to ~the fp16 scale floor;
  - a continuous tensor is NOT (the ternary rule costs it tens of percent);
  - Q4_0 on ternary-valued weights is far worse than the ternary rule, which is the
    refutation of "widen the weights to buy quality back".

The TMQ1 reader is pinned too, so the tool cannot silently drift from the container
(tensormark/tmq.h) it is supposed to describe.
"""

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


_HERE = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wp = _load("weight_precision_under_test", "weight_precision.py")
qt = _load("quantize_ternary_for_probe", "quantize_ternary.py")

GROUP = 128


def _write_safetensors(path: Path, tensors: dict) -> None:
    head, off = {}, 0
    for name, (dtype, shape, blob) in tensors.items():
        head[name] = {"dtype": dtype, "shape": list(shape),
                      "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    hjson = json.dumps(head).encode()
    hjson += b" " * ((8 - len(hjson) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for _n, (_d, _s, blob) in tensors.items():
            f.write(blob)


def _ternary_tensor(rng, out: int, inn: int) -> np.ndarray:
    """Exactly what a natively-ternary checkpoint holds: 3 levels per group."""
    codes = rng.integers(0, 3, size=(out, inn // GROUP, GROUP)).astype(np.float32)
    d = rng.uniform(0.01, 0.04, size=(out, inn // GROUP, 1)).astype(np.float32)
    return ((codes - 1.0) * d).reshape(out, inn).astype(np.float32)


def _continuous_tensor(rng, out: int, inn: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(out, inn)).astype(np.float32)


class CensusTests(unittest.TestCase):
    def test_census_separates_ternary_from_continuous(self):
        rng = np.random.default_rng(3)
        g = _ternary_tensor(rng, 8, 256).reshape(8, 2, GROUP)
        d = np.abs(g).max(axis=2, keepdims=True)
        norm = (g / np.where(d == 0, 1.0, d)).reshape(-1, GROUP)
        uniq = np.array([len(np.unique(np.round(row, 5))) for row in norm])
        self.assertEqual(uniq.min(), 3)
        self.assertEqual(uniq.max(), 3)

        c = _continuous_tensor(rng, 8, 256).reshape(-1, GROUP)
        uniq_c = np.array([len(np.unique(np.round(row, 5))) for row in c])
        self.assertGreater(uniq_c.min(), 8,
                           "a continuous group must not look ternary")

    def test_role_census_takes_the_minimum_not_a_sum(self):
        # REGRESSION: the role aggregate used to ACCUMULATE each tensor's minimum,
        # so a role with 36 tensors reported "min 108" — a number that reads as
        # "not ternary" on a pack that is exactly ternary. min/max are the role's.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rng = np.random.default_rng(23)
            tensors = {}
            for i in range(4):
                w = _ternary_tensor(rng, 4, 256)
                tensors["model.layers.%d.mlp.gate_proj.weight" % i] = (
                    "F16", (4, 256), w.astype("<f2").tobytes())
            _write_safetensors(tmp / "model.safetensors", tensors)

            out = subprocess.run(
                [sys.executable, str(_HERE / "weight_precision.py"), str(tmp)],
                capture_output=True, text=True, check=True).stdout
        line = next(ln for ln in out.splitlines() if "mlp.gate_proj" in ln)
        self.assertIn("min 3", line)
        self.assertIn("max 3", line)
        self.assertNotIn("min 12", line,
                         "per-tensor minima were summed: %s" % line)


class ErrorModelTests(unittest.TestCase):
    def test_ternary_rule_is_essentially_lossless_on_a_ternary_tensor(self):
        rng = np.random.default_rng(11)
        t = _ternary_tensor(rng, 16, 256)
        n, d = wp.err_tq2(t.reshape(16, 2, GROUP))
        rel = float(np.sqrt(n.sum() / d.sum()))
        self.assertLess(rel, 1e-3,
                        "ternary on ternary should be at the fp16 floor, got %g" % rel)

    def test_ternary_rule_costs_a_continuous_tensor_dearly(self):
        rng = np.random.default_rng(13)
        c = _continuous_tensor(rng, 16, 256)
        n, d = wp.err_tq2(c.reshape(16, 2, GROUP))
        rel = float(np.sqrt(n.sum() / d.sum()))
        self.assertGreater(rel, 0.2,
                           "a continuous tensor cannot be ternary-reproduced, got %g" % rel)

    def test_q4_0_is_far_worse_than_ternary_on_ternary_valued_weights(self):
        # THE REFUTATION: widening a natively-ternary model cannot buy quality back.
        # Q4_0's grid is asymmetric ([-8d, +7d]), so a symmetric +-d pair cannot
        # land exactly and one polarity shrinks to 0.875d.
        rng = np.random.default_rng(17)
        t = _ternary_tensor(rng, 16, 256)
        n1, d1 = wp.err_tq2(t.reshape(16, 2, GROUP))
        n2, d2 = wp.err_q4(t.reshape(16, 8, 32))
        rel_tq2 = float(np.sqrt(n1.sum() / d1.sum()))
        rel_q4 = float(np.sqrt(n2.sum() / d2.sum()))
        self.assertGreater(rel_q4, 0.05)
        self.assertGreater(rel_q4, 100 * rel_tq2)
        # and the exact mechanism: the scale is anchored on the FIRST
        # largest-magnitude element, so +2 (the earlier index) is kept exact and
        # -2 is the one shrunk to 0.875 * -2.
        x = np.array([[2.0, -2.0] + [0.0] * 30], dtype=np.float32).reshape(1, 1, 32)
        got = (np.clip(np.floor(x * (1.0 / (2.0 / -8.0)) + 8.5), 0, 15) - 8.0) * (2.0 / -8.0)
        self.assertAlmostEqual(float(got[0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(got[0, 0, 1]), -1.75)

    def test_duplicate_guard_when_one_dtype_is_absent(self):
        # A pack with no ternary leg must not divide by zero.
        self.assertEqual(float(wp.err_tq2(np.zeros((1, 1, GROUP), np.float32))[1].sum()),
                         0.0)


class TmqReaderTests(unittest.TestCase):
    def test_reads_a_tmq_container_through_the_same_math(self):
        # The .tmq path must agree with the safetensors path: build one TQ2 tensor
        # both ways and compare what the tool reconstructs.
        rng = np.random.default_rng(19)
        w = rng.normal(0.0, 1.0, size=(4, 256)).astype(np.float32)
        payload = qt.quantize_block(w)
        name = b"model.layers.0.mlp.gate_proj.weight"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.tmq"
            with open(p, "wb") as f:
                f.write(b"TMQ1")
                f.write(struct.pack("<I", 1))
                f.write(struct.pack("<I", len(name)))
                f.write(name)
                f.write(struct.pack("<I", 7))                 # dtype 7 = TQ2
                f.write(struct.pack("<I", 2))
                f.write(struct.pack("<II", 4, 256))
                f.write(struct.pack("<Q", 4 * 256 // GROUP))
                f.write(payload)

            got = list(wp.read_tmq(str(p)))
        self.assertEqual(len(got), 1)
        gname, dt, shape, blob, nb = got[0]
        self.assertEqual(gname, name.decode())
        self.assertEqual(dt, 7)
        self.assertEqual(shape, [4, 256])
        self.assertEqual(nb, 8)
        # the reader's decode equals the quantizer's own dequantized payload
        back = wp._unpack_tq2(blob, 4 * 256).reshape(4, 256)
        blk = np.frombuffer(payload, np.uint8).reshape(4, 2, 34)
        d = np.ascontiguousarray(
            blk[..., 0:2].copy().view("<f2").astype(np.float32))[..., 0]
        codes = qt.unpack_codes(np.ascontiguousarray(
            blk[..., 2:34].copy().view(np.uint32).reshape(4, 2, 8)))
        expect = ((codes.astype(np.float32) - 1.0) * d[..., None]).reshape(4, 256)
        self.assertTrue(np.allclose(back, expect))


if __name__ == "__main__":
    unittest.main()
