"""Unit tests for tools/quantize_ternary.py — the dense-fp16 -> TQ2 quantizer.

These do not need a checkpoint: they pin the parts that fail QUIETLY in the engine.
A wrong bit order, a wrong grouping or an off-by-one in the `code - 1` offset all
still produce a loadable model that simply answers worse — the `--selfcheck` mode
catches that against a real pack, and these catch it without one.
"""

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "quantize_ternary_under_test",
    Path(__file__).resolve().parents[1] / "quantize_ternary.py",
)
qt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qt)


def _q4_golden_vector():
    """The exact vector the engine's C++ reference is fed."""
    return np.array([np.float32(((i * 37) % 23) - 11) * np.float32(0.37)
                     for i in range(32)], dtype=np.float32).reshape(1, 32)


def _write_safetensors(path: Path, tensors: dict) -> None:
    """tensors: {name: (dtype, shape, payload bytes)} — HF safetensors layout."""
    head, off = {}, 0
    for name, (dtype, shape, blob) in tensors.items():
        head[name] = {"dtype": dtype, "shape": list(shape),
                      "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    hjson = json.dumps(head).encode()
    hjson += b" " * ((8 - len(hjson) % 8) % 8)     # safetensors pads to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for _n, (_d, _s, blob) in tensors.items():
            f.write(blob)


class TernaryCodeTests(unittest.TestCase):
    def test_code_roundtrip_is_exact(self):
        rng = np.random.default_rng(7)
        codes = rng.integers(0, 3, size=(3, 2, 128)).astype(np.uint32)
        self.assertTrue(np.array_equal(qt.unpack_codes(qt.pack_codes(codes)), codes))

    def test_pack_uses_little_endian_two_bit_fields(self):
        # Value j must live in bits 2j..2j+1 of word j//16 — the engine reads the
        # block that way, so a big-endian or 16-value-per-word packing is a silent
        # quality regression, not a crash.
        codes = np.zeros((1, 1, 128), dtype=np.uint32)
        codes[0, 0, 0] = 2      # first value -> bits 0..1 of word 0
        codes[0, 0, 16] = 1     # 17th value  -> bits 0..1 of word 1
        words = qt.pack_codes(codes)
        self.assertEqual(words[0, 0, 0], 2)
        self.assertEqual(words[0, 0, 1], 1)
        self.assertEqual(words[0, 0, 2], 0)

    def test_quantize_block_reproduces_the_absmax_rule(self):
        # Values that are exact multiples of each group's absmax, so the rule's
        # output is determined. 1.5 rather than 1.0: at d = 2 that is 0.75, which
        # rounds unambiguously (1.0 would land on .5, where numpy's rint is
        # ties-to-even — a tie-break the pack's own weights never exercise).
        w = np.array([[0.0, 2.0, -2.0, 1.5] + [0.0] * 124], dtype=np.float32)
        payload = qt.quantize_block(w)
        self.assertEqual(len(payload), qt.BLOCK_BYTES)
        blk = np.frombuffer(payload, np.uint8).reshape(1, 1, qt.BLOCK_BYTES)
        scale = blk[..., 0:2].copy().view("<f2").astype(np.float32).ravel()[0]
        self.assertEqual(scale, 2.0)
        codes = qt.unpack_codes(np.ascontiguousarray(
            blk[..., 2:qt.BLOCK_BYTES].copy().view(np.uint32).reshape(1, 1, 8)))
        self.assertEqual(list(codes.ravel()[:4]), [1, 2, 0, 2])

    def test_quantize_block_handles_an_all_zero_group(self):
        payload = qt.quantize_block(np.zeros((1, 128), dtype=np.float32))
        blk = np.frombuffer(payload, np.uint8).reshape(1, 1, qt.BLOCK_BYTES)
        self.assertEqual(blk[..., 0:2].copy().view("<f2")[0, 0], 0.0)
        self.assertEqual(len(payload), qt.BLOCK_BYTES)

    def test_quantize_block_rejects_a_partial_group(self):
        with self.assertRaises(SystemExit):
            qt.quantize_block(np.zeros((1, 100), dtype=np.float32))

    def test_quantize_block_shape_and_error_bound(self):
        rng = np.random.default_rng(11)
        w = rng.normal(0.0, 1.0, size=(2, 256)).astype(np.float32)
        payload = qt.quantize_block(w)
        self.assertEqual(len(payload), 2 * 2 * qt.BLOCK_BYTES)   # 2 rows x 2 groups
        blk = np.frombuffer(payload, np.uint8).reshape(2, 2, qt.BLOCK_BYTES)
        words = np.ascontiguousarray(blk[..., 2:qt.BLOCK_BYTES].copy()
                                     .view(np.uint32).reshape(2, 2, 8))
        d = np.ascontiguousarray(blk[..., 0:2].copy().view("<f2")
                                 .astype(np.float32).reshape(2, 2))
        got = (qt.unpack_codes(words).astype(np.float32) - 1.0) * d[..., None]
        # Every reconstructed value is within half a step of the original.
        self.assertLessEqual(
            np.abs(got - w.reshape(2, 2, 128)).max(), 0.5 * d.max() + 1e-5)


class Q4Tests(unittest.TestCase):
    """The mixed-precision rule, pinned against the engine's own C++."""

    def test_q4_matches_the_engine_reference_exactly(self):
        # A known-answer test against the C++ quantizer, not against this file's
        # own round-trip: scale and all 16 nibble bytes must be the ones
        # tensormark/quant.h's quantize_row_q4_0 produced for this vector.
        blk = np.frombuffer(qt.quantize_block_q4_0(_q4_golden_vector()),
                            np.uint8).reshape(1, qt.Q4_BLOCK_BYTES)
        self.assertEqual("%04x" % blk[0, 0:2].copy().view("<u2")[0], "3812")
        self.assertEqual(bytes(blk[0, 2:qt.Q4_BLOCK_BYTES]).hex(),
                         "a0e4174b8fc1f5296c9fd3074a7eb1f4")

    def test_q4_layout_and_error_bound(self):
        rng = np.random.default_rng(13)
        w = rng.normal(0.0, 1.0, size=(2, 64)).astype(np.float32)
        payload = qt.quantize_block_q4_0(w)
        self.assertEqual(len(payload), 2 * 2 * qt.Q4_BLOCK_BYTES)
        blk = np.frombuffer(payload, np.uint8).reshape(2, 2, qt.Q4_BLOCK_BYTES)
        d = np.ascontiguousarray(blk[..., 0:2].copy().view("<f2")
                                 .astype(np.float32).reshape(2, 2))
        qs = blk[..., 2:qt.Q4_BLOCK_BYTES]
        lo = (qs & 0x0F).astype(np.float32)                # element 2j
        hi = ((qs >> 4) & 0x0F).astype(np.float32)         # element 2j+1
        got = np.stack([lo, hi], axis=-1).reshape(2, 2, 32)
        got = (got - 8.0) * d[..., None]
        # Q4_0's range is [-8d, +7d], so the step is d: reconstruct within it.
        self.assertLessEqual(
            np.abs(got - w.reshape(2, 2, 32)).max(), d.max() + 1e-5)

    def test_q4_handles_an_all_zero_group(self):
        blk = np.frombuffer(
            qt.quantize_block_q4_0(np.zeros((1, 32), dtype=np.float32)),
            np.uint8).reshape(1, qt.Q4_BLOCK_BYTES)
        self.assertEqual(blk[0, 0:2].copy().view("<f2")[0], 0.0)
        # Both nibbles are 8, i.e. (8-8)*d == 0 whatever d is — the byte is 0x88,
        # not 0x08: element 2j is the LOW nibble and 2j+1 the high one.
        self.assertTrue((blk[0, 2:qt.Q4_BLOCK_BYTES] == 0x88).all())


class DriverTests(unittest.TestCase):
    """main(): streaming, and a working-set bound that never changes the bytes."""

    @unittest.skipUnless(hasattr(qt, "CHUNK_VALUES"), "chunking not present")
    def test_row_chunking_does_not_change_a_single_byte(self):
        rng = np.random.default_rng(29)
        w = rng.normal(0.0, 1.0, size=(37, 256)).astype(np.float32)
        for name, group, block in (("quantize_block", qt.GROUP, qt.BLOCK_BYTES),
                                   ("quantize_block_q4_0", qt.Q4_GROUP,
                                    qt.Q4_BLOCK_BYTES)):
            with self.subTest(rule=name):
                with mock.patch.object(qt, "CHUNK_VALUES", 1 << 30):
                    one = getattr(qt, name)(w)
                with mock.patch.object(qt, "CHUNK_VALUES", 256):   # one row each
                    many = getattr(qt, name)(w)
                    # the bound really engaged, checked INSIDE the patch — outside
                    # it CHUNK_VALUES is back to the shipped value and this would
                    # assert the arithmetic of the default instead.
                    self.assertEqual(qt._chunk_rows(256), 1)
                self.assertEqual(one, many)
                self.assertEqual(len(one), 37 * (256 // group) * block)
        # and the shipped default still chunks in whole rows at this width
        self.assertGreaterEqual(qt._chunk_rows(256), 1)

    def test_main_streams_the_source_instead_of_materializing_it(self):
        # REGRESSION, and a whole-model hang: main() used to build
        # `list(read_tensors(src))` BEFORE opening the destination, which holds the
        # entire checkpoint in RAM — 8.06 GB for the fp16 Bonsai on a 7 GB host, so
        # it swapped forever and never even created the output file. Detect the
        # shape of that bug: the destination must already be open by the time the
        # first tensor is yielded.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "src"
            src.mkdir()
            tensors = {}
            for i in range(2):
                w = (np.arange(256, dtype=np.float32) % 3 - 1).reshape(1, 256)
                tensors["model.layers.%d.mlp.gate_proj.weight" % i] = (
                    "F16", (1, 256), w.astype("<f2").tobytes())
            _write_safetensors(src / "model.safetensors", tensors)

            dst = tmp / "out.tmq"
            seen = []
            original = qt.read_tensors

            def spy(s):
                for t in original(s):
                    seen.append(dst.exists())
                    yield t

            with mock.patch.object(qt, "read_tensors", spy), \
                    mock.patch.object(sys, "argv",
                                      ["quantize_ternary.py", str(src), str(dst)]):
                self.assertEqual(qt.main(), 0)
            self.assertTrue(seen, "the source produced no tensors")
            self.assertTrue(all(seen),
                            "destination not open when tensors were read — it materializes")
            self.assertGreater(dst.stat().st_size, 12)


if __name__ == "__main__":
    unittest.main()
