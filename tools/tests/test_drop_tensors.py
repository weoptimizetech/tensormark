"""drop_tensors.py: the tied-head repair, and explicit prefix drops.

The tool's security argument is "the payloads are identical, or you said
--force", so the tests exercise the outcomes: a head identical to the embedding
is dropped; a head that differs is refused (and dropped only under --force); a
prefix drop removes exactly the matching group; nothing matching writes nothing.
"""

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "drop_tensors", ROOT / "tools" / "drop_tensors.py")
dt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dt
SPEC.loader.exec_module(dt)

EMBED = "model.embed_tokens.weight"
HEAD = "lm_head.weight"


def pack(path, tensors):
    """tensors: [(name, payload)] as raw F32 (4, 2) — dtype 2 needs no blocks."""
    with open(path, "wb") as f:
        f.write(b"TMQ1")
        f.write(struct.pack("<I", len(tensors)))
        for name, payload in tensors:
            f.write(struct.pack("<I", len(name)))
            f.write(name.encode())
            f.write(struct.pack("<II", 2, 2))          # dtype F32, rank 2
            f.write(struct.pack("<II", 4, 2))          # dims
            f.write(struct.pack("<Q", 8))              # nblocks (unused for F32)
            f.write(payload)


def payload(seed):
    return struct.pack("<8f", *[seed + i for i in range(8)])


class DropTensors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def names(self, path):
        with open(path, "rb") as f:
            return [r.name for r in dt.read_index(f)]

    # ---- tied-head repair (no flags) ----

    def test_identical_head_is_dropped(self):
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [(EMBED, payload(1.0)), (HEAD, payload(1.0))])
        self.assertEqual(dt.main([str(src), str(dst)]), 0)
        self.assertEqual(self.names(dst), [EMBED])
        with open(dst, "rb") as f:                     # payload survives untouched
            rec = dt.read_index(f)[0]
            f.seek(rec.off)
            self.assertEqual(f.read(rec.size), payload(1.0))

    def test_different_head_is_refused(self):
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [(EMBED, payload(1.0)), (HEAD, payload(99.0))])
        self.assertEqual(dt.main([str(src), str(dst)]), 1)
        self.assertFalse(dst.exists(), "refusal must not leave a file behind")

    def test_force_drops_a_different_head(self):
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [(EMBED, payload(1.0)), (HEAD, payload(99.0))])
        self.assertEqual(dt.main(["--force", str(src), str(dst)]), 0)
        self.assertEqual(self.names(dst), [EMBED])

    def test_no_head_is_a_noop(self):
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [(EMBED, payload(1.0))])
        self.assertEqual(dt.main([str(src), str(dst)]), 0)
        self.assertFalse(dst.exists(), "nothing to do must not write a copy")

    def test_order_and_other_tensors_are_preserved(self):
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [("model.norm.weight", payload(7.0)),
                   (HEAD, payload(1.0)),
                   (EMBED, payload(1.0)),
                   ("model.layers.0.self_attn.q_proj.weight", payload(3.0))])
        self.assertEqual(dt.main([str(src), str(dst)]), 0)
        self.assertEqual(self.names(dst),
                         ["model.norm.weight", EMBED,
                          "model.layers.0.self_attn.q_proj.weight"])

    # ---- explicit prefix drops ----

    def test_prefix_drop_removes_exactly_the_group(self):
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [("model.layers.0.mlp.up_proj.weight", payload(1.0)),
                   ("mtp.eh_proj.weight", payload(2.0)),
                   ("mtp.mlp.up_proj.weight", payload(3.0)),
                   ("mtp.enorm.weight", payload(4.0))])
        self.assertEqual(dt.main(["--drop-prefix", "mtp.", str(src), str(dst)]), 0)
        self.assertEqual(self.names(dst), ["model.layers.0.mlp.up_proj.weight"])

    def test_prefix_drop_that_matches_nothing_writes_nothing(self):
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [("model.layers.0.mlp.up_proj.weight", payload(1.0))])
        self.assertEqual(dt.main(["--drop-prefix", "mtp.", str(src), str(dst)]), 0)
        self.assertFalse(dst.exists())

    def test_prefix_drop_still_guards_the_head(self):
        # A prefix that catches lm_head must not sneak past the identity check.
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [(EMBED, payload(1.0)), (HEAD, payload(99.0))])
        self.assertEqual(dt.main(["--drop-prefix", "lm_head", str(src), str(dst)]), 1)
        self.assertFalse(dst.exists())

    def test_multiple_prefixes_accumulate(self):
        src, dst = self.dir / "a.tmq", self.dir / "b.tmq"
        pack(src, [("mtp.a.weight", payload(1.0)),
                   ("model.layers.0.mlp.up_proj.weight", payload(2.0)),
                   ("mtp.b.weight", payload(3.0))])
        self.assertEqual(
            dt.main(["--drop-prefix", "mtp.", "--drop-prefix", "mlp.", str(src), str(dst)]), 0)
        self.assertEqual(self.names(dst), ["model.layers.0.mlp.up_proj.weight"])


if __name__ == "__main__":
    unittest.main()
