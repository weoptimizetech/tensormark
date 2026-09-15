"""Small graph/cache regression tests; no real CoreML conversion or prediction."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location(
    "anepack_under_test", Path(__file__).resolve().parents[1] / "anepack.py"
)
anepack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anepack)


class AnepackTests(unittest.TestCase):
    D, H, KVH, DH, S, LAYERS = 32, 4, 2, 8, 4, 2

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tmq = self.root / "tiny.tmq"
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "hidden_size": self.D,
            "num_attention_heads": self.H,
            "num_key_value_heads": self.KVH,
            "head_dim": self.DH,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        }))
        self.write_tmq()

    def write_tmq(self):
        """Real Q8_0 blocks, deterministic small projections and unit norms."""
        rng = np.random.default_rng(7)
        shapes = {
            "self_attn.q_proj": (self.D, self.D),
            "self_attn.k_proj": (self.KVH * self.DH, self.D),
            "self_attn.v_proj": (self.KVH * self.DH, self.D),
            "self_attn.o_proj": (self.D, self.D),
            "mlp.gate_proj": (2 * self.D, self.D),
            "mlp.up_proj": (2 * self.D, self.D),
            "mlp.down_proj": (self.D, 2 * self.D),
            "input_layernorm": (self.D,),
            "post_attention_layernorm": (self.D,),
        }
        with self.tmq.open("wb") as f:
            f.write(b"TMQ1" + struct.pack("<I", self.LAYERS * len(shapes)))
            for layer in range(self.LAYERS):
                for suffix, shape in shapes.items():
                    name = f"model.layers.{layer}.{suffix}.weight".encode()
                    count = int(np.prod(shape))
                    if len(shape) == 1:
                        q = np.full(count, 8, dtype=np.int8)
                        scale = 0.125
                    else:
                        q = rng.integers(-4, 5, count, dtype=np.int8)
                        scale = 0.015625
                    blocks = q.reshape(-1, 32)
                    f.write(struct.pack("<I", len(name)) + name)
                    f.write(struct.pack("<II", 0, len(shape)))
                    f.write(struct.pack("<" + "I" * len(shape), *shape))
                    f.write(struct.pack("<Q", len(blocks)))
                    for block in blocks:
                        f.write(struct.pack("<e", scale) + block.tobytes())

    def trace(self, emit_kv):
        package = mock.Mock()
        ct = types.ModuleType("coremltools")
        ct.TensorType = lambda **kwargs: types.SimpleNamespace(**kwargs)
        ct.precision = types.SimpleNamespace(FLOAT16="fp16")
        ct.target = types.SimpleNamespace(iOS16="ios16")
        ct.convert = mock.Mock(return_value=package)
        out = str(self.root / "test.mlpackage")
        with mock.patch.dict(sys.modules, {"coremltools": ct}):
            with contextlib.redirect_stdout(io.StringIO()):
                anepack.build(str(self.tmq), self.LAYERS, self.S, out, emit_kv)
        ct.convert.assert_called_once()
        package.save.assert_called_once_with(out)
        traced = ct.convert.call_args.args[0]
        options = ct.convert.call_args.kwargs
        self.assertEqual(options["inputs"][0].name, "x")
        self.assertEqual(options["inputs"][0].shape, (self.S, self.D))
        self.assertIs(options["inputs"][0].dtype, np.float16)
        self.assertEqual(options["compute_precision"], "fp16")
        return traced, options

    def test_fp16_finite_additive_causal_bias(self):
        traced, _ = self.trace(False)
        graph = traced.inlined_graph
        self.assertNotIn("aten::masked_fill", str(graph))
        softmaxes = [node for node in graph.nodes() if node.kind() == "aten::softmax"]
        self.assertEqual(len(softmaxes), self.LAYERS)
        expected = torch.triu(torch.full((self.S, self.S), -30000.0,
                                         dtype=torch.float16), diagonal=1)
        for node in softmaxes:
            scores = list(node.inputs())[0]
            self.assertEqual(scores.type().scalarType(), "Half")
            self.assertEqual(node.output().type().scalarType(), "Half")
            add = scores.node()
            self.assertEqual(add.kind(), "aten::add")
            score_input, bias_input = list(add.inputs())[:2]
            self.assertEqual(score_input.type().scalarType(), "Half")
            self.assertEqual(score_input.node().kind(), "aten::div")
            bias = bias_input.toIValue()
            self.assertIsInstance(bias, torch.Tensor)
            self.assertEqual(bias.dtype, torch.float16)
            self.assertTrue(torch.isfinite(bias).all().item())
            torch.testing.assert_close(bias, expected, rtol=0, atol=0)
            probabilities = torch.softmax(bias, -1)
            self.assertTrue(torch.isfinite(probabilities).all().item())
            self.assertEqual(torch.count_nonzero(torch.triu(probabilities, 1)).item(), 0)
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(self.S).half())

    def test_hidden_and_named_kv_outputs_are_causal_and_agree(self):
        hidden_graph, hidden_options = self.trace(False)
        kv_graph, kv_options = self.trace(True)
        self.assertNotIn("outputs", hidden_options)
        self.assertEqual([output.name for output in kv_options["outputs"]],
                         ["h", "k_0", "v_0", "k_1", "v_1"])
        x = torch.randn(self.S, self.D, generator=torch.Generator().manual_seed(9)).half()
        changed = x.clone()
        changed[2:] = -3 * changed[2:] + 1
        with torch.no_grad():
            hidden = hidden_graph(x)
            outputs = kv_graph(x)
            changed_hidden = hidden_graph(changed)
            changed_outputs = kv_graph(changed)
        self.assertIsInstance(hidden, torch.Tensor)
        self.assertEqual(tuple(hidden.shape), (self.S, self.D))
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 1 + 2 * self.LAYERS)
        torch.testing.assert_close(hidden, outputs[0], rtol=0, atol=0)
        torch.testing.assert_close(hidden[:2], changed_hidden[:2], rtol=0, atol=0)
        self.assertFalse(torch.equal(hidden[2:], changed_hidden[2:]))
        for index, (output, changed_output) in enumerate(zip(outputs, changed_outputs)):
            self.assertEqual(output.dtype, torch.float16)
            self.assertTrue(torch.isfinite(output).all().item())
            if index:
                self.assertEqual(tuple(output.shape), (self.KVH, self.S, self.DH))
                torch.testing.assert_close(output[:, :2], changed_output[:, :2],
                                           rtol=0, atol=0)

    def key(self, **kwargs):
        options = dict(nlayers=self.LAYERS, tokens=self.S, emit_kv=True)
        options.update(kwargs)
        return anepack.cache_key(str(self.tmq), **options)

    def test_cache_identity_includes_source_config_and_build_options(self):
        baseline = self.key()
        self.assertEqual(self.key(), baseline)
        for options in ({"nlayers": 1}, {"tokens": 8}, {"emit_kv": False}):
            with self.subTest(options=options):
                self.assertNotEqual(self.key(**options), baseline)
        original_config = self.config.read_bytes()
        stat = self.config.stat()
        self.config.write_bytes(original_config.replace(b"10000.0", b"20000.0"))
        os.utime(self.config, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(self.key(), baseline)
        self.config.write_bytes(original_config)
        self.assertEqual(self.key(), baseline)
        self.config.unlink()
        absent_config = self.key()
        self.assertNotEqual(absent_config, baseline)
        self.config.write_bytes(b"")
        self.assertNotEqual(self.key(), absent_config)
        self.config.write_bytes(original_config)
        source = self.root / "builder.py"
        source.write_bytes(Path(anepack.__file__).read_bytes())
        with mock.patch.object(anepack, "__file__", str(source)):
            self.assertEqual(self.key(), baseline)
            source.write_bytes(source.read_bytes() + b"\n# graph fix\n")
            self.assertNotEqual(self.key(), baseline)
        stat = self.tmq.stat()
        os.utime(self.tmq, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        self.assertNotEqual(self.key(), baseline)

    def run_main(self, *args):
        with mock.patch.object(sys, "argv", ["anepack", str(self.tmq), *args]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                anepack.main()
        return output.getvalue().strip()

    def fake_build(self, tmq, layers, tokens, out_pkg, emit_kv):
        Path(out_pkg).mkdir(parents=True, exist_ok=True)

    def test_cache_hit_requires_package_and_force_rebuilds(self):
        args = ("--cache", str(self.root / "cache"), "--layers", "2", "--tokens", "4")
        with mock.patch.object(anepack, "build", side_effect=self.fake_build) as build:
            package = self.run_main(*args)
            self.assertTrue(Path(package).is_dir())
            self.assertEqual(build.call_count, 1)
            self.assertEqual(self.run_main(*args), package)
            self.assertEqual(build.call_count, 1)
            Path(package).rmdir()
            self.assertEqual(self.run_main(*args), package)
            self.assertEqual(build.call_count, 2)
            self.run_main(*args, "--force")
            self.assertEqual(build.call_count, 3)

    def test_explicit_output_checks_cache_identity(self):
        package = self.root / "explicit.mlpackage"
        args = ("--out", str(package), "--layers", "2", "--tokens", "4")
        with mock.patch.object(anepack, "build", side_effect=self.fake_build) as build:
            self.run_main(*args)
            self.run_main(*args)
            self.assertEqual(build.call_count, 1)
            config = json.loads(self.config.read_text())
            config["rope_theta"] = 20000.0
            self.config.write_text(json.dumps(config))
            self.run_main(*args)
            self.assertEqual(build.call_count, 2)
            self.run_main(*args, "--no-kv")
            self.assertEqual(build.call_count, 3)
            self.assertFalse(build.call_args.kwargs["emit_kv"])


if __name__ == "__main__":
    unittest.main()
