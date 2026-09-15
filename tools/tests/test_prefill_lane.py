"""Real worker protocol/routing gates; requires the built binary and local TMQ."""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "tensormark/build/bench_prefill_lane"
MODEL = ROOT / "tensormark/data/tinyllama/tinyllama_q40.tmq"


@unittest.skipUnless(BINARY.is_file() and MODEL.is_file(), "build worker and provide TinyLlama TMQ")
class PrefillLaneTests(unittest.TestCase):
    def worker(self, lane="cpu", *, commands="run\nquit\n", args=None, env=None):
        environment = {k: v for k, v in os.environ.items() if not k.startswith("TM_")}
        environment.update(env or {})
        return subprocess.run(
            [str(BINARY), *(args if args is not None else [str(MODEL), "8", "7", lane])],
            input=commands, text=True, capture_output=True, env=environment, timeout=90,
        )

    def test_cpu_and_metal_protocol_and_repeatability(self):
        for lane in ("cpu", "metal"):
            with self.subTest(lane=lane):
                result = self.worker(lane)
                self.assertEqual(result.returncode, 0, result.stderr)
                messages = [json.loads(line) for line in result.stdout.splitlines()]
                self.assertEqual([m["event"] for m in messages], ["ready", "result"])
                for message in messages:
                    self.assertEqual(message["protocol"], "tensormark.prefill-lane/1")
                    self.assertEqual(message["lane"], lane)
                    self.assertEqual(message["tokens"], 8)
                    self.assertEqual(message["seed"], 7)
                    self.assertFalse(message["ane_executed"])
                    self.assertEqual(message["metal_executed"], lane == "metal")
                    self.assertGreater(message["wall_ms"], 0)
                    for metric in ("wall_ms", "cpu_maxdiff", "cpu_relative_l2", "replay_maxdiff"):
                        self.assertTrue(math.isfinite(message[metric]))
                        self.assertGreaterEqual(message[metric], 0)
                    self.assertLessEqual(message["replay_maxdiff"], 1e-4)
                    if lane == "cpu":
                        self.assertEqual(message["cpu_maxdiff"], 0)
                        self.assertTrue(message["cpu_top1_match"])

    def test_invalid_arguments(self):
        for args in ([], [str(MODEL), "0", "7", "cpu"], [str(MODEL), "8", "0", "cpu"],
                     [str(MODEL), "8", "7", "invalid"], [str(MODEL), "8x", "7", "cpu"],
                     [str(MODEL), "2147483648", "7", "cpu"]):
            with self.subTest(args=args):
                result = self.worker(args=args)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("prefill lane failed", result.stderr)

    def test_ane_refusal_is_not_a_successful_worker(self):
        for environment in ({}, {"TM_ANE_PATH": "/nonexistent/tensormark-test.mlmodelc"}):
            with self.subTest(environment=environment):
                result = self.worker("ane", env=environment)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("ANE", result.stderr)

    def test_metal_fallback_is_rejected(self):
        result = self.worker("metal", env={"TM_LLAMA_GPU_STACK": "0"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("refusing fallback", result.stderr)

    def test_invalid_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            tokens = Path(folder) / "tokens.txt"
            for text in ("1 2 3", "1 2 3 4 5 6 7 -1", "1 2 3 4 5 6 7 32000",
                         "1 2 3 4 5 6 7 nope", "1 2 3 4 5 6 7 8 9"):
                with self.subTest(text=text):
                    tokens.write_text(text)
                    result = self.worker(env={"TM_BENCH_TOKENS": str(tokens)})
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("TM_BENCH_TOKENS", result.stderr)

    def test_invalid_command_and_clean_eof(self):
        bad = self.worker(commands="unknown\n")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("expected run or quit", bad.stderr)
        self.assertEqual([json.loads(line)["event"] for line in bad.stdout.splitlines()], ["ready"])
        eof = self.worker(commands="")
        self.assertEqual(eof.returncode, 0, eof.stderr)
        self.assertEqual([json.loads(line)["event"] for line in eof.stdout.splitlines()], ["ready"])


if __name__ == "__main__":
    unittest.main()
