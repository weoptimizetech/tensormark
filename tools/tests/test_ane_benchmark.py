"""Input/rejection contracts for the built ANE benchmark, using local weights."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "tensormark/build/bench_ane_split"
MODEL = ROOT / "tensormark/data/tinyllama/tinyllama_q40.tmq"


@unittest.skipUnless(BINARY.exists() and MODEL.exists(), "build benchmark and provide TinyLlama TMQ")
class AneBenchmarkTests(unittest.TestCase):
    def run_rejection(self, message, tokens=None, enabled="1", args=()):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, TM_ANE=enabled, TM_ANE_PATH=str(Path(directory) / "missing.mlpackage"),
                       TM_DECODE_GPU="0", TM_PREFILL_GPU="0")
            env.pop("TM_BENCH_TOKENS", None)
            if tokens is not None:
                path = Path(directory) / "ids.txt"
                path.write_text(tokens)
                env["TM_BENCH_TOKENS"] = str(path)
            result = subprocess.run([str(BINARY), str(MODEL), "4", "1", *args],
                                    env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(message, result.stderr)
            self.assertNotIn("tok_per_s", result.stdout)

    def test_disabled_backend_is_not_a_benchmark(self):
        self.run_rejection("requires TM_ANE=1", enabled="0")

    def test_valid_tokens_missing_backend(self):
        self.run_rejection("not benchmarking fallback", "1 2 3 4\n")

    def test_invalid_token_files(self):
        for text in ("1 2 3", "1 2 nope 4", "1 2 -1 4", "1 2 32000 4"):
            with self.subTest(tokens=text):
                self.run_rejection("needs T valid token IDs", text)
        self.run_rejection("more than T token IDs", "1 2 3 4 5")

    def test_seed_and_decode_steps_validation(self):
        self.run_rejection("expected a positive integer", args=("not-a-seed",))
        self.run_rejection("exceeds context", args=("7", "2147483647"))


if __name__ == "__main__":
    unittest.main()
