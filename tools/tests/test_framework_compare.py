import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_framework_compare as fwc


class GenerateIdsTest(unittest.TestCase):
    def test_deterministic_shared_workload(self):
        a = fwc.generate_ids(64, 7)
        b = fwc.generate_ids(64, 7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        self.assertTrue(all(0 <= i < 32000 for i in a))
        self.assertNotEqual(a, fwc.generate_ids(64, 8))

    def test_write_read_roundtrip(self):
        import tempfile
        p = Path(tempfile.mkstemp(suffix=".txt")[1])
        ids = fwc.generate_ids(10, 3)
        fwc.write_ids(p, ids)
        self.assertEqual([int(x) for x in p.read_text().split()], ids)
        p.unlink()


class TableTest(unittest.TestCase):
    def test_table_uses_measurement_values_only(self):
        m = {"times_seconds": [0.5, 0.6], "tokens_per_run": 1024,
             "tokens_per_second": [1706.6, 1420.4]}
        report = {"task": "prefill", "pytorch_device": "mps",
                  "pytorch_dtype": "float16", "measurements": {"pytorch": m}}
        rows = fwc.table_rows(report)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["device"], "mps (float16)")
        self.assertIn("0.550", rows[0]["mean"])  # mean of 0.6, 0.5
        self.assertIn("tok/s", rows[0]["rate"])
        # mean time derived from times_seconds, not tokens_per_second list
        self.assertEqual(rows[0]["mean"], "0.550")

    def test_na_reason_renders_single_row(self):
        report = {"task": "train", "pytorch_device": "mps",
                  "measurements": {}, "na_reason": "needs 17.6 GiB"}
        rows = fwc.table_rows(report)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["framework"], "tensormark")
        self.assertIn("17.6", rows[0]["spread"])

    def test_table_format(self):
        rows = [{"task": "prefill", "framework": "tensormark-metal",
                 "device": "Metal+CPU", "mean": "1.100", "spread": "1.0–1.2 (1024 tok/run)",
                 "rate": "931.0 tok/s"}]
        table = fwc.build_table(rows)
        self.assertTrue(table.startswith("| Task | Framework | Device |"))
        self.assertEqual(len(table.strip().splitlines()), 3)


class ArgTest(unittest.TestCase):
    def test_defaults_and_validation(self):
        args = fwc.parse_args(["prefill"])
        self.assertEqual(args.tokens, 1024)
        self.assertEqual(args.hybrid, 0.25)
        with self.assertRaises(SystemExit):
            fwc.parse_args(["prefill", "--rounds", "0"])
        with self.assertRaises(SystemExit):
            fwc.parse_args(["prefill", "--timeout", "-1"])

    def test_optimizer_state_gb(self):
        class Fake:
            def parameters(self):
                class P:
                    def __init__(self, n):
                        self.n = n
                    def numel(self):
                        return self.n
                return self
        model = type("M", (), {})()
        model.parameters = lambda: iter([type("P", (), {"numel": lambda s: 10**9})()])
        # 1e9 params * 16 bytes / 2**30
        self.assertAlmostEqual(fwc.optimizer_state_gb(model), 1e9 * 16 / 2**30, places=3)


if __name__ == "__main__":
    unittest.main()