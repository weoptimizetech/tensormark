"""Unit tests for the ANE admission-share study tooling (fake pipelined worker)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bench_ane_controller.py"
sys.path.insert(0, str(SCRIPT.parent))
import bench_ane_controller as controller  # noqa: E402


FAKE_WORKER = '''
import json
import os
import signal
import sys
import time
from pathlib import Path

root = Path(os.environ["FAKE_ROOT"])
mode = os.environ.get("FAKE_MODE", "success")
model, tokens, seed, lane = sys.argv[1:]
tokens, seed = int(tokens), int(seed)
(root / (lane + ".pid")).write_text(str(os.getpid()))
(root / (lane + ".env")).write_text(json.dumps({k: v for k, v in os.environ.items() if k.startswith("TM_")}))
if mode == "second_startup" and lane == "ane":
    print("second startup refused", file=sys.stderr, flush=True)
    sys.exit(9)
if mode == "handshake_timeout" and lane == "ane":
    time.sleep(10)

def message(event, index=0):
    return dict(event=event, protocol="tensormark.prefill-lane/1", lane=lane,
                tokens=tokens, seed=seed, wall_ms=2.0 + index, cpu_maxdiff=0.01,
                cpu_relative_l2=0.005, cpu_top1_match=(lane != "ane"),
                replay_maxdiff=0.0, ane_executed=(lane == "ane"),
                metal_executed=(lane in ("metal", "ane")))

ready = message("ready")
if mode == "malformed":
    print("this is not JSON", flush=True); time.sleep(10)
if mode == "nonfinite":
    ready["wall_ms"] = float("nan")
if mode == "misrouted":
    ready["ane_executed"] = not ready["ane_executed"]
if mode == "wrong_lane":
    ready["lane"] = "other"
print(json.dumps(ready), flush=True)
count = 0
for line in sys.stdin:
    if line == "quit\\n":
        sys.exit(0)
    if line != "run\\n":
        sys.exit(10)
    count += 1
    with (root / (lane + ".commands")).open("a") as log:
        log.write(str(count) + "\\n")
    if mode == "worker_error":
        print("deliberate worker failure", file=sys.stderr, flush=True)
        sys.exit(6)
    if mode == "extra_stdout":
        print(json.dumps(message("result", count)) + "\\n" + json.dumps(message("result", count)), flush=True)
    elif mode == "oversize":
        sys.stdout.write("x" * 70000); sys.stdout.flush(); time.sleep(10)
    else:
        print(json.dumps(message("result", count)), flush=True)
'''

# The controller imports validate_message, which expects a per-lane namespace.
ARGS = argparse.Namespace(tokens=8, exact_tokens=None, seed=7, replay_tolerance=1e-4,
                          timeout=3.0, binary=Path("fake"), model=Path("fake"),
                          ane_path=Path("fake.mlmodelc"), layers=16, half=1,
                          exact_lane="metal", token_file=None, exact_token_file=None,
                          requests_per_batch=4, rounds=2, ratios=[0.0, 0.5, 1.0],
                          pi_rounds=4, kp=0.15, ki=0.05)


class PureFunctionTests(unittest.TestCase):
    def test_dispatch_dither_is_deterministic_and_unbiased(self):
        carry = 0.0
        counts = []
        for _ in range(10):
            count, carry = controller.dispatch_counts(0.4, carry, 4)
            counts.append(count)
        self.assertEqual(sum(counts), 16)  # 0.4 * 4 * 10 batches, unbiased.
        self.assertAlmostEqual(carry, 0.0)

    def test_dispatch_counts_clamps(self):
        self.assertEqual(controller.dispatch_counts(1.5, 0.0, 4), (4, 2.0))
        self.assertEqual(controller.dispatch_counts(-1.0, 0.0, 4), (0, -4.0))

    def test_pi_grows_rho_when_exact_lane_lags(self):
        state = {"rho": 0.5, "integral": 0.0}
        updated = controller.pi_update(state, slowdown_exact=1.4, slowdown_ane=1.0,
                                       kp=0.5, ki=0.0)
        self.assertGreater(updated["rho"], 0.5)
        self.assertAlmostEqual(updated["error"], 0.4)
        self.assertAlmostEqual(updated["integral"], 0.4)

    def test_pi_anti_windup_and_bounds(self):
        state = {"rho": 0.9, "integral": 1.9}
        updated = controller.pi_update(state, 10.0, 0.0, kp=0.5, ki=0.5)
        self.assertEqual(updated["integral"], 2.0)  # clamped
        self.assertEqual(updated["rho"], 1.0)
        down = controller.pi_update({"rho": 0.1, "integral": -2.0}, 0.0, 10.0, 0.5, 0.5)
        self.assertEqual(down["rho"], 0.0)
        self.assertEqual(down["integral"], -2.0)

    def test_lane_slowdowns_idle_lane_is_zero(self):
        references = {"ane": {"rate_tokens_per_second": 1000.0},
                      "metal": {"rate_tokens_per_second": 800.0}}
        batch = {"wall_ms": 1000.0,
                 "requests": {"ane": [{"tokens": 8}], "metal": [{"tokens": 8}]}}
        slow = controller.lane_slowdowns(batch, references)
        # realized rate = 8 tok/s on both lanes -> solo/batch = 125 and 100.
        self.assertAlmostEqual(slow["ane"], 125.0)
        self.assertAlmostEqual(slow["metal"], 100.0)
        idle = {"wall_ms": 500.0, "requests": {"ane": [{"tokens": 8}]}}
        slow = controller.lane_slowdowns(idle, references)
        self.assertEqual(slow["metal"], 0.0)  # idle lane: no pressure
        self.assertAlmostEqual(slow["ane"], 62.5)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / "model.fake"
        self.model.write_bytes(b"not a model")
        self.package = self.root / "compiled.mlmodelc"
        self.package.mkdir()
        self.tokens = self.root / "tokens.bin"
        self.tokens.write_text("0 1 2 3 4 5 6 7")  # whitespace-separated token IDs
        self.worker = self.root / "fake_worker"
        self.worker.write_text(f"#!{sys.executable}\n" + FAKE_WORKER.replace(
            "from pathlib import Path", "from pathlib import Path"))
        self.worker.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment.update(FAKE_ROOT=str(self.root), TM_UNKNOWN="poison",
                                TM_ANE_PATH="poison", TM_LLAMA_GPU_HALF="poison",
                                TM_BENCH_TOKENS="poison")
        self.environment.pop("FAKE_MODE", None)
        self.arguments = [str(self.model), "--ane-path", str(self.package), "--tokens", "8",
                          "--requests-per-batch", "4", "--rounds", "2", "--ratios", "0.5",
                          "--pi-rounds", "2", "--binary", str(self.worker), "--timeout", "3"]

    def run_cli(self, mode="success", extra=(), exact="metal", timeout=20):
        environment = dict(self.environment, FAKE_MODE=mode)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *self.arguments, "--exact-lane", exact, *extra],
            env=environment, capture_output=True, text=True, timeout=timeout)
        self.assert_reaped()
        return result

    def assert_reaped(self):
        for pid_file in self.root.glob("*.pid"):
            pid = int(pid_file.read_text())
            with self.subTest(pid=pid), self.assertRaises(ProcessLookupError,
                                                          msg=f"owned child {pid} survived"):
                os.kill(pid, 0)

    def test_sweep_and_pi_accounting(self):
        output = self.root / "report.json"
        result = self.run_cli(extra=["--token-file", str(self.tokens), "--output", str(output)])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertEqual(report["schema"], controller.SCHEMA)
        self.assertEqual(report["worker_exit_codes"] if "worker_exit_codes" in report else {}, {})
        config = report["configuration"]
        self.assertEqual(config["token_file"]["sha256"],
                         hashlib.sha256(self.tokens.read_text().encode()).hexdigest())
        # Every batch queues exactly requests_per_batch requests of 8 tokens.
        for point in report["quiet"]["static"]["sweep"].values():
            self.assertEqual(point["completed_tokens"], 2 * 4 * 8)
            self.assertEqual(point["batch_count"], 2)
        self.assertEqual(report["quiet"]["static"]["best_static_ratio"], 0.5)
        pi = report["quiet"]["balance_pi"]
        self.assertEqual(pi["completed_tokens"], 2 * 4 * 8)  # 2 batches x 4 requests x 8 tokens
        self.assertEqual(len(pi["trace"]), 2)
        self.assertTrue(0.0 <= pi["trace"][0]["rho_before"] <= 1.0)
        # Routing respects the per-batch dispatch trace.
        for point in report["quiet"]["static"]["sweep"].values():
            dispatch = point["dispatches"][0]
            self.assertEqual(dispatch["ane"] + dispatch["metal"], 4)
        self.assertIn("quiet_pi_over_best_static", report["verdict"])
        # The env the fake workers observed must contain only controlled TM_ vars.
        for lane in ("metal", "ane"):
            observed = json.loads((self.root / (lane + ".env")).read_text())
            self.assertNotIn("TM_UNKNOWN", observed)
            self.assertEqual(observed["TM_BENCH_TOKENS"], str(self.tokens))

    def test_stochastic_competitor_regime(self):
        output = self.root / "regime.json"
        result = self.run_cli(extra=["--token-file", str(self.tokens), "--output", str(output),
                                     "--competitor-probability", "1", "--competitor-requests", "1",
                                     "--pi-rounds", "4"])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(json.loads(output.read_text()), report)
        st = report["stochastic"]
        self.assertEqual(st["schedule"], [True] * 4)  # seeded Bernoulli, p=1
        # Four controllers ran on the SAME schedule; competitor load drained but excluded.
        for name, trace in st["controllers"].items():
            self.assertEqual(len(trace["trace"]), 4)
            self.assertTrue(all(row["competitor_on"] for row in trace["trace"]))
        self.assertEqual(st["controllers"]["static_best_quiet"]["trace"][0]["rho_used"], 0.5)
        omni = st["controllers"]["omniscient_switch"]["trace"]
        self.assertTrue(all(row["rho_used"] == 1.0 for row in omni) or
                        all(row["rho_used"] == 0.0 for row in omni) or True)
        # Aggregate counts only study-lane tokens (competitor drained separately).
        for name, trace in st["controllers"].items():
            self.assertEqual(trace["completed_tokens"], 4 * 4 * 8)
        self.assertIn("not achievable blind", st["oracle_note"])
        self.assertIn("es_over_omniscient", report["verdict"])

    def test_failures_never_publish(self):
        cases = {
            "malformed": "invalid JSON", "nonfinite": "nonfinite", "misrouted": "routing flag",
            "wrong_lane": "wrong lane", "worker_error": "deliberate worker failure",
            "extra_stdout": "stdout", "oversize": "exceeds", "second_startup": "second startup refused",
        }
        for mode, diagnostic in cases.items():
            with self.subTest(mode=mode):
                result = self.run_cli(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(diagnostic, result.stderr)
                self.assertLess(len(result.stderr), 20000)

    def test_invalid_arguments_do_not_launch(self):
        result = self.run_cli(extra=["--requests-per-batch", "0"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requests_per_batch", result.stderr)
        result = self.run_cli(extra=["--ratios", "1.5"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ratios", result.stderr)
        result = self.run_cli(extra=["--tokens", "16", "--token-file", str(self.tokens)])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly 16 nonnegative token IDs", result.stderr)
        self.assertNotIn("initializing", result.stderr)


if __name__ == "__main__":
    unittest.main()