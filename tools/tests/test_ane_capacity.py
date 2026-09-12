"""Portable capacity-controller tests: tiny fake workers, never model execution."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "bench_ane_capacity.py"
SPEC = importlib.util.spec_from_file_location("bench_ane_capacity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capacity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capacity)

# Startup writes PID/environment evidence before emitting ready. Modes exercise
# pipe backpressure, split lines, idle-peer failure, replay, and exit handling.
FAKE_WORKER = r'''
import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(os.environ["FAKE_ROOT"])
mode = os.environ.get("FAKE_MODE", "success")
model, tokens, seed, lane = sys.argv[1:]
tokens, seed = int(tokens), int(seed)
(root / (lane + ".pid")).write_text(str(os.getpid()))
(root / (lane + ".env")).write_text(json.dumps({k: v for k, v in os.environ.items() if k.startswith("TM_")}))
if mode == "second_startup" and lane == "ane":
    print("second startup refused", file=sys.stderr, flush=True)
    sys.exit(9)
if mode == "ignore_terminate" and lane != "ane":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode == "ignore_terminate" and lane == "ane":
    sys.exit(9)
if mode == "handshake_timeout" and lane == "ane":
    time.sleep(10)
if mode in ("idle_stdout", "idle_exit") and lane == "ane":
    time.sleep(0.2)
if mode == "stderr_flood":
    sys.stderr.write("x" * 200000 + "\nTAIL-MARKER\n")
    sys.stderr.flush()
if mode == "oversize":
    sys.stdout.write("x" * 70000)
    sys.stdout.flush()
    time.sleep(10)


def message(event):
    return dict(event=event, protocol="tensormark.prefill-lane/1", lane=lane,
                tokens=tokens, seed=seed, wall_ms=2.0, cpu_maxdiff=0.01,
                cpu_relative_l2=0.005, cpu_top1_match=(lane != "ane"),
                replay_maxdiff=0.0, ane_executed=(lane == "ane"),
                metal_executed=(lane in ("metal", "ane")))


def emit(value):
    text = json.dumps(value) + "\n"
    if mode == "split_line":
        os.write(sys.stdout.fileno(), text[:17].encode())
        time.sleep(0.002)
        os.write(sys.stdout.fileno(), text[17:].encode())
    else:
        print(text, end="", flush=True)


ready = message("ready")
if mode == "malformed":
    print("this is not JSON", flush=True)
    time.sleep(10)
if mode == "nonfinite":
    ready["wall_ms"] = float("nan")
if mode == "misrouted":
    ready["ane_executed"] = not ready["ane_executed"]
if mode == "wrong_lane":
    ready["lane"] = "other"
emit(ready)
if mode in ("idle_stdout", "idle_exit") and lane != "ane":
    time.sleep(0.05)
    if mode == "idle_exit":
        sys.exit(0)
    emit(message("result"))
count = 0
for line in sys.stdin:
    if line == "quit\n":
        (root / (lane + ".quit")).write_text("graceful")
        if mode == "shutdown_failure" and lane == "ane":
            print("failed during shutdown", file=sys.stderr, flush=True)
            sys.exit(7)
        if mode == "shutdown_stdout":
            print("extra output", flush=True)
        if mode == "shutdown_timeout":
            time.sleep(10)
        sys.exit(0)
    if line != "run\n":
        sys.exit(10)
    count += 1
    with (root / (lane + ".commands")).open("a") as log:
        log.write(str(count) + "\n")
    if mode == "worker_error":
        print("deliberate worker failure", file=sys.stderr, flush=True)
        sys.exit(6)
    if mode == "eof":
        sys.exit(0)
    if mode == "timeout":
        time.sleep(10)
    if mode == "stderr_flood":
        sys.exit(8)
    # Two rounds before + two paired + two after. At paired requests the
    # barrier ensures the controller cannot await one result before sending
    # the peer's command. Delay ANE so arrival-order timing is testable.
    if count in (3, 4):
        (root / (lane + ".barrier." + str(count))).touch()
        peer = "ane" if lane != "ane" else os.environ.get("FAKE_EXACT", "metal")
        deadline = time.monotonic() + 2
        while not (root / (peer + ".barrier." + str(count))).exists():
            if time.monotonic() > deadline:
                print("paired dispatch barrier timed out", file=sys.stderr, flush=True)
                sys.exit(11)
            time.sleep(0.001)
        if lane == "ane":
            time.sleep(0.06)
    else:
        time.sleep(0.002)
    result = message("result")
    if mode == "replay":
        result["replay_maxdiff"] = 0.2
    if mode == "result_nonfinite":
        result["cpu_relative_l2"] = float("inf")
    if mode == "wrong_event":
        result["event"] = "ready"
    if mode == "extra_stdout":
        print(json.dumps(result) + "\n" + json.dumps(result), flush=True)
    else:
        emit(result)
'''


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(tokens=8, exact_tokens=None, seed=7, replay_tolerance=1e-4)
        self.good = {
            "event": "result", "protocol": capacity.PROTOCOL, "lane": "ane", "tokens": 8,
            "seed": 7, "wall_ms": 1.0, "cpu_maxdiff": 0, "cpu_relative_l2": 0,
            "cpu_top1_match": True, "replay_maxdiff": 1e-5,
            "ane_executed": True, "metal_executed": True,
        }

    def validate(self, message):
        return capacity.validate_message(json.dumps(message).encode(), "ane", "result", self.args)

    def test_valid_and_observational_quality(self):
        message = dict(self.good, cpu_maxdiff=1e10, cpu_relative_l2=1e5, cpu_top1_match=False)
        self.assertEqual(self.validate(message), message)
        self.validate(dict(self.good, replay_maxdiff=self.args.replay_tolerance))

    def test_invalid_fields(self):
        changes = {
            "event": ["ready", None], "protocol": ["wrong", None], "lane": ["cpu", None],
            "tokens": [True, 8.0, 9, None], "seed": [True, 7.0, 8, None],
            "wall_ms": [True, False, 0, -1, float("nan"), float("inf"), "1", None, 10**400],
            "cpu_maxdiff": [True, -1, float("nan"), float("inf"), None],
            "cpu_relative_l2": [False, -1, float("inf"), None],
            "replay_maxdiff": [True, -1, float("nan"), 0.001, None],
            "cpu_top1_match": [0, 1, "true", None],
            "ane_executed": [False, 1, None], "metal_executed": [False, 1, None],
        }
        for field, values in changes.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(capacity.BenchmarkError):
                    self.validate(dict(self.good, **{field: value}))
            with self.subTest(missing=field), self.assertRaises(capacity.BenchmarkError):
                message = self.good.copy()
                del message[field]
                self.validate(message)

    def test_strict_json(self):
        values = [b"[]", b"null", b"garbage", b"\xff", b'{"x":1,"x":2}', b"{} {}"]
        for value in values:
            with self.subTest(value=value), self.assertRaises(capacity.BenchmarkError):
                capacity.validate_message(value, "ane", "result", self.args)

    def test_ready_replay_must_be_zero(self):
        with self.assertRaisesRegex(capacity.BenchmarkError, "must be zero"):
            capacity.validate_message(json.dumps(dict(self.good, event="ready")).encode(), "ane", "ready", self.args)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / "model.fake"
        self.model.write_bytes(b"not a model")
        self.package = self.root / "compiled.mlmodelc"
        self.package.mkdir()
        self.tokens = self.root / "tokens.bin"
        self.tokens.write_bytes(b"\x01\x00\x00\x00" * 8)
        self.worker = self.root / "fake_worker"
        self.worker.write_text(f"#!{sys.executable}\n" + FAKE_WORKER)
        self.worker.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment.update(FAKE_ROOT=str(self.root), TM_UNKNOWN="poison", TM_GEMV_THREADS="99",
                                TM_ANE_PATH="poison", TM_LLAMA_GPU_HALF="poison", TM_BENCH_TOKENS="poison")
        self.environment.pop("FAKE_MODE", None)
        self.environment.pop("FAKE_EXACT", None)
        self.arguments = [str(self.model), "--ane-path", str(self.package), "--tokens", "8", "--rounds", "2",
                          "--binary", str(self.worker), "--timeout", "3"]

    def run_cli(self, mode="success", extra=(), timeout=12, exact="metal"):
        environment = dict(self.environment, FAKE_MODE=mode, FAKE_EXACT=exact)
        result = subprocess.run([sys.executable, str(SCRIPT), *self.arguments, "--exact-lane", exact, *extra],
                                env=environment, capture_output=True, text=True, timeout=timeout)
        self.assert_reaped()
        return result

    def assert_reaped(self):
        for pid_file in self.root.glob("*.pid"):
            pid = int(pid_file.read_text())
            with self.subTest(pid=pid), self.assertRaises(ProcessLookupError, msg=f"owned child {pid} survived"):
                os.kill(pid, 0)

    def test_success_accounting_dispatch_arrival_and_environment(self):
        output = self.root / "report.json"
        result = self.run_cli(extra=["--token-file", str(self.tokens), "--output", str(output)])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertEqual(report["schema"], capacity.SCHEMA)
        self.assertEqual(report["worker_exit_codes"], {"metal": 0, "ane": 0})
        self.assertIn("Co-resident idle-peer", report["measurement_scope"]["baseline"])
        self.assertIn("not steady-state", report["measurement_scope"]["capacity"])
        self.assertIn("baseline_after round 2/2", result.stderr)
        config = report["configuration"]
        self.assertIsInstance(config["model"]["mtime_ns"], int)
        self.assertNotIn("sha256", config["model"])
        expected_env = {"TM_ANE_PATH": str(self.package), "TM_ANE_LAYERS": "16", "TM_LLAMA_GPU_HALF": "1",
                        "TM_BENCH_TOKENS": str(self.tokens)}
        self.assertEqual(config["controlled_environment"], {"metal": expected_env, "ane": expected_env})
        for lane in ("metal", "ane"):
            self.assertEqual(json.loads((self.root / (lane + ".env")).read_text()), expected_env)
            self.assertTrue((self.root / (lane + ".quit")).exists())
            self.assertEqual(len((self.root / (lane + ".commands")).read_text().splitlines()), 6)
            self.assertEqual(config["token_files"][lane]["sha256"],
                             hashlib.sha256(self.tokens.read_bytes()).hexdigest())
            self.assertEqual(config["token_files"][lane]["size_bytes"], 32)
        observations = report["observations"]
        for phase in ("baseline_before", "baseline_after"):
            self.assertEqual([row["dispatch_order"] for row in observations[phase]], [["metal"], ["ane"], ["ane"], ["metal"]])
        paired = observations["concurrent"]
        self.assertEqual([row["dispatch_order"] for row in paired], [["metal", "ane"], ["ane", "metal"]])
        for batch in paired:
            fast, slow = batch["requests"]["metal"], batch["requests"]["ane"]
            self.assertLess(fast["completion_ms"], slow["completion_ms"])
            self.assertGreater(slow["completion_ms"] - fast["completion_ms"], 20)
            self.assertGreaterEqual(batch["wall_ms"], slow["completion_ms"])
            self.assertEqual(slow["service_ms"], 2)
        summary = report["summary"]
        self.assertEqual(summary["concurrent_completed_tokens"], 32)
        self.assertEqual(summary["concurrent_batch_count"], 2)
        self.assertAlmostEqual(summary["concurrent_aggregate_tokens_per_second"], 32000 / sum(row["wall_ms"] for row in paired))
        self.assertEqual(summary["lanes"]["ane"]["quality"]["top1_match_count"], 0)
        self.assertEqual(summary["lanes"]["metal"]["quality"]["observation_count_including_warmup"], 7)
        self.assertIn("not equivalent-quality", summary["comparison_caveat"])

    def test_cpu_and_split_lines(self):
        result = self.run_cli(mode="split_line", exact="cpu", extra=["--half", "0", "--layers", "3"])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["ready"]["cpu"]["message"]["metal_executed"])
        self.assertEqual(report["configuration"]["controlled_environment"], {
            "cpu": {"TM_ANE_PATH": str(self.package), "TM_ANE_LAYERS": "3", "TM_LLAMA_GPU_HALF": "0"},
            "ane": {"TM_ANE_PATH": str(self.package), "TM_ANE_LAYERS": "3", "TM_LLAMA_GPU_HALF": "0"},
        })

    def test_heterogeneous_exact_tokens(self):
        exact_tokens = self.root / "exact_tokens.bin"
        exact_tokens.write_bytes(b"\x02\x00\x00\x00" * 8)
        result = self.run_cli(extra=["--token-file", str(self.tokens),
                                     "--exact-tokens", "16", "--exact-token-file", str(exact_tokens)])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        config = report["configuration"]
        self.assertEqual(config["tokens"], 8)
        self.assertEqual(config["exact_tokens"], 16)
        self.assertEqual(config["token_files"]["ane"]["path"], str(self.tokens))
        self.assertEqual(config["token_files"]["metal"]["path"], str(exact_tokens))
        # The fake worker echoes its argv token count; per-lane validation and
        # accounting must accept the differing lengths end to end.
        self.assertEqual(report["ready"]["ane"]["message"]["tokens"], 8)
        self.assertEqual(report["ready"]["metal"]["message"]["tokens"], 16)
        for rows in report["observations"].values():
            for batch in rows:
                for lane, message in batch["requests"].items():
                    self.assertEqual(message["result"]["tokens"], 8 if lane == "ane" else 16)
        summary = report["summary"]
        self.assertEqual(summary["concurrent_completed_tokens"], 2 * (8 + 16))
        self.assertAlmostEqual(summary["concurrent_aggregate_tokens_per_second"],
                               24 * 2000 / sum(row["wall_ms"] for row in report["observations"]["concurrent"]))
        self.assertEqual(summary["lanes"]["ane"]["baseline_rate_tokens_per_second"],
                         8000 / summary["lanes"]["ane"]["baseline_bracketed_mean_completion_ms"])
        self.assertEqual(summary["lanes"]["metal"]["baseline_rate_tokens_per_second"],
                         16000 / summary["lanes"]["metal"]["baseline_bracketed_mean_completion_ms"])
        # An ANE-sized shared token file with different exact tokens is ambiguous.
        result = self.run_cli(extra=["--token-file", str(self.tokens), "--exact-tokens", "16"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--exact-token-file", result.stderr)
        self.assertNotIn("initializing", result.stderr)

    def test_failures_never_publish_success(self):
        cases = {
            "malformed": "invalid JSON", "nonfinite": "nonfinite", "misrouted": "routing flag",
            "wrong_lane": "wrong lane", "worker_error": "deliberate worker failure", "eof": "EOF",
            "replay": "exceeds tolerance", "result_nonfinite": "nonfinite", "wrong_event": "wrong event",
            "extra_stdout": "stdout", "oversize": "exceeds", "second_startup": "second startup refused",
            "idle_stdout": "stdout", "idle_exit": "EOF",
            "shutdown_failure": "exited 7", "shutdown_stdout": "stdout", "stderr_flood": "TAIL-MARKER",
        }
        for mode, diagnostic in cases.items():
            with self.subTest(mode=mode):
                output = self.root / (mode + ".json")
                result = self.run_cli(mode, extra=["--output", str(output)])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertFalse(output.exists())
                self.assertIn(diagnostic, result.stderr)
                self.assertLess(len(result.stderr), 20000)

    def test_timeouts_and_kill_escalation(self):
        for mode in ("timeout", "handshake_timeout", "shutdown_timeout", "ignore_terminate"):
            with self.subTest(mode=mode):
                start = time.monotonic()
                result = self.run_cli(mode, extra=["--timeout", "0.4"])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertLess(time.monotonic() - start, 7)
                if mode != "ignore_terminate":
                    self.assertIn("timeout", result.stderr)

    def test_invalid_arguments_do_not_launch(self):
        cases = [
            ["--tokens", "1"], ["--rounds", "1"], ["--seed", "0"], ["--layers", "0"],
            ["--tokens", str(2**31)], ["--timeout", "0"], ["--timeout", "nan"], ["--timeout", "inf"],
            ["--replay-tolerance", "-1"], ["--replay-tolerance", "nan"], ["--replay-tolerance", "inf"],
            ["--half", "2"], ["--exact-lane", "gpu"], ["--token-file", str(self.root / "missing")],
            ["--binary", str(self.model)], ["--ane-path", str(self.root)],
            ["--output", str(self.root / "missing" / "report.json")],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.run_cli(extra=arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertEqual(list(self.root.glob("*.pid")), [])
        self.model.unlink()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.root.glob("*.pid")), [])

    def test_existing_and_dangling_output_refused(self):
        output = self.root / "report.json"
        output.write_text("keep this")
        for path in (output, self.root / "dangling.json"):
            if path != output:
                path.symlink_to(self.root / "not-created")
            result = self.run_cli(extra=["--output", str(path)])
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("refusing overwrite", result.stderr)
        self.assertEqual(output.read_text(), "keep this")
        self.assertEqual(list(self.root.glob("*.pid")), [])

    def test_output_race_is_no_clobber(self):
        output = self.root / "raced.json"

        def raced_benchmark(args):
            output.write_text("other writer")
            return {"schema": capacity.SCHEMA}

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(capacity, "benchmark", side_effect=raced_benchmark), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = capacity.main([*self.arguments, "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(output.read_text(), "other writer")


class SummaryTests(unittest.TestCase):
    def test_deterministic_bracketed_accounting(self):
        def request(lane, completion, service, quality=0):
            return {"completion_ms": completion, "service_ms": service,
                    "result": {"cpu_maxdiff": quality, "cpu_relative_l2": quality / 2,
                               "replay_maxdiff": quality / 100, "cpu_top1_match": lane == "metal"}}

        def serial(metal_ms, ane_ms):
            return [{"requests": {"metal": request("metal", metal_ms, metal_ms / 2)}},
                    {"requests": {"ane": request("ane", ane_ms, ane_ms / 2)}}] * 2

        observations = {
            "baseline_before": serial(10, 20), "baseline_after": serial(30, 40),
            "concurrent": [
                {"wall_ms": 50, "requests": {"metal": request("metal", 40, 20), "ane": request("ane", 50, 25, 0.01)}},
                {"wall_ms": 150, "requests": {"metal": request("metal", 80, 40), "ane": request("ane", 150, 75, 0.02)}},
            ],
        }
        ready = {lane: {"message": request(lane, 1, 1)["result"]} for lane in ("metal", "ane")}
        summary = capacity.summarize(observations, ready,
                                         argparse.Namespace(tokens=100, exact_tokens=None), "metal")
        self.assertEqual(summary["concurrent_completed_tokens"], 400)
        self.assertEqual(summary["concurrent_total_wall_seconds"], 0.2)
        self.assertEqual(summary["concurrent_aggregate_tokens_per_second"], 2000)
        self.assertEqual(summary["concurrent_batch_tokens_per_second"], [4000, 200000 / 150])
        self.assertEqual(summary["aggregate_over_exact_lane_baseline_rate"], 0.4)
        metal, ane = summary["lanes"]["metal"], summary["lanes"]["ane"]
        self.assertEqual(metal["baseline_bracketed_mean_completion_ms"], 20)
        self.assertEqual(metal["baseline_rate_tokens_per_second"], 5000)
        self.assertEqual(metal["completion_latency_slowdown"], 3)
        self.assertEqual(metal["service_latency_slowdown"], 3)
        self.assertEqual(metal["baseline_completion_drift_after_over_before"], 3)
        self.assertEqual(ane["baseline_completion_drift_after_over_before"], 2)
        self.assertAlmostEqual(ane["completion_latency_slowdown"], 100 / 30)
        self.assertEqual(ane["quality"]["cpu_maxdiff_max"], 0.02)
        self.assertEqual(ane["quality"]["replay_maxdiff_max"], 0.0002)
        self.assertEqual(ane["quality"]["observation_count_including_warmup"], 7)
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
