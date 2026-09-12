#!/usr/bin/env python3
"""Measure finite, process-isolated exact/ANE prefill bursts; never a server.

The two workers stay resident throughout both bracketed idle-peer baselines and
paired measurements. Host timing cannot establish operation-level overlap.
Only the Python standard library is required, including by the fake-worker tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

PROTOCOL = "tensormark.prefill-lane/1"
SCHEMA = "tensormark.ane-capacity/1"
MAX_LINE_BYTES = 64 * 1024
STDERR_TAIL_BYTES = 8192
EXIT_GRACE_SECONDS = 2.0


class BenchmarkError(RuntimeError):
    """A run failed closed; no capacity report may be published."""


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def tolerance_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("model", type=Path)
    result.add_argument("--ane-path", type=Path, required=True)
    result.add_argument("--tokens", type=int, default=1024)
    # Heterogeneous serving probe: the ANE package has one fixed token count,
    # while the ordinary lane can serve a different prompt length.
    result.add_argument("--exact-tokens", type=int)
    result.add_argument("--exact-token-file", type=Path)
    result.add_argument("--rounds", type=int, default=3)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--layers", type=int, default=16)
    result.add_argument("--exact-lane", choices=("cpu", "metal"), default="metal")
    result.add_argument("--binary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tensormark/build/bench_prefill_lane")
    result.add_argument("--token-file", type=Path)
    result.add_argument("--timeout", type=positive_float, default=180.0)
    result.add_argument("--replay-tolerance", type=tolerance_float, default=1e-4)
    result.add_argument("--half", choices=(0, 1), type=int, default=1)
    # Three-way blend: pin the ordinary lane's CPU/GPU row split
    # (TM_LLAMA_HYBRID; fraction of rows given to CPU, [0, 0.5), 0 = off).
    result.add_argument("--metal-hybrid", type=float)
    result.add_argument("--output", type=Path)
    return result


def validate_args(args: argparse.Namespace) -> None:
    for name, minimum in (("tokens", 2), ("rounds", 2), ("seed", 1), ("layers", 1)):
        value = getattr(args, name)
        if type(value) is not int or not minimum <= value <= 2**31 - 1:
            raise BenchmarkError(f"{name} must be an integer in [{minimum}, {2**31 - 1}]")
        if args.exact_tokens is not None and (
                type(args.exact_tokens) is not int or not 2 <= args.exact_tokens <= 2**31 - 1):
            raise BenchmarkError("exact-tokens must be an integer in [2, 2**31 - 1]")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise BenchmarkError("timeout must be finite and positive")
    if not math.isfinite(args.replay_tolerance) or args.replay_tolerance < 0:
        raise BenchmarkError("replay tolerance must be finite and nonnegative")
    if args.exact_lane not in ("cpu", "metal") or args.half not in (0, 1):
        raise BenchmarkError("invalid exact lane or half setting")
    if args.metal_hybrid is not None and (
            type(args.metal_hybrid) is not float or not math.isfinite(args.metal_hybrid)
            or not 0.0 <= args.metal_hybrid < 0.5):
        raise BenchmarkError("metal-hybrid must be a float in [0, 0.5)")
    for name in ("model", "binary", "token_file", "exact_token_file"):
        path = getattr(args, name)
        if path is None:
            continue
        path = path.expanduser().resolve()
        setattr(args, name, path)
        # A directory model arg is an F32 safetensors snapshot for
        # precision-matched comparisons (bench loads it via Llama::load).
        if name == "model" and path.is_dir():
            continue
        if not path.is_file() or not os.access(path, os.R_OK):
            raise BenchmarkError(f"{name} must be an existing readable file: {path}")
    if not os.access(args.binary, os.X_OK):
        raise BenchmarkError(f"binary must be executable: {args.binary}")
    package = args.ane_path.expanduser().resolve()
    if package.suffix != ".mlmodelc" or not package.is_dir() or not os.access(package, os.R_OK | os.X_OK):
        raise BenchmarkError("ane-path must be an existing readable compiled .mlmodelc directory")
    args.ane_path = package
    if (args.token_file is not None and args.exact_tokens is not None
            and args.exact_tokens != args.tokens and args.exact_token_file is None):
        # The exact worker would reject the ANE-sized token file at runtime;
        # refuse the ambiguous configuration before launching anything.
        raise BenchmarkError(
            "--exact-tokens differs from --tokens: provide --exact-token-file for the exact lane")
    if args.output is not None:
        # Do not resolve the final component: a dangling symlink also exists for
        # no-clobber purposes, and must never redirect the report write.
        args.output = Path(os.path.abspath(args.output.expanduser()))
        if os.path.lexists(args.output):
            raise BenchmarkError(f"output already exists (refusing overwrite): {args.output}")
        if not args.output.parent.is_dir() or not os.access(args.output.parent, os.W_OK | os.X_OK):
            raise BenchmarkError("output parent must be an existing writable directory")


def lane_tokens(args: argparse.Namespace, lane: str) -> int:
    """Per-lane prompt length: the ANE package shape is fixed; the exact lane may differ."""
    if lane == "ane":
        return args.tokens
    return args.tokens if args.exact_tokens is None else args.exact_tokens


def lane_token_file(args: argparse.Namespace, lane: str) -> Path | None:
    if lane == "ane":
        return args.token_file
    return args.exact_token_file if args.exact_token_file is not None else args.token_file


def identity(path: Path, *, hash_contents: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_contents:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def worker_environment(args: argparse.Namespace, lane: str) -> tuple[dict[str, str], dict[str, str]]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TM_")}
    controlled = {
        "TM_ANE_PATH": str(args.ane_path),
        "TM_ANE_LAYERS": str(args.layers),
        "TM_LLAMA_GPU_HALF": str(args.half),
    }
    if getattr(args, "metal_hybrid", None) is not None:
        controlled["TM_LLAMA_HYBRID"] = repr(args.metal_hybrid)
        # bench_prefill_lane forces the hybrid split off unless this explicit
        # opt-in is also present (no ambient lane change).
        controlled["TM_BENCH_HYBRID"] = "1"
    token_file = lane_token_file(args, lane)
    if token_file is not None:
        controlled["TM_BENCH_TOKENS"] = str(token_file)
    environment.update(controlled)
    return environment, controlled


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def validate_message(line: bytes, lane: str, event: str, args: argparse.Namespace) -> dict[str, Any]:
    try:
        message = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise BenchmarkError(f"{lane}: invalid JSON: {error}") from error
    if not isinstance(message, dict):
        raise BenchmarkError(f"{lane}: expected a JSON object")
    for field, expected in (("event", event), ("protocol", PROTOCOL), ("lane", lane),
                            ("tokens", lane_tokens(args, lane)), ("seed", args.seed)):
        actual = message.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise BenchmarkError(f"{lane}: wrong {field}: {actual!r}, expected {expected!r}")
    for field in ("wall_ms", "cpu_maxdiff", "cpu_relative_l2", "replay_maxdiff"):
        value = message.get(field)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid or (field == "wall_ms" and value == 0):
            raise BenchmarkError(f"{lane}: invalid {field}: {value!r}")
    if type(message.get("cpu_top1_match")) is not bool:
        raise BenchmarkError(f"{lane}: cpu_top1_match must be boolean")
    for field, expected in (("ane_executed", lane == "ane"),
                            ("metal_executed", lane in ("metal", "ane"))):
        if message.get(field) is not expected:
            raise BenchmarkError(f"{lane}: unexpected routing flag {field}")
    replay = message["replay_maxdiff"]
    if event == "ready" and replay != 0:
        raise BenchmarkError(f"{lane}: ready replay_maxdiff must be zero")
    if event == "result" and replay > args.replay_tolerance:
        raise BenchmarkError(f"{lane}: replay_maxdiff exceeds tolerance {args.replay_tolerance}")
    return message


class Worker:
    def __init__(self, process: subprocess.Popen[bytes], lane: str, stderr: Any):
        self.process = process
        self.lane = lane
        self.stderr = stderr
        self.buffer = bytearray()

    def tail(self) -> str:
        self.stderr.seek(0, os.SEEK_END)
        size = self.stderr.tell()
        self.stderr.seek(max(0, size - STDERR_TAIL_BYTES))
        return self.stderr.read(STDERR_TAIL_BYTES).decode("utf-8", errors="replace")


class Workers:
    """Own exactly two PIDs, bounded stdout buffers, and file-backed stderr."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.workers: dict[str, Worker] = {}
        self.selector = selectors.DefaultSelector()

    def start(self, lane: str) -> dict[str, Any]:
        if len(self.workers) >= 2 or lane in self.workers:
            raise BenchmarkError("at most two distinct workers are permitted")
        self.check_idle()
        stderr = tempfile.TemporaryFile(mode="w+b")
        try:
            environment, _ = worker_environment(self.args, lane)
            process = subprocess.Popen(
                [str(self.args.binary), str(self.args.model), str(lane_tokens(self.args, lane)),
                 str(self.args.seed), lane],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                env=environment, bufsize=0,
            )
        except BaseException:
            stderr.close()
            raise
        worker = Worker(process, lane, stderr)
        self.workers[lane] = worker  # Register ownership before any fallible setup.
        assert process.stdout is not None and process.stdin is not None
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stdin.fileno(), False)
        self.selector.register(process.stdout, selectors.EVENT_READ, worker)
        start = time.perf_counter()
        received = self.receive({lane}, "ready", start)
        return {"message": received[lane]["result"], "handshake_ms": received[lane]["completion_ms"]}

    def send(self, worker: Worker, command: bytes) -> None:
        if worker.process.poll() is not None:
            raise BenchmarkError(f"{worker.lane}: worker exited {worker.process.returncode}")
        assert worker.process.stdin is not None
        try:
            written = os.write(worker.process.stdin.fileno(), command)
        except OSError as error:
            raise BenchmarkError(f"{worker.lane}: command write failed: {error}") from error
        if written != len(command):
            raise BenchmarkError(f"{worker.lane}: incomplete command write")

    def receive(self, pending: set[str], event: str, start: float) -> dict[str, Any]:
        pending = set(pending)
        observations: dict[str, Any] = {}
        deadline = start + self.args.timeout
        while pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise BenchmarkError(f"timeout waiting for {event} from {', '.join(sorted(pending))}")
            events = self.selector.select(remaining)
            if not events:
                raise BenchmarkError(f"timeout waiting for {event} from {', '.join(sorted(pending))}")
            for key, _ in events:
                worker = key.data
                try:
                    chunk = os.read(key.fd, MAX_LINE_BYTES + 1)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise BenchmarkError(f"{worker.lane}: unexpected EOF (exit={worker.process.poll()})")
                if worker.lane not in pending:
                    raise BenchmarkError(f"{worker.lane}: unsolicited stdout while idle")
                worker.buffer.extend(chunk)
                if len(worker.buffer) > MAX_LINE_BYTES:
                    raise BenchmarkError(f"{worker.lane}: protocol line exceeds {MAX_LINE_BYTES} bytes")
                if b"\n" not in worker.buffer:
                    continue
                line, extra = bytes(worker.buffer).split(b"\n", 1)
                worker.buffer.clear()
                if extra:
                    raise BenchmarkError(f"{worker.lane}: extra stdout after protocol line")
                message = validate_message(line, worker.lane, event, self.args)
                completion_ms = (time.perf_counter() - start) * 1000
                if time.perf_counter() > deadline:
                    raise BenchmarkError(f"{worker.lane}: {event} validation exceeded timeout")
                observations[worker.lane] = {
                    "result": message, "service_ms": message["wall_ms"], "completion_ms": completion_ms,
                }
                pending.remove(worker.lane)
        return observations

    def check_idle(self) -> None:
        for worker in self.workers.values():
            if worker.process.poll() is not None:
                raise BenchmarkError(f"{worker.lane}: worker exited {worker.process.returncode} while idle")
        for key, _ in self.selector.select(0):
            try:
                chunk = os.read(key.fd, MAX_LINE_BYTES + 1)
            except BlockingIOError:
                continue
            detail = "unsolicited stdout" if chunk else "unexpected EOF"
            raise BenchmarkError(f"{key.data.lane}: {detail} while idle")

    def batch(self, lanes: list[str], round_index: int) -> dict[str, Any]:
        self.check_idle()
        # One common controller clock, and both writes BEFORE the first wait.
        start = time.perf_counter()
        for lane in lanes:
            self.send(self.workers[lane], b"run\n")
        received = self.receive(set(lanes), "result", start)
        wall_ms = (time.perf_counter() - start) * 1000
        return {"round": round_index, "dispatch_order": lanes, "wall_ms": wall_ms, "requests": received}

    def shutdown(self) -> None:
        for worker in self.workers.values():
            self.send(worker, b"quit\n")
        deadline = time.perf_counter() + min(self.args.timeout, EXIT_GRACE_SECONDS)
        # Drain only EOF; any data after the last response violates the protocol.
        open_streams = set(self.workers)
        while open_streams:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise BenchmarkError("timeout waiting for worker shutdown")
            for key, _ in self.selector.select(remaining):
                worker = key.data
                try:
                    chunk = os.read(key.fd, MAX_LINE_BYTES + 1)
                except BlockingIOError:
                    continue
                if chunk:
                    raise BenchmarkError(f"{worker.lane}: unexpected stdout during shutdown")
                self.selector.unregister(key.fileobj)
                open_streams.remove(worker.lane)
        for worker in self.workers.values():
            try:
                code = worker.process.wait(timeout=max(0.001, deadline - time.perf_counter()))
            except subprocess.TimeoutExpired as error:
                raise BenchmarkError(f"{worker.lane}: timeout waiting for exit") from error
            if code != 0:
                raise BenchmarkError(f"{worker.lane}: worker exited {code}")

    def abort(self) -> None:
        # Signal only Popen objects created by this controller, never names/groups.
        for worker in self.workers.values():
            if worker.process.poll() is None:
                try:
                    worker.process.terminate()
                except ProcessLookupError:
                    pass
        deadline = time.perf_counter() + EXIT_GRACE_SECONDS
        for worker in self.workers.values():
            try:
                worker.process.wait(timeout=max(0.001, deadline - time.perf_counter()))
            except subprocess.TimeoutExpired:
                try:
                    worker.process.kill()
                except ProcessLookupError:
                    pass
        for worker in self.workers.values():
            try:
                worker.process.wait(timeout=EXIT_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                print(f"{worker.lane}: could not reap owned PID {worker.process.pid}", file=sys.stderr, flush=True)

    def close(self) -> None:
        self.selector.close()
        for worker in self.workers.values():
            for stream in (worker.process.stdin, worker.process.stdout, worker.stderr):
                if stream is not None:
                    stream.close()


def summarize(observations: dict[str, list[dict[str, Any]]], ready: dict[str, Any],
              args: argparse.Namespace, exact_lane: str) -> dict[str, Any]:
    """Ratios of measured means; never a sum of reciprocal service times."""
    paired = observations["concurrent"]
    lanes: dict[str, Any] = {}
    for lane in (exact_lane, "ane"):
        tokens = lane_tokens(args, lane)
        before = [batch["requests"][lane] for batch in observations["baseline_before"] if lane in batch["requests"]]
        after = [batch["requests"][lane] for batch in observations["baseline_after"] if lane in batch["requests"]]
        concurrent = [batch["requests"][lane] for batch in paired]
        completion = statistics.mean(row["completion_ms"] for row in before + after)
        service = statistics.mean(row["service_ms"] for row in before + after)
        before_ms = statistics.mean(row["completion_ms"] for row in before)
        after_ms = statistics.mean(row["completion_ms"] for row in after)
        before_service = statistics.mean(row["service_ms"] for row in before)
        after_service = statistics.mean(row["service_ms"] for row in after)
        messages = [ready[lane]["message"]] + [row["result"] for row in before + concurrent + after]
        lanes[lane] = {
            "baseline_before_count": len(before), "baseline_after_count": len(after),
            "concurrent_count": len(concurrent),
            "baseline_before_mean_completion_ms": before_ms,
            "baseline_after_mean_completion_ms": after_ms,
            "baseline_bracketed_mean_completion_ms": completion,
            "baseline_bracketed_mean_service_ms": service,
            "baseline_rate_tokens_per_second": tokens * 1000 / completion,
            "baseline_completion_drift_after_over_before": after_ms / before_ms,
            "baseline_service_drift_after_over_before": after_service / before_service,
            "concurrent_mean_completion_ms": statistics.mean(row["completion_ms"] for row in concurrent),
            "concurrent_mean_service_ms": statistics.mean(row["service_ms"] for row in concurrent),
            "completion_latency_slowdown": statistics.mean(row["completion_ms"] for row in concurrent) / completion,
            "service_latency_slowdown": statistics.mean(row["service_ms"] for row in concurrent) / service,
            "quality": {
                "cpu_maxdiff_max": max(message["cpu_maxdiff"] for message in messages),
                "cpu_relative_l2_max": max(message["cpu_relative_l2"] for message in messages),
                "replay_maxdiff_max": max(message["replay_maxdiff"] for message in messages),
                "top1_match_observations": [message["cpu_top1_match"] for message in messages],
                "top1_match_count": sum(message["cpu_top1_match"] for message in messages),
                "observation_count_including_warmup": len(messages),
            },
        }
    completed = sum(sum(lane_tokens(args, lane) for lane in batch["requests"]) for batch in paired)
    total_seconds = sum(batch["wall_ms"] for batch in paired) / 1000
    aggregate = completed / total_seconds
    return {
        "lanes": lanes,
        "concurrent_batch_count": len(paired),
        "concurrent_completed_tokens": completed,
        "concurrent_total_wall_seconds": total_seconds,
        "concurrent_aggregate_tokens_per_second": aggregate,
        "concurrent_batch_tokens_per_second": [
            sum(lane_tokens(args, lane) for lane in batch["requests"]) * 1000 / batch["wall_ms"]
            for batch in paired],
        "aggregate_over_exact_lane_baseline_rate": aggregate / lanes[exact_lane]["baseline_rate_tokens_per_second"],
        "comparison_caveat": "Mixed exact-lane and fp16 ANE workload; not equivalent-quality throughput.",
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    configuration = {
        "tokens": args.tokens, "exact_tokens": args.exact_tokens,
        "rounds": args.rounds, "seed": args.seed,
        "exact_lane": args.exact_lane, "layers": args.layers, "half": args.half,
        "timeout_seconds": args.timeout, "replay_tolerance": args.replay_tolerance,
        "model": identity(args.model), "compiled_package": identity(args.ane_path),
        "binary": identity(args.binary),
        "token_files": {lane: (identity(token, hash_contents=True) if token is not None else None)
                        for lane, token in ((args.exact_lane, lane_token_file(args, args.exact_lane)),
                                            ("ane", lane_token_file(args, "ane")))},
        "controlled_environment": {lane: worker_environment(args, lane)[1]
                                   for lane in (args.exact_lane, "ane")},
        "environment_policy": "Copy os.environ after clearing EVERY TM_* variable; add only controlled_environment.",
        "thread_control": "Unset: no verified thread knob in the worker contract (TM_GEMV_THREADS is not forwarded).",
        "identity_scope": "Model and binary stat only, no huge-model hashing; package directory stat, not recursive content identity.",
    }
    workers = Workers(args)
    ready: dict[str, Any] = {}
    observations: dict[str, list[dict[str, Any]]] = {}
    try:
        for lane in (args.exact_lane, "ane"):
            print(f"initializing {lane} (sequential load/reference/warmup)", file=sys.stderr, flush=True)
            ready[lane] = workers.start(lane)
        for phase in ("baseline_before", "concurrent", "baseline_after"):
            observations[phase] = []
            for index in range(args.rounds):
                order = [args.exact_lane, "ane"] if index % 2 == 0 else ["ane", args.exact_lane]
                print(f"{phase} round {index + 1}/{args.rounds}", file=sys.stderr, flush=True)
                groups = [order] if phase == "concurrent" else [[lane] for lane in order]
                for group in groups:
                    observations[phase].append(workers.batch(group, index))
        print("waiting for both workers to exit successfully", file=sys.stderr, flush=True)
        workers.shutdown()
    except BaseException as error:
        workers.abort()
        tails = "\n".join(f"{lane} stderr tail:\n{worker.tail()}" for lane, worker in workers.workers.items())
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            if tails:
                print(tails, file=sys.stderr, flush=True)
            raise
        raise BenchmarkError(f"{error}\n{tails}") from error
    finally:
        workers.close()
    return {
        "schema": SCHEMA,
        "measurement_scope": {
            "workload": "Cache-reset prefill only; no decode, download, conversion, or model archive operations.",
            "baseline": "Co-resident idle-peer baseline, not standalone memory footprint; serial measurements bracket paired batches.",
            "capacity": "Finite paired bursts, not steady-state capacity; no operation-level overlap or residency claim from host timing alone.",
            "timing": "Controller perf_counter from before first dispatch through receipt and validation of both results; worker wall_ms is reset+forward only.",
            "quality": "Warmup CPU comparisons and top1 are observational, not a corpus-quality gate; result replay differences are tolerance-gated.",
            "mix": "Exact lane plus fp16 ANE; per-lane prompt lengths may differ (--exact-tokens); "
                       "aggregate comparison is not equivalent-quality throughput.",
            "initialization": "Two children maximum, sequential initialization; warmups excluded from measured token accounting.",
        },
        "configuration": configuration,
        "ready": ready,
        "observations": observations,
        "summary": summarize(observations, ready, args, args.exact_lane),
        "worker_exit_codes": {lane: worker.process.returncode for lane, worker in workers.workers.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = benchmark(args)
        serialized = json.dumps(report, allow_nan=False, indent=2) + "\n"
        if args.output is not None:
            # Exclusive creation is the final race-safe no-clobber check.
            with args.output.open("x", encoding="utf-8") as target:
                target.write(serialized)
        sys.stdout.write(serialized)
        sys.stdout.flush()
    except (BenchmarkError, OSError, ValueError, OverflowError) as error:
        print(f"ane-capacity: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("ane-capacity: interrupted", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
