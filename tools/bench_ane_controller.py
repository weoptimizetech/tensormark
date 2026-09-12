#!/usr/bin/env python3
"""ANE admission-share study: static routing sweep vs balance-PI controller.

Serving-style batches over the same persistent workers as bench_ane_capacity.
Each batch queues a FIXED total workload: `--requests-per-batch` m requests of
the ANE package's fixed token count, routed between the two lanes. Only the
eligible share routed to the ANE lane changes; total tokens per batch are
constant, so static sweep points are matched comparisons. The share is either
swept statically or adapted by a balance PI controller on per-lane rate
slowdowns — the direct analogue of the GPU/CPU blend balance controller
(bench_prefill_lane workers accept one fixed prompt length per process, so the
ordinary lane here serves the same eligible length rather than shorter mixed
requests).

A lane with no requests in a batch has slowdown 0 (idle), so the PI escapes
extremes: if the ordinary lane lags, rho grows and shifts eligible work to the
ANE lane. This is a finite-burst admission-share study over two persistent
workers, NOT a production scheduler, an SLA policy, or evidence of per-operation
ANE residency. The ANE tier is fp16 with different numerics from the exact
lane; all metrics are capacity accounting, never quality claims.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
import selectors
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ane_capacity import (  # noqa: E402
    BenchmarkError, identity, lane_tokens, tolerance_float, validate_args,
    validate_message, worker_environment,
)

SCHEMA = "tensormark.ane-admission/1"
STDERR_TAIL_BYTES = 8192
MAX_LINE_BYTES = 64 * 1024
EXIT_GRACE_SECONDS = 5.0


def ratio_list(value: str) -> list[float]:
    try:
        points = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ratio list: {error}") from error
    if not points or any(not math.isfinite(point) or not 0.0 <= point <= 1.0 for point in points):
        raise argparse.ArgumentTypeError("ratios must be in [0, 1]")
    return points


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("model", type=Path)
    result.add_argument("--ane-path", type=Path, required=True)
    result.add_argument("--tokens", type=int, default=1024,
                        help="eligible prompt length; must match the ANE package shape")
    result.add_argument("--requests-per-batch", type=int, default=4)
    result.add_argument("--rounds", type=int, default=2, help="batches per static ratio point")
    result.add_argument("--ratios", type=ratio_list, default="0,0.25,0.5,0.75,1",
                        help="static ANE-share sweep points")
    result.add_argument("--pi-rounds", type=int, default=12)
    # Stochastic contention regime: a third metal-lane worker runs one or more
    # prefills concurrently in batches drawn by a seeded Bernoulli schedule.
    result.add_argument("--competitor-probability", type=tolerance_float, default=0.0)
    result.add_argument("--competitor-requests", type=int, default=1)
    result.add_argument("--es-delta", type=tolerance_float, default=0.125)
    result.add_argument("--es-step", type=tolerance_float, default=0.2)
    result.add_argument("--kp", type=tolerance_float, default=0.15)
    result.add_argument("--ki", type=tolerance_float, default=0.05)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--layers", type=int, default=16)
    result.add_argument("--exact-lane", choices=("cpu", "metal"), default="metal")
    result.add_argument("--binary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tensormark/build/bench_prefill_lane")
    result.add_argument("--token-file", type=Path, help="eligible prompt token IDs")
    result.add_argument("--exact-token-file", type=Path,
                        help="optional ordinary-lane token IDs; defaults to --token-file")
    result.add_argument("--timeout", type=float, default=180.0)
    result.add_argument("--replay-tolerance", type=tolerance_float, default=1e-4)
    result.add_argument("--half", choices=(0, 1), type=int, default=1)
    result.add_argument("--metal-hybrid", type=float,
                        help="pin the metal lane's CPU/GPU row split (TM_LLAMA_HYBRID, [0, 0.5))")
    result.add_argument("--output", type=Path)
    return result


def validate_controller_args(args: argparse.Namespace) -> None:
    for name, minimum in (("tokens", 2), ("requests_per_batch", 1), ("rounds", 1),
                          ("pi_rounds", 2), ("seed", 1)):
        value = getattr(args, name)
        if type(value) is not int or not minimum <= value <= 2**31 - 1:
            raise BenchmarkError(f"{name} must be an integer in [{minimum}, {2**31 - 1}]")
    if type(args.timeout) is not float or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise BenchmarkError("timeout must be finite and positive")
    if args.metal_hybrid is not None and (
            type(args.metal_hybrid) is not float or not math.isfinite(args.metal_hybrid)
            or not 0.0 <= args.metal_hybrid < 0.5):
        raise BenchmarkError("metal-hybrid must be a float in [0, 0.5)")
    for name in ("competitor_requests",):
        value = getattr(args, name)
        if type(value) is not int or not 0 <= value <= 2**31 - 1:
            raise BenchmarkError(f"{name} must be an integer in [0, {2**31 - 1}]")
    # Both lanes serve the same eligible length here (one fixed worker prompt);
    # the ordinary lane's token count mirrors --tokens. validate_args still gets
    # it so its ambiguity guard stays meaningful for per-lane token files.
    args.exact_tokens = args.tokens
    validate_args(args)  # paths, package shape, output no-clobber, replay tolerance.
    for name, path in (("token-file", args.token_file), ("exact-token-file", args.exact_token_file)):
        if path is None:
            continue
        try:
            ids = [int(item) for item in path.read_text().split()]
        except ValueError as error:
            raise BenchmarkError(f"{name} must contain whitespace-separated integer token IDs: {error}")
        if len(ids) != args.tokens or any(value < 0 for value in ids):
            raise BenchmarkError(
                f"{name} must contain exactly {args.tokens} nonnegative token IDs, got {len(ids)}")


class Worker:
    def __init__(self, process: subprocess.Popen[bytes], lane: str, stderr):
        self.process = process
        self.lane = lane
        self.stderr = stderr
        self.buffer = bytearray()

    def tail(self) -> str:
        self.stderr.seek(0, os.SEEK_END)
        size = self.stderr.tell()
        self.stderr.seek(max(0, size - STDERR_TAIL_BYTES))
        return self.stderr.read(STDERR_TAIL_BYTES).decode("utf-8", errors="replace")


class Pool:
    """Pipelined two-lane pool: several queued requests per lane per batch."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.workers: dict[str, Worker] = {}
        self.selector = selectors.DefaultSelector()

    def start(self, alias: str, protocol_lane: str | None = None) -> dict:
        if len(self.workers) >= 3 or alias in self.workers:
            raise BenchmarkError("at most three distinct workers are permitted")
        protocol_lane = protocol_lane or alias
        self.check_idle()
        stderr = tempfile.TemporaryFile(mode="w+b")
        try:
            environment, _ = worker_environment(self.args, protocol_lane)
            process = subprocess.Popen(
                [str(self.args.binary), str(self.args.model),
                 str(lane_tokens(self.args, protocol_lane)), str(self.args.seed), protocol_lane],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                env=environment, bufsize=0,
            )
        except BaseException:
            stderr.close()
            raise
        worker = Worker(process, protocol_lane, stderr)
        worker.alias = alias
        self.workers[alias] = worker
        assert process.stdout is not None and process.stdin is not None
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stdin.fileno(), False)
        self.selector.register(process.stdout, selectors.EVENT_READ, worker)
        start = time.perf_counter()
        received = self._read_until({alias: 1}, "ready", start)
        return {"message": received[alias][0]["result"], "handshake_ms": received[alias][0]["arrival_ms"]}

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

    def _lines(self, worker: Worker, chunk: bytes) -> list[bytes]:
        worker.buffer.extend(chunk)
        if len(worker.buffer) > MAX_LINE_BYTES:
            raise BenchmarkError(f"{worker.lane}: protocol line exceeds {MAX_LINE_BYTES} bytes")
        parts = worker.buffer.split(b"\n")
        worker.buffer = bytearray(parts.pop())
        return parts

    def _result(self, worker: Worker, line: bytes, event: str, start: float, deadline: float) -> dict:
        message = validate_message(line, worker.lane, event, self.args)
        if time.perf_counter() > deadline:
            raise BenchmarkError(f"{worker.lane}: {event} validation exceeded timeout")
        return {"result": message, "arrival_ms": (time.perf_counter() - start) * 1000}

    def _read_until(self, expected: dict[str, int], event: str, start: float) -> dict[str, list[dict]]:
        deadline = start + self.args.timeout * max(1, max(expected.values()))
        received: dict[str, list[dict]] = {lane: [] for lane in expected}
        while any(len(received[lane]) < expected[lane] for lane in expected):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise BenchmarkError(f"timeout waiting for {event} from "
                                     f"{', '.join(sorted(lane for lane in expected if len(received[lane]) < expected[lane]))}")
            for key, _ in self.selector.select(remaining):
                worker = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), MAX_LINE_BYTES + 1)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise BenchmarkError(f"{worker.lane}: unexpected EOF (exit={worker.process.poll()})")
                for line in self._lines(worker, chunk):
                    received[worker.alias].append(self._result(worker, line, event, start, deadline))
                if len(received[worker.alias]) > expected[worker.alias]:
                    raise BenchmarkError(
                        f"{worker.lane}: extra stdout beyond the {expected[worker.lane]} expected {event} result(s)")
        return received

    def batch(self, dispatch: dict[str, int], label: str) -> dict:
        """Queue every request on both lanes BEFORE waiting for any result."""
        self.check_idle()
        start = time.perf_counter()
        for lane, count in dispatch.items():
            for _ in range(count):
                self.send(self.workers[lane], b"run\n")
        received = self._read_until(dict(dispatch), "result", start)
        wall_ms = (time.perf_counter() - start) * 1000
        expected_tokens = {lane: count * self.args.tokens for lane, count in dispatch.items()}
        requests: dict[str, list[dict]] = {}
        for lane, rows in received.items():
            if len(rows) != dispatch[lane]:
                raise BenchmarkError(f"{lane}: received {len(rows)} results, expected {dispatch[lane]}")
            if sum(row["result"]["tokens"] for row in rows) != expected_tokens[lane]:
                raise BenchmarkError(f"{lane}: token accounting mismatch")
            previous = 0.0
            lane_requests = []
            for row in rows:
                service = row["result"]["wall_ms"]
                arrival = row["arrival_ms"]
                lane_requests.append({
                    "tokens": row["result"]["tokens"], "service_ms": service, "arrival_ms": arrival,
                    # The worker runs queued requests back to back, so the wait in
                    # front of this request is the previous request's arrival time.
                    "queue_ms": max(0.0, arrival - service - previous),
                    "cpu_relative_l2": row["result"]["cpu_relative_l2"],
                    "cpu_top1_match": row["result"]["cpu_top1_match"],
                    "replay_maxdiff": row["result"]["replay_maxdiff"],
                })
                previous = arrival
            requests[lane] = lane_requests
        return {"label": label, "wall_ms": wall_ms, "requests": requests}

    def check_idle(self) -> None:
        for worker in self.workers.values():
            if worker.process.poll() is not None:
                raise BenchmarkError(f"{worker.lane}: worker exited {worker.process.returncode} while idle")
            if worker.buffer:
                raise BenchmarkError(f"{worker.lane}: unparsed stdout while idle")
        for key, _ in self.selector.select(0):
            try:
                chunk = os.read(key.fileobj.fileno(), MAX_LINE_BYTES + 1)
            except BlockingIOError:
                continue
            detail = "unsolicited stdout" if chunk else "unexpected EOF"
            raise BenchmarkError(f"{key.data.lane}: {detail} while idle")

    def shutdown(self) -> None:
        for worker in self.workers.values():
            self.send(worker, b"quit\n")
        deadline = time.perf_counter() + min(self.args.timeout, EXIT_GRACE_SECONDS)
        open_streams = set(self.workers)
        while open_streams:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise BenchmarkError("timeout waiting for worker shutdown")
            for key, _ in self.selector.select(remaining):
                worker = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), MAX_LINE_BYTES + 1)
                except BlockingIOError:
                    continue
                if chunk:
                    raise BenchmarkError(f"{worker.lane}: unexpected stdout during shutdown")
                self.selector.unregister(key.fileobj)
                open_streams.remove(worker.alias)
        for worker in self.workers.values():
            try:
                code = worker.process.wait(timeout=max(0.001, deadline - time.perf_counter()))
            except subprocess.TimeoutExpired as error:
                raise BenchmarkError(f"{worker.lane}: timeout waiting for exit") from error
            if code != 0:
                raise BenchmarkError(f"{worker.lane}: worker exited {code}")

    def abort(self) -> None:
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
                print(f"{worker.lane}: could not reap owned PID {worker.process.pid}",
                      file=sys.stderr, flush=True)

    def close(self) -> None:
        self.selector.close()
        for worker in self.workers.values():
            for stream in (worker.process.stdin, worker.process.stdout, worker.stderr):
                if stream is not None:
                    stream.close()


def pi_update(state: dict, slowdown_exact: float, slowdown_ane: float,
              kp: float, ki: float, low: float = 0.0, high: float = 1.0) -> dict:
    """Balance PI on rate slowdowns: route eligible work toward the lagging lane.

    rho is the ANE lane's share of eligible requests. If the ordinary lane is
    slower than the ANE lane the error is positive and rho grows. The integral
    is clamped for anti-windup at the extremes where one lane idles.
    """
    error = slowdown_exact - slowdown_ane
    integral = max(-2.0, min(2.0, state["integral"] + error))
    rho = min(high, max(low, state["rho"] + kp * error + ki * integral))
    return {"integral": integral, "rho": rho, "error": error, "kp": kp, "ki": ki,
            "slowdown_exact": slowdown_exact, "slowdown_ane": slowdown_ane}


def dispatch_counts(rho: float, carry: float, eligible: int) -> tuple[int, float]:
    """Deterministic dithered rounding: the running ANE share stays unbiased."""
    carry += rho * eligible
    count = min(max(int(carry), 0), eligible)
    return count, carry - count


def dispatch_for(rho: float, carry: float, args: argparse.Namespace,
                 competitor_on: bool) -> tuple[dict[str, int], float]:
    count, carry = dispatch_counts(rho, carry, args.requests_per_batch)
    dispatch = {"ane": count, args.exact_lane: args.requests_per_batch - count}
    if competitor_on:
        dispatch["competitor"] = args.competitor_requests
    return dispatch, carry


def lane_slowdowns(batch: dict, references: dict) -> dict:
    """Solo-rate / batch-rate per lane; a lane with no requests has slowdown 0."""
    result = {}
    for lane, reference in references.items():
        requests = batch["requests"].get(lane, [])
        tokens = sum(item["tokens"] for item in requests)
        realized = tokens * 1000 / batch["wall_ms"] if tokens else math.inf
        result[lane] = 0.0 if not tokens else reference["rate_tokens_per_second"] / realized
    return result


def summarize_mode(batches: list[dict], references: dict, dispatches: list[dict],
                   study_lanes: tuple[str, ...] = ("ane", "metal")) -> dict:
    tokens = sum(sum(item["tokens"] for lane in study_lanes
                     for item in batch["requests"].get(lane, [])) for batch in batches)
    competitor_requests = sum(len(batch["requests"].get("competitor", [])) for batch in batches)
    seconds = sum(batch["wall_ms"] for batch in batches) / 1000
    lanes = {}
    for lane in references:
        service = [item["service_ms"] for batch in batches for item in batch["requests"].get(lane, [])]
        queue = [item["queue_ms"] for batch in batches for item in batch["requests"].get(lane, [])]
        rate_slowdowns = [lane_slowdowns(batch, references)[lane]
                          for batch in batches if batch["requests"].get(lane)]
        lanes[lane] = {
            "request_count": len(service),
            "mean_service_ms": statistics.mean(service) if service else None,
            "mean_queue_ms": statistics.mean(queue) if queue else None,
            "rate_slowdown_mean": statistics.mean(rate_slowdowns) if rate_slowdowns else None,
            "rate_slowdown_max": max(rate_slowdowns) if rate_slowdowns else None,
        }
    return {
        "batch_count": len(batches), "completed_tokens": tokens, "total_wall_seconds": seconds,
        "batch_wall_ms": [batch["wall_ms"] for batch in batches],
        "batch_lane_service_ms": {
            lane: [statistics.mean([item["service_ms"] for item in batch["requests"].get(lane, [])])
                   if batch["requests"].get(lane) else None for batch in batches]
            for lane in study_lanes},
        "aggregate_tokens_per_second": tokens / seconds if seconds else None,
        "batch_tokens_per_second": [
            sum(item["tokens"] for lane in study_lanes for item in batch["requests"].get(lane, []))
            * 1000 / batch["wall_ms"] for batch in batches],
        "lanes": lanes, "dispatches": dispatches,
        "competitor_requests_drained": competitor_requests,
    }


def run_static(pool: Pool, args: argparse.Namespace, references: dict,
               study_lanes: tuple[str, ...], competitor: bool = False) -> dict:
    points = {}
    for rho in args.ratios:
        carry, batches, dispatches = 0.0, [], []
        for index in range(args.rounds):
            count, carry = dispatch_counts(rho, carry, args.requests_per_batch)
            dispatch = {"ane": count, args.exact_lane: args.requests_per_batch - count}
            if competitor:
                dispatch["competitor"] = args.competitor_requests
            label = f"static rho={rho:.3g} batch {index + 1}/{args.rounds}"
            print(label, file=sys.stderr, flush=True)
            batch = pool.batch(dispatch, label)
            batches.append(batch)
            dispatches.append({"rho": rho, "dither_carry": carry, "ane": count,
                               args.exact_lane: args.requests_per_batch - count})
        points[rho] = summarize_mode(batches, references, dispatches, study_lanes)
    best_rho, best = max(points.items(), key=lambda item: item[1]["aggregate_tokens_per_second"])
    return {"sweep": {repr(rho): value for rho, value in points.items()},
            "best_static_ratio": best_rho,
            "best_static_aggregate_tokens_per_second": best["aggregate_tokens_per_second"]}


def run_pi(pool: Pool, args: argparse.Namespace, references: dict,
           study_lanes: tuple[str, ...]) -> dict:
    rho, integral, carry = 0.5, 0.0, 0.0
    batches, trace = [], []
    for index in range(args.pi_rounds):
        count, carry = dispatch_counts(rho, carry, args.requests_per_batch)
        dispatch = {"ane": count, args.exact_lane: args.requests_per_batch - count}
        label = f"pi batch {index + 1}/{args.pi_rounds} rho={rho:.3f}"
        print(label, file=sys.stderr, flush=True)
        batch = pool.batch(dispatch, label)
        slow = lane_slowdowns(batch, references)
        state = pi_update({"rho": rho, "integral": integral},
                          slow[args.exact_lane], slow["ane"], args.kp, args.ki)
        batches.append(batch)
        trace.append({"batch": index + 1, "rho_before": rho, "integral_before": integral,
                      "k_ane": count, "slowdown_exact": slow[args.exact_lane],
                      "slowdown_ane": slow["ane"], "error": state["error"],
                      "rho_after": state["rho"], "integral_after": state["integral"],
                      "dither_carry": carry})
        rho, integral = state["rho"], state["integral"]
    summary = summarize_mode(batches, references, trace, study_lanes)
    last = summary["batch_tokens_per_second"][-min(3, len(batches)):]
    summary["tail_mean_tokens_per_second"] = statistics.mean(last)
    return {"trace": trace, **summary}


def build_schedule(args: argparse.Namespace, rounds: int, salt: int) -> list[bool]:
    """Seeded Bernoulli competitor schedule; identical across controllers."""
    rng = random.Random(args.seed * 1000003 + salt)
    return [rng.random() < args.competitor_probability for _ in range(rounds)]


class FixedController:
    name = "static"
    def __init__(self, rho: float): self.rho = rho
    def next(self, competitor_on: bool) -> float: return self.rho
    def observe(self, batch, competitor_on, references, study_lanes): return {}


class OmniscientController(FixedController):
    """Upper reference: switches rho with the KNOWN schedule (not achievable blind)."""
    name = "omniscient"
    def __init__(self, quiet: float, loaded: float):
        super().__init__(quiet)
        self.loaded = loaded
    def next(self, competitor_on: bool) -> float:
        return self.loaded if competitor_on else self.rho


class BalancePiController:
    name = "balance_pi"
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = {"rho": 0.5, "integral": 0.0}
    def next(self, competitor_on: bool) -> float: return self.state["rho"]
    def observe(self, batch, competitor_on, references, study_lanes):
        slow = lane_slowdowns(batch, references)
        state = pi_update(self.state, slow[self.args.exact_lane], slow["ane"],
                          self.args.kp, self.args.ki)
        self.state = state
        return {"error": state["error"], "rho_after": state["rho"],
                "integral_after": state["integral"]}


class ExtremumSeekingController:
    """Pair-based Kiefer-Wolfowitz on per-token batch wall time.

    Alternating +/- delta arms (drift cancellation), EWMA relative-noise gate,
    cumulative direction-flip step damping: the lessons that fixed the frozen
    split and limit cycles in the GPU/CPU blend controller.
    """
    name = "extremum_seeking"
    def __init__(self, delta: float, step: float, gate_factor: float = 1.5, rho0: float = 0.5):
        self.center, self.delta, self.step = rho0, delta, step
        self.gate_factor = gate_factor
        self.damp, self.last_sign, self.sigma = 1.0, 0, None
        self.pending = None
        self.direction = 1
    def next(self, competitor_on: bool) -> float:
        self.probe_sign = self.direction
        return min(1.0, max(0.0, self.center + self.probe_sign * self.delta))
    def observe(self, batch, competitor_on, references, study_lanes):
        tokens = sum(item["tokens"] for lane in study_lanes
                     for item in batch["requests"].get(lane, []))
        per_token = batch["wall_ms"] / max(1, tokens)
        if self.pending is None:
            self.pending = {"sign": self.probe_sign, "per_token": per_token}
            self.direction = -self.direction
            return {"probe": "open"}
        pair = [self.pending, {"sign": self.probe_sign, "per_token": per_token}]
        self.pending = None
        wall_plus = next(a["per_token"] for a in pair if a["sign"] > 0)
        wall_minus = next(a["per_token"] for a in pair if a["sign"] < 0)
        mean = (wall_plus + wall_minus) / 2
        diff = wall_plus - wall_minus
        noise = abs(diff) / mean
        self.sigma = noise if self.sigma is None else 0.3 * noise + 0.7 * self.sigma
        gate = self.gate_factor * self.sigma * math.sqrt(2.0)
        move_sign = -1 if diff > 0 else 1  # descend per-token wall
        if self.last_sign and move_sign != self.last_sign:
            self.damp *= 0.5
        elif self.last_sign:
            self.damp = min(1.0, self.damp * 2.0)
        self.last_sign = move_sign
        moved = abs(diff) > gate * mean
        if moved:
            self.center = min(1.0, max(0.0, self.center + move_sign * self.step * self.damp))
        return {"per_token_ms": pair[1]["per_token"], "pair_diff_ms": diff,
                "noise": noise, "gate": gate * mean, "moved": moved,
                "center_after": self.center, "damp": self.damp}


def run_scheduled(pool: Pool, args: argparse.Namespace, references: dict,
                  schedule: list[bool], controller, study_lanes: tuple[str, ...]) -> dict:
    batches, rows, dispatches = [], [], []
    carry = 0.0
    for index, competitor_on in enumerate(schedule):
        rho = controller.next(competitor_on)
        dispatch, carry = dispatch_for(rho, carry, args, competitor_on)
        label = (f"{controller.name} batch {index + 1}/{len(schedule)} "
                 f"rho={rho:.3f} competitor={int(competitor_on)}")
        print(label, file=sys.stderr, flush=True)
        batch = pool.batch(dispatch, label)
        batch["dispatch"] = {lane: dispatch[lane] for lane in study_lanes}
        row = controller.observe(batch, competitor_on, references, study_lanes)
        batches.append(batch)
        dispatches.append({"rho_used": rho, "competitor_on": competitor_on, **dispatch})
        rows.append({"batch": index + 1, "rho_used": rho, "competitor_on": competitor_on,
                     "k_ane": dispatch["ane"], **row})
    return {"trace": rows, **summarize_mode(batches, references, dispatches, study_lanes)}



def study(args: argparse.Namespace) -> dict:
    pool = Pool(args)
    study_lanes = (args.exact_lane, "ane")
    references: dict[str, dict] = {}
    try:
        for lane in (args.exact_lane, "ane"):
            print(f"initializing {lane} (sequential load/reference/warmup)", file=sys.stderr, flush=True)
            ready = pool.start(lane)
            solo = pool.batch({lane: 1}, f"{lane} solo reference")
            references[lane] = {
                "rate_tokens_per_second": args.tokens * 1000 / solo["wall_ms"],
                "solo_wall_ms": solo["wall_ms"],
                "handshake_ms": ready["handshake_ms"], "warmup_message": ready["message"],
            }
        if args.competitor_probability > 0:
            print("initializing competitor (metal co-tenant)", file=sys.stderr, flush=True)
            pool.start("competitor", protocol_lane=args.exact_lane)
            pool.batch({"competitor": 1}, "competitor solo reference")
        static_quiet = run_static(pool, args, references, study_lanes)
        quiet_pi = run_pi(pool, args, references, study_lanes)
        stochastic = None
        if args.competitor_probability > 0:
            static_loaded = run_static(pool, args, references, study_lanes, competitor=True)
            schedule = build_schedule(args, args.pi_rounds, salt=1)
            best_quiet = static_quiet["best_static_ratio"]
            best_loaded = static_loaded["best_static_ratio"]
            traces = {
                "static_best_quiet": run_scheduled(pool, args, references, schedule,
                                                   FixedController(best_quiet), study_lanes),
                "omniscient_switch": run_scheduled(pool, args, references, schedule,
                                                   OmniscientController(best_quiet, best_loaded), study_lanes),
                "balance_pi": run_scheduled(pool, args, references, schedule,
                                            BalancePiController(args), study_lanes),
                "extremum_seeking": run_scheduled(pool, args, references, schedule,
                                                  ExtremumSeekingController(args.es_delta, args.es_step),
                                                  study_lanes),
            }
            aggregates = {name: trace["aggregate_tokens_per_second"] for name, trace in traces.items()}
            stochastic = {
                "schedule": schedule,
                "competitor_probability": args.competitor_probability,
                "competitor_requests_per_batch": args.competitor_requests,
                "best_ratio_quiet": best_quiet, "best_ratio_loaded": best_loaded,
                "static_loaded_sweep": static_loaded["sweep"],
                "static_quiet_sweep": static_quiet["sweep"],
                "controllers": {name: {
                    "completed_tokens": trace["completed_tokens"],
                    "aggregate_tokens_per_second": trace["aggregate_tokens_per_second"],
                    "tail_mean_tokens_per_second": statistics.mean(
                        trace["batch_tokens_per_second"][-min(3, len(trace["batch_tokens_per_second"])):]),
                    "trace": trace["trace"],
                } for name, trace in traces.items()},
                "aggregate_tokens_per_second": {
                    name: trace["aggregate_tokens_per_second"] for name, trace in traces.items()},
                "oracle_note": "omniscient_switch knows the schedule and is an upper reference, not achievable blind",
            }
        pool.shutdown()
    except BaseException as error:
        pool.abort()
        tails = "\n".join(f"{lane} stderr tail:\n{worker.tail()}"
                          for lane, worker in pool.workers.items())
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            if tails:
                print(tails, file=sys.stderr, flush=True)
            raise
        import traceback
        raise BenchmarkError(f"{error}\n{tails}\n{traceback.format_exc()}") from error
    finally:
        pool.close()
    best_static = static_quiet["best_static_aggregate_tokens_per_second"]
    report = {
        "schema": SCHEMA,
        "measurement_scope": {
            "workload": "Fixed-size cache-reset prefills only; every batch queues the same study work and only the ANE routing share changes.",
            "controller": "Balance PI on per-lane rate slowdowns vs a solo warm reference; extremum seeking on per-token batch wall; not an SLA/SLO controller.",
            "regime": "Optional seeded Bernoulli metal co-tenant (competitor worker) emulates stochastic contention; same schedule across controllers.",
            "capacity": "Finite queued bursts over two/three persistent workers, not steady-state capacity; no per-operation ANE residency claim.",
            "quality": "ANE tier is fp16 with different numerics; aggregate is capacity accounting, not equivalent-quality throughput.",
            "queue": "Per-request queue_ms derives from arrival times of back-to-back worker runs; controller-clock approximation.",
        },
        "configuration": {
            "tokens": args.tokens, "requests_per_batch": args.requests_per_batch,
            "rounds": args.rounds, "ratios": args.ratios, "pi_rounds": args.pi_rounds,
            "kp": args.kp, "ki": args.ki, "seed": args.seed, "exact_lane": args.exact_lane,
            "layers": args.layers, "half": args.half, "timeout_seconds": args.timeout,
            "replay_tolerance": args.replay_tolerance,
            "competitor_probability": args.competitor_probability,
            "competitor_requests": args.competitor_requests,
            "es_delta": args.es_delta, "es_step": args.es_step,
            "model": identity(args.model), "compiled_package": identity(args.ane_path),
            "binary": identity(args.binary),
            "token_file": identity(args.token_file, hash_contents=True) if args.token_file else None,
            "exact_token_file": identity(args.exact_token_file, hash_contents=True) if args.exact_token_file else None,
            "environment_policy": "Copy os.environ after clearing every TM_* variable; add only the controlled per-lane set.",
        },
        "references": references,
        "quiet": {"static": static_quiet, "balance_pi": quiet_pi},
        "stochastic": stochastic,
        "verdict": {
            "quiet_pi_aggregate_tokens_per_second": quiet_pi["aggregate_tokens_per_second"],
            "quiet_best_static_aggregate_tokens_per_second": static_quiet["best_static_aggregate_tokens_per_second"],
            "quiet_pi_over_best_static": (quiet_pi["aggregate_tokens_per_second"]
                                          / static_quiet["best_static_aggregate_tokens_per_second"]
                                          if static_quiet["best_static_aggregate_tokens_per_second"] else None),
        },
    }
    if stochastic is not None:
        es = stochastic["aggregate_tokens_per_second"]["extremum_seeking"]
        omni = stochastic["aggregate_tokens_per_second"]["omniscient_switch"]
        static_q = stochastic["aggregate_tokens_per_second"]["static_best_quiet"]
        report["verdict"].update({
            "stochastic_aggregate_tokens_per_second": stochastic["aggregate_tokens_per_second"],
            "es_over_static_quiet": es / static_q if static_q else None,
            "es_over_omniscient": es / omni if omni else None,
            "caveat": "Single fixed-size workload, seeded schedule and short traces; a benefit here does not establish production-scheduler value.",
        })
    return report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        validate_controller_args(args)
        report = study(args)
        serialized = json.dumps(report, allow_nan=False, indent=2) + "\n"
        if args.output is not None:
            args.output = Path(os.path.abspath(args.output.expanduser()))
            with args.output.open("x", encoding="utf-8") as target:
                target.write(serialized)
        sys.stdout.write(serialized)
        sys.stdout.flush()
    except (BenchmarkError, OSError, ValueError, OverflowError) as error:
        print(f"ane-controller: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("ane-controller: interrupted", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())