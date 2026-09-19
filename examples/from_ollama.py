#!/usr/bin/env python3
"""from_ollama.py — run TensorMark behind an Ollama-compatible HTTP API.

If your tooling already speaks Ollama's `/api/generate` (LangChain, shell
pipelines, `curl`, editor plugins), point it here instead of re-learning a
new CLI. The server drives `build/llama_chat_metal --batch` as a subprocess
and translates its line protocol (PROMPT / EOT / <<<EOT>>>) into Ollama's
JSON-line streaming format.

    ./tensormark/build_llama.sh                 # once
    python3 examples/from_ollama.py \
        --model data/tinyllama/tinyllama_q40.tmq \
        --tokenizer data/tinyllama/tokenizer.model \
        --port 11435

    curl -s localhost:11435/api/generate -d '{"model":"tensormark",
        "prompt":"Name the primary colors.","stream":false}'
    # or streaming:
    curl -N localhost:11435/api/generate -d '{"prompt":"...", "stream":true}'

Differences from a real Ollama server (kept deliberately small and honest):
  * generation is greedy-deterministic, so a conversation replays byte-for-byte.
    Of Ollama's `options`, `repeat_penalty` and `repeat_last_n` ARE honoured
    (default off: 1.0). Greedy decoding loops on list-shaped answers — asked for
    the authors of "Attention Is All You Need" this engine listed eight correct
    names, invented several more, then restarted and cycled, never reaching the
    year — and a repetition penalty fixes that while costing the determinism
    nothing, because it is a pure function of the tokens already emitted.
    `temperature`, `top_k`, `top_p` and the rest remain accepted-but-ignored:
    real sampling would break the replay contract the cache reuse depends on;
  * `budget` is the per-turn token limit (default 256). A re-sent conversation is
    not re-read from the start: the turns the engine's cache already holds are
    skipped, so Ollama's `context` reuse parameter is unnecessary rather than
    ignored. `TM_SHIM_NO_REUSE=1` restores a full replay per request, and
    `TM_SHIM_DEBUG=1` reports what each request reused;
  * /api/generate, /api/chat, /api/tags, /api/embeddings and /api/embed are
    implemented;
  * a `system` message inside /api/chat is loaded as a system turn, but the
    server's own --system prompt already opened the conversation, so a client
    that sends one should start the shim with --system "".

  Structured output is the reason this shim exists. Pass a GBNF grammar as
  `"grammar"` (inline text) and every token the engine samples extends a
  string the grammar still accepts, so the reply is valid by construction
  rather than by luck — a small model can then be trusted to emit a JSON
  object or a tuple list:

    curl -s localhost:11435/api/generate -d '{
      "prompt": "Extract the relation as {\"subject\":..,\"verb\":..}",
      "grammar": "root ::= \"{\" ws \"\\\"subject\\\"\" ws \":\" ws str ws \",\" ws \"\\\"verb\\\"\" ws \":\" ws str ws \"}\"\nws ::= [ ]*\nstr ::= \"\\\"\" [a-z ]+ \"\\\"\"",
      "stream": false}'

  The grammar is switched on the long-lived engine process (`/grammar <file>`
  over its batch protocol), so a caller can use a different schema per request
  without paying a model reload. A grammar the engine rejects is reported as
  HTTP 400 with the parse error — never silently downgraded to unconstrained
  decoding, which would hand you well-formed-looking garbage.

The model is NOT bundled: convert an HF safetensors checkpoint with
`tensormark/convert_tmq.cpp` (see examples/from_llama_cpp.md), or download
one with examples/fetch_model.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "tensormark" / "build" / "llama_chat_metal"

# The same preference order the engine binary uses for its own default, so the
# shim and the CLI cannot disagree about which model "the model" is. A clone
# ships no weights at all (tensormark/data/ is gitignored), so this picks the
# best one PRESENT, and falls back to the preferred path so a missing model
# reports something a reader can act on.
DEFAULT_MODELS = (
    "qwen38-4b/qwen38_4b_q6k.tmq",
    "qwen38-4b/qwen38_4b_q40.tmq",
    "bonsai4b/bonsai4b.tmq",
    "qwen35-4b/qwen35_4b_q40.tmq",
    "tinyllama/tinyllama_q40.tmq",
    "qwen35-0.8b/qwen35_0.8b_q40.tmq",
)


def default_model() -> Path:
    for rel in DEFAULT_MODELS:
        p = REPO / "tensormark" / "data" / rel
        if p.exists():
            return p
    return REPO / "tensormark" / "data" / DEFAULT_MODELS[0]


def default_tokenizer(model: Path) -> Path:
    """The tokenizer that belongs to `model`, by the engine's own rule.

    BPE sidecars beside the weights when they are there, else a sentencepiece
    tokenizer.model in that directory. Getting this wrong is not a loud failure —
    it is the wrong tokenizer silently applied to the right weights.
    """
    stem = model.name[:-4] if model.name.endswith(".tmq") else model.name
    bpe = model.parent / (stem + ".tokenizer.vocab.json")
    return bpe if bpe.exists() else model.parent / "tokenizer.model"


# How long the shim waits for the engine to say something before calling the
# protocol dead. Generous on purpose: it is a deadlock breaker, not a generation
# limit, so a legitimately long completion never trips it.
READ_TIMEOUT_S = 600.0

# ---- idle unload -------------------------------------------------------------
# The engine holds its weights in unified memory for as long as it is resident
# (672 MB for Qwen3.5-0.8B Q4_0, ~2 GB for the 4B), and a serving process that
# never lets go keeps them for the life of the machine. Ollama's contract is that a
# model is unloaded after a period of inactivity, overridable per request with
# `keep_alive`; this shim follows it by REAPING the engine — its memory comes back
# only when the process exits, so an unload is a close, not a cache eviction.
DEFAULT_IDLE_TTL_S = 300.0
IDLE_REAPER_TICK_S = 0.5
# Lane-pool controller. Two lanes are not free: a second engine costs a second set
# of activations and KV, and each lane keeps ~2/3 of its solo rate while both are
# busy (measured, see LanePool). The controller therefore keeps asking whether the
# extra lane still pays — LANE_KEEP_FRACTION is how much the pool's aggregate must
# beat the best solo lane, LANE_REPROBE_S is how long a parked lane stays parked
# before it is tried again. Same dither/measure/retreat shape as the prefill ESC,
# with a lane instead of a row count as the actuator.
#
# Overridable on purpose, so the retreat can be TESTED rather than assumed: the lane
# gate sets TM_LANE_KEEP_FRACTION above any reachable aggregate, which forces the
# park branch to fire on a quiet machine. A controller nobody has watched retreat is
# a controller whose retreat path is unverified.
def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


CONTROL_TICK_S = _envf("TM_LANE_TICK_S", 5.0)
LANE_REPROBE_S = _envf("TM_LANE_REPROBE_S", 60.0)
LANE_KEEP_FRACTION = _envf("TM_LANE_KEEP_FRACTION", 1.05)


def parse_keep_alive(v) -> float | None:
    """Ollama's `keep_alive` -> seconds, or None when the request has no opinion.

    Ollama accepts a number of seconds or a duration string ("30s", "5m", "1h30m").
    0 means unload as soon as the request finishes; a negative value means never
    unload. Anything else raises, so a typo is a 400 and not an ignored field.
    """
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"keep_alive: {v!r} is not a duration")
    if isinstance(v, (int, float)):
        secs = float(v)
    else:
        s = v.strip().lower()
        if s in ("", "default"):
            return None
        neg, s = s.startswith("-"), s.lstrip("+-")
        units = {"s": 1.0, "m": 60.0, "h": 3600.0}
        total, num = 0.0, ""
        for ch in s:
            if ch.isdigit() or ch == ".":
                num += ch
            elif ch in units and num:
                total += float(num) * units[ch]
                num = ""
            else:
                raise ValueError(f"keep_alive: {v!r} is not a duration")
        total += float(num) if num else 0.0
        secs = -total if neg else total
    return float("inf") if secs < 0 else secs


LANE_ENV = {"gpu": {"TM_DECODE_GPU": "1"}, "cpu": {"TM_DECODE_GPU": "0"}}


class Lane:
    """One engine process on one compute lane, with its own KV cache.

    `rate` is the measurement the controller acts on: an EWMA of the tokens/s this
    lane achieves while it is being used, seeded from the paired measurement below
    so the first request is not scheduled blind. It is a service-rate estimate, not
    a benchmark reading — under a pool it reflects the contended rate, which is
    exactly what the park decision needs.
    """

    def __init__(self, name: str, state: dict):
        self.name = name              # "gpu" | "cpu" | "auto"
        self.state = state
        self.session = None
        self.lock = threading.Lock()  # serialises use of THIS engine
        self.inflight = 0             # requests admitted and not yet answered
        self.parked = False           # the controller's retreat
        self.parked_at = 0.0
        self.last_used = time.monotonic()
        self.deadline = float("inf")
        self.tokens = 0.0             # total measured tokens, for the EWMA
        self.seconds = 0.0
        self.served = 0
        self.rate = {"gpu": 76.97, "cpu": 62.07}.get(name, 62.07)

    # --- lifecycle ---------------------------------------------------------
    def ensure(self):
        """The resident engine on this lane, loading it if the reaper let it go."""
        if self.session is not None:
            return self.session
        t0 = time.monotonic()
        st = self.state
        # The lane's own env, plus the context window when the caller sized one:
        # TM_CTX is the engine's override, and it has to be set on the PROCESS, so
        # it belongs here rather than on any single request.
        env = dict(LANE_ENV.get(self.name) or {})
        if st.get("ctx"):
            env["TM_CTX"] = str(st["ctx"])
        self.session = LlamaChatSession(st["model"], st["tokenizer"], st["system"],
                                        greedy=st["greedy"], lane_env=env or None)
        self.arm(None)
        print(f"[shim] lane {self.name}: loaded {st['model_name']} in "
              f"{time.monotonic() - t0:.1f}s ({st['model_size'] / 1e6:.0f} MB)", flush=True)
        return self.session

    def drop(self, reason: str) -> None:
        """Release this lane's engine, and with it its copy of the model.

        Takes the lane's own lock: a request may be mid-flight on this engine, and
        pulling its session out from under it is the one thing eviction must never
        do. No caller may hold the STATE lock here — a request holds the lane lock
        and takes the state lock to report what it cost, so holding state while
        waiting for a lane deadlocks. Acquisition order is lane-then-state, never
        the reverse.
        """
        with self.lock:
            session, self.session = self.session, None
            if session is None:
                return
            held = time.monotonic() - self.last_used
            session.close()             # /quit, then SIGKILL if it will not go
            self.deadline = float("inf")
            print(f"[shim] lane {self.name}: unloaded {self.state['model_name']} "
                  f"({reason}; unused for {held:.1f}s)", flush=True)

    def arm(self, keep: float | None) -> None:
        ttl = self.state["idle_ttl"] if keep is None else keep
        self.last_used = time.monotonic()
        self.deadline = self.last_used + ttl if ttl > 0 else float("inf")

    def observe(self, tokens: float, seconds: float) -> None:
        self.served += 1
        if seconds <= 0 or tokens <= 0:
            # An embedding generates nothing, so it carries no service-rate sample.
            # Folding a zero in would decay the lane's rate toward zero and park a
            # lane that is perfectly healthy.
            return
        self.tokens += tokens
        self.seconds += seconds
        sample = tokens / seconds
        # alpha 0.3: fast enough to react to a contention episode, slow enough that
        # one slow request does not park a lane.
        self.rate = 0.7 * self.rate + 0.3 * sample


class LanePool:
    """The lanes this server may use, and which one takes the next request.

    Why a pool at all: an engine process serves one request at a time
    (LlamaChatSession's contract), so using both compute units means two
    processes, each pinned to a lane (`TM_DECODE_GPU`). This is the decode-side
    counterpart of the prefill row split (`HybridSplit`): that one divides ROWS of
    a single forward and is meaningless at T=1, so the actuator here is which lane
    serves which request.

    Measured on this machine (M1, 8 GB, 2026-09-15, one window, load 2.7-3.2,
    tinyllama_q40, 256 tokens per leg): GPU chain alone 76.97 t/s, CPU decode alone
    62.07 t/s, BOTH AT ONCE 54.53 + 41.22 = 95.75 t/s. Each lane keeps ~2/3 of its
    solo rate, the pool is 1.24x the better single lane and 1.54x the CPU lane, and
    both greedy streams stayed byte-identical to their solo runs.

    Two lanes are not always right: they cost a second set of activations and KV,
    and under outside load the pair can end up slower than one lane. Hence the
    controller below, which parks a lane that stops earning and re-probes it later
    — the same measure-then-retreat discipline as the prefill split.
    """

    def __init__(self, state: dict, names: list[str]):
        self.state = state
        self.lanes = [Lane(n, state) for n in names]
        self.best_solo = max(ln.rate for ln in self.lanes)

    # --- scheduling --------------------------------------------------------
    def admit(self, est_tokens: float) -> Lane:
        """Reserve the lane with the earliest predicted finish. Caller holds the state lock."""
        live = [ln for ln in self.lanes if not ln.parked] or self.lanes
        def finish(ln: Lane) -> float:
            # Least time to finish, not least queue: a slow lane with one request
            # queued can still be the better choice, and a fast lane behind two
            # requests is not. `est_tokens` is the request's own budget, which is
            # the only length hint a caller gives us.
            return (ln.inflight * est_tokens + est_tokens) / max(ln.rate, 1e-6)
        lane = min(live, key=finish)
        lane.inflight += 1
        return lane

    def settle(self, lane: Lane, tokens: float, seconds: float) -> None:
        """Record what the request cost and free the lane. Caller holds the state lock."""
        lane.inflight = max(0, lane.inflight - 1)
        lane.observe(tokens, seconds)
        # The one-lane reference the controller compares against: a lane's own best
        # uncontended reading. Seeded from the same paired measurement as `rate`.
        if lane.inflight == 0 and len(self.lanes) == 1:
            self.best_solo = max(self.best_solo, lane.rate)

    def served_total(self) -> int:
        """Requests this pool has answered, across lanes.

        The judgement waits for evidence the pool is in USE, and that is measured
        across lanes rather than per lane on purpose: when requests are short enough
        that a lane always drains before the next arrives, the scheduler keeps giving
        them to the faster lane and the extra lane serves nothing — which is the
        correct decision, and would leave a per-lane test waiting forever. The extra
        lane's rate stays at its seed while it is unused, which is optimistic, so an
        unused lane is KEPT rather than parked: keeping is the status quo.
        """
        return sum(ln.served for ln in self.lanes)

    def judge(self) -> None:
        """Park an extra lane when the pool stops beating one lane; re-probe it later.

        The test is the one the measurement supports: the pool's aggregate service
        rate must be worth more than the best single lane by LANE_KEEP_FRACTION.

        The clock is the parked lane's own `parked_at`, and that is deliberate. A
        first version gated on a separate `next_probe` deadline initialized to
        infinity, which made the gate `now < next_probe` true forever: the pool could
        never retreat, and nothing on a healthy machine would ever have shown it. A
        parked lane measures nothing, so the re-probe is to UN-park it and let it
        serve again — fresh samples, then the next judgement decides. The retreat is a
        probe, not a verdict.
        """
        if len(self.lanes) < 2:
            return
        now = time.monotonic()
        extras = self.lanes[1:]
        for e in extras:
            if e.parked:
                if now - e.parked_at >= LANE_REPROBE_S:
                    e.parked = False
                    e.served = 0        # judge the probe on fresh samples, not the old ones
                    print(f"[shim] lane {e.name}: re-probed after "
                          f"{LANE_REPROBE_S:.0f}s parked", flush=True)
                continue                # a parked lane is not judged again while parked
        if any(e.parked for e in extras):
            return                      # wait for the re-probe to produce evidence
        if os.environ.get("TM_LANE_DEBUG"):
            print(f"[shim] judge: served={self.served_total()} "
                  f"aggregate={sum(ln.rate for ln in self.lanes):.1f} "
                  f"best_solo={self.best_solo:.1f} keep={LANE_KEEP_FRACTION:.2f}",
                  flush=True)
        if self.served_total() < 4:
            return                      # wait for evidence that the pool is in use
        aggregate = sum(ln.rate for ln in self.lanes)
        if aggregate >= LANE_KEEP_FRACTION * self.best_solo:
            return
        for e in extras:
            e.parked = True
            e.parked_at = now
            print(f"[shim] lane {e.name}: parked (pool {aggregate:.1f} t/s does not "
                  f"beat best solo {self.best_solo:.1f} by "
                  f"{LANE_KEEP_FRACTION:.2f}x)", flush=True)


def arm_idle(state: dict, keep: float | None, lane=None) -> None:
    """Set when a lane becomes eligible for unloading. Caller holds the state lock."""
    for ln in ([lane] if lane is not None else state["pool"].lanes):
        ln.arm(keep)


def expires_at(state: dict, lane=None) -> str | None:
    """When a lane is due to go, for /api/ps. None = not resident, or never."""
    lanes = [lane] if lane is not None else state["pool"].lanes
    for ln in lanes:
        if ln.session is not None and ln.deadline != float("inf"):
            left = ln.deadline - time.monotonic()
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + left))
    return None


def idle_reaper(state: dict) -> None:
    """Unload each lane once nothing has used it for `idle_ttl` seconds.

    Eviction takes the LANE lock, which a request holds for its whole duration, so
    a request in progress can never have its engine pulled out from under it — the
    same guarantee as before, now per lane instead of per server.
    """
    while not state["stop"].wait(IDLE_REAPER_TICK_S):
        # Collect under the lock, drop outside it: Lane.drop waits for the lane, and
        # a request holds its lane lock while taking the state lock to report cost.
        with state["lock"]:
            now = time.monotonic()
            due = [ln for ln in state["pool"].lanes
                   if ln.session is not None and now >= ln.deadline]
        for ln in due:
            ln.drop(f"idle over {state['idle_ttl']:.0f}s")


def lane_controller(state: dict) -> None:
    """Decide, every CONTROL_TICK_S, whether the extra lanes are still earning."""
    while not state["stop"].wait(CONTROL_TICK_S):
        with state["lock"]:
            state["pool"].judge()


def run_on_lane(state: dict, est_tokens: float, fn, keep: float | None = None):
    """Serve one request on the best lane. `fn(session)` -> (result, tokens).

    The state lock is taken only to choose a lane and to record what the request
    cost — never while an engine is running — so two requests on two lanes overlap
    instead of queueing, which is the whole point of the pool. Returns the lane
    used as well, so the caller can act on it afterwards (keep_alive 0 unloads).
    """
    with state["lock"]:
        lane = state["pool"].admit(est_tokens)
    t0 = time.monotonic()
    tokens, result = 0.0, None
    try:
        with lane.lock:
            result, tokens = fn(lane.ensure())
    finally:
        with state["lock"]:
            state["pool"].settle(lane, float(tokens or 0.0), time.monotonic() - t0)
            arm_idle(state, keep, lane)
    return lane, result


# The JSON Schema -> GBNF compiler ships
# The JSON Schema -> GBNF compiler ships with the engine (tensormark/), not with
# the shim, and the repo root is not on sys.path when this runs as a script.
sys.path.insert(0, str(REPO / "tensormark"))
from jsonschema_to_gbnf import Unsupported, compile_schema   # noqa: E402


class GrammarError(RuntimeError):
    """The engine rejected a grammar; the caller's schema is at fault."""


class ContextTooSmall(RuntimeError):
    """A history turn does not fit the context the engine was started with.

    Its own type because it must be answered differently from an engine failure:
    nothing crashed, the caller sent a conversation larger than the window. A
    client reads 400 as "fix the request" and 500 as "the server is broken", and
    the distinction was invisible — a 2 849-token turn against the default 2048
    context came back as HTTP 500 "would not load the history", which reads as a
    crash and names neither the limit nor the knob.
    """


class LlamaChatSession:
    """One --batch subprocess; one request at a time (the engine is single-tenant)."""

    def __init__(self, model: str, tokenizer: str, system: str, greedy: bool = True,
                 lane_env: dict | None = None):
        # Raised, never sys.exit: a session is now created and destroyed repeatedly
        # (the idle reaper unloads the model, the next request loads it again), and
        # SystemExit on a handler thread ends that thread silently — the client gets
        # an empty body and nothing in the log says why.
        if not BIN.exists():
            raise RuntimeError(f"missing {BIN}\nrun ./tensormark/build_llama.sh first")
        self.proc = None
        # The engine writes its diagnostics (grammar banners, parse errors) to
        # stderr, which is NOT the protocol channel. Keep it in a file so a
        # rejected grammar can be reported with the reason the engine gave.
        self._errlog = tempfile.NamedTemporaryFile(
            prefix="tm-shim-stderr-", suffix=".log", delete=False)
        try:
            self.proc = subprocess.Popen(
                # NOT --quiet: this driver sends slash-commands, and their `ACK`
                # lines are how it knows a command landed — `--quiet` suppresses
                # them. Instead of --quiet the protocol is read strictly: completions
                # are taken up to `<<<EOT>>>`, commands take exactly one `ACK` line
                # followed by `PROMPT`, and anything else is a loud desync error.
                [str(BIN), "-m", model, "-t", tokenizer, "-s", system, "--batch"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errlog,
                # errors="replace": the engine's output is a byte stream, and a
                # truncated multi-byte codepoint (the engine holds those back now,
                # but a byte-level tokenizer can still produce one) must degrade to
                # U+FFFD rather than raise UnicodeDecodeError, which kills the
                # request and returns an empty body with a traceback in the log.
                text=True, errors="replace", cwd=str(REPO),
                # Which compute lane this engine takes. The engine's own auto
                # policy picks one; a pool pins each process so two of them can
                # hold two different lanes instead of racing for the same one.
                env={**os.environ, **(lane_env or {})},
            )
            self._grammar_key: str | None = None     # hash of the grammar in effect
            self._grammar_dir: str | None = None
            # The (role, text) turns the engine's cache represents, oldest first — the
            # shim's model of engine state. It advances only where the shim observed that
            # state (note_loaded, canonicalize_turn), never where it assumed one: reusing
            # a prefix the cache does not actually hold is a wrong answer, not a slow one.
            self.loaded: list[tuple[str, str]] = []
            # Only used to stay clear of the context edge when deciding what to reuse. The
            # lane env is what the ENGINE actually receives, so it is the authority here;
            # reading os.environ alone would miss a shim started with --ctx.
            self.ctx = int((lane_env or {}).get("TM_CTX")
                           or os.environ.get("TM_CTX", 2048))
            # The engine's stdout is drained by its own thread into a queue, never
            # read inline by a request handler: an inline read blocks that handler
            # forever when the engine stops answering, with no way to tell a slow
            # generation from a dead protocol.
            self._q: queue.Queue[str | None] = queue.Queue()
            # Read through an alias held by the pump thread: then closing the session
            # at ANY moment cannot leave that thread reading a closed file — it keeps
            # its own reference and simply sees EOF when the child goes.
            self._out = self.proc.stdout
            threading.Thread(target=self._pump, args=(self._out,), daemon=True).start()
            banner = self._readline()                # "ACK ready"
            if banner != "ACK ready":
                # Include the engine's stderr: without it this failure reads as an
                # opaque empty banner, and the reason (a bad path, a config the loader
                # rejected) is in a temp file nobody looks at.
                detail = Path(self._errlog.name).read_text(errors="replace").strip()
                raise RuntimeError(f"unexpected engine banner: {banner!r}\n"
                                   f"engine stderr: {detail[-800:]}")
            ready = self._readline()                 # then "PROMPT"
            if ready != "PROMPT":
                raise RuntimeError(f"engine did not signal readiness: {ready!r}")
            # Invariant from here: the engine has printed PROMPT and is blocked
            # reading message lines — every complete() must leave the same state.
            # /greedy TOGGLES, so what it does depends on the engine's own
            # default — and that default is now greedy (private 21a8683).
            # Sending it blindly, as this did while the default was sampling,
            # turns greedy OFF and serves SAMPLED replies: not a subtle
            # difference, it breaks the byte-identical replay this shim
            # documents and that the cache-reuse gate asserts. Drive it until
            # the engine REPORTS the state we need — the invariant is the
            # state, not the command.
            self._ensure_greedy(greedy)
            self._budget = 1024                      # the engine's own starting budget
        except BaseException:
            self.close()      # never leave a half-built session: no orphan, no temp file
            raise

    def _pump(self, out) -> None:
        """Drain the engine's stdout on its own thread into a queue."""
        for line in out:                       # blocks; ends at EOF
            self._q.put(line.rstrip("\n"))
        self._q.put(None)                      # EOF sentinel

    def _readline(self) -> str:
        """One protocol line, or a loud error. Never an unbounded wait."""
        try:
            line = self._q.get(timeout=READ_TIMEOUT_S)
        except queue.Empty:
            raise RuntimeError(
                f"engine silent for {READ_TIMEOUT_S:.0f}s — the protocol is dead, "
                f"not slow") from None
        if line is None:
            raise RuntimeError(
                f"engine closed stdout (crash?) — engine stderr: "
                f"{self.engine_stderr()[-300:]}")
        return line

    def command(self, cmd: str) -> str:
        """Send a slash-command; return its ACK payload (protocol-stable).

        The engine answers a command with exactly one `ACK <what>` line and then
        re-arms with `PROMPT`. Anything else means the stream is out of step,
        which must be a loud failure: a desynchronised driver silently reads the
        next request's answer as this one's.
        """
        p = self.proc.stdin
        p.write(cmd + "\nEOT\n")
        p.flush()
        ack = self._readline()
        if not ack.startswith("ACK "):
            raise RuntimeError(f"engine did not acknowledge {cmd!r}: {ack!r}")
        if self._readline() != "PROMPT":
            raise RuntimeError(f"engine out of sync after {cmd!r}")
        return ack[4:]

    def _ensure_greedy(self, want: bool) -> None:
        """Leave the engine decoding greedily iff `want`, whatever its default.

        Written against the STATE rather than the command because /greedy is a
        toggle: whether it enables or disables depends on how the engine was
        started, and that starting default has already changed once under this
        code (21a8683), which is how the served path silently became sampled.
        Two attempts cover the whole space, since one flip is all that
        separates the two states.
        """
        for _ in range(2):
            ack = self.command("/greedy")
            if ("greedy=on" in ack) == want:
                return
        raise RuntimeError(f"engine would not settle greedy={want}: {ack!r}")
    def engine_stderr(self, limit: int = 800) -> str:
        # Read by PATH, not through the file object: this is called after a failure,
        # and a failure may already have closed the session. Reading a closed object
        # turns the one useful diagnostic into a ValueError.
        try:
            return Path(self._errlog.name).read_text(errors="replace")[-limit:].strip()
        except OSError:
            return ""

    def set_grammar(self, gbnf: str | None) -> None:
        """Constrain decoding to `gbnf`, or clear the constraint with None.

        The grammar is compiled inside the engine, which reports a rejected one
        as `ACK grammar=FAILED` rather than falling back to unconstrained
        decoding — a caller that asked for a schema must never get
        well-formed-looking garbage back. Switching costs a grammar parse, not a
        model reload, because the engine keeps its vocabulary tables.
        """
        if gbnf is None:
            if self._grammar_key is not None:
                self.command("/grammar off")
                self._grammar_key = None
            return
        key = hashlib.sha256(gbnf.encode()).hexdigest()[:16]
        if key == self._grammar_key:
            return                                     # already in effect
        if self._grammar_dir is None:
            self._grammar_dir = tempfile.mkdtemp(prefix="tm-grammar-")
        path = Path(self._grammar_dir) / f"{key}.gbnf"
        path.write_text(gbnf)
        if self.command(f"/grammar {path}").startswith("grammar=FAILED"):
            raise GrammarError(f"engine rejected the grammar: {self.engine_stderr()}")
        self._grammar_key = key

    def embed(self, text: str) -> list[float]:
        """L2-normalised pooled embedding of `text`, via the engine's /embed.

        Unit-norm, so cosine similarity is a dot product. Raw text: a chat
        template is a generation concern.
        """
        payload = self.command("/embed " + text.replace("\n", " "))
        if not payload.startswith("embed="):
            raise RuntimeError(f"engine did not return an embedding: {payload[:80]!r}")
        return json.loads(payload[len("embed="):])["embedding"]

    # NOTE: this class is a SECOND implementation of the engine protocol —
    # tensormark/tmchat.py::Engine is the other, and it drives the same
    # subprocess the same way. A new protocol command has to be added to both;
    # forgetting is silent here and surfaces only as an AttributeError at
    # request time, which is exactly how the /embed command first failed.
    def set_budget(self, tokens: int) -> None:
        """Apply the per-turn budget and leave it in effect.

        `complete` used to take this argument and drop it, so every request ran
        at the engine's own default (1024 tokens) whatever the caller asked for.
        Inaudible on a model that stops at EOS; ruinous on one that does not.
        """
        tokens = max(1, int(tokens))
        if tokens == self._budget:
            return
        self.command(f"/n {tokens}")
        self._budget = tokens

    def set_repeat(self, penalty: float, window: int = 64) -> None:
        """Arm the engine's repetition penalty for this session.

        The served path runs greedy (`/greedy`, above) and greedy decoding has one
        signature failure: a list-shaped answer enters a cycle and repeats it.
        Observed live on qwen38_4b_q40 — asked for the authors of "Attention Is All
        You Need" it gave eight correct names, drifted into inventions, then
        restarted the list and looped, never reaching the year. At penalty 1.15 the
        same prompt answers Vaswani/Shazeer/Parmar, notes the list has 13 researchers
        rather than inventing them, and gives 2017.

        The penalty is a DETERMINISTIC function of the emitted tokens, so this does
        not disturb the greedy contract or the byte-identical replay the cache-reuse
        path is built on — which is why it is a penalty and not sampling.
        """
        if penalty <= 1.0:
            return
        self.command(f"/repeat {penalty} {window}")

    def prefill(self, role: str, text: str) -> None:
        """Load one templated turn into the KV cache, generating nothing."""
        ack = self.command("/prefill " + role + " " + text.replace("\n", " "))
        if ack != f"prefill={role}":
            raise ContextTooSmall(
                f"this conversation does not fit the engine's context window "
                f"(engine said {ack!r}): a history turn is longer than the window "
                f"the engine was started with. Raise it with --ctx (or TM_CTX — see "
                f"the engine's own note on auto-sizing), or send less history.")
        self.note_loaded(role, text)

    def replay(self, messages: list[dict]) -> None:
        """Make the cache match the client's history, ready for the last turn.

        A client re-sends the whole conversation every turn and the engine's cache holds
        the previous one, so the work is the DELTA: turns already cached are skipped and
        only what is new is loaded. Six turns through this path cost 49.2 s while
        generating ~500 tokens (docs/BENCHMARKS.md), and nearly all of it was re-reading a
        transcript the engine already had.

        Skipping a prefix is only sound when the client's history IS the one in the cache,
        so the test is equality of normalized (role, text) turns. A mismatch — an edited
        reply, a different conversation, a client that summarised its own history — falls
        back to a full replay, which is what this function always did. `loaded` advances
        only where the shim observed the engine's state, never where it assumed one.
        """
        want: list[tuple[str, str]] = []
        for m in messages:
            role, text = self._turn_key(m)
            if text:
                want.append((role, text))
        keep = min(len(want), len(self.loaded))
        # TM_SHIM_NO_REUSE=1 turns reuse off, so the two legs of a pairing can run in one
        # window against the same machine state instead of across sessions.
        if os.environ.get("TM_SHIM_NO_REUSE"):
            keep = 0
        elif keep and self._loaded_tokens() > 0.6 * self.ctx:
            # A near-full cache is where the engine resets a conversation mid-turn, and a
            # prefix that survived that is not the one we think it is. Stay clear of the
            # edge rather than discovering it with a wrong answer.
            keep = 0
        elif want[:keep] != self.loaded[:keep] or len(self.loaded) > keep:
            keep = 0                        # diverged, or the client dropped turns
        if keep == 0:
            self.reset_turns()
        else:
            # Observability on purpose: "reuse is on" and "reuse never fired" look
            # identical from the outside, and the difference is the whole feature.
            if os.environ.get("TM_SHIM_DEBUG"):
                print(f"[shim] reuse: {keep}/{len(want)} turns already cached, "
                      f"loading {len(want) - keep}", flush=True)
        for role, text in want[keep:]:
            self.prefill(role, text)
        self.loaded = want

    @staticmethod
    def _turn_key(m: dict) -> tuple[str, str]:
        """One history message as (role, text), normalized the way prefill sends it."""
        role = str(m.get("role", "user"))
        if role not in ("user", "assistant", "system"):
            role = "user"
        return role, str(m.get("content", "")).replace("\n", " ").strip()

    def _loaded_tokens(self) -> int:
        """Rough size of what the cache holds; characters/4 is this shim's proxy."""
        return sum(len(text) for _, text in self.loaded) // 4

    def note_loaded(self, role: str, text: str) -> None:
        """Record a turn this session just loaded into the cache."""
        key = text.replace("\n", " ").strip()
        if key:
            self.loaded.append((role, key))

    def reset_turns(self) -> None:
        """Drop the conversation from the cache and forget that we had one."""
        self.command("/reset")
        self.loaded = []

    def canonicalize_turn(self, role: str, prompt: str, reply: str) -> None:
        """Leave the turn that just ran in the cache in its REPLAYED form.

        A generated reply's tokens are the ones that were SAMPLED, and a replay builds
        the same turn by tokenizing text — one token different at the closing marker is
        enough to change every answer that follows, which the shim learned the hard way
        (31 % faster and wrong; docs/BENCHMARKS.md). So the turn is rewound to the
        boundary the engine reports for it and re-prefilled FROM TEXT, which is exactly
        what a replay would have built. Cheap, because the reply is the short half and
        everything before the boundary stays cached.
        """
        key = prompt.replace("\n", " ").strip()
        if key:
            self.loaded.append((role, key))
        ack = self.command("/turnpos")
        boundary = int(ack.split("=")[1]) if ack.startswith("turnpos=") else -1
        if boundary >= 0 and self.command(f"/rewind {boundary}") == f"rewind={boundary}":
            if not reply.strip():
                # The rewind above already removed the generated turn, and a replay builds
                # NO turn for empty content (`replay` skips empty text). Loading an empty
                # assistant turn here would leave the cache one turn longer than `loaded`
                # says it is — a divergence that only shows up as a different answer on the
                # NEXT turn, which is why the gate compares whole conversations.
                return
            try:
                self.prefill("assistant", reply)
                return
            except ContextTooSmall:
                pass
        # No boundary, or the re-prefill does not fit: the cache holds a SAMPLED turn,
        # which is not the replay form, so nothing may be reused from it.
        if os.environ.get("TM_SHIM_DEBUG"):
            print(f"[shim] canonicalize failed (turnpos={ack!r}) — reuse disabled for "
                  f"this conversation", flush=True)
        self.loaded = []

    def complete(self, prompt: str, budget: int, on_piece=None) -> str:
        """Run one turn, handing each decoded piece to `on_piece` as it arrives.

        The engine streams — it prints what it has decoded so far and flushes —
        so a caller can see the reply while it is still being generated.
        Buffering the whole turn and replaying it in fixed-size slices looks
        identical on the wire and is useless to somebody waiting for the first
        token.
        """
        self.set_budget(budget)
        p = self.proc.stdin
        p.write(prompt.replace("\n", " ") + "\nEOT\n")  # single-line: batch protocol is line-based
        p.flush()
        reply: list[str] = []
        while True:
            line = self._readline()
            if line == "<<<EOT>>>":
                break
            # The engine REPRINTS the document as it grows, so a line that extends
            # the previous one carries new text, while a line that does not is a
            # newline the model itself produced.
            if reply and line.startswith(reply[-1]):
                piece = line[len(reply[-1]):]
                reply[-1] = line
            else:
                piece = ("\n" + line) if reply else line
                reply.append(line)
            if piece and on_piece is not None:
                on_piece(piece)
        while reply and not reply[-1]:
            reply.pop()                                # engine ends the turn with a blank line
        armed = self._readline()                       # re-arm: the next PROMPT marker
        if armed != "PROMPT":
            # Raised, never sys.exit: this runs on the HTTP server's handler
            # threads, where SystemExit ends the thread silently and the client
            # gets an empty response with nothing in the log.
            raise RuntimeError(f"engine out of sync: {armed!r}")
        text = "\n".join(reply)
        # Both halves of this turn are in the cache now, but the reply half is SAMPLED —
        # so the turn is put back into the form a replay would build before anyone is
        # allowed to reuse it. Doing it here, where the turn just happened, is what keeps
        # the cache one answer rather than a mixture of computed and sampled history.
        self.canonicalize_turn("user", prompt, text)
        return text

    def close(self):
        """End the engine and give back what it held. Safe to call more than once.

        The engine is its own process, so this is where the model's memory actually
        returns: it does not exit when its stdin closes, so it is asked politely and
        then killed. The stderr temp file goes too — one per session, and a session
        is now created per load, so leaving them behind leaks one file per unload.
        """
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.stdin.write("/quit\nEOT\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
                try:
                    proc.wait(timeout=10)     # reap it: a kill without a wait zombies
                except Exception:
                    pass
        try:
            self._errlog.close()
        except Exception:
            pass
        try:
            os.unlink(self._errlog.name)
        except OSError:
            pass


def make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, status=200):
            body = (json.dumps(obj) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                # `size` was hardcoded 0 and the family said "llama" whatever the
                # model was — a client sizing its context budget off this got told
                # nothing, and a Qwen served under a llama label is a small lie that
                # costs somebody an hour.
                self._json({"models": [{"name": state["model_name"], "model": state["model_name"],
                                        "size": state["model_size"],
                                        "details": {"format": "tmq", "family": "tensormark"}}]})
            elif self.path == "/api/ps":
                # Ollama's "what is loaded right now". Served so the idle policy is
                # observable rather than inferred: an empty list means the model is gone
                # and its memory is back, which is the whole point of the TTL.
                # One entry per RESIDENT LANE, with the lane named: with a pool there
                # can be two, and this is the only way to see that from outside.
                with state["lock"]:
                    self._json({"models": [
                        {"name": state["model_name"], "model": state["model_name"],
                         "size": state["model_size"],
                         "expires_at": expires_at(state, ln),
                         "details": {"format": "tmq", "family": "tensormark",
                                     "lane": ln.name}}
                        for ln in state["pool"].lanes if ln.session is not None]})
            else:
                self._json({"error": "not found (only /api/generate, /api/chat, "
                                     "/api/embeddings, /api/embed, /api/tags and "
                                     "/api/ps)"}, 404)

        def do_POST(self):
            # Ollama's two spellings of the same capability: /api/embeddings takes
            # "prompt" and returns one vector, /api/embed takes "input" (a string or
            # a list) and returns a list. Both are served by one engine call each.
            # A grammar is deliberately ignored here — it constrains what a model
            # GENERATES, and an embedding generates nothing.
            if self.path in ("/api/embeddings", "/api/embed"):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/embeddings":
                    text = req.get("prompt", "")
                else:
                    text = req.get("input", "")
                texts = text if isinstance(text, list) else [text]
                try:
                    # An embedding generates nothing, so it carries no service-rate
                    # sample (see Lane.observe) — but it still goes through the pool,
                    # so it cannot land on a lane another request is already using.
                    _, vectors = run_on_lane(
                        state, 8.0,
                        lambda s: ([s.embed(str(t)) for t in texts], 0.0))
                except Exception as exc:   # a bare traceback becomes an empty 500
                    self._json({"error": f"engine: {exc}"}, 500)
                    return
                if self.path == "/api/embeddings":
                    self._json({"embedding": vectors[0]})
                else:
                    self._json({"model": req.get("model", state["model_name"]),
                                "embeddings": vectors})
                return
            if self.path not in ("/api/generate", "/api/chat"):
                self._json({"error": "not found (only /api/generate, /api/chat, "
                                     "/api/embeddings, /api/embed)"}, 404)
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            stream = bool(req.get("stream", False))
            # Ollama's per-request override of the idle policy: seconds, or a duration
            # string like "5m"; 0 unloads as soon as this reply is sent, a negative
            # value keeps the model resident. A typo is the caller's bug, so it is a
            # 400 and not a silently ignored field.
            try:
                keep = parse_keep_alive(req.get("keep_alive"))
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return
            if self.path == "/api/chat":
                # Ollama's /api/chat is stateless: the client re-sends the whole
                # conversation every time. Load every turn but the last from the
                # TEXT the client sent, then generate the last one. Generating
                # the history instead would condition the answer on the model's
                # own past samples rather than on the conversation that happened.
                messages = req.get("messages")
                if not isinstance(messages, list) or not messages \
                        or not all(isinstance(m, dict) for m in messages):
                    self._json({"error": "messages is required: a list of "
                                         "{role, content} objects"}, 400)
                    return
                history, prompt = messages[:-1], str(messages[-1].get("content", ""))
            else:
                history, prompt = None, req.get("prompt", "")

            def now() -> str:
                return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"

            def payload(content: str, done: bool) -> dict:
                """Ollama shapes the two endpoints differently: `response` for
                /api/generate, a `message` object for /api/chat."""
                obj = {"model": state["model_name"], "created_at": now(), "done": done}
                if self.path == "/api/chat":
                    obj["message"] = {"role": "assistant", "content": content}
                else:
                    obj["response"] = content
                return obj

            def emit(content: str, done: bool, duration_us: int | None = None) -> None:
                obj = payload(content, done)
                if duration_us is not None:
                    obj["total_duration"] = duration_us
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()

            # The engine is single-tenant — one --batch subprocess, one document
            # at a time — but this HTTP server is threaded. Without the lock two
            # concurrent requests interleave on the same pipes and each reads the
            # other's tokens: silent corruption rather than an error, which is
            # the worst way for a pipeline to fail.
            def body(sess):  # -> (text | None, measured tokens); None = answered
                # `grammar` carries inline GBNF text; `grammar_path` reads it from
                # this host; `schema` carries a JSON Schema, compiled to a grammar
                # here. A schema is the unit a caller actually has — it says what
                # the answer MEANS — and compiling it to a CLOSED grammar is what
                # makes the document terminate: a general JSON grammar lets a
                # small model free-run inside it and never close. With none of
                # the three, any previously-set constraint is CLEARED, so a
                # request never inherits one its caller did not ask for.
                grammar, gpath = req.get("grammar"), req.get("grammar_path")
                schema = req.get("schema")
                try:
                    if schema is not None:
                        sess.set_grammar(compile_schema(schema))
                    elif grammar is None and gpath is None:
                        sess.set_grammar(None)
                    else:
                        sess.set_grammar(
                            grammar if grammar is not None else Path(gpath).read_text())
                except (GrammarError, OSError, Unsupported) as exc:
                    self._json({"error": str(exc)}, 400)
                    return None, 0.0
                except RuntimeError as exc:      # the engine failed, not the request
                    self._json({"error": f"engine: {exc}"}, 500)
                    return None, 0.0
                # The cache is made to match the CALLER's history, not inherited from
                # whatever ran last. /api/generate carries no history, so it starts
                # clean; /api/chat re-sends the conversation every turn, and `replay`
                # loads only the turns the cache does not already hold — refusing the
                # shortcut the moment the client's history diverges from what the shim
                # observed the engine holding. Inheriting by accident is the failure
                # mode: request N would be answered in the context of 1..N-1 with no
                # way for the caller to tell.
                try:
                    if history is None:
                        sess.reset_turns()
                    else:
                        sess.replay(history)
                except ContextTooSmall as exc:
                    # The request is too big for the window, not the server broken.
                    self._json({"error": str(exc)}, 400)
                    return None, 0.0
                except (RuntimeError, OSError) as exc:
                    self._json({"error": f"engine: {exc}"}, 500)
                    return None, 0.0
                t0 = time.time()
                if stream:
                    # Headers before the first token, not after the last: the
                    # point of streaming is that the opening words arrive while
                    # the rest of the reply still does not exist.
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()

                    def on_piece(piece: str) -> None:
                        emit(piece, False)
                else:
                    on_piece = None
                # Ollama's `options.repeat_penalty` — its own default is 1.1, and
                # honouring it is the whole point: the field used to be accepted
                # and dropped, which is how a greedy loop reached a caller.
                _opts = req.get("options") or {}
                _rp = _opts.get("repeat_penalty", req.get("repeat_penalty"))
                if _rp is not None:
                    sess.set_repeat(float(_rp),
                                    int(_opts.get("repeat_last_n", 64)))
                try:
                    text = sess.complete(
                        prompt, int(req.get("budget", 256)), on_piece)
                except (RuntimeError, OSError) as exc:
                    # OSError covers BrokenPipeError: the engine process died and
                    # the shim's write to its stdin failed. Uncaught, that ends the
                    # handler thread and returns an empty response with nothing in
                    # the log — a dead engine looked like a dead server.
                    detail = (f"engine: {exc}; stderr: "
                              f"{sess.engine_stderr()[-300:]}")
                    if stream:
                        emit(detail, True)   # the status line is already sent
                    else:
                        self._json({"error": detail}, 500)
                    return None, 0.0
                if stream:
                    emit("", True, int((time.time() - t0) * 1e9))
                else:
                    # `total_duration` is what a client turns into tokens/s, so it is
                    # measured rather than estimated.
                    out = payload(text, True)
                    out["total_duration"] = int((time.time() - t0) * 1e9)
                    self._json(out)
                # The engine reports no token count, so this is a proxy at four
                # characters per token. It exists only to compare lanes with each other
                # — the scheduler's `rate` — and is never itself quoted as throughput.
                return text, max(1.0, len(text) / 4.0)

            lane, _ = run_on_lane(state, float(int(req.get("budget", 256))), body, keep)
            if keep == 0 and lane is not None:
                # The reply is sent; now act on an explicit keep_alive 0. Lane.drop
                # takes the lane's lock itself, so this must not run under the state
                # lock: a request holds its lane while taking the state lock.
                lane.drop(reason="keep_alive 0")

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Resolved, not hardcoded, and by the same rule the engine binary uses for its
    # own default — so the shim and the CLI cannot disagree about which model "the
    # model" is. Absolute: the engine subprocess runs with cwd set to the repository
    # root, where `data/...` does not exist.
    ap.add_argument("--model", default=None,
                    help="quantized model (.tmq); default: the best one present, "
                         "Qwen3.5-4B if it is there")
    ap.add_argument("--tokenizer", default=None,
                    help="sentencepiece .model, or the byte-level BPE .vocab.json; "
                         "default: the one beside --model")
    ap.add_argument("--system", default="You are a helpful, concise assistant.")
    ap.add_argument("--sample", action="store_true",
                    help="use the engine's chat sampling instead of greedy decoding "
                         "(greedy is the default: a grammar-constrained caller wants the "
                         "same document back for the same prompt)")
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--idle-ttl", type=float, default=DEFAULT_IDLE_TTL_S,
                    help="unload the model after this many seconds of inactivity, "
                         "releasing the memory it holds (default 300). 0 or "
                         "negative keeps it resident; a request can override this "
                         "with Ollama's keep_alive")
    ap.add_argument("--lanes", type=int, default=1, choices=(1, 2),
                    help="engine lanes to serve from: 1 (default) is the engine's own "
                         "auto policy; 2 pins one process to the GPU-preferred lane and "
                         "one to the CPU, so two requests can run at once. Measured on "
                         "an M1/8GB with tinyllama-q40: 95.75 t/s across both lanes "
                         "against 76.97 t/s on the better single lane (each lane keeps "
                         "~2/3 of its solo rate). Costs a second set of activations and "
                         "KV; the pool parks the extra lane again if it stops paying")
    ap.add_argument("--ctx", type=int, default=None,
                    help="context window in tokens for each engine (default: the model's "
                         "own max_position_embeddings — 2048 for TinyLlama). This is the "
                         "knob that decides how large a conversation a client may send: a "
                         "history turn that does not fit is rejected with HTTP 400, not "
                         "silently truncated, because dropping earlier turns changes the "
                         "answer. Costs KV memory per lane")
    a = ap.parse_args()
    # A tokenizer the caller did not name belongs to the model they did name.
    if a.model is None:
        a.model = str(default_model())
    if a.tokenizer is None:
        a.tokenizer = str(default_tokenizer(Path(a.model)))
    model = Path(a.model)
    state = {
        "model": a.model, "tokenizer": a.tokenizer, "system": a.system,
        "ctx": a.ctx,
        "greedy": not a.sample,
        "model_name": model.stem,
        "model_size": model.stat().st_size if model.exists() else 0,
        "lock": threading.Lock(),         # pool bookkeeping and lane lifecycle only
        "stop": threading.Event(),
        "idle_ttl": a.idle_ttl,
    }
    # One lane by default, which is the engine's own policy and nothing else: the
    # pool exists so that a second lane is a decision the controller keeps making,
    # not a startup choice nobody revisits.
    state["pool"] = LanePool(state, ["auto"] if a.lanes == 1 else ["gpu", "cpu"])
    # Load before announcing the port, as this server always has: a bad model path or
    # an engine that refuses its config is then a startup failure a reader can act on,
    # instead of a 500 on every request for the life of the process. The reaper can
    # unload a lane later — Lane.drop takes that lane's lock, so a request in flight
    # can never have its engine taken away.
    try:
        for ln in state["pool"].lanes:
            ln.ensure()
    except RuntimeError as exc:
        sys.exit(f"cannot start the engine: {exc}")
    threading.Thread(target=idle_reaper, args=(state,), daemon=True).start()
    if len(state["pool"].lanes) > 1:
        threading.Thread(target=lane_controller, args=(state,), daemon=True).start()
        print("[shim] lanes: " + ", ".join(ln.name for ln in state["pool"].lanes)
              + " (the controller parks the extra lane if the pool stops beating one)",
              flush=True)
    print(f"tensormark ollama shim on http://127.0.0.1:{a.port} (model: {a.model})")
    if a.idle_ttl > 0:
        print(f"[shim] idle unload after {a.idle_ttl:.0f}s of inactivity", flush=True)
    else:
        print("[shim] idle unload disabled: the model stays resident", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state)).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state["stop"].set()
        for ln in state["pool"].lanes:
            ln.drop("shutting down")


if __name__ == "__main__":
    main()