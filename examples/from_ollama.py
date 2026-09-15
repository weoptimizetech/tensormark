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
  * generation is greedy-deterministic; `options` (temperature, top_k, ...)
    are accepted but ignored;
  * `budget` is the per-turn token limit (default 256). Every request starts from
    a clean cache, so Ollama's `context` reuse parameter is ignored;
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


def arm_idle(state: dict, keep: float | None) -> None:
    """Set when the model becomes eligible for unloading. Caller holds the lock."""
    ttl = state["idle_ttl"] if keep is None else keep
    state["last_used"] = time.monotonic()
    state["deadline"] = time.monotonic() + ttl if ttl > 0 else float("inf")


def expires_at(state: dict) -> str | None:
    """When the model is due to go, for /api/ps. None = not resident, or never."""
    deadline = state["deadline"]
    if state["session"] is None or deadline == float("inf"):
        return None
    left = deadline - time.monotonic()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + left))


def ensure_session(state: dict):
    """The resident engine, loading it if the reaper let it go. Caller holds the lock."""
    if state["session"] is not None:
        return state["session"]
    t0 = time.monotonic()
    state["session"] = LlamaChatSession(state["model"], state["tokenizer"],
                                        state["system"], greedy=state["greedy"])
    arm_idle(state, None)
    print(f"[shim] loaded {state['model_name']} in {time.monotonic() - t0:.1f}s "
          f"({state['model_size'] / 1e6:.0f} MB)", flush=True)
    return state["session"]


def drop_session(state: dict, reason: str) -> None:
    """Release the engine, and with it the model. Caller holds the lock."""
    session, state["session"] = state["session"], None
    if session is None:
        return
    held = time.monotonic() - state["last_used"]
    session.close()                     # /quit, then SIGKILL if it will not go
    state["deadline"] = float("inf")
    print(f"[shim] unloaded {state['model_name']} ({reason}; unused for {held:.1f}s)",
          flush=True)


def idle_reaper(state: dict) -> None:
    """Unload the model once nothing has used it for `idle_ttl` seconds.

    Eviction runs under the same lock every request takes, so a request in
    progress can never have the engine pulled out from under it, and this thread
    can never see a half-idle request: the lock is held for its whole duration.
    """
    while not state["stop"].wait(IDLE_REAPER_TICK_S):
        with state["lock"]:
            if state["session"] is None or time.monotonic() < state["deadline"]:
                continue
            drop_session(state, reason=f"idle over {state['idle_ttl']:.0f}s")


# The JSON Schema -> GBNF compiler ships
# The JSON Schema -> GBNF compiler ships with the engine (tensormark/), not with
# the shim, and the repo root is not on sys.path when this runs as a script.
sys.path.insert(0, str(REPO / "tensormark"))
from jsonschema_to_gbnf import Unsupported, compile_schema   # noqa: E402


class GrammarError(RuntimeError):
    """The engine rejected a grammar; the caller's schema is at fault."""


class LlamaChatSession:
    """One --batch subprocess; one request at a time (the engine is single-tenant)."""

    def __init__(self, model: str, tokenizer: str, system: str, greedy: bool = True):
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
            )
            self._grammar_key: str | None = None     # hash of the grammar in effect
            self._grammar_dir: str | None = None
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
            if greedy:
                self.command("/greedy")              # the documented contract
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

    def prefill(self, role: str, text: str) -> None:
        """Load one templated turn into the KV cache, generating nothing."""
        ack = self.command("/prefill " + role + " " + text.replace("\n", " "))
        if ack != f"prefill={role}":
            raise RuntimeError(f"engine would not load the history: {ack!r}")

    def replay(self, messages: list[dict]) -> None:
        """Make the cache match the client's history, ready for the last turn.

        Reset first: a client that re-sends the whole conversation gives us no way
        to know it is the same one, and a stale prefix is not a slow answer — it
        is a wrong one. Every turn but the last is loaded from the TEXT the client
        sent, never re-generated, so the model conditions on the conversation that
        actually happened rather than on its own past samples.
        """
        self.command("/reset")
        for m in messages:
            role = str(m.get("role", "user"))
            if role not in ("user", "assistant", "system"):
                role = "user"
            text = str(m.get("content", "")).replace("\n", " ").strip()
            if text:
                self.prefill(role, text)

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
        return "\n".join(reply)

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
                with state["lock"]:
                    loaded = state["session"] is not None
                    self._json({"models": [{
                        "name": state["model_name"], "model": state["model_name"],
                        "size": state["model_size"],
                        "expires_at": expires_at(state),
                        "details": {"format": "tmq", "family": "tensormark"},
                    }] if loaded else []})
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
                    with state["lock"]:
                        sess = ensure_session(state)
                        vectors = [sess.embed(str(t)) for t in texts]
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
            with state["lock"]:
                # A request may find the model unloaded: the reaper releases it after
                # the idle TTL, and this is where it comes back — a second or two on a
                # warm page cache, which is the price of not holding a model nobody is
                # using. It is also why every use below goes through `sess`.
                sess = ensure_session(state)
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
                    return
                except RuntimeError as exc:      # the engine failed, not the request
                    self._json({"error": f"engine: {exc}"}, 500)
                    return
                # Every request starts from a clean cache. The engine carries the
                # conversation across turns by design — that is what makes an
                # interactive session cheap — so a stateless API has to say so
                # explicitly, or request N is answered in the context of requests
                # 1..N-1 and the caller has no way to tell.
                try:
                    if history is None:
                        sess.command("/reset")
                    else:
                        sess.replay(history)
                except (RuntimeError, OSError) as exc:
                    self._json({"error": f"engine: {exc}"}, 500)
                    return
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
                    return
                # The reply is done with the engine: restart the idle clock, or act on an
                # explicit keep_alive 0 straight away.
                arm_idle(state, keep)
                if keep == 0:
                    drop_session(state, reason="keep_alive 0")
                if stream:
                    emit("", True, int((time.time() - t0) * 1e9))
                    return
                # `total_duration` is what a client turns into tokens/s, so it is
                # measured rather than estimated.
                out = payload(text, True)
                out["total_duration"] = int((time.time() - t0) * 1e9)
                self._json(out)

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
    a = ap.parse_args()
    # A tokenizer the caller did not name belongs to the model they did name.
    if a.model is None:
        a.model = str(default_model())
    if a.tokenizer is None:
        a.tokenizer = str(default_tokenizer(Path(a.model)))
    model = Path(a.model)
    state = {
        "session": None,                  # the resident engine, or None when unloaded
        "model": a.model, "tokenizer": a.tokenizer, "system": a.system,
        "greedy": not a.sample,
        "model_name": model.stem,
        "model_size": model.stat().st_size if model.exists() else 0,
        "lock": threading.Lock(),         # serialises requests AND the reaper
        "stop": threading.Event(),
        "idle_ttl": a.idle_ttl,
        "last_used": time.monotonic(),
        "deadline": float("inf"),
    }
    # Load before announcing the port, as this server always has: a bad model path or
    # an engine that refuses its config is then a startup failure a reader can act on,
    # instead of a 500 on every request for the life of the process. The reaper can
    # unload this same session later — it takes the lock, so there is no race.
    try:
        with state["lock"]:
            ensure_session(state)
    except RuntimeError as exc:
        sys.exit(f"cannot start the engine: {exc}")
    threading.Thread(target=idle_reaper, args=(state,), daemon=True).start()
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
        with state["lock"]:
            drop_session(state, reason="shutting down")


if __name__ == "__main__":
    main()