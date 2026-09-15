#!/usr/bin/env python3
"""tmchat — drive the tensormark chat engine from Python.

The engine's `--batch` mode is a line protocol on stdin/stdout: it prints
`PROMPT`, reads a message up to a line containing only `EOT`, streams the
completion, prints `<<<EOT>>>`, and repeats. Slash-commands are acknowledged
with `ACK <what>`. This module wraps that protocol so a Python pipeline does not
have to re-implement it, and so the two failure modes that bite are handled once:

  * **Desynchronisation.** A driver that reads the wrong number of lines does
    not fail loudly — it reads the NEXT request's answer as this one's. Every
    read here is exact and anything unexpected raises.
  * **Hangs.** A read that never returns is indistinguishable from a slow model.
    Timeouts are enforced from a reader thread feeding a queue; `select()` on the
    raw fd does NOT work with text-mode `readline()`, because readline() fills an
    internal buffer and the fd then looks idle while complete lines sit in
    Python's buffer.

Engine output is read with `errors="replace"`: the engine holds back incomplete
codepoints, but a byte-level tokenizer can still hand one over, and a stray
fraction of a character must degrade to U+FFFD rather than raise.

    from tmchat import Engine
    with Engine("data/tinyllama/tinyllama_q40.tmq") as e:
        e.set_grammar('root ::= "yes" | "no"')
        print(e.complete("Answer yes or no."))
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BIN = REPO / "tensormark" / "build" / "llama_chat_metal"
# TM_TEST_MODEL / TM_TEST_TOKENIZER let the capability gates run against a
# generated fixture (tensormark/gen_tiny_llama.py) instead of the fetched 590 MB
# checkpoint, so the assertion that a reply VALIDATES against its schema is
# exercised on every push rather than only on a machine that has the model.
DEFAULT_MODEL = Path(os.environ.get(
    "TM_TEST_MODEL", REPO / "tensormark" / "data" / "tinyllama" / "tinyllama_q40.tmq"))
DEFAULT_TOKENIZER = Path(os.environ.get(
    "TM_TEST_TOKENIZER", REPO / "tensormark" / "data" / "tinyllama" / "tokenizer.model"))
READ_TIMEOUT_S = 600.0


class GrammarError(RuntimeError):
    """The engine rejected a grammar; the caller's schema is at fault."""


class Engine:
    """A long-lived `llama_chat_metal --batch` process.

    One process, many turns: the model stays loaded and the vocabulary tables
    behind the token mask are built once, so switching grammars between calls
    costs a grammar parse rather than a model reload.
    """

    def __init__(self, model: str | os.PathLike = DEFAULT_MODEL,
                 tokenizer: str | os.PathLike = DEFAULT_TOKENIZER,
                 system: str = "You are a helpful, concise assistant.",
                 binary: str | os.PathLike = DEFAULT_BIN,
                 budget: int = 256, greedy: bool = True,
                 timeout: float = READ_TIMEOUT_S):
        self.binary = Path(binary)
        if not self.binary.exists():
            raise FileNotFoundError(f"missing {self.binary} — run ./tensormark/build_llama.sh")
        self.timeout = timeout
        # stderr goes to a FILE, never a pipe: nothing drains a pipe while the
        # driver is blocked on stdout, so a chatty engine would fill the 64 KB
        # pipe buffer and block forever inside write().
        self._errlog = tempfile.NamedTemporaryFile(
            prefix="tmchat-stderr-", suffix=".log", delete=False)
        self._grammar_key: str | None = None
        self._grammar_dir: str | None = None
        self.proc = subprocess.Popen(
            # NOT --quiet: this driver sends slash-commands and their ACK lines
            # are how it knows a command landed.
            [str(self.binary), "-m", str(model), "-t", str(tokenizer),
             "-s", system, "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errlog,
            text=True, errors="replace", bufsize=1, cwd=str(REPO),
        )
        self._q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        if self.expect("PROMPT") != ["ACK ready"]:
            raise RuntimeError("unexpected engine banner; is this the chat binary?")
        if greedy:
            self.command("/greedy")
        self.command(f"/n {budget}")

    # ------------------------------------------------------------------ plumbing
    def _pump(self) -> None:
        for line in self.proc.stdout:              # blocks; ends at EOF
            self._q.put(line.rstrip("\n"))
        self._q.put(None)                          # EOF sentinel

    def _readline(self) -> str:
        try:
            line = self._q.get(timeout=self.timeout)
        except queue.Empty:
            raise RuntimeError(f"engine silent for {self.timeout:.0f}s") from None
        if line is None:
            raise RuntimeError("engine closed stdout (crash?)")
        return line

    def expect(self, until: str) -> list[str]:
        """Read lines up to and including `until`; return the lines before it."""
        out: list[str] = []
        while True:
            line = self._readline()
            if line == until:
                return out
            out.append(line)

    def send(self, message: str) -> None:
        self.proc.stdin.write(message + "\nEOT\n")
        self.proc.stdin.flush()

    def command(self, cmd: str) -> str:
        """Send a slash-command; return its ACK payload."""
        self.send(cmd)
        ack = self._readline()
        if not ack.startswith("ACK "):
            raise RuntimeError(f"engine did not acknowledge {cmd!r}: {ack!r}")
        if self._readline() != "PROMPT":
            raise RuntimeError(f"engine out of sync after {cmd!r}")
        return ack[4:]

    def engine_stderr(self) -> str:
        try:
            return Path(self._errlog.name).read_text()
        except OSError:
            return ""

    # ------------------------------------------------------------- the two calls
    def set_grammar(self, gbnf: str | None) -> None:
        """Constrain decoding to `gbnf` (inline text), or clear it with None.

        A grammar the engine rejects raises `GrammarError` instead of falling
        back to unconstrained decoding: a caller that asked for a schema must
        never get well-formed-looking garbage back.
        """
        if gbnf is None:
            if self._grammar_key is not None:
                self.command("/grammar off")
                self._grammar_key = None
            return
        key = hashlib.sha256(gbnf.encode()).hexdigest()[:16]
        if key == self._grammar_key:
            return
        if self._grammar_dir is None:
            self._grammar_dir = tempfile.mkdtemp(prefix="tm-grammar-")
        path = Path(self._grammar_dir) / f"{key}.gbnf"
        path.write_text(gbnf)
        if self.command(f"/grammar {path}").startswith("grammar=FAILED"):
            raise GrammarError(f"engine rejected the grammar: {self.engine_stderr()[-400:]}")
        self._grammar_key = key

    def complete(self, prompt: str) -> str:
        """One turn; returns the completion text (constrained, if a grammar is set)."""
        self.send(prompt.replace("\n", " "))       # the protocol is line-based
        lines = self.expect("<<<EOT>>>")
        self.expect("PROMPT")                      # re-arm
        while lines and not lines[-1]:
            lines.pop()                            # the turn ends with a blank line
        return "\n".join(lines)

    def set_budget(self, tokens: int) -> None:
        self.command(f"/n {int(tokens)}")

    def embed(self, text: str) -> list[float]:
        """L2-normalised pooled embedding of `text` — the mean of the final-layer
        hidden state over the prompt's positions (the engine's `embed_begin`).

        Raw text, no chat template: a template is a generation concern, and a
        caller that wants one can put it in `text`. Length is the model's hidden
        size; the vector is unit-norm, so cosine similarity is a dot product.
        """
        payload = self.command("/embed " + text.replace("\n", " "))
        if not payload.startswith("embed="):
            raise RuntimeError(f"engine did not return an embedding: {payload[:80]!r}")
        return json.loads(payload[len("embed="):])["embedding"]

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> int:
        """Ask the engine to quit and reap it.

        `/quit` breaks the engine loop and exits without another `PROMPT`, so
        waiting for one would block until the pipe closes.
        """
        try:
            if self.proc.poll() is None:
                self.send("/quit")
                self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=20)
        finally:
            self._errlog.close()
        return self.proc.returncode

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def main(argv: list[str] | None = None) -> int:
    """Tiny CLI so the driver is usable without writing Python:
    `python3 tensormark/tmchat.py "prompt" [grammar.gbnf]`."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt")
    ap.add_argument("grammar", nargs="?", help="a .gbnf file to constrain decoding")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--embed", action="store_true",
                    help="print the prompt's pooled embedding as JSON instead of completing")
    a = ap.parse_args(argv)
    with Engine(budget=a.budget) as e:
        if a.embed:
            print(json.dumps({"embedding": e.embed(a.prompt)}))
            return 0
        if a.grammar:
            e.set_grammar(Path(a.grammar).read_text())
        sys.stdout.write(e.complete(a.prompt) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
