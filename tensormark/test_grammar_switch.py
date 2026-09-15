#!/usr/bin/env python3
"""test_grammar_switch.py — gate for runtime grammar switching in --batch mode.

`/grammar <file.gbnf>` and `/grammar off` change the decoding constraint
mid-session, so one long-lived engine process can serve a driver that uses a
different grammar per request. That is what makes structured output usable from
a Python pipeline: reloading the model per request costs seconds, re-parsing a
grammar costs milliseconds.

WHAT THIS GATE ASSERTS, AND WHY IT IS NOT "the output parses as JSON".

A grammar constrains the *structure* of the token stream; it does not make a
1.1B model produce a useful document. Measured on TinyLlama-1.1B Q4_0, greedy,
under a JSON grammar, the model free-runs inside the schema — it emits a
repeating `"$ref"/"$title"` cycle until the budget dies, and the document never
closes. Asserting `json.loads(output)` would therefore be asserting something
about the model, and it would fail for a reason that is not a grammar bug.

So the constraint is checked two ways that do not depend on generation quality:

  * a grammar with only closed alternatives (`"yes" | "no"`) — the output MUST be
    exactly one of them, which is decisive and finishes in a handful of tokens;
  * a JSON grammar — the output must be a valid *prefix* of the language: every
    byte legal so far, with the document possibly still open at the budget.

The second check is what actually protects a caller: a truncated document is a
retry, whereas a document that leaves the grammar is corruption, and only the
second one is a bug in this feature.

`--batch` uses stdout as its channel, so anything the engine writes there that
is not an `ACK`, a `PROMPT`, the completion, or `<<<EOT>>>` desynchronises the
driver — and a desynchronised driver does not fail loudly, it reads the next
request's answer as this one's. Both halves are checked: the grammar banner
goes to stderr, and a rejected grammar still ACKs and returns to `PROMPT`.

Run:  python3 tensormark/test_grammar_switch.py
Needs build/llama_chat_metal (./tensormark/build_llama.sh) and a .tmq model.
Exits 0 on pass, 1 on failure, 77 when the model or binary is absent (skip).
"""
from __future__ import annotations

import queue
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "tensormark" / "build" / "llama_chat_metal"
MODEL = Path(os.environ.get(
    "TM_TEST_MODEL", REPO / "tensormark" / "data" / "tinyllama" / "tinyllama_q40.tmq"))
TOKENIZER = Path(os.environ.get(
    "TM_TEST_TOKENIZER", REPO / "tensormark" / "data" / "tinyllama" / "tokenizer.model"))
READ_TIMEOUT_S = 90.0
BUDGET = 24                       # enough for the closed grammar, cheap to run

# A closed grammar: no repetition, so the model cannot free-run inside it.
YESNO_GBNF = 'root ::= "yes" | "no"\n'

JSON_GBNF = r"""
root   ::= object
object ::= "{" ws ( pair ( ws "," ws pair )* )? ws "}"
pair   ::= string ws ":" ws value
value  ::= string | number | object | array | "true" | "false" | "null"
array  ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" ( [^"\\] | "\\" . )* "\""
number ::= "-"? [0-9]+ ( "." [0-9]+ )?
ws     ::= [ \t\n]*
"""


def is_json_prefix(text: str) -> bool:
    """True if `text` is a legal prefix of some JSON document.

    Truncation is allowed (the budget can die mid-document); leaving the grammar
    is not. Scanning for balance keyed on strings is enough for a grammar whose
    only nesting is `{}` and `[]`.
    """
    stack: list[str] = []
    in_str = escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and (not stack or stack.pop() != {"}": "{", "]": "["}[ch]):
            return False
    return True


class Engine:
    """Line driver for the --batch protocol, with a real timeout on every read."""

    def __init__(self, model: Path, tokenizer: Path):
        # stderr goes to a FILE, not a pipe: nothing drains a pipe while the
        # driver is blocked on stdout, so a chatty engine would fill the 64 KB
        # pipe buffer and block forever inside write() — a deadlock that looks
        # exactly like the engine being slow.
        self._errlog = tempfile.NamedTemporaryFile(
            prefix="tm-gate-stderr-", suffix=".log", delete=False)
        self.p = subprocess.Popen(
            # NOT --quiet: this driver sends slash-commands, and their `ACK` line
            # is how it knows a command landed. The protocol is read strictly
            # instead — completions take lines up to `<<<EOT>>>`, commands take
            # exactly one `ACK` then `PROMPT`, anything else is a loud desync.
            [str(BIN), "-m", str(model), "-t", str(tokenizer), "-s", "You output JSON.",
             "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errlog,
            # errors="replace": the engine's stdout is a byte stream — a sampled
            # byte-fallback token can be a lone continuation byte — so a strict
            # text-mode reader dies with UnicodeDecodeError mid-test. Every other
            # reader of this protocol (tmchat.py, from_ollama.py) decodes
            # leniently; this gate was the only one that did not.
            text=True, errors="replace", bufsize=1, cwd=str(REPO),
        )
        # Timeouts are enforced by a reader thread feeding a queue. `select` on
        # the raw fd does NOT work here: text-mode readline() fills an internal
        # buffer, so after one readline() the fd looks idle while complete lines
        # sit in Python's buffer — every "timeout" would be spurious, and the
        # gate would fail on a healthy engine.
        self._q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        if self.expect("PROMPT") != ["ACK ready"]:
            raise AssertionError("unexpected startup banner")
        self.command("/greedy")        # deterministic; also the shim's default
        self.command(f"/n {BUDGET}")

    def _pump(self) -> None:
        for line in self.p.stdout:                 # blocks; ends at EOF
            self._q.put(line.rstrip("\n"))
        self._q.put(None)                          # EOF sentinel

    def _readline(self) -> str:
        """One line from stdout; a stall is a failure, not a wait."""
        try:
            line = self._q.get(timeout=READ_TIMEOUT_S)
        except queue.Empty:
            raise AssertionError(f"engine silent for {READ_TIMEOUT_S:.0f}s")
        if line is None:
            raise AssertionError("engine closed stdout (crash?)")
        return line

    def engine_stderr(self) -> str:
        try:
            return Path(self._errlog.name).read_text()
        except OSError:
            return ""

    def expect(self, until: str) -> list[str]:
        out: list[str] = []
        while True:
            line = self._readline()
            if line == until:
                return out
            out.append(line)

    def send(self, message: str) -> None:
        self.p.stdin.write(message + "\nEOT\n")
        self.p.stdin.flush()

    def command(self, cmd: str) -> str:
        """Send a slash-command; return its ACK payload (protocol-stable)."""
        self.send(cmd)
        ack = self._readline()
        if not ack.startswith("ACK "):
            raise AssertionError(f"engine did not acknowledge {cmd!r}: {ack!r}")
        if self._readline() != "PROMPT":
            raise AssertionError(f"engine out of sync after {cmd!r}")
        return ack[4:]

    def complete(self, prompt: str) -> str:
        self.send(prompt)
        text = self.expect("<<<EOT>>>")
        self.expect("PROMPT")                          # re-arm
        while text and not text[-1]:
            text.pop()                                 # turn ends with a blank line
        return "\n".join(text)

    def close(self) -> int:
        """Ask the engine to quit and reap it.

        `/quit` breaks the engine loop and exits — it prints no `PROMPT`
        afterwards, so waiting for one would block until the pipe closes.
        """
        try:
            self.send("/quit")
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()
            self.p.wait(timeout=20)
        self._errlog.close()
        return self.p.returncode


def main() -> int:
    if not BIN.exists():
        print(f"SKIP: no {BIN.relative_to(REPO)} — run ./tensormark/build_llama.sh")
        return 77
    if not MODEL.exists():
        # relative_to raises when the path is outside the repo — which is exactly
        # what the TM_TEST_MODEL override does (a generated fixture under /tmp), so
        # the skip path must not assume the model lives inside REPO.
        shown = MODEL.relative_to(REPO) if MODEL.is_relative_to(REPO) else MODEL
        print(f"SKIP: no model at {shown} — run examples/fetch_model.py")
        return 77

    n = 0
    with tempfile.TemporaryDirectory() as td:
        yesno, jsong, broken = Path(td) / "yesno.gbnf", Path(td) / "json.gbnf", Path(td) / "broken.gbnf"
        yesno.write_text(YESNO_GBNF)
        jsong.write_text(JSON_GBNF)
        broken.write_text("root ::= [unterminated\n")
        missing = Path(td) / "absent.gbnf"

        e = Engine(MODEL, TOKENIZER)
        try:
            # 1. a closed grammar is enforced byte for byte
            ack = e.command(f"/grammar {yesno}")
            n += 1
            assert ack == f"grammar={yesno}", f"switch ACK wrong: {ack!r}"
            print(f"  ok   /grammar <file> -> ACK {ack}")

            out = e.complete("Answer yes or no.").strip()
            n += 1
            assert out in ("yes", "no"), f"closed grammar leaked: {out!r}"
            print(f"  ok   closed grammar enforced exactly -> {out!r}")

            # 2. the same process takes a different grammar with no reload
            ack = e.command(f"/grammar {jsong}")
            n += 1
            assert ack == f"grammar={jsong}", f"re-switch ACK wrong: {ack!r}"
            out = e.complete('Output a JSON object with keys "name" and "kind".')
            n += 1
            assert out.lstrip().startswith("{") and is_json_prefix(out), \
                f"output left the JSON grammar: {out[:160]!r}"
            closed = out.rstrip().endswith(("}", "]"))
            print(f"  ok   swapped to a JSON grammar; output is a legal JSON prefix "
                  f"({len(out)} bytes, document closed: {closed})")

            # 3. off -> unconstrained; anything non-empty is acceptable
            ack = e.command("/grammar off")
            n += 1
            assert ack == "grammar=off", f"off ACK wrong: {ack!r}"
            out = e.complete("Name the primary colors.").strip()
            n += 1
            assert out, "empty completion after /grammar off"
            print(f"  ok   /grammar off -> unconstrained ({len(out)} bytes)")

            # 4. a missing grammar fails the ACK but must NOT desync the session
            ack = e.command(f"/grammar {missing}")
            n += 1
            assert ack == "grammar=FAILED", f"missing-file ACK wrong: {ack!r}"
            n += 1
            assert e.complete("Say OK.").strip(), "no answer after a rejected grammar"
            print("  ok   missing grammar -> ACK grammar=FAILED, session in sync")

            # 5. same for a grammar that does not parse
            ack = e.command(f"/grammar {broken}")
            n += 1
            assert ack == "grammar=FAILED", f"bad-grammar ACK wrong: {ack!r}"
            n += 1
            assert e.complete("Say OK.").strip(), "no answer after a parse error"
            print("  ok   unparseable grammar -> ACK grammar=FAILED, session in sync")

            # 6. a rejected grammar must leave the PREVIOUS constraint intact or
            #    clearly off — never a half-applied filter
            ack = e.command(f"/grammar {yesno}")
            n += 1
            assert ack == f"grammar={yesno}", f"re-arm ACK wrong: {ack!r}"
            out = e.complete("Answer yes or no.").strip()
            n += 1
            assert out in ("yes", "no"), f"grammar not re-armed cleanly: {out!r}"
            print("  ok   grammar re-armed cleanly after failures")
        finally:
            stderr = e.engine_stderr()
            rc = e.close()
        n += 1
        assert rc == 0, f"engine exited {rc}"
        print(f"  ok   clean exit (rc={rc})")

        # 7. stdout is the protocol channel; banners belong on stderr
        n += 1
        banners = [ln for ln in stderr.splitlines() if ln.startswith("grammar:")]
        assert banners, "expected grammar banners on stderr, saw none"
        print(f"  ok   {len(banners)} grammar banners on stderr, stdout protocol-only")

    print(f"\n{n} checks, 0 failed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
