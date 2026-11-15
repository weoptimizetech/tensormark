#!/usr/bin/env python3
"""test_shim_reuse.py — the cache-reuse path must answer what the non-reuse path answers.

Why this exists. An agent re-sends its whole conversation every turn, and the engine's
cache almost always already holds it, so the shim loads only the turns it has not seen
(docs/BENCHMARKS.md: 17.78 s -> 9.14 s over three turns, 1.95x). The first version of that
idea was 31% faster and answered DIFFERENT questions, because the cache built by GENERATING
a turn is not the cache built by PREFILLING the same text. Speed is the easy half; this gate
is the other half.

Two legs, one window, same machine state, only the switch differing:

  A  TM_SHIM_NO_REUSE=1   every request resets the cache and replays the whole history from
                          the client's text — exactly the behaviour before reuse existed,
                          and therefore the reference the fast path has to reproduce
  B  reuse (default)      only the new turns are loaded; each finished turn is canonicalized
                          (rewind to the engine's turn boundary, re-prefill from text) so the
                          cache holds the replay form rather than the sampled one

WHAT IT ASSERTS
  1. Every reply hash is identical between the legs. This is the whole gate: the bench in
     docs/BENCHMARKS.md records the same hashes, and a faster path that moves one of them is
     a bug, not a speed-up.
  2. The switch works in both directions — leg B's log shows reuse firing, leg A's log shows
     it never fires. A "reuse is on" that silently never reuses is indistinguishable from off
     from the outside, which is why the debug line exists and why it is asserted.
  3. On a real model, leg B beats leg A on wall-clock by a floor under the measured range. On
     the random-weight fixture CI uses, a request costs about a millisecond, so the ratio
     measures process noise and is REPORTED rather than asserted.

Skipping (exit 77) when the engine binary, the model or the tokenizer is absent.

Run:  python3 tensormark/test_shim_reuse.py
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
MODEL = os.environ.get("TM_TEST_MODEL", "tensormark/data/tinyllama/tinyllama_q40.tmq")
TOKENIZER = os.environ.get("TM_TEST_TOKENIZER", "tensormark/data/tinyllama/tokenizer.model")
PREAMBLE = "tensormark/sp_tokenizer.h"

TURNS = 3
BUDGET = 64
CTX = 4096
# Measured 1.95x (17.78 s -> 9.14 s, docs/BENCHMARKS.md). The floor sits well under it so a
# loaded machine does not fail the gate, while a reuse path that silently stopped firing —
# or one that re-loads the prefix anyway — still does.
RATIO_FLOOR = 1.25


def die(msg: str, code: int) -> None:
    print(msg)
    raise SystemExit(code)


if not (REPO / "tensormark/build/llama_chat_metal").exists():
    die("SKIP: run ./tensormark/build_llama.sh", 77)
for _p, _hint in ((MODEL, "set TM_TEST_MODEL"), (TOKENIZER, "set TM_TEST_TOKENIZER")):
    if not (REPO / _p).exists():
        die(f"SKIP: no model/tokenizer at {_p} ({_hint})", 77)

# A real checkpoint (tinyllama by default) makes the speed-up assertable; TM_TEST_MODEL marks
# the CI fixture, whose replies are near-instant and whose timings mean nothing.
REAL_MODEL = not os.environ.get("TM_TEST_MODEL")


def post(port: int, messages: list[dict], timeout: float = 600.0) -> tuple[float, str]:
    """One /api/chat request. Returns (seconds, reply text)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps({"messages": messages, "budget": BUDGET, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.time() - t0, str((body.get("message") or {}).get("content", ""))


def wait_ready(port: int, proc: subprocess.Popen, log: pathlib.Path) -> None:
    for _ in range(240):                              # an engine start can be slow cold
        if proc.poll() is not None:
            die(f"FAIL  shim exited before it served (rc={proc.returncode}):\n"
                f"{log.read_text()[-1500:]}", 1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ps", timeout=1).read()
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    die(f"FAIL  shim never came up on {port}:\n{log.read_text()[-1500:]}", 1)


def run_leg(port: int, env: dict, words: list[str]) -> tuple[list[str], float, pathlib.Path]:
    """Drive the conversation shape an agent produces. Returns (hashes, seconds, log)."""
    messages = [
        {"role": "system", "content": "You are a terse coding assistant."},
        {"role": "user", "content": "Here is a file:\n" + " ".join(words) +
                                    "\n\nTurn 1: describe it in one sentence."},
    ]
    log = pathlib.Path(tempfile.mkstemp(prefix=f"tm-reuse-{port}-", suffix=".log")[1])
    with log.open("w") as fh:
        proc = subprocess.Popen(
            [sys.executable, str(REPO / "examples/from_ollama.py"),
             "--model", MODEL, "--tokenizer", TOKENIZER, "--port", str(port),
             "--ctx", str(CTX), "--idle-ttl", "0", "--lanes", "1"],
            env={**os.environ, **env}, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
        try:
            wait_ready(port, proc, log)
            hashes, total = [], 0.0
            for turn in range(1, TURNS + 1):
                wall, reply = post(port, messages)
                total += wall
                hashes.append(hashlib.sha1(reply.encode()).hexdigest()[:12])
                print(f"    turn {turn}: {wall:6.2f}s  {len(reply):5d} chars  {hashes[-1]}")
                if turn < TURNS:
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user",
                                     "content": f"Turn {turn + 1}: say more about that."})
            print(f"    total {total:.2f}s")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()                            # the engine should be gone already
                proc.wait(timeout=10)
    return hashes, total, log


words = (REPO / PREAMBLE).read_text().split()[:1200]
failures = 0

print("== shim cache reuse gate ==")
print(f"  model {MODEL}, {TURNS} turns, budget {BUDGET}, ctx {CTX}")

print("  leg A: no reuse (reference — a full replay per request)")
ha, ta, path_a = run_leg(11450, {"TM_SHIM_NO_REUSE": "1"}, words)

print("  leg B: reuse, debug on")
hb, tb, path_b = run_leg(11451, {"TM_SHIM_DEBUG": "1"}, words)

log_a, log_b = path_a.read_text(), path_b.read_text()

if ha == hb:
    print(f"  ok   every reply identical across legs ({' '.join(ha)})")
else:
    print(f"  FAIL reuse changed the answers: no-reuse {' '.join(ha)} vs reuse {' '.join(hb)}")
    for i, (x, y) in enumerate(zip(ha, hb), 1):
        if x != y:
            print(f"       first divergence on turn {i}: {x} -> {y}")
    failures += 1

fired = [ln for ln in log_b.splitlines() if "reuse:" in ln]
if fired:
    print(f"  ok   reuse fired: {fired[-1].strip()}")
else:
    print("  FAIL reuse never fired — the leg measured the reference, not the fast path"
          f"\n       {log_b[-600:]}")
    failures += 1

if "reuse:" in log_a:
    print("  FAIL TM_SHIM_NO_REUSE=1 still reused a prefix")
    failures += 1
else:
    print("  ok   TM_SHIM_NO_REUSE=1 replayed from scratch, as the reference must")

ratio = ta / tb if tb > 0 else 0.0
print(f"  wall-clock: no reuse {ta:.2f}s, reuse {tb:.2f}s -> {ratio:.2f}x")
if REAL_MODEL:
    if ratio >= RATIO_FLOOR:
        print(f"  ok   reuse beat the full replay by >= {RATIO_FLOOR}x")
    else:
        print(f"  FAIL reuse was not faster than the full replay ({ratio:.2f}x)")
        failures += 1
else:
    print("  note fixture model: the ratio is reported but not asserted (a request there"
          "\n       costs about a millisecond, so it measures startup, not the prefix)")

for p in (path_a, path_b):
    p.unlink(missing_ok=True)
print("PASS  shim cache reuse" if failures == 0 else "FAIL  shim cache reuse")
raise SystemExit(1 if failures else 0)
