#!/usr/bin/env python3
"""test_shim_ctx_edge.py — reuse runs to the window's edge, and past it is an error.

Two defects this gate exists for, both on the conversation shape an agent produces
(system prompt + a history that grows every turn):

  1. The shim used to stop reusing the cache once its chars/4 proxy passed 60 % of
     the window, so the LONGEST conversations — the ones where reuse pays most —
     silently paid a full re-prefill per turn. Now the engine's own accounting is
     the only limit.
  2. When a conversation outgrew the window, the engine RESET it mid-replay and
     answered anyway: the system prompt and the early turns were gone, the reply was
     generated without them, and the client got a 200. Now the engine refuses the
     turn with the cache untouched (`/overflow refuse`), and the shim answers with
     the error the protocol has for it (HTTP 400, `context_length_exceeded`).

Two legs, one window, TinyLlama at a deliberately small --ctx so the edge is reached
in a few turns:

  A  TM_SHIM_NO_REUSE=1   full replay per request (the reference)
  B  reuse (default), debug on

WHAT IT ASSERTS
  * every reply that fit is hash-identical across legs — reuse near the edge does not
    move an answer;
  * reuse FIRED on the last fitting turn, whose history is past 60 % of the window
    (the band the proxy guard used to forfeit);
  * the first turn that does not fit is HTTP 400 with code context_length_exceeded
    on BOTH legs — never a 200 with a reply;
  * the session is still consistent afterwards: a short fresh conversation on the
    same shim answers 200, and answers the same on both legs.

Skipping (exit 77) when the engine binary, the model or the tokenizer is absent.

Run:  python3 tensormark/test_shim_ctx_edge.py
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

CTX = 512  # small on purpose: the edge must be reached in a handful of turns
BUDGET = 16
MAX_TURNS = 40  # safety stop; the window is expected to fill long before this
# Long, common words: SentencePiece spends few tokens per character on them, so the
# history is LARGE by the shim's old chars/4 proxy while still fitting the window —
# which is what puts the last fitting turn well past the 60 % line that guard drew.
# (Short words go the other way: ~2 chars/token, and the band is never reached.)
FILLER = (
    "international understanding communication organization development "
    "information environment government management responsibility"
)


def die(msg: str, code: int) -> None:
    print(msg)
    raise SystemExit(code)


if not (REPO / "tensormark/build/llama_chat_metal").exists():
    die("SKIP: run ./tensormark/build_llama.sh", 77)
for _p, _hint in ((MODEL, "set TM_TEST_MODEL"), (TOKENIZER, "set TM_TEST_TOKENIZER")):
    if not (REPO / _p).exists():
        die(f"SKIP: no model/tokenizer at {_p} ({_hint})", 77)


def post(port: int, messages: list[dict]) -> tuple[int, dict]:
    """POST /v1/chat/completions; returns (status, body) without raising on 4xx."""
    body = json.dumps({"messages": messages, "max_tokens": BUDGET, "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_ready(port: int, proc: subprocess.Popen, log: pathlib.Path) -> None:
    for _ in range(240):
        if proc.poll() is not None:
            die(
                f"FAIL  shim exited before it served (rc={proc.returncode}):\n"
                f"{log.read_text()[-1500:]}",
                1,
            )
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ps", timeout=1).read()
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    die(f"FAIL  shim never came up on {port}:\n{log.read_text()[-1500:]}", 1)


def run_leg(port: int, env: dict) -> tuple[dict, pathlib.Path]:
    """Grow one conversation until the window refuses it. Returns the record + log."""
    messages = [{"role": "system", "content": "You are a terse assistant. One short sentence."}]
    rec: dict = {"hashes": [], "chars_before_last_ok": 0, "overflow": None, "after": None}
    log = pathlib.Path(tempfile.mkstemp(prefix=f"tm-ctxedge-{port}-", suffix=".log")[1])
    with log.open("w") as fh:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(REPO / "examples/from_ollama.py"),
                "--model",
                MODEL,
                "--tokenizer",
                TOKENIZER,
                "--port",
                str(port),
                "--ctx",
                str(CTX),
                "--idle-ttl",
                "0",
                "--lanes",
                "1",
            ],
            env={**os.environ, **env},
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO),
        )
        try:
            wait_ready(port, proc, log)
            for turn in range(1, MAX_TURNS + 1):
                messages.append(
                    {
                        "role": "user",
                        "content": f"Turn {turn}. Note {turn}: {FILLER}. Which turn is this?",
                    }
                )
                history_chars = sum(len(m["content"]) for m in messages)
                status, body = post(port, messages)
                if status == 200:
                    reply = body["choices"][0]["message"]["content"]
                    h = hashlib.sha1(reply.encode()).hexdigest()[:12]
                    rec["hashes"].append(h)
                    rec["chars_before_last_ok"] = history_chars
                    print(f"    turn {turn:2d}: 200  history {history_chars:5d} chars  {h}")
                    messages.append({"role": "assistant", "content": reply})
                    continue
                rec["overflow"] = (turn, status, body)
                code = (body.get("error") or {}).get("code")
                print(
                    f"    turn {turn:2d}: {status}  history {history_chars:5d} chars  "
                    f"-> {code} ({str((body.get('error') or {}).get('message'))[:60]}...)"
                )
                break
            else:
                print(f"    never overflowed in {MAX_TURNS} turns — raise the filler or lower CTX")
            # The session must still be consistent: a short fresh conversation answers.
            status, body = post(
                port,
                [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Say hello."},
                ],
            )
            reply = body["choices"][0]["message"]["content"] if status == 200 else ""
            rec["after"] = (status, hashlib.sha1(reply.encode()).hexdigest()[:12])
            print(f"    after : {status}  fresh conversation  {rec['after'][1]}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    return rec, log


failures = 0
print("== shim context-edge gate ==")
print(f"  model {MODEL}, ctx {CTX}, budget {BUDGET}")

print("  leg A: no reuse (reference — a full replay per request)")
ra, log_a = run_leg(11452, {"TM_SHIM_NO_REUSE": "1"})
print("  leg B: reuse, debug on")
rb, log_b = run_leg(11453, {"TM_SHIM_DEBUG": "1"})
text_b = log_b.read_text()

n = min(len(ra["hashes"]), len(rb["hashes"]))
if n and ra["hashes"][:n] == rb["hashes"][:n]:
    print(f"  ok   {n} fitting replies identical across legs")
else:
    print(f"  FAIL replies differ: A {' '.join(ra['hashes'])} vs B {' '.join(rb['hashes'])}")
    failures += 1

if len(ra["hashes"]) != len(rb["hashes"]):
    print(
        f"  FAIL the legs overflowed on different turns: A after {len(ra['hashes'])}, "
        f"B after {len(rb['hashes'])}"
    )
    failures += 1

fired = [ln for ln in text_b.splitlines() if "reuse:" in ln]
proxy_share = rb["chars_before_last_ok"] / 4 / CTX
if fired and proxy_share > 0.6:
    print(
        f"  ok   reuse fired on the last fitting turn, history at {proxy_share:.0%} of the "
        f"window by the old proxy (> 60 %): {fired[-1].strip()}"
    )
elif not fired:
    print("  FAIL reuse never fired in leg B")
    failures += 1
else:
    print(
        f"  FAIL the conversation never passed 60 % of the window ({proxy_share:.0%}); "
        f"the band this gate exists for was not exercised"
    )
    failures += 1

for name, rec in (("A", ra), ("B", rb)):
    ov = rec["overflow"]
    if ov is None:
        print(f"  FAIL leg {name} never hit the window")
        failures += 1
        continue
    turn, status, body = ov
    code = (body.get("error") or {}).get("code")
    if status == 400 and code == "context_length_exceeded":
        print(f"  ok   leg {name}: turn {turn} did not fit -> 400 context_length_exceeded")
    else:
        print(
            f"  FAIL leg {name}: turn {turn} did not fit -> {status} {code!r} (want 400 "
            f"context_length_exceeded); body {json.dumps(body)[:200]}"
        )
        failures += 1

if ra["after"] and rb["after"] and ra["after"][0] == 200 and ra["after"] == rb["after"]:
    print(
        f"  ok   a fresh conversation after the overflow answers 200 and identically "
        f"({ra['after'][1]})"
    )
else:
    print(f"  FAIL after the overflow: A {ra['after']} vs B {rb['after']}")
    failures += 1

for p in (log_a, log_b):
    if failures:
        print(f"  log kept: {p}")
    else:
        p.unlink(missing_ok=True)
print("PASS  shim context edge" if failures == 0 else "FAIL  shim context edge")
raise SystemExit(1 if failures else 0)
