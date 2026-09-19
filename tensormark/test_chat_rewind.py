#!/usr/bin/env python3
"""test_chat_rewind.py — a rewound cache must be the cache a replay would build.

Why this exists. The shim tried to keep a conversation in the engine's cache and load
only the new turns. It was 31% faster over six turns and it answered DIFFERENT
questions, because the cache built by GENERATING a turn is not the cache built by
PREFILLING the same text: the turn loop closes a reply with the end-of-turn marker,
while `prefill_turn` loads "{text}<eos>\\n", and the tokens reaching that marker are the
sampled ones, not a re-encoding of the decoded text. One token of difference moves the
continuation (docs/BENCHMARKS.md).

The sound form is to rewind to the end of the last turn whose tokens came from the
tokenizer and re-prefill from there, so the cache holds exactly what a replay would
hold. That is what /pos and /rewind exist for, and this gate is the proof obligation:

  A  reference   reset, prefill U, generate reply1, RESET, replay the whole
                 conversation from text, ask Q2
  B  rewind      generate reply1, rewind to the end of the user turn, re-prefill
                 reply1 FROM TEXT, ask Q2        <- must equal A, byte for byte
  C  raw reuse   keep the generated reply, ask Q2 <- reported, expected to differ

B == A is the assertion. C is reported rather than asserted: on a random-weight fixture
the reply is often empty, and an empty reply cannot show the difference this gate
exists to catch, so failing on C there would be failing on the fixture instead of on
the product.

Run:  python3 tensormark/test_chat_rewind.py       (exit 77 when it cannot run)
"""
import hashlib
import importlib.util
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
MODEL = os.environ.get("TM_TEST_MODEL",
                       "tensormark/data/tinyllama/tinyllama_q40.tmq")
TOKENIZER = os.environ.get("TM_TEST_TOKENIZER",
                           "tensormark/data/tinyllama/tokenizer.model")


def die(msg: str, code: int) -> None:
    print(msg)
    raise SystemExit(code)


if not (REPO / "tensormark/build/llama_chat_metal").exists():
    die("SKIP: run ./tensormark/build_llama.sh", 77)
if not (REPO / MODEL).exists():
    die(f"SKIP: no model at {MODEL} (set TM_TEST_MODEL)", 77)
if not (REPO / TOKENIZER).exists():
    die(f"SKIP: no tokenizer at {TOKENIZER} (set TM_TEST_TOKENIZER)", 77)

spec = importlib.util.spec_from_file_location("shim", REPO / "examples/from_ollama.py")
shim = importlib.util.module_from_spec(spec)
sys.modules["shim"] = shim
spec.loader.exec_module(shim)

U = " ".join((REPO / "tensormark/sp_tokenizer.h").read_text().split()[:900])
Q1 = "Turn 1: describe the file in one sentence."
Q2 = "Turn 2: name one function from it."
# U is ~2.1k tokens; the default window is 2048 and a history turn that does not fit
# is refused, so the session is sized for it explicitly.
ENV = {"TM_CTX": "4096"}


def sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def pos(s) -> int:
    return int(s.command("/pos").split("=")[1])


def session():
    return shim.LlamaChatSession(MODEL, TOKENIZER, "You are terse.", lane_env=ENV)


failures = 0

a = session()
a.command("/reset")
a.prefill("user", U)
r1a = a.complete(Q1, 32)
a.command("/reset")
a.prefill("user", U)
a.prefill("assistant", r1a)
a2 = a.complete(Q2, 32)
a.close()

b = session()
b.command("/reset")
b.prefill("user", U)
after_u = pos(b)
r1b = b.complete(Q1, 32)
after_r1 = pos(b)
if b.command(f"/rewind {after_u}") != f"rewind={after_u}":
    die(f"FAIL  /rewind {after_u} was refused (cache {after_r1} tokens)", 1)
b.prefill("assistant", r1b)
b2 = b.complete(Q2, 32)
b.close()

d = session()
d.command("/reset")
d.prefill("user", U)
r1d = d.complete(Q1, 32)
# The engine's OWN boundary, not a guess: the row where this turn's tokens start.
tp_ack = d.command("/turnpos")
if not tp_ack.startswith("turnpos=") or tp_ack == "turnpos=none":
    die(f"FAIL  /turnpos gave {tp_ack!r} after a turn", 1)
turn_start = int(tp_ack.split("=")[1])
if d.command(f"/rewind {turn_start}") != f"rewind={turn_start}":
    die(f"FAIL  /rewind {turn_start} (the engine's own turn boundary) was refused", 1)
d.prefill("assistant", r1d)
d2 = d.complete(Q2, 32)
d.close()

# D's reference is NOT A. A's context is [U][reply1]; D's is [U][Q1][reply1] — the
# Q1 turn is what the client actually sent, and /turnpos points at its end, so the
# comparison has to be against the same conversation replayed from text.
e = session()
e.command("/reset")
e.prefill("user", U)
e.prefill("user", Q1)
e.prefill("assistant", r1d)
e2 = e.complete(Q2, 32)
e.close()

c = session()
c.command("/reset")
c.prefill("user", U)
r1c = c.complete(Q1, 32)
c2 = c.complete(Q2, 32)          # the fast-but-wrong path, for contrast
c.close()

print(f"  reply1:  A {sha(r1a)}  B {sha(r1b)}  C {sha(r1c)}  D {sha(r1d)}")
print(f"  cache:   {after_u} tok after the user turn, {after_r1} after reply1 "
      f"(reply1 added {after_r1 - after_u}); /turnpos said {turn_start}")
if r1a != r1b or r1a != r1c or r1a != r1d:
    print("  FAIL reply1 differs between legs — the legs are not comparable")
    failures += 1
if d2 == e2:
    print(f"  ok   /turnpos boundary + re-prefill reproduces that replay ({sha(e2)})")
else:
    print(f"  FAIL /turnpos leg answered {sha(d2)}, its replay answered {sha(e2)}")
    failures += 1
if b2 == a2:
    print(f"  ok   rewind + re-prefill reproduces the replayed cache ({sha(a2)})")
else:
    print(f"  FAIL rewind leg answered {sha(b2)}, replay answered {sha(a2)}")
    failures += 1
if c2 == a2:
    print(f"  note raw reuse also matched here ({sha(c2)}) — a fixture reply cannot "
          f"show the difference; the real-model run is the one that must not match")
else:
    print(f"  note raw reuse differs ({sha(c2)}) — the defect this primitive exists for")

print("PASS  chat rewind" if failures == 0 else "FAIL  chat rewind")
raise SystemExit(1 if failures else 0)
