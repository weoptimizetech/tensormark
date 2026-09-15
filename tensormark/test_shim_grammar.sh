#!/usr/bin/env bash
# test_shim_grammar.sh — end-to-end gate for the Ollama-shaped shim's grammar path.
#
# test_grammar_switch.py gates the engine's protocol; this gates the HTTP surface
# a pipeline actually talks to: an inline `"grammar"` in the JSON body must reach
# the engine and constrain the reply, a grammar the engine rejects must be a 400
# and never a silent fallback to unconstrained decoding, and a request that asks
# for no grammar must not inherit the previous request's schema.
#
# It gates the serving contract too, because each of these was silently wrong at
# some point: `budget` must actually limit the reply (it was accepted and dropped,
# so every request ran at the engine's own 1024), a request must start from a clean
# cache (the engine carries a conversation across turns by design, so a stateless
# API has to say so out loud), and /api/chat and /api/tags must answer with the
# shape and the model they claim.
#
# Starts the shim on a private port and kills EXACTLY the PID it started — a
# server left running is how a laptop wakes up to a 100% CPU process (the
# llama-cli incident), so this script never exits with the shim alive.
#
# Run:  bash tensormark/test_shim_grammar.sh
# Skips (exit 77) when the binary or the model is absent.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PORT=11439
LOG=$(mktemp -t tm-shim-gate.XXXXXX)
R=0
pass() { echo "  ok   $1"; }
fail() { echo "  FAIL $1"; R=1; }

[ -x tensormark/build/llama_chat_metal ] || { echo "SKIP: run ./tensormark/build_llama.sh"; exit 77; }
# TM_TEST_MODEL / TM_TEST_TOKENIZER point at a generated fixture
# (tensormark/gen_tiny_llama.py) that satisfies the same contract as the fetched
# checkpoint, so this gate runs in CI instead of skipping for want of a model.
TM_MODEL="${TM_TEST_MODEL:-tensormark/data/tinyllama/tinyllama_q40.tmq}"
TM_TOKENIZER="${TM_TEST_TOKENIZER:-tensormark/data/tinyllama/tokenizer.model}"
[ -f "$TM_MODEL" ] || { echo "SKIP: run python3 examples/fetch_model.py"; exit 77; }

# -u matters: without it the readiness banner sits in Python's stdout buffer and
# the wait loop below times out on a shim that is already serving.
python3 -u examples/from_ollama.py \
  --model "$TM_MODEL" \
  --tokenizer "$TM_TOKENIZER" \
  --port "$PORT" >"$LOG" 2>&1 &
SHIM=$!
# EXIT plus the signals a killed runner actually sends — only SIGKILL is
# uncatchable. Cleaning up on EXIT alone is not enough: a run stopped with TERM
# leaked the shim AND its engine child, which between them hold a ~600 MB model
# open (one was found still resident 6h26m after the run it belonged to).
# The engine is a child of the shim and does not exit when its stdin closes, so it
# has to be killed explicitly.
cleanup() {
  for child in $(pgrep -P "$SHIM" 2>/dev/null); do kill "$child" 2>/dev/null; done
  kill "$SHIM" 2>/dev/null
  wait "$SHIM" 2>/dev/null
  rm -f "$LOG"
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
  grep -q "ollama shim on" "$LOG" 2>/dev/null && break
  sleep 1
done
if ! grep -q "ollama shim on" "$LOG"; then
  echo "FAIL: shim never came up"; cat "$LOG"; exit 1
fi
pass "shim up (pid $SHIM)"

# Budgets are kept small throughout: under a permissive grammar a 1.1B model
# free-runs until its budget dies, so an unbounded request here would take
# minutes rather than seconds.
post() {
  curl -s --max-time 240 -X POST "http://127.0.0.1:$PORT/api/generate" \
       -H 'Content-Type: application/json' -d "$1"
}

# 1. a closed grammar must be honoured byte for byte
OUT=$(post '{"prompt":"Answer yes or no.","grammar":"root ::= \"yes\" | \"no\"\n","stream":false,"budget":24}')
printf '%s' "$OUT" | python3 -c '
import json,sys
r=json.load(sys.stdin).get("response","").strip()
assert r in ("yes","no"), f"closed grammar leaked: {r!r}"
print(f"  ok   closed grammar over HTTP -> {r!r}")' || fail "closed grammar: $OUT"

# 2. an inline JSON grammar yields a legal JSON prefix, and the bytes are valid UTF-8
OUT=$(post '{"prompt":"Output a JSON object.","grammar":"root ::= \"{\" ws \"\\\"a\\\"\" ws \":\" ws num ws \"}\"\nnum ::= [0-9]+\nws ::= [ ]*\n","stream":false,"budget":24}')
printf '%s' "$OUT" | python3 -c '
import json,sys
r=json.load(sys.stdin).get("response","")
assert r.lstrip().startswith("{"), f"not JSON-shaped: {r!r}"
r.encode("utf-8")                     # must round-trip: no truncated codepoints
print(f"  ok   JSON grammar over HTTP -> {r[:44]!r}")' || fail "JSON grammar: $OUT"

# 2b. a JSON SCHEMA (not a hand-written grammar) produces a document that both
#     parses AND satisfies the schema. This is the capability the shim exists
#     for; parsing alone is not enough — a document can be legal JSON and
#     still be missing a required key.
OUT=$(post '{"prompt":"Fill in the fields.","schema":{"type":"object","additionalProperties":false,"required":["subject","relation"],"properties":{"subject":{"type":"string","maxLength":8},"relation":{"enum":["is-a","part-of"]}}},"stream":false,"budget":96}')
printf '%s' "$OUT" | python3 -c '
import json,sys
r=json.load(sys.stdin).get("response","")
d=json.loads(r)                                  # must be JSON at all
assert set(d)=={"subject","relation"}, f"wrong keys: {sorted(d)}"
assert d["relation"] in ("is-a","part-of"), f"bad enum: {d}"
s = d["subject"]
assert 1 <= len(s) <= 8, f"maxLength violated: {s!r}"
print(f"  ok   JSON schema over HTTP -> {r[:56]!r}")' || fail "schema: $OUT"

# 3. a grammar the engine rejects is a 400, never a silent fallback
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 \
         -X POST "http://127.0.0.1:$PORT/api/generate" -H 'Content-Type: application/json' \
         -d '{"prompt":"hi","grammar":"root ::= [unterminated\n","stream":false,"budget":24}')
[ "$CODE" = "400" ] && pass "bad grammar -> HTTP 400 (no silent fallback)" \
                    || fail "bad grammar returned $CODE (expected 400)"

# 4. no grammar still works, and does not inherit the rejected schema
OUT=$(post '{"prompt":"Name three colors.","stream":false,"budget":24}')
printf '%s' "$OUT" | python3 -c '
import json,sys
r=json.load(sys.stdin).get("response","")
assert r.strip(), "empty response without a grammar"
print(f"  ok   no grammar -> unconstrained ({len(r)} bytes)")' || fail "unconstrained: $OUT"

# 5. streaming still emits Ollama ndjson, terminates, and the terminal chunk
#    carries `total_duration` — the field a caller turns into tokens/s. Nothing
#    asserted that, and a restructure of the handler dropped it silently while the
#    non-streaming path went on reporting it.
N=$(curl -s --max-time 240 -X POST "http://127.0.0.1:$PORT/api/generate" \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"Say hi.","stream":true,"budget":24}' | python3 -c '
import json,sys
lines=[json.loads(l) for l in sys.stdin if l.strip()]
assert lines and lines[-1]["done"] is True, "no terminal done:true line"
assert lines[-1].get("total_duration", 0) > 0, f"terminal chunk has no total_duration: {lines[-1]}"
print(len(lines))')
if [ -n "${N:-}" ]; then pass "streaming ndjson ($N lines, ends done:true, with total_duration)"; else fail "streaming"; fi

# 6. the embedding surface: Ollama's two spellings, unit-norm vectors. Without
#    this the HTTP path for embeddings could rot silently — the CLI gate would
#    keep passing while every request over HTTP 500'd.
OUT=$(curl -s --max-time 240 -X POST "http://127.0.0.1:$PORT/api/embeddings" \
        -H 'Content-Type: application/json' -d '{"prompt":"The cat sat on the mat."}')
printf '%s' "$OUT" | python3 -c '
import json,math,sys
v=json.load(sys.stdin)["embedding"]
n=math.sqrt(sum(x*x for x in v))
assert 0.999 < n < 1.001, f"not unit-norm: |v| = {n}"
print(f"  ok   /api/embeddings -> {len(v)} values, |v| = {n:.6f}")' || fail "embeddings: $OUT"

OUT=$(curl -s --max-time 240 -X POST "http://127.0.0.1:$PORT/api/embed" \
        -H 'Content-Type: application/json' -d '{"input":["a cat","a dog"]}')
printf '%s' "$OUT" | python3 -c '
import json,sys
e=json.load(sys.stdin)["embeddings"]
assert len(e)==2, f"expected 2 vectors, got {len(e)}"
assert e[0]!=e[1], "the two inputs returned the same vector"
print(f"  ok   /api/embed -> {len(e)} distinct vectors, dim {len(e[0])}")' || fail "embed: $OUT"

# 7. the budget is APPLIED, not accepted and dropped. The grammar forces exactly
#    16 digits, so the reply length IS the budget unless the budget went missing —
#    in which case the engine's own 1024 applies and all 16 come back. This is the
#    regression that turned every request into a minutes-long monologue on a model
#    whose answers do not end by themselves.
BODY='{"prompt":"digits","grammar":"root ::= d d d d d d d d d d d d d d d d\nd ::= [0-9]\n","stream":false,"budget":4}'
OUT=$(post "$BODY")
printf '%s' "$OUT" | python3 -c '
import json,sys
r=json.load(sys.stdin).get("response","")
assert 1 <= len(r) <= 8, f"budget 4 was ignored: {len(r)} chars back: {r!r}"
print(f"  ok   budget honoured (asked 4, got {len(r)} chars)")' || fail "budget: $OUT"

# 8. a request starts from a CLEAN cache. The engine carries a conversation across
#    batch turns by design, so without an explicit reset the second identical
#    request is answered in the context of the first. Greedy decoding makes that
#    visible: same request, byte-identical reply, or the shim is stateful without
#    saying so — and a caller cannot tell a leaked context from a model quirk.
# Compare the COMPLETIONS, not the envelopes: `created_at` and `total_duration`
# differ between any two runs, so comparing the raw JSON fails whatever the cache
# did — and a check that cannot pass is worse than no check at all.
completion() { post "$1" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("response",""))'; }
A=$(completion '{"prompt":"Name a color.","stream":false,"budget":16}')
B=$(completion '{"prompt":"Name a color.","stream":false,"budget":16}')
[ -n "$A" ] && [ "$A" = "$B" ] && pass "repeated request -> identical completion (cache reset per request)" \
                || fail "identical requests differ: context leaked between them ($A | $B)"

# 9. /api/chat exists, is shaped like Ollama's, and refuses an empty history
CHAT='{"messages":[{"role":"user","content":"Say hi."}],"stream":false,"budget":16}'
OUT=$(curl -s --max-time 240 -X POST "http://127.0.0.1:$PORT/api/chat" \
        -H 'Content-Type: application/json' -d "$CHAT")
printf '%s' "$OUT" | python3 -c '
import json,sys
d=json.load(sys.stdin)
m=d.get("message") or {}
assert m.get("role")=="assistant", f"no assistant message: {d}"
c=m.get("content")
assert isinstance(c,str), f"content is not text: {m}"
print(f"  ok   /api/chat -> message.content ({len(c)} chars)")' || fail "chat: $OUT"

CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 -X POST "http://127.0.0.1:$PORT/api/chat" \
         -H 'Content-Type: application/json' -d '{"messages":[]}')
[ "$CODE" = "400" ] && pass "/api/chat with no messages -> HTTP 400" \
                    || fail "/api/chat with no messages returned $CODE (expected 400)"

# 10. /api/tags reports the model it is actually serving. `size` was hardcoded 0
#     and the family said "llama" whatever was loaded.
OUT=$(curl -s --max-time 60 "http://127.0.0.1:$PORT/api/tags")
printf '%s' "$OUT" | python3 -c '
import json,sys
m=json.load(sys.stdin)["models"][0]
name, size, fam = m["name"], m["size"], m["details"]["family"]
assert size > 0, "size is 0 — a client cannot size a context budget off that"
assert fam == "tensormark", f"family says {fam!r}"
print(f"  ok   /api/tags -> {name}, {size/1e6:.0f} MB")' || fail "tags: $OUT"

# `cleanup`, not a bare kill: the engine is a child that outlives its shim, so
# signalling only the shim leaves a ~600 MB process behind — the exact failure the
# trap above exists to prevent, and this line used to disable that trap without
# doing its job.
cleanup
trap - EXIT
echo
if [ "$R" = 0 ]; then echo "shim grammar gate: PASS"; else echo "shim grammar gate: FAIL"; fi
exit "$R"
