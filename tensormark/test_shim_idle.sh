#!/usr/bin/env bash
# test_shim_idle.sh — end-to-end gate for the shim's idle-unload contract.
#
# The engine holds its weights in unified memory for as long as it is resident, and
# it is a separate PROCESS, so "unload" means "reap": the memory comes back when the
# child exits, not when a cache is dropped. This gate checks the whole cycle over
# HTTP — resident at startup, gone after the idle TTL, back on demand with a new
# PID, unloadable on request with keep_alive 0, pinnable with keep_alive -1 — and it
# checks the two things that make an unload a real one rather than a bookkeeping
# change: the child PID no longer exists (a zombie or a leaked process would still
# answer `kill -0`), and a reload does not accumulate stderr temp files.
#
# THE ASSERTIONS RACE THE TTL, and the first CI run lost that race. Residency is
# asserted moments after a load, so every microsecond between the load and the probe
# is spent out of the TTL's budget: the poll loop below adds up to one granularity,
# `pgrep` and `curl` add their own, and CI's first engine start is slow (the Metal ops
# library compiles from source). It failed there with
#     FAIL the model should be resident after startup
#     FAIL no engine child process after startup
# while passing locally, which is exactly what a thin margin looks like. So: TTL is
# 5s (not 2), the poll is fine-grained, and the probe is `curl | grep` with no
# interpreter to cold-start. A gate that flakes is worse than no gate.
#
# Skips (exit 77) when the binary or the model is absent.
#
# Run:  bash tensormark/test_shim_idle.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PORT=11441
# Overridable so the gate can also be run deliberately broken: TM_SHIM_IDLE_TTL=0 pins
# the model, which must make the eviction checks FAIL. A gate that cannot fail is not
# evidence that anything works.
TTL="${TM_SHIM_IDLE_TTL:-5}"
LOG=$(mktemp -t tm-shim-idle.XXXXXX)
R=0
pass() { echo "  ok   $1"; }
fail() { echo "  FAIL $1"; R=1; }

[ -x tensormark/build/llama_chat_metal ] || { echo "SKIP: run ./tensormark/build_llama.sh"; exit 77; }
# TM_TEST_MODEL / TM_TEST_TOKENIZER point at a generated fixture
# (tensormark/gen_tiny_llama.py) that satisfies the same contract as the fetched
# checkpoint, so this gate runs in CI instead of skipping for want of a model. None of
# what it asserts depends on the weights: replies are checked for FORMAT, and the
# lifecycle is about processes.
TM_MODEL="${TM_TEST_MODEL:-tensormark/data/tinyllama/tinyllama_q40.tmq}"
TM_TOKENIZER="${TM_TEST_TOKENIZER:-tensormark/data/tinyllama/tokenizer.model}"
[ -f "$TM_MODEL" ] || { echo "SKIP: run python3 examples/fetch_model.py"; exit 77; }

# -u matters: without it the readiness banner sits in Python's stdout buffer and
# the wait loop below times out on a shim that is already serving.
python3 -u examples/from_ollama.py \
  --model "$TM_MODEL" \
  --tokenizer "$TM_TOKENIZER" \
  --port "$PORT" --idle-ttl "$TTL" >"$LOG" 2>&1 &
SHIM=$!
T0=$(date +%s.%N)
# EXIT plus the signals a killed runner actually sends. The engine is a child of the
# shim and does not exit when its stdin closes, so it has to be killed explicitly;
# a run stopped with TERM otherwise leaks the shim AND a ~600 MB engine with it.
cleanup() {
  for child in $(pgrep -P "$SHIM" 2>/dev/null); do kill "$child" 2>/dev/null; done
  kill "$SHIM" 2>/dev/null
  wait "$SHIM" 2>/dev/null
  rm -f "$LOG"
}
trap cleanup EXIT INT TERM

# 0.2s steps, not 1s: the loop's granularity is part of the budget residency is
# asserted out of (see the header), and 450 of them still bound this at 90s.
for _ in $(seq 1 450); do
  grep -q "ollama shim on" "$LOG" 2>/dev/null && break
  sleep 0.2
done
if ! grep -q "ollama shim on" "$LOG"; then
  echo "FAIL: shim never came up"; cat "$LOG"; exit 1
fi
T1=$(date +%s.%N)
pass "shim up (pid $SHIM) after $(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.2f", b-a}')s"

# The engine is the shim's direct child. Its existence, not a flag in the JSON, is
# what decides whether the memory is actually back.
engine_pid() { pgrep -P "$SHIM" 2>/dev/null | head -1; }
# /api/ps straight to a grep: no interpreter to cold-start in the middle of the window
# the TTL is ticking through. `expires_at` is present exactly when a model is loaded.
loaded() {
  curl -s --max-time 30 "http://127.0.0.1:$PORT/api/ps" | grep -q '"expires_at"'
}
ps_dump() { curl -s --max-time 10 "http://127.0.0.1:$PORT/api/ps" | head -c 200; }
generate() {
  curl -s --max-time 240 -X POST "http://127.0.0.1:$PORT/api/generate" \
    -H 'Content-Type: application/json' -d "$1"
}
# Temp files are named tm-shim-stderr-*.log and there is one per engine session;
# a reload that does not reap its file is how a server "unloads" models and still
# fills /tmp.
before=$(ls "${TMPDIR:-/tmp}"/tm-shim-stderr-*.log 2>/dev/null | wc -l | tr -d ' ')

# Dump what the shim itself recorded. Its load and unload lines carry the REASON
# ("idle over Ns", "keep_alive 0", "shutting down") and the idle time that triggered
# it, which is the only place that information exists.
dump() {
  echo "  ---- shim log ----"
  tail -20 "$LOG" | sed 's/^/  | /'
  echo "  ---- processes ----"
  ps -o pid,ppid,stat,etime,command -p "$SHIM" $(pgrep -P "$SHIM" 2>/dev/null) 2>/dev/null | sed 's/^/  | /'
  echo "  ------------------"
}

# The engine is the shim's direct child, and that it exists is half the contract: the
# weights are in ITS address space.
PID1=$(engine_pid)
if [ -n "$PID1" ]; then
  pass "engine child at startup (pid $PID1)"
else
  fail "no engine child process after startup"
  dump
fi

# ---------------------------------------------------------------- a real reply
# This is the assertion that has to hold, and it cannot race: a served request proves
# the model was loaded, and it RE-ARMS the idle clock, so the residency check below
# happens at a known instant inside a fresh window.
out=$(generate '{"prompt":"Name a color.","budget":8}')
echo "$out" | grep -q '"response"' || fail "generate returned no response: ${out:0:200}"
echo "$out" | grep -q '"done": true' || fail "generate did not report done"
if [ -n "$out" ]; then pass "generate served by pid ${PID1:-none}"; fi

# Residency is asserted HERE and not before the request. Probing immediately after the
# load means the probe's own latency (shell, pgrep, curl, and on a CI runner a cold
# python3) is spent out of the TTL's budget, which is a race the gate lost in CI and
# won locally — with --idle-ttl 2 it failed "should be resident at startup" while the
# engine it had just started was alive. The property worth asserting is not "loaded at
# instant T"; it is that a request is served and the model is then resident, both of
# which are deterministic.
if loaded; then
  pass "/api/ps reports the model resident after a served request"
else
  fail "/api/ps reports nothing loaded right after a served request ($(ps_dump))"
  dump
fi

# ---------------------------------------------------------------- idle unload
# The reply above re-armed the clock, so this waits TTL past a known instant.
sleep $((TTL + 3))
if [ -n "$(engine_pid)" ]; then
  fail "engine still alive $((TTL + 3))s after the last request"
else
  pass "engine reaped after ${TTL}s idle"
fi
if [ -n "${PID1:-}" ] && kill -0 "$PID1" 2>/dev/null; then
  # A zombie passes kill -0 too, which is exactly the leak worth catching: close()
  # must wait() the child, not just signal it.
  fail "engine pid $PID1 still exists (running or unreaped zombie)"
else
  pass "engine pid ${PID1:-none} is gone, not a zombie"
fi
if loaded; then fail "/api/ps still reports the model as resident"; else
  pass "/api/ps reports nothing loaded"
fi

# ---------------------------------------------------------------- reload on demand
out=$(generate '{"prompt":"Name a color.","budget":8}')
echo "$out" | grep -q '"response"' || fail "reload returned no response: ${out:0:200}"
PID2=$(engine_pid)
if [ -n "$PID2" ] && [ "$PID2" != "${PID1:-}" ]; then
  pass "reloaded on demand (new engine pid $PID2)"
elif [ -n "$out" ]; then
  fail "expected a fresh engine process, got '${PID2:-none}' (was ${PID1:-none})"
fi

# ---------------------------------------------------------------- keep_alive 0
generate '{"prompt":"Name a color.","budget":8,"keep_alive":0}' >/dev/null
sleep 2
if [ -n "$(engine_pid)" ]; then fail "keep_alive 0 did not unload the model"; else
  pass "keep_alive 0 unloads as soon as the reply is sent"
fi

# ---------------------------------------------------------------- keep_alive -1
generate '{"prompt":"Name a color.","budget":8,"keep_alive":-1}' >/dev/null
sleep $((TTL + 3))
if loaded; then
  pass "keep_alive -1 keeps it resident past the TTL"
else
  fail "keep_alive -1 must keep the model resident past the TTL ($(ps_dump))"
fi

# ---------------------------------------------------------------- malformed input
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 \
  -X POST "http://127.0.0.1:$PORT/api/generate" \
  -H 'Content-Type: application/json' -d '{"prompt":"hi","keep_alive":"5x"}')
[ "$code" = "400" ] || fail "malformed keep_alive should be 400, got $code"
[ "$code" = "400" ] && pass "malformed keep_alive -> 400"

# ---------------------------------------------------------------- no file leak
after=$(ls "${TMPDIR:-/tmp}"/tm-shim-stderr-*.log 2>/dev/null | wc -l | tr -d ' ')
# Three loads happened above (startup, reload, keep_alive -1), but each unloaded
# session must have removed its own file, so the count must not have grown by them.
if [ "$after" -le "$((before + 1))" ]; then
  pass "no stderr temp files accumulated ($before -> $after over 3 loads)"
else
  fail "stderr temp files leaked: $before -> $after over 3 loads"
fi

[ "$R" = 0 ] && echo "PASS  shim idle unload contract" || echo "FAIL  shim idle unload contract"
exit $R
