#!/usr/bin/env bash
# Grammar and structured-output gates.
#
# These gates need a built engine and (for their model half) a local model, so
# they self-skip with exit 77 when a prerequisite is absent — the convention the
# other gates in this repo use. 77 must NOT be treated as a failure: a gate that
# could not run has proved nothing, but it is not a broken build either. Every
# other non-zero exit is a real failure and is propagated.
#
# CI prepares a synthetic fixture, so all five gates below run their model halves
# there and none of them skips. A gate that is only ever run by hand is a gate that
# rots; a gate that is only ever SKIPPED is a gate that is not there.
#
# TM_GATES_REQUIRE=1 flips that: a skip becomes a FAILURE. That is what CI wants —
# there the fixture IS prepared, so a skip means the preparation broke, and the
# point of the step is that "the gate ran" must not quietly degrade into "the gate
# proved nothing".
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

rc=0
run() {
    local label="$1"; shift
    "$@"
    local r=$?
    case "$r" in
        0)  printf 'PASS  %s\n' "$label" ;;
        77) if [ -n "${TM_GATES_REQUIRE:-}" ]; then
                printf 'FAIL  %s (skipped, but TM_GATES_REQUIRE=1 — prepare the fixture)\n' "$label"
                rc=1
            else
                printf 'SKIP  %s (prerequisite absent)\n' "$label"
            fi ;;
        *)  printf 'FAIL  %s (exit %s)\n' "$label" "$r"; rc=1 ;;
    esac
}
run "grammar switch (engine /grammar protocol)" python3 -u tensormark/test_grammar_switch.py
run "json schema -> gbnf -> valid document"     python3 -u tensormark/test_jsonschema_to_gbnf.py
run "pooled embeddings (CLI / protocol / norm)" python3 -u tensormark/test_embeddings.py

# The shim gate runs on the generated fixture too. It used to opt out
# (`env TM_TEST_MODEL= TM_TEST_TOKENIZER=`) on the grounds that a random-weight
# model "stalls the shim mid-stream, so subsequent requests go unanswered". Measured
# 2026-09-15 against exactly this fixture: 14 of 14 pass in 8 seconds, the schema
# check included. What this gate asserts is response FORMAT — a terminal done:true,
# a document that validates, the budget honoured, the cache clean between requests —
# and the grammar mask, not the weights, decides every one of those. The opt-out made
# the serving path the one thing CI never exercised.
run "shim grammar + schema over HTTP" bash tensormark/test_shim_grammar.sh

# The serving contract has a second half that is pure lifecycle: the model must come
# back out of memory when nobody is using it. That is not observable from a reply —
# it is a child process, and /api/ps is only honest because the gate also checks the
# PID — so it gets its own gate rather than a line in the one above.
run "shim idle unload (reap, reload, keep_alive)" bash tensormark/test_shim_idle.sh

# ...and a third half, which only exists because the machine has two compute units:
# the lanes. One engine process serves one request at a time, so using both units
# means two processes, and then somebody has to decide which request goes where -
# and whether the second lane is still earning its memory. That decision is a
# controller, so it gets a gate that measures it rather than a paragraph that
# claims it.
run "shim lane pool (two engines, schedule, park)" bash tensormark/test_shim_lanes.sh

# ...and the primitive that makes a cached conversation reusable at all. The shim
# tried to reuse one without it and answered different questions (docs/BENCHMARKS.md),
# so the rewind has to be shown to reproduce a replayed cache byte for byte before
# anything is allowed to depend on it.
run "chat rewind reproduces a replayed cache"   python3 -u tensormark/test_chat_rewind.py

# ...and the decision to actually DEPEND on it. The rewind gate proves the mechanism on one
# turn; this one proves the shim's whole conversation path — every turn is canonicalized as
# it is produced, every following request reuses it — answers exactly what a full replay
# answers, through HTTP, on all three turns. That is the assertion the first (31% faster,
# wrong) attempt would have failed, and the reason a speed-up here is allowed to ship at all.
run "shim cache reuse reproduces a full replay" python3 -u tensormark/test_shim_reuse.py
exit "$rc"
