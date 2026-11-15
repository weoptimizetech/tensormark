#!/usr/bin/env bash
# Does the attention-gate gate actually catch the bug it exists for?
#
# Method: reintroduce the stray write that used to sit in the head-interleaved
# split — one row past its slot, the write that SIGSEGV'd on long prompts — rebuild,
# and require the gate to fail. Then restore and require it to pass.
#
# EVERYTHING HAPPENS IN A COPY. An earlier revision patched llama.h in place and
# held that patch for the whole (~15 minute) ASan compile — so for minutes at a
# time the working tree contained a deliberately buggy engine, and anything else
# running concurrently (an export, a commit) would have shipped it. A verification
# tool must not mutate the tree it verifies; this one copies the headers to a
# temporary directory and builds there.
#
# Run the buggy binary REPEATEDLY rather than once, and read the aggregate. The
# stray write is 64 bytes past a page-rounded buffer, so whether it lands in a
# mapped page is malloc's decision, not the test's. Measured 2026-09-15 by putting
# the stray write back, and the measurement is itself the lesson: five samples of
# the same binary gave 5/12, 2/12 and 0/12 across the whole sweep and 5/24, 4/24 and
# 10/24 at one size per fresh process — roughly one run in four, with a spread wide
# enough that no single count should be quoted. Smaller samples were worse than
# useless: "3 of 6", "2 of 6", "0 of 6" and one "6 of 6" were the same coin landing
# differently (that last one was really the fixture aborting on a context
# overflow). A single exit code is not evidence of anything; the two loops cover
# both regimes precisely because neither is reliable, and the ASan report is what
# settles it.
#
# TM_ASAN_GATES=1 additionally builds it under AddressSanitizer, which is the only
# configuration that catches the overrun on EVERY run. Two caveats, both learned
# the hard way. (1) The sanitizer has to be one that actually reports: Apple's
# clang deadlocks in its own initializer on macOS 26 (the stack is recorded in
# probe_asan_mechanism.cpp), so every instrumented run below is preceded by that
# probe and TM_ASAN_CC picks a compiler whose runtime works. (2) Every instrumented
# run is BOUNDED: a deadlocked sanitizer prints nothing and never exits, which is
# indistinguishable from work in progress, and a gate that hangs is worse than one
# that fails.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
WORK="$(mktemp -d /tmp/tm_gate_verify.XXXXXX)" || exit 2
trap 'rm -rf "$WORK"' EXIT

# Only the headers and the test: this is a CPU build, and the fixture is generated
# in-process, so no converter, tokenizer or checkpoint is involved. The layout has
# to mirror the repository (tensormark/ beside the root source), because
# safetensors.h and engine.h include "../neural_demo.cpp" from there.
mkdir -p "$WORK/tensormark" || exit 2
cp "$ROOT"/tensormark/*.h "$ROOT"/tensormark/test_llama_attn_gate.cpp \
   "$WORK/tensormark/" || exit 2
cp "$ROOT"/neural_demo.cpp "$WORK/" || exit 2
cd "$WORK/tensormark" || exit 2

# Wall-clock limit for anything that can hang (see run_bounded.sh): the sanitizer
# deadlock is a property of the host, not of the code under test.
bounded() { bash "$ROOT/tensormark/run_bounded.sh" "$@"; }

python3 - <<'PY' || exit 2
import pathlib, sys
p = pathlib.Path("llama.h")
s = p.read_text()
anchor = """                        for (int d = 0; d < dh; ++d)
                            dstq[(std::size_t)hd * dh + d] = src[(std::size_t)hd * 2 * dh + d];"""
stray = """                        std::copy_n(src + (std::size_t)hd * 2 * dh, dh,
                                    g + (std::size_t)(hd + 1) * dh);
"""
if anchor not in s:
    sys.exit("anchor for the interleaved split not found — did the code move?")
if stray in s:
    sys.exit("the stray write is already present; nothing to reintroduce")
p.write_text(s.replace(anchor, stray + anchor, 1))
print("reintroduced the stray write (in the copy)")
PY

# The two loops are not equally good at this, and the numbers say which to trust.
# Measured across four runs: the focused loop caught 5/24, 4/24 and 10/24, while the
# whole sweep caught 5/12, 2/12 and 0/12. The sweep is the weaker detector — it runs
# every size in one process, so the buffers grown by the earlier sizes leave exactly
# the capacity slack the stray row needs to land inside the allocation. Both loops
# still run (the sweep is what CI exercises, and it costs seconds), and the verdict
# needs only one of them to fire: at p ≈ 0.25 for the focus loop alone, both missing
# together is ~1e-4.
SWEEP_RUNS=12   # the whole sweep in one process, sizes 1 .. 1024
FOCUS_T=32      # one size per fresh process: no capacity slack for the stray row
FOCUS_RUNS=24
clang++ -std=c++23 -O2 -I. -DACCELERATE_NEW_LAPACK test_llama_attn_gate.cpp \
    -o gate_buggy -framework Accelerate || exit 2
hits=0
for _ in $(seq 1 "$SWEEP_RUNS"); do
    ./gate_buggy > /dev/null 2>&1 || hits=$((hits + 1))
done
focus_hits=0
for _ in $(seq 1 "$FOCUS_RUNS"); do
    TM_GATE_T="$FOCUS_T" ./gate_buggy > /dev/null 2>&1 || focus_hits=$((focus_hits + 1))
done
caught_plain=$((hits + focus_hits))
echo "  buggy build:   sweep $hits/$SWEEP_RUNS, T=$FOCUS_T $focus_hits/$FOCUS_RUNS"

asan_hit=0
asan_ok=0
if [ -n "${TM_ASAN_GATES:-}" ]; then
    ASAN_CC=${TM_ASAN_CC:-clang++}
    # Does THIS sanitizer report at all? A silent one turns every "ASan clean"
    # below into a false pass, and a deadlocked one hangs. The probe is a
    # self-contained 64-byte overrun (same shape as this bug) plus a control that
    # stays in bounds; the control exits 42 by design — 42 is the probe's way of
    # saying "no report here, as intended", so only the absence of a report in the
    # buggy case is a failure.
    if "$ASAN_CC" -std=c++23 -O0 -g -fsanitize=address \
            "$ROOT/tensormark/probe_asan_mechanism.cpp" -o probe_asan; then
        bounded 120 ./probe_asan buggy > probe_buggy.out 2>&1
        rc_probe=$?
        bounded 120 ./probe_asan control > probe_control.out 2>&1
        rc_control=$?
        if [ "$rc_probe" -eq 124 ] || [ "$rc_control" -eq 124 ]; then
            echo "  sanitizer:     never returned — deadlocked before main (probe_asan_mechanism.cpp)"
            echo "                 try: TM_ASAN_CC=\$(brew --prefix llvm)/bin/clang++ $0"
        elif grep -q "heap-buffer-overflow" probe_buggy.out \
             && ! grep -q "heap-buffer-overflow" probe_control.out; then
            asan_ok=1
            echo "  sanitizer:     reports a 64-byte overrun ($ASAN_CC); in-bounds control stays quiet"
        else
            echo "  sanitizer:     SILENT on a deliberate 64-byte overrun (buggy=$rc_probe control=$rc_control)"
            echo "                 whatever the gate says below, this host cannot judge it"
        fi
    else
        echo "  sanitizer:     $ASAN_CC cannot build the probe"
    fi
fi

if [ "$asan_ok" -eq 1 ]; then
    "$ASAN_CC" -std=c++23 -O1 -g -fsanitize=address -I. -DACCELERATE_NEW_LAPACK \
        test_llama_attn_gate.cpp -o gate_buggy_asan -framework Accelerate || exit 2
    bounded 600 ./gate_buggy_asan > asan_buggy.out 2>&1
    asan_hit=$(grep -cE "ERROR: AddressSanitizer" asan_buggy.out)
    echo "  buggy + ASan:  $asan_hit report(s)"
    grep -E "heap-buffer-overflow|WRITE of size" asan_buggy.out | head -2 | sed 's/^/      /'
fi

# The unmodified headers, compiled fresh: the fix must pass the same gate.
clang++ -std=c++23 -O2 -I"$ROOT/tensormark" -DACCELERATE_NEW_LAPACK \
    "$ROOT/tensormark/test_llama_attn_gate.cpp" -o gate_fixed -framework Accelerate || exit 2
rc_fixed=0
./gate_fixed > fixed.out 2>&1 || rc_fixed=$?
echo "  fixed build:   exit=$rc_fixed ($(grep -cE '^ok ' fixed.out) checks passed)"

echo
# Order matters. An over-strict gate is the most alarming outcome, so it is
# reported first; then a broken mechanical guard; only then the plain catch —
# whose absence is NOT evidence that the overrun has gone away.
if [ "$rc_fixed" -ne 0 ]; then
    echo "VERDICT: the fixed engine fails the gate — the gate is over-strict"
    exit 1
fi
if [ -n "${TM_ASAN_GATES:-}" ] && [ "$asan_ok" -eq 0 ]; then
    echo "VERDICT: the sanitizer on this host cannot be trusted (see 'sanitizer:' above),"
    echo "         so the mechanical guard cannot be judged from here. Nothing is wrong"
    echo "         with the test — re-run it against a working runtime:"
    echo "           TM_ASAN_CC=\$(brew --prefix llvm)/bin/clang++ TM_ASAN_GATES=1 $0"
    exit 1
fi
if [ -n "${TM_ASAN_GATES:-}" ] && [ "$asan_hit" -eq 0 ]; then
    echo "VERDICT: ASan did not report the overrun — the mechanical guard is broken"
    exit 1
fi
if [ "$caught_plain" -eq 0 ] && [ -z "${TM_ASAN_GATES:-}" ]; then
    echo "VERDICT: the plain build missed the bug in all $((SWEEP_RUNS + FOCUS_RUNS)) runs —"
    echo "         either the test is worthless, or the overrun no longer happens."
    echo "         Re-run with TM_ASAN_GATES=1 to tell those apart."
    exit 1
fi
if [ "$caught_plain" -eq 0 ]; then
    echo "VERDICT: the gate catches the bug — ASan reported it ($asan_hit report(s)) and the"
    echo "         fix passes. The plain build missed it in all $((SWEEP_RUNS + FOCUS_RUNS))"
    echo "         runs, which is allocator placement, not health: it has never caught this"
    echo "         reliably at any size."
else
    echo "VERDICT: the gate catches the bug — sweep $hits/$SWEEP_RUNS, T=$FOCUS_T $focus_hits/$FOCUS_RUNS${TM_ASAN_GATES:+, ASan $asan_hit report(s)} — and passes on the fix"
fi
echo "         the working tree was never modified; everything ran in $WORK"
