#!/usr/bin/env bash
# Run a command under a wall-clock limit.
#
# Why not `timeout`: stock macOS has no GNU timeout, and these gates have to run on
# the machine the engine targets. Why at all: the AddressSanitizer runtime on
# macOS 26 deadlocks inside its own initializer (the stack is in
# probe_asan_mechanism.cpp), so an instrumented build can hang BEFORE main — no
# output, no exit code, no clue. A gate that blocks forever is worse than one that
# fails: a failure gets reported and fixed, a hang just looks like work in
# progress.
#
# Usage:  run_bounded.sh <seconds> <command> [args...]
# Exit:   the command's own status, or 124 if the limit was reached.
#
# It reaps the process it started, not its descendants: every caller here runs a
# single-threaded test binary, so nothing else is left behind.
#
# The watchdog's file descriptors are closed off deliberately. An earlier revision
# let it inherit the caller's stdout: the watchdog was killed when the command
# exited, but the `sleep` it had already started survived for the rest of the limit
# HOLDING THE PIPE OPEN — so `run_bounded.sh ... | tail` never saw EOF and a
# finished build looked exactly like a hang. Nothing here may outlive its own limit
# by more than a second.
set -uo pipefail
if [ $# -lt 2 ]; then
    echo "usage: $0 <seconds> <command> [args...]" >&2
    exit 125
fi
limit=$1
shift
marker="$(mktemp)" || exit 125
trap 'rm -f "$marker"' EXIT

"$@" &
child=$!
(
    # Detach from the caller's streams first: this subshell must not be able to
    # keep a pipe open once the command it guards has finished.
    exec </dev/null >/dev/null 2>&1
    # Tick in one-second slices rather than one long sleep, so killing the watchdog
    # cannot leave a multi-minute sleeper behind.
    waited=0
    while [ "$waited" -lt "$limit" ]; do
        sleep 1
        kill -0 "$child" 2>/dev/null || exit 0
        waited=$((waited + 1))
    done
    # Written, not truncating (`: >`): the file is tested with -s below, and an
    # empty marker file would report a limit that was hit as a normal exit.
    echo triggered > "$marker"
    kill -TERM "$child" 2>/dev/null
    sleep 1
    kill -KILL "$child" 2>/dev/null
) &
watchdog=$!

wait "$child"
rc=$?
kill "$watchdog" 2>/dev/null
wait "$watchdog" 2>/dev/null
if [ -s "$marker" ]; then
    exit 124
fi
exit "$rc"
