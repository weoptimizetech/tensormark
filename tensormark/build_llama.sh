#!/bin/bash
# build_llama.sh — compile the LLM inference engine tests and benchmarks.
#
# The engine is header-only C++23; the Metal paths additionally need the
# ObjC++ shim (metal_shim.mm) and the Metal/Foundation frameworks. Targets
# macOS on Apple Silicon with Xcode Command Line Tools (Accelerate, Metal).
#
#   ./build_llama.sh                 # everything below into build/
#
# Binaries that read a quantized model accept the model path as the first
# argument (default data/tinyllama/tinyllama_q40.tmq; weights are not
# bundled with the repository).
set -e
cd "$(dirname "$0")"
mkdir -p build

CXX=${CXX:-clang++}
# ACCELERATE_NEW_LAPACK: Apple deprecated the classic CBLAS interface in
# macOS 13.3; the updated one is ABI-compatible for our int-typed calls.
# Deprecations stay visible on purpose (no -Wno-...): the define is the fix.
FLAGS="-std=c++23 -O3 -march=native -I. -DACCELERATE_NEW_LAPACK"
FRW="-framework Foundation -framework Metal -framework MetalPerformanceShaders -framework Accelerate"

# Pure C++ (no Metal): the hybrid-split controller gate on a synthetic plant.
THYBRID=tests/test_hybrid.cpp
[ -f "$THYBRID" ] || THYBRID=../tests/test_hybrid.cpp
TESTINC=tests
[ -d "$TESTINC" ] || TESTINC=../tests
$CXX -std=c++23 -O2 -I. -I"$TESTINC" -DACCELERATE_NEW_LAPACK "$THYBRID" -o build/test_hybrid

# ANE routing-policy gate retired: the in-engine ANE dispatch API
# (ane_ready/set_ane_enabled/ane_prefills) was removed from llama.h when the
# ANE lane concluded it cannot add to the hybrid; the ANE segment tooling
# (ane_shim.mm, bench_ane_split, bench_prefill_lane) is kept as the record of
# that work — it no longer compiles against the engine (see the note below).
# Suffix-stack test retired with it: tm_metal_llama_prefill_suffix was the
# removed dispatch entry point; nothing in the current engine has a suffix
# prefill API to pin.

# Metal engine: numerics gate, decode benchmark, hybrid-split prefill bench.
$CXX $FLAGS -DTM_HAVE_METAL -ObjC++ -fobjc-arc test_llama_gpu_stack.cpp metal_shim.mm -o build/test_llama_gpu_stack $FRW
$CXX $FLAGS -DTM_HAVE_METAL -ObjC++ -fobjc-arc hybrid_bench.cpp metal_shim.mm -o build/hybrid_bench $FRW
$CXX $FLAGS -DTM_HAVE_METAL -ObjC++ -fobjc-arc bench_llama_decode.cpp metal_shim.mm -o build/bench_llama_decode_gpu $FRW

# Chat / completion binary: interactive REPL, --batch line protocol, and
# --oneshot single-shot completion (greedy-deterministic, banner-free).
$CXX $FLAGS -DTM_HAVE_METAL -ObjC++ -fobjc-arc llama_chat.cpp metal_shim.mm -o build/llama_chat_metal $FRW

# GGUF -> TMQ converter: lets llama.cpp / Ollama users convert the .gguf
# files they already have (Q4_0 / Q8_0, tokenizer exported alongside).
$CXX -std=c++23 -O2 -I. -DACCELERATE_NEW_LAPACK convert_gguf.cpp -o build/convert_gguf

# safetensors -> TMQ converter. It reads a HuggingFace checkpoint and writes the
# container the engine loads, so it belongs in the build the README calls "every
# binary below" — leaving it out meant a first-time user had to compile it by
# hand (or rely on fetch_model.py, which built it into a different directory).
$CXX -std=c++23 -O2 -I. -DACCELERATE_NEW_LAPACK convert_tmq.cpp -o build/convert_tmq

# Grammar-constrained decoding gate: pure logic — no model, no Metal — so it
# runs wherever the headers compile. It pins the accept/reject boundary the
# grammar filter depends on, recursion and malformed UTF-8 included.
$CXX -std=c++23 -O2 -I. test_gbnf.cpp -o build/test_gbnf
# RUN it, not just build it. A gate that is compiled but never executed
# catches only the one failure mode that stops the build, which is the rarest
# of them; the 95 checks below cost milliseconds and were previously only run
# by hand.
./build/test_gbnf

# Attention-output-gate split: the regression gate for the write that ran one
# row past `agate_` and SIGSEGV'd on long prompts. Self-contained — it writes its
# own tiny quantized fixture — so it needs no converter, tokenizer or checkpoint.
# This plain run is NOT the mechanical guard for the overrun — that is the ASan
# build below, because whether a 64-byte stray write lands in a mapped page is
# malloc's decision (measured: roughly one run in four). What this run
# does deterministically is pin the split's meaning, at shapes from the decode
# row to T=1024. Both split layouts, because TM_QGATE_SPLIT is resolved once per
# process. Guarded by -f for the same reason as the optional tools below: the
# public export tracks a curated subset of sources, and a gate that is not
# shipped must not break the build.
if [ -f test_llama_attn_gate.cpp ]; then
    $CXX -std=c++23 -O2 -I. -DACCELERATE_NEW_LAPACK \
        test_llama_attn_gate.cpp -o build/test_llama_attn_gate -framework Accelerate
    ./build/test_llama_attn_gate
    TM_QGATE_SPLIT=halves ./build/test_llama_attn_gate
    # TM_ASAN_GATES=1 builds the same checks with AddressSanitizer: CI sets it,
    # and it is the only configuration that catches the overrun on EVERY run.
    # Measured 2026-09-15: 8.9 s to compile and 1.7 s for the sweep — redzones
    # turn the stray write into a report with a stack, rather than a signal
    # that may or may not arrive.
    #
    # The probe comes first, because a sanitizer is only evidence if it reports:
    # probe_asan_mechanism.cpp is a deliberate 64-byte overrun of the same shape
    # (plus an in-bounds control), and Apple's clang deadlocks in the ASan
    # runtime's own initializer on macOS 26 — before main, so the failure mode is
    # a silent hang, not a report. TM_ASAN_CC picks a compiler whose runtime works
    # (Homebrew's llvm does). Every instrumented run is bounded for the same
    # reason: a deadlocked sanitizer prints nothing and never exits, so a gate
    # waiting on it is indistinguishable from work in progress.
    if [ -n "${TM_ASAN_GATES:-}" ]; then
        ASAN_CXX=${TM_ASAN_CC:-$CXX}
        $ASAN_CXX -std=c++23 -O0 -g -fsanitize=address \
            probe_asan_mechanism.cpp -o build/probe_asan_mechanism
        rc_probe=0
        bash run_bounded.sh 120 ./build/probe_asan_mechanism buggy \
            > build/probe_buggy.out 2>&1 || rc_probe=$?
        if ! grep -q heap-buffer-overflow build/probe_buggy.out; then
            if [ "$rc_probe" -eq 124 ]; then
                echo "attention-gate split: the sanitizer never returned — it deadlocked before"
                echo "  main. Apple clang's ASan runtime hangs on macOS 26: set TM_ASAN_CC to a"
                echo "  compiler whose runtime works, e.g. TM_ASAN_CC=\$(brew --prefix llvm)/bin/clang++"
            else
                echo "attention-gate split: the sanitizer ($ASAN_CXX) stayed silent on a deliberate"
                echo "  64-byte overrun, so a clean run below would mean nothing. Set TM_ASAN_CC."
            fi
            exit 1
        fi
        $ASAN_CXX -std=c++23 -O1 -g -fsanitize=address -I. -DACCELERATE_NEW_LAPACK \
            test_llama_attn_gate.cpp -o build/test_llama_attn_gate_asan \
            -framework Accelerate
        bash run_bounded.sh 600 ./build/test_llama_attn_gate_asan > /dev/null
        TM_QGATE_SPLIT=halves bash run_bounded.sh 600 ./build/test_llama_attn_gate_asan > /dev/null
        echo "attention-gate split: ASan clean (both layouts)"
    fi
fi

# Warm-KV correctness pin: cache-continued inference must be argmax-identical
# to a full re-prefill of the same context.
$CXX $FLAGS -DTM_HAVE_METAL -ObjC++ -fobjc-arc llq4.cpp metal_shim.mm -o build/llq4 $FRW

# Decode-chain benchmark: argmax + embedding on the GPU, command buffers
# committed back-to-back (the pipeline the README decode numbers measure).
$CXX $FLAGS -DTM_HAVE_METAL -ObjC++ -fobjc-arc bench_chain_decode.cpp metal_shim.mm -o build/bench_chain_decode $FRW

# Decode diagnostics — the instruments behind the M1 bandwidth work. Decode
# is memory-bandwidth-bound, so these answer "which tensor, how many bytes,
# at what effective GB/s, and is the answer right?":
#   microbench_lmhead    per-tensor GEMV timing + effective GB/s
#   eval_ppl_llama       perplexity (and top-1) for any .tmq
#   compare_logits       two builds' logits: max abs/relative delta, top-k
#   check_kquant_kernel  K-quant GEMV kernels vs the reference dequantizers
#
# OPTIONAL on purpose. The public export tracks a curated subset of these
# sources, so a tool that is not shipped must not break the build. An ABSENT
# file is skipped; one that is present and fails to compile still aborts
# under `set -e`.
for tool in microbench_lmhead eval_ppl_llama compare_logits check_kquant_kernel; do
    [ -f "$tool.cpp" ] || continue
    $CXX -std=c++23 -O3 -march=native -I. -DACCELERATE_NEW_LAPACK \
        "$tool.cpp" -o "build/$tool" -framework Accelerate
done

# ANE fused prefill segment: retired from the default build. The tools
# (bench_ane_split, bench_prefill_lane) drive the in-engine ANE dispatch API
# (ane_ready/set_ane_enabled/ane_prefills) that was removed from llama.h when
# the ANE lane concluded it cannot add to the hybrid; they no longer compile
# against the current headers. ane_shim.mm remains for standalone ANE probes.

echo "built: build/test_hybrid build/test_llama_gpu_stack" \
     "build/hybrid_bench build/bench_llama_decode_gpu build/llama_chat_metal build/llq4" \
     "build/bench_chain_decode build/convert_gguf build/convert_tmq"
# Only the optional diagnostics whose sources shipped (see above). Written as
# an if rather than `[ -n ] && echo`: as this script's last statement the &&
# form leaves a non-zero exit status when no tool shipped, which `set -e`
# then turns into a CI failure.
built_tools=""
for tool in microbench_lmhead eval_ppl_llama compare_logits check_kquant_kernel; do
    if [ -x "build/$tool" ]; then built_tools="$built_tools build/$tool"; fi
done
if [ -n "$built_tools" ]; then echo "built (tools):$built_tools"; fi
