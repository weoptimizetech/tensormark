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
# (ane_shim.mm, bench_ane_split, bench_prefill_lane) remains standalone.
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

# Warm-KV correctness pin: cache-continued inference must be argmax-identical
# to a full re-prefill of the same context.
$CXX $FLAGS -DTM_HAVE_METAL -ObjC++ -fobjc-arc llq4.cpp metal_shim.mm -o build/llq4 $FRW

# Decode-chain benchmark: argmax + embedding on the GPU, command buffers
# committed back-to-back (the pipeline the README decode numbers measure).
$CXX $FLAGS -DTM_HAVE_METAL -ObjC++ -fobjc-arc bench_chain_decode.cpp metal_shim.mm -o build/bench_chain_decode $FRW

# ANE fused prefill segment: retired from the default build. The tools
# (bench_ane_split, bench_prefill_lane) drive the in-engine ANE dispatch API
# (ane_ready/set_ane_enabled/ane_prefills) that was removed from llama.h when
# the ANE lane concluded it cannot add to the hybrid; they no longer compile
# against the current headers. ane_shim.mm remains for standalone ANE probes.

echo "built: build/test_hybrid build/test_llama_gpu_stack" \
     "build/hybrid_bench build/bench_llama_decode_gpu build/llama_chat_metal build/llq4" \
     "build/bench_chain_decode build/convert_gguf"
