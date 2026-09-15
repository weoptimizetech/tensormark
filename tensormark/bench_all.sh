#!/bin/zsh
# tensormark/bench_all.sh — single entry point for ALL performance tables.
#
# Runs every paired measurement (kernels vs torch.compile, MNIST training vs
# torch.compile, SFT step cost, LLM inference vs llama.cpp), validates output
# checksums so both sides provably compute the same math, computes medians and
# speed-ups PROGRAMMATICALLY (no hand transcription), and emits:
#   build/PERF_TABLES.md      markdown tables (README source of truth)
#   build/perf_charts.tex     pgfplots bar charts (weoptimizetech.com identity)
#   build/perf_charts.pdf     compiled charts (when latexmk is available)
#   build/perf_evidence.json  raw medians, checksums, speed-ups
#
# Sections are skipped (with a marker) when their data/binary is absent.
set -euo pipefail
cd "$(dirname "$0")"
ROUNDS=${ROUNDS:-3}
mkdir -p build

echo "== [0/5] build + shared inputs =="
./build.sh microbench.cpp >/dev/null
./build_llama.sh > /dev/null   # hybrid_bench, bench_llama_decode_gpu
if [ ! -x build/ml_system ] && [ -f ../ml_system.cpp ]; then
  clang++ -std=c++23 -O3 -march=native -DACCELERATE_NEW_LAPACK -I.. \
    ../ml_system.cpp -o build/ml_system -framework Accelerate -lz
fi
python3 gen_bench_inputs.py build/bench_inputs.bin

echo "== [1/5] kernels: engine vs torch.compile ($ROUNDS rounds) =="
for r in $(seq 1 $ROUNDS); do
  ./build/microbench > "build/ours_$r.json"
  python3 microbench_torch.py --compile > "build/torchc_$r.json"
done

echo "== [2/5] training: MNIST full epoch, engine vs torch.compile =="
if [ -d ../mnist ] && [ -x build/ml_system ]; then
  for r in $(seq 1 $ROUNDS); do
    ./build/ml_system --mode mnist --mnist-epochs 1 \
      --mnist-dir ../mnist 2>&1 | grep "samples_s" \
      > "build/mnist_ours_$r.txt"
    python3 mnist_torch_compile_e2e.py 1 > "build/mnist_torch_$r.txt" 2>/dev/null || true
  done
else
  echo "SKIP: mnist data or ml_system binary missing"
fi

echo "== [3/5] fine-tuning: GPT-2 124M SFT step cost =="
if [ -f data/gpt2/model.safetensors ] && [ -f /tmp/sft_pairs.jsonl ]; then
  python3 tensormark/train_sft.py --data /tmp/sft_pairs.jsonl --steps 24 --batch 2 \
    --grad-accum 4 --warmup 1 --out build/sft_bench 2>&1 \
    | grep "^step" > build/sft_steps.txt || true
else
  echo "SKIP: gpt2 snapshot or /tmp/sft_pairs.jsonl missing"
fi

echo "== [4/5] inference: LLM prefill + decode vs llama.cpp + mlx-lm (TinyLlama 1.1B) =="
if [ -f data/tinyllama/tinyllama_q40.tmq ] && command -v llama-bench >/dev/null; then
  ./build/hybrid_bench data/tinyllama/tinyllama_q40.tmq 2000 2 > build/llm_ours_prefill.txt
  : > build/llm_llama_prefill.txt
  # Production decode path: the GPU-resident greedy chain (argmax + embedding
  # on the GPU, command buffers committed back-to-back) — the same path
  # llama_chat ships. llama-bench's tg128 is llama.cpp's production decode
  # loop, so this is the apples-to-apples comparison; the per-token
  # host-synchronized loop (bench_llama_decode_gpu) is the fallback and is
  # reported separately in the README, not as the headline.
  : > build/llm_ours_decode.txt
  : > build/llm_llama_decode.txt
  for r in $(seq 1 $ROUNDS); do
    ./build/bench_chain_decode data/tinyllama/tinyllama_q40.tmq 128 \
      >> build/llm_ours_decode.txt 2>&1
    llama-bench -m data/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -p 0 -n 128 -t 4 \
      2>/dev/null | grep tg128 >> build/llm_llama_decode.txt
    llama-bench -m data/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -p 2000 -n 0 -t 4 \
      2>/dev/null | grep pp2000 >> build/llm_llama_prefill.txt
  done
  # mlx-lm evidence: skipped when the venv or the MLX model dir is absent
  MLXPY="$HOME/.venvs/mlx/bin/python"
  if [ -x "$MLXPY" ] && [ -d data/tinyllama/mlx-4bit-g32 ]; then
    "$MLXPY" bench_mlx.py --model data/tinyllama/mlx-4bit-g32 --out-dir build \
      --prefill-tokens 2000 --decode-tokens 128 --rounds 3 > /dev/null
  else
    echo "SKIP: mlx venv or MLX model dir missing"
  fi
else
  echo "SKIP: tinyllama data or llama-bench missing"
fi

echo "== [5/5] join + tables + charts =="
python3 join_perf.py \
  --engine build/ours_{1..$ROUNDS}.json \
  --torch-compiled build/torchc_{1..$ROUNDS}.json \
  --out-md build/PERF_TABLES.md --out-tex build/perf_charts.tex \
  --out-json build/perf_evidence.json
python3 emit_sections.py --dir build || true

# Publish the evidence where a reader can audit it. build/ is gitignored, so an
# evidence file left there is invisible to everyone but the machine that made
# it — which is how the README came to claim its tables were extracted from raw
# files while most of them had no artifact behind them at all.
python3 collect_perf_evidence.py --dir build --out ../docs/perf_evidence.json
# No --readme: the checker resolves the README (and the evidence) against the
# repo root itself, so this works from any cwd — and no `|| true`, which used to
# turn a real complaint about an unevidenced number into silence. --report is
# what keeps this advisory here: bench_all.sh generates, CI enforces.
python3 check_readme_perf.py --report
if command -v latexmk >/dev/null; then
  (cd build && latexmk -pdf -interaction=nonstopmode perf_charts.tex >/dev/null 2>&1 || true)
  [ -f build/perf_charts.pdf ] && echo "charts: build/perf_charts.pdf"
fi
echo "ALL DONE — tables: build/PERF_TABLES.md, charts: build/perf_charts.pdf"
