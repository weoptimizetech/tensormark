# Coming from llama.cpp

If you use `llama.cpp`, `llama-bench` or `ollama run`, the TensorMark engine
will feel familiar: same Llama-family models, same Q4_0 block quantization,
same ideas of prefill/decode. The differences: TensorMark ships as
header-only C++ you compile with one script, reads its own `.tmq` container
(converted from HF `safetensors`), and hybridizes prefill across the GPU
and the CPU's AMX automatically.

## 1. Get a model

TensorMark reads its own `.tmq` container. Convert what you already have —
GGUF files work directly:

```bash
# a) convert an existing GGUF (llama.cpp / Ollama downloads work as-is):
#    Q4_0 blocks are transcoded bit-exactly (same fp16 scale, same
#    d = max_signed/-8 rule — only the nibble order differs, which the
#    converter reorders), q/k head permutation is undone, and an embedded
#    SentencePiece tokenizer is exported next to the .tmq. Q8_0, Q6_K,
#    Q5_K, Q4_1, F16, BF16 and F32 tensors inside the file are handled too —
#    the K-quant families are stored at their source byte width, not widened.
./tensormark/build_llama.sh          # or: c++ -std=c++23 -O2 -Itensormark \
                                     #   tensormark/convert_gguf.cpp -o build/convert_gguf
./build/convert_gguf model.gguf model_q40.tmq
# -> model_q40.tmq + model_q40.tokenizer.model

# b) convert the HF safetensors checkpoint — the same source your GGUF
#    was quantized from; the .tmq q40 format is the same 32-wide block
#    quantization Q4_0 uses:
./tensormark/build_llama.sh   # builds tensormark/build/convert_tmq along with the engine
./tensormark/build/convert_tmq model.safetensors model_q40.tmq q40 asis

# c) or download a ready checkpoint + tokenizer:
python3 examples/fetch_model.py --out tensormark/data/tinyllama
```

The tokenizer is the model repo's SentencePiece `tokenizer.model` — parsed
in-tree (`tensormark/sp_tokenizer.h`, no protobuf dependency).

## 2. Flag mapping

| llama.cpp / llama-cli | TensorMark |
|---|---|
| `llama-cli -m m.gguf -p "..." -n 64 --temp 0` | `build/llama_chat_metal --oneshot -m m.tmq -t tokenizer.model -p "..." -n 64` (always greedy) |
| `llama-cli -m m.gguf -c 2048` (chat REPL) | `build/llama_chat_metal -m m.tmq -t tokenizer.model` (interactive; `/greedy` `/n <tok>` `/reset` `/quit`) |
| `--system-prompt "..."` | `-s "..."` |
| `-f prompt.txt` | `-f prompt.txt` |
| `llama-bench -p 2000` (prefill) | `build/hybrid_bench m.tmq 2000 4` |
| `llama-bench -tg 128` (decode) | `build/bench_llama_decode_gpu m.tmq` |
| `OLLAMA_HOST=...` HTTP server | `python3 examples/from_ollama.py` (see below) |

Sampling: `--oneshot` is greedy-deterministic (audit-friendly, reproducible);
interactive and batch modes sample at temperature 0.7, top-k 40, top-p 0.9
(`/greedy` toggles argmax). Context: 2048, chat template auto-detected
(Zephyr for TinyLlama-Chat, ChatML for OpenHermes; override with
`TM_CHAT_TEMPLATE=chatml|zephyr`).

## 3. If you drive Ollama's HTTP API

`examples/from_ollama.py` exposes an Ollama-compatible `/api/generate`
(streaming and not), `/api/embeddings` and `/api/embed` backed by the engine, so
existing Ollama clients work unchanged. With no flags it uses the fetched
TinyLlama under `tensormark/data/tinyllama/`:

```bash
python3 examples/from_ollama.py --port 11435
curl -s localhost:11435/api/generate -d '{"model":"tensormark","prompt":"...","stream":false}'
curl -s localhost:11435/api/embeddings -d '{"prompt":"..."}'    # {"embedding":[...]}
```

For your own conversion, pass absolute paths. The engine subprocess runs with the
repository root as its working directory, so a bare `model_q40.tmq` (the file
step 2 writes next to your checkpoint) resolves from there, not from where you
invoked the shim:

```bash
python3 examples/from_ollama.py \
    --model "$PWD/model_q40.tmq" --tokenizer "$PWD/tokenizer.model" --port 11435
```

The model does not stay in memory just because the server is running: after 300 s
with no request the engine process is **reaped**, which is the only thing that gives
its memory back, and the next request loads it again transparently. `--idle-ttl`
changes the delay (`0` keeps the model resident for the life of the server), and
Ollama's per-request `keep_alive` overrides it per call — `0` unloads the moment
that reply is sent, `-1` pins the model. `GET /api/ps` reports what is loaded and
when it is due to go, and returns an empty list once the model is gone.

## 4. What you gain / give up

Gain: hybrid GPU+AMX prefill with a self-tuning controller (prefill and
decode at parity to slightly ahead of llama.cpp on M1 at 1.1B — 0.95–1.07×
across windows — and ~1.3× faster than mlx-lm on the same machine), warm-KV
multi-turn chat, bounded-memory design for 8 GB Macs. Give up: GGUF files
are read through a one-way converter (`convert_gguf`) rather than loaded
directly, no GPU offload beyond Metal on Apple Silicon, and a far smaller
feature surface — TensorMark is an engine to experiment with, not a
llama.cpp replacement.
