#!/usr/bin/env python3
"""Build a tiny llama-arch .tmq whose vocabulary matches a SentencePiece tokenizer.

Why this exists
---------------
The structured-output gates assert the property that matters and that a
parse-only assertion misses: the model's reply both *parses* and *validates*
against its schema. That assertion needs a model, and a real one (TinyLlama Q4_0,
590 MB, fetched over the network) cannot be a prerequisite for CI — which is how
the feature ended up shipping with six live bugs behind passing gates.

But the property under test is that decoding *enforces* the grammar, not that the
model is intelligent. Random weights of the right shape exercise it exactly as
well, because the grammar mask — not the weights — decides which tokens are
reachable: a token that would leave the language is masked out, whatever the
logits say. So the gate runs against a 2-layer model with deterministic
pseudo-random weights and a REAL vocabulary, in milliseconds, with nothing to
download beyond the tokenizer.

The one thing that cannot be faked is the vocabulary: the grammar mask is built
over the tokenizer's pieces and the sampled id indexes the model's embeddings, so
the model's vocab_size must equal the tokenizer's piece count. That count is read
straight out of the SentencePiece protobuf here (field 1 of ModelProto, the
repeated `pieces`), so the two cannot drift apart.

Run: python3 tensormark/gen_tiny_llama.py <tokenizer.model> [out_dir] [convert_tmq]
Writes <out_dir>/model.tmq, model.safetensors and config.json.
"""
import json
import os
import struct
import subprocess
import sys

# Geometry. Matches the shape contract tests/test_llama_reload.cpp pins: D=32,
# H=2 heads of dh=16, KVH=1, and every 2-D weight with a multiple-of-32
# contraction dim so the Q4_0 32-wide blocks divide exactly.
D, L, H, KVH, F = 32, 2, 2, 1, 64
DH = D // H


def sp_piece_count(path):
    """Count the `pieces` entries in a SentencePiece .model (protobuf, field 1).

    A minimal top-level walk is enough and avoids a protobuf dependency: the
    trainer/normalizer specs are nested length-delimited fields, so skipping them
    by length keeps their contents from being mistaken for pieces.
    """
    data = open(path, "rb").read()
    i, n, pieces = 0, len(data), 0
    while i < n:
        key, shift = 0, 0
        while True:
            byte = data[i]
            i += 1
            key |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        field, wire = key >> 3, key & 7
        if wire == 0:  # varint
            while data[i] & 0x80:
                i += 1
            i += 1
        elif wire == 2:  # length-delimited
            length, shift = 0, 0
            while True:
                byte = data[i]
                i += 1
                length |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    break
                shift += 7
            if field == 1:
                pieces += 1
            i += length
        elif wire == 5:
            i += 4
        elif wire == 1:
            i += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire} at byte {i}")
    return pieces


def rnd(count, salt):
    """Deterministic pseudo-random values in [-0.5, 0.5]."""
    out, x = [], 2654435761 ^ (salt * 2246822519)
    for _ in range(count):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append((x % 200001) / 200000.0 - 0.5)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    tokenizer = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else "/tmp/tinyllama_fixture"
    here = os.path.dirname(os.path.abspath(__file__))
    conv = (os.path.abspath(sys.argv[3]) if len(sys.argv) > 3
            else os.path.join(here, "build", "convert_tmq"))
    if not os.path.exists(conv):
        print(f"gen_tiny_llama: converter not found at {conv}; run "
              f"./tensormark/build_llama.sh", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)

    V = sp_piece_count(tokenizer)
    if V <= 0:
        raise SystemExit(f"gen_tiny_llama: no pieces parsed from {tokenizer}")

    tensors, salt = {}, 1

    def put(name, shape, fill=None):
        nonlocal salt
        count = 1
        for dim in shape:
            count *= dim
        tensors[name] = (shape, list(fill) if fill is not None else rnd(count, salt))
        salt += 1

    put("model.embed_tokens.weight", [V, D])
    put("lm_head.weight", [V, D])
    put("model.norm.weight", [D], [1.0] * D)
    for layer in range(L):
        p = f"model.layers.{layer}."
        put(p + "input_layernorm.weight", [D], [1.0] * D)
        put(p + "post_attention_layernorm.weight", [D], [1.0] * D)
        put(p + "self_attn.q_proj.weight", [H * DH, D])
        put(p + "self_attn.k_proj.weight", [KVH * DH, D])
        put(p + "self_attn.v_proj.weight", [KVH * DH, D])
        put(p + "self_attn.o_proj.weight", [D, H * DH])
        put(p + "mlp.gate_proj.weight", [F, D])
        put(p + "mlp.up_proj.weight", [F, D])
        put(p + "mlp.down_proj.weight", [D, F])

    header, blobs, offset = {}, [], 0
    for name, (shape, values) in tensors.items():
        blob = struct.pack("<%df" % len(values), *values)
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with open(f"{out}/model.safetensors", "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(b"".join(blobs))

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": D, "num_hidden_layers": L, "num_attention_heads": H,
        "num_key_value_heads": KVH, "head_dim": DH, "intermediate_size": F,
        "vocab_size": V, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
    }
    with open(f"{out}/config.json", "w") as handle:
        json.dump(config, handle)

    # `asis`: HF nn.Linear weights are already (out, in).
    result = subprocess.run([conv, f"{out}/model.safetensors", f"{out}/model.tmq",
                             "q40", "asis"], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-800:], result.stderr[-800:], sep="", file=sys.stderr)
        return result.returncode
    print(f"gen_tiny_llama: vocab {V}, {L} layers, D={D} -> {out}/model.tmq")
    print(f"gen_tiny_llama: tokenizer {tokenizer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
