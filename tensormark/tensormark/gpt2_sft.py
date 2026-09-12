"""Supervised fine-tuning of pretrained GPT-2 124M, entirely on tensormark ops.

The model graph mirrors tensormark.gpt2_runtime.GPT2 op-for-op (pre-norm blocks,
fused c_attn Conv1D projection, tanh-approx GELU, tied lm_head), but every
weight is a tape leaf: tm.param for trainable tensors, tm.const_ for frozen
ones (embeddings by default — no gradient buffer is even allocated).

Padding design (stated choice per the research pipeline's request): sequences
are RIGHT-padded inside a batch and no attention mask is needed — with causal
attention a real query position can only attend to keys at or before it, and
every pad token sits AFTER all real tokens of its row, so pad keys are
unreachable from real queries. Pad *queries* produce garbage logits that the
loss simply ignores via cross_entropy_rows(..., ignore_index=-100). This keeps
the validated mha_causal kernel untouched while making masked batching exact.

    python3 train_sft.py --data pairs.jsonl --steps 200
"""

from __future__ import annotations

import json
import os
import re
import struct

import numpy as np

import tensormark as tm


# ---------------------------------------------------------------------------
# GPT-2 byte-level BPE (stdlib only: json + re). vocab.json/merges.txt are the
# standard OpenAI tokenizer files shipped with the model snapshot.
# ---------------------------------------------------------------------------
def _bytes_to_unicode() -> dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class GPT2BPE:
    """Minimal GPT-2 encoder: byte-level BPE with the classic regex split.

    The reference pattern's \\p{L}/\\p{N} classes become ASCII approximations
    so the tokenizer stays on the stdlib `re` module.
    """

    PAT = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
    )

    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
            self.encoder: dict[str, int] = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_u = _bytes_to_unicode()
        self.u_byte = {c: b for b, c in self.byte_u.items()}
        merges: list[tuple[str, str]] = []
        with open(os.path.join(model_dir, "merges.txt"), encoding="utf-8") as f:
            for line in f.read().split("\n")[1:]:
                parts = line.split()
                if len(parts) == 2:
                    merges.append(tuple(parts))
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.cache: dict[str, list[int]] = {}
        self.eot = self.encoder.get("<|endoftext|>", 50256)

    def _bpe(self, token: str) -> list[int]:
        if token in self.cache:
            return self.cache[token]
        word = list(token)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            first, second = best
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        self.cache[token] = ids = [self.encoder[t] for t in word]
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for token in self.PAT.findall(text):
            out.extend(self._bpe("".join(self.byte_u[b] for b in token.encode("utf-8"))))
        return out

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder[int(i)] for i in ids)
        return bytes(self.u_byte[c] for c in text).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GPT2SFT:
    """Trainable GPT-2 over tape ops, loaded from HF-layout safetensors keys.

    Layout follows tensormark.gpt2_runtime.GPT2 exactly (fused c_attn Conv1D,
    tanh GELU, tied lm_head), so exported weights reload with zero code changes.
    """

    def __init__(self, weights: dict[str, np.ndarray], n_layer: int = 12,
                 n_head: int = 12, train_embeddings: bool = False):
        self.w: dict[str, object] = {}
        self.n_layer, self.H = n_layer, n_head
        self.D = weights["wte.weight"].shape[1]
        self.V, self.ctx = weights["wte.weight"].shape[0], weights["wpe.weight"].shape[0]
        self.dh = self.D // self.H
        f = tm.param if train_embeddings else tm.const_

        def p(name: str, arr: np.ndarray, trainable: bool = True):
            src = weights.get(name)
            if src is None:
                raise KeyError(f"missing weight: {name}")
            self.w[name] = tm.param(np.ascontiguousarray(src, dtype=np.float32)) if trainable \
                else f(np.ascontiguousarray(src, dtype=np.float32))

        p("wte.weight", weights["wte.weight"], train_embeddings)
        p("wpe.weight", weights["wpe.weight"], train_embeddings)
        for l in range(n_layer):
            pre = f"h.{l}."
            for k, train in (("ln_1.weight", True), ("ln_1.bias", True),
                             ("attn.c_attn.weight", True), ("attn.c_attn.bias", True),
                             ("attn.c_proj.weight", True), ("attn.c_proj.bias", True),
                             ("ln_2.weight", True), ("ln_2.bias", True),
                             ("mlp.c_fc.weight", True), ("mlp.c_fc.bias", True),
                             ("mlp.c_proj.weight", True), ("mlp.c_proj.bias", True)):
                p(pre + k, weights[pre + k], train)
        p("ln_f.weight", weights["ln_f.weight"])
        p("ln_f.bias", weights["ln_f.bias"])

    # -- bookkeeping --------------------------------------------------------
    def param_lists(self) -> tuple[list, list[bool]]:
        """(params, decay flags) for tm.AdamW: 2-D tensors decay, 1-D do not."""
        params, decay = [], []
        for name, v in self.w.items():
            if v is self.w["wte.weight"] or v is self.w["wpe.weight"]:
                continue  # frozen (const_) unless train_embeddings
            params.append(v)
            decay.append(len(v.shape) >= 2)
        return params, decay

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: np.array(v.view, copy=True) for k, v in self.w.items()}

    def load_state_dict(self, sd: dict[str, np.ndarray]) -> None:
        for k, v in self.w.items():
            if k in sd:
                np.copyto(v.view, sd[k])

    # -- forward ------------------------------------------------------------
    def forward(self, ids: np.ndarray) -> object:
        """ids: (B, T) int array, right-padded. Returns logits (B*T, V)."""
        B, T = ids.shape
        D = self.D
        flat = np.ascontiguousarray(ids.reshape(-1), dtype=np.int32)
        pos = np.tile(np.arange(T, dtype=np.int32), B)
        x = tm.add(tm.gather_rows(self.w["wte.weight"], flat),
                   tm.gather_rows(self.w["wpe.weight"], pos))
        for l in range(self.n_layer):
            pre = f"h.{l}."
            w = lambda k, pre=pre: self.w[pre + k]
            h = tm.layernorm(x, w("ln_1.weight"), w("ln_1.bias"))
            qkv = tm.add_rowvec(tm.matmul(h, w("attn.c_attn.weight")),
                                w("attn.c_attn.bias"))
            q = tm.slice2d(qkv, 0, B * T, 0, D)
            k = tm.slice2d(qkv, 0, B * T, D, 2 * D)
            v = tm.slice2d(qkv, 0, B * T, 2 * D, 3 * D)
            a = tm.mha_causal(q, k, v, B, T, self.H)
            x = tm.add(x, tm.add_rowvec(tm.matmul(a, w("attn.c_proj.weight")),
                                        w("attn.c_proj.bias")))
            h2 = tm.layernorm(x, w("ln_2.weight"), w("ln_2.bias"))
            m = tm.gelu(tm.add_rowvec(tm.matmul(h2, w("mlp.c_fc.weight")),
                                      w("mlp.c_fc.bias")))
            x = tm.add(x, tm.add_rowvec(tm.matmul(m, w("mlp.c_proj.weight")),
                                        w("mlp.c_proj.bias")))
        x = tm.layernorm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])
        return tm.matmul_nt(x, self.w["wte.weight"])

    def loss(self, ids: np.ndarray, targets: np.ndarray, ignore_index: int = -100):
        """Cross-entropy over ALL rows; ignored rows carry ignore_index."""
        logits = self.forward(ids)
        labels = np.ascontiguousarray(targets.reshape(-1)).astype(np.int32).tolist()
        return tm.cross_entropy_rows(logits, labels, ignore_index=ignore_index)


# ---------------------------------------------------------------------------
# Data: JSONL {"prompt": str, "target": str} pairs, right-padded batches
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100


class SFTBatcher:
    def __init__(self, pairs: list[dict[str, str]], bpe: GPT2BPE,
                 max_len: int = 512, seed: int = 0):
        self.rows = []
        for r in pairs:
            prompt = bpe.encode(r["prompt"])
            target = bpe.encode(r["target"]) + [bpe.eot]
            if len(prompt) + len(target) > max_len:
                prompt = prompt[-(max_len - len(target)):]
            self.rows.append((prompt, target))
        self.rng = np.random.default_rng(seed)

    def batch(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        """(ids (B,T), labels (B,T)): labels are -100 on prompt and pad rows."""
        ix = self.rng.integers(0, len(self.rows), size=batch)
        chosen = [self.rows[int(i)] for i in ix]
        tmax = max(len(p) + len(t) for p, t in chosen)
        ids = np.full((len(chosen), tmax), 50256, dtype=np.int32)   # <|endoftext|>
        labels = np.full((len(chosen), tmax), IGNORE_INDEX, dtype=np.int32)
        for b, (prompt, target) in enumerate(chosen):
            seq = prompt + target
            ids[b, :len(seq)] = seq
            labels[b, len(prompt):len(seq)] = target
        return ids, labels


# ---------------------------------------------------------------------------
# Export: HF-layout model.safetensors (F32) that gpt2_runtime.GPT2 consumes as is
# ---------------------------------------------------------------------------
def save_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        nbytes = arr.nbytes
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        blobs.append(arr.tobytes())
    hj = json.dumps(header).encode("utf-8")
    pad = (8 - len(hj) % 8) % 8
    hj += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for blob in blobs:
            f.write(blob)


def export_safetensors(model: GPT2SFT, path: str) -> None:
    save_safetensors(path, model.state_dict())
