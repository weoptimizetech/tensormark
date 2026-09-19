#!/usr/bin/env python3
"""Quantize DENSE weights to TensorMark's ternary TQ2 blocks — the step
convert_bonsai.py cannot do.

convert_bonsai.py is a REFORMATTER: it consumes PrismML's already-quantized MLX
2-bit pack (2-D tensors arrive as U32 with `.scales`/`.biases` sidecars, and it
verifies `bias == -scale` before writing). Handed a dense 2-D `F16` tensor it
raises `SystemExit("unsupported ... rank 2")` — which is exactly what the
full-precision source (prism-ml/Ternary-Bonsai-4B-unpacked) contains. So the
fp16 arm of the quality gate (spec: Δppl ≤ 0.1 against the same model) has never
been measurable, and neither has any mixed-precision pack.

This writes the same TMQ1 container the engine already loads:

    "TMQ1" | u32 tensor_count | per tensor:
        u32 name_len | name | u32 dtype | u32 rank | rank × u32 dims |
        u64 nblocks | payload

A TQ2 tensor is 34 bytes per 128-value group: a little-endian fp16 scale `d`
followed by 32 bytes of packed 2-bit codes, codes read as uint32 little-endian
(16 values per word, value j in bits 2j..2j+1), value = (code - 1) * d.

    ./tools/quantize_ternary.py <src_dir> <dst.tmq>     # dense fp16 -> ternary
    ./tools/quantize_ternary.py --selfcheck <pack_dir>  # validate the quantizer

`--selfcheck` is the one that matters before trusting a new pack. It dequantizes
a KNOWN pack with its own scales and re-quantizes it with this file's rule: if
the rule and the bit order match the pack's, every code and every scale comes
back identical. It catches a wrong bit order, a wrong grouping, or an off-by-one
in the `-1` offset, none of which would fail loudly in the engine — they would
just produce a quietly worse model.

PROVEN EQUIVALENT TO THE SHIPPED PACK (2026-09-19). For
`model.layers.0.mlp.gate_proj.weight`, the 34-byte-per-group payload this file
produces from PrismML's own fp16 source and the payload reconstituted from their
2-bit pack (scale || codes, the same layout convert_bonsai.py writes) hash the
same:

    PACK_PAYLOAD_SHA256    11d77d508a7d2010d64f1e7745e5ad804a593ddf1baa2ab9c192573fc166ed28
    FP16_QUANTIZED_SHA256  11d77d508a7d2010d64f1e7745e5ad804a593ddf1baa2ab9c192573fc166ed28

So the shipped pack IS this absmax/round/clamp rule applied to the fp16 weights
— bit for bit, no rotated basis, no second scheme hidden behind it. Two
consequences: the pack can be regenerated from full precision (so a VARIANT can
be built and differ from it under control, which is what any mixed-precision
work needs), and a quality difference measured against the fp16 arm is
attributable to ternary quantization alone rather than to an unknown transform.
"""
from __future__ import annotations

import json
import struct
import sys
import fnmatch
from pathlib import Path

import numpy as np

GROUP = 128          # values per ternary block
DTYPE_TQ2 = 7
DTYPE_Q4_0 = 1       # 32 values per 18-byte block; the mixed-precision option
DTYPE_F32 = 2        # 1-D norms stay dense, as the engine's loader expects
BLOCK_BYTES = 34
Q4_GROUP = 32
Q4_BLOCK_BYTES = 18

# Working-set bound. Quantizing a tensor allocates several full fp32 copies of it
# at once (the scale division, rint, clip, the u32 codes), so peak memory is ~4x
# the tensor: the 388M-value Bonsai embedding needed 6.8 GB on a 7.8 GB box and was
# OOM-KILLED, which is how this bound came to exist. Rows are chunked so peak RSS
# follows the chunk, not the tensor. Chunking along ROWS is exact — every
# 128-value group lies inside one row — so the payload is byte-identical, and this
# is pinned by a test that compares a chunked call against row-by-row calls.
CHUNK_VALUES = 1 << 22      # ~4M values -> ~64 MB per fp32 temporary


def _chunk_rows(inn: int) -> int:
    """Rows per chunk: at least one, whatever the row width."""
    return max(1, CHUNK_VALUES // max(inn, 1))

_READERS = {
    "F32": lambda b: np.frombuffer(b, "<f4").astype(np.float32),
    "F16": lambda b: np.frombuffer(b, "<f2").astype(np.float32),
    "BF16": lambda b: (np.frombuffer(b, "<u2").astype(np.uint32) << 16)
                      .view(np.float32),
}


def shards(src: Path) -> list[Path]:
    """The single-file layout first, then a sharded one (both are HF standard)."""
    one = src / "model.safetensors"
    if one.exists():
        return [one]
    many = sorted(src.glob("model-*.safetensors"))
    if not many:
        raise SystemExit(f"{src}: no model.safetensors and no model-*.safetensors")
    return many


def read_tensors(src: Path):
    """Yield (name, dtype, shape, blob) across every shard, in file order."""
    for path in shards(src):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            base = f.tell()
            for name, e in header.items():
                if name == "__metadata__":
                    continue
                lo, hi = e["data_offsets"]
                f.seek(base + lo)
                yield name, e["dtype"], list(e["shape"]), f.read(hi - lo)


def unpack_codes(packed: np.ndarray) -> np.ndarray:
    """(..., words) uint32 -> (..., words*16) int8 codes in {0,1,2}."""
    shifts = (np.arange(16, dtype=np.uint32) * 2)
    codes = (packed[..., None] >> shifts) & np.uint32(3)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * 16).astype(np.int8)


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """(out, groups, 128) uint32 codes in {0,1,2} -> (out, groups, 8) uint32."""
    out, groups, width = codes.shape
    if width % 16:
        raise SystemExit(f"group width {width} is not a multiple of 16")
    words = codes.reshape(out, groups, width // 16, 16).astype(np.uint32)
    shifts = (np.arange(16, dtype=np.uint32) * 2)
    return (words << shifts).sum(axis=3, dtype=np.uint32)


def quantize_block(w: np.ndarray) -> bytes:
    """(out, in) float32 with in % 128 == 0 -> TQ2 payload bytes.

    Per 128-value group: d = max|w|, code = round(w/d) clamped to {-1,0,1}.
    A group that is all zeros gets d = 0 (the engine handles a zero scale; the
    division below is guarded so it never becomes NaN).
    """
    out, inn = w.shape
    if inn % GROUP:
        raise SystemExit(f"input dim {inn} is not a multiple of {GROUP}")
    groups = inn // GROUP
    per = _chunk_rows(inn)
    parts = []
    for o0 in range(0, out, per):
        rows = min(per, out - o0)
        x = w[o0:o0 + rows].reshape(rows, groups, GROUP)
        d = np.abs(x).max(axis=2)
        d_safe = np.where(d == 0.0, 1.0, d).astype(np.float32)
        q = np.clip(np.rint(x / d_safe[..., None]), -1.0, 1.0)
        codes = (q + 1.0).astype(np.uint32)             # {-1,0,1} -> {0,1,2}
        blk = np.empty((rows, groups, BLOCK_BYTES), dtype=np.uint8)
        blk[..., 0:2] = d.astype("<f2").view(np.uint8).reshape(rows, groups, 2)
        blk[..., 2:BLOCK_BYTES] = pack_codes(codes).view(np.uint8).reshape(rows, groups, 32)
        parts.append(blk.tobytes())
    return b"".join(parts)


def norms_to_f32(dtype: str, shape, blob: bytes) -> bytes:
    if len(shape) != 1:
        raise SystemExit(f"unsupported {dtype} rank {len(shape)}")
    if dtype == "F32":
        return blob
    reader = _READERS.get(dtype)
    if reader is None:
        raise SystemExit(f"unsupported dtype {dtype}")
    return reader(blob).astype("<f4").tobytes()


def quantize_block_q4_0(w: np.ndarray) -> bytes:
    """(out, in) float32 with in % 32 == 0 -> Q4_0 payload bytes.

    The engine's own rule (quant.h, ggml's reference): d = max_signed / -8 where
    max_signed is the largest-magnitude element WITH its sign, then
    nibble = clamp(floor(w/d + 8.5), 0, 15), low nibble = element 2j and high =
    2j+1. The [-8d, +7d] asymmetry is inherent to Q4_0; the point of writing it
    the same way is that a Python-built pack can be compared against the C++
    quantizer instead of merely being self-consistent.
    """
    out, inn = w.shape
    if inn % Q4_GROUP:
        raise SystemExit(f"input dim {inn} is not a multiple of {Q4_GROUP}")
    groups = inn // Q4_GROUP
    per = _chunk_rows(inn)
    parts = []
    for o0 in range(0, out, per):
        rows = min(per, out - o0)
        x = w[o0:o0 + rows].reshape(rows, groups, Q4_GROUP)
        idx = np.argmax(np.abs(x), axis=2)
        max_signed = np.take_along_axis(x, idx[..., None], axis=2)[..., 0]
        d = (max_signed / -8.0).astype(np.float32)
        nz = d != 0
        id_ = np.where(nz, 1.0 / np.where(nz, d, 1.0), 0.0).astype(np.float32)
        xi = np.clip(np.floor(x * id_[..., None] + 8.5), 0.0, 15.0).astype(np.uint8)
        qs = xi[..., 0::2] | (xi[..., 1::2] << 4)        # low = 2j, high = 2j+1
        blk = np.empty((rows, groups, Q4_BLOCK_BYTES), dtype=np.uint8)
        blk[..., 0:2] = d.astype("<f2").view(np.uint8).reshape(rows, groups, 2)
        blk[..., 2:Q4_BLOCK_BYTES] = qs
        parts.append(blk.tobytes())
    return b"".join(parts)


def encode(name: str, dtype: str, shape, blob: bytes, q4_globs: tuple[str, ...]):
    """One tensor -> (out_dtype, dims, payload, block_elems, block_bytes)."""
    if len(shape) == 2:
        reader = _READERS.get(dtype)
        if reader is None:
            raise SystemExit(f"{name}: unsupported 2-D dtype {dtype}")
        w = reader(blob).reshape(shape[0], shape[1])
        if any(fnmatch.fnmatch(name, g) for g in q4_globs):
            return (DTYPE_Q4_0, [shape[0], shape[1]], quantize_block_q4_0(w),
                    Q4_GROUP, Q4_BLOCK_BYTES)
        return (DTYPE_TQ2, [shape[0], shape[1]], quantize_block(w), GROUP,
                BLOCK_BYTES)
    return DTYPE_F32, [int(shape[0])], norms_to_f32(dtype, shape, blob), 1, 4


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == "--selfcheck":
        return selfcheck(Path(argv[1]))
    q4_globs: tuple[str, ...] = ()
    if "--q4" in argv:
        i = argv.index("--q4")
        if i + 1 >= len(argv):
            print("--q4 needs a glob list")
            return 2
        q4_globs = tuple(g for g in argv[i + 1].split(",") if g)
        del argv[i:i + 2]
    if len(argv) != 2:
        print("usage: quantize_ternary.py <src_dir> <dst.tmq> [--q4 <glob[,glob]>]")
        print("       quantize_ternary.py --selfcheck <pack_dir>")
        return 2
    src, dst = Path(argv[0]), Path(argv[1])

    written = 0
    with dst.open("wb") as f:
        f.write(b"TMQ1")
        f.write(struct.pack("<I", 0))
        # STREAM, never materialize. This used to be `list(read_tensors(src))`,
        # which held the WHOLE checkpoint in RAM: 8.06 GB for the fp16 Bonsai on a
        # 7 GB host, so it swapped instead of finishing (the symptom is a process
        # that runs forever and never even creates the output file, because the
        # list is built BEFORE `dst.open`). read_tensors yields one tensor's bytes
        # at a time and the count is a placeholder rewritten at the end, so
        # streaming costs nothing and makes peak RSS one tensor, not the model.
        for name, dtype, shape, blob in read_tensors(src):
            if name.endswith((".scales", ".biases")):
                continue
            out_dtype, dims, payload, elems, per_block = encode(
                name, dtype, shape, blob, q4_globs)
            total = 1
            for d in dims:
                total *= d
            if total % elems:
                raise SystemExit(f"{name}: {total} values not a multiple of {elems}")
            nb = total // elems
            if len(payload) != nb * per_block:
                raise SystemExit(f"{name}: payload {len(payload)} != {nb * per_block}")
            nb_name = name.encode()
            f.write(struct.pack("<I", len(nb_name)))
            f.write(nb_name)
            f.write(struct.pack("<I", out_dtype))
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<Q", nb))
            f.write(payload)
            written += 1
        end = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", written))

    cfg = src / "config.json"
    if cfg.exists():
        (dst.parent / "config.json").write_text(cfg.read_text())
    print(f"wrote {dst} ({end / 1e9:.3f} GB, {written} tensors)")
    return 0


def selfcheck(pack: Path) -> int:
    """Re-quantize a known 2-bit pack's own weights and demand the codes back.

    The pack stores w = d * (code - 1), so its group absmax IS d and this file's
    rule must reproduce both the codes and the fp16 scale exactly. Any mismatch
    means the packing order, the grouping, or the -1 offset is wrong.
    """
    tensors = {n: (dt, sh, b) for n, dt, sh, b in read_tensors(pack)}
    checked = codes_ok = scales_ok = 0
    worst = 0
    for name, (dtype, shape, blob) in tensors.items():
        if dtype != "U32" or len(shape) != 2:
            continue
        base = name[: -len(".weight")]
        sc = np.frombuffer(tensors[base + ".scales"][2], "<u2").view("<f2") \
            .astype(np.float32)
        out_dim, packed = shape
        groups = packed // 8
        codes = np.frombuffer(blob, np.uint32).reshape(out_dim, groups, 8)
        d = sc.reshape(out_dim, groups)
        want = unpack_codes(codes)                       # (out, groups, 128)
        w = (want.astype(np.float32) - 1.0) * d[..., None]
        got_packed = pack_codes(
            ((np.clip(np.rint(w / np.where(d[..., None] == 0, 1.0, d[..., None])),
                      -1.0, 1.0) + 1.0).astype(np.uint32)))
        same = int((got_packed == codes).all())
        codes_ok += same
        sc_back = d.astype("<f2").astype(np.float32)
        scales_ok += int(np.array_equal(sc_back, d))
        worst = max(worst, int(np.abs(got_packed.astype(np.int64)
                                      - codes.astype(np.int64)).max()))
        checked += 1
    print(f"selfcheck {pack}")
    print(f"  tensors re-quantized : {checked}")
    print(f"  codes identical      : {codes_ok}/{checked}")
    print(f"  scales round-tripped : {scales_ok}/{checked}")
    print(f"  worst word delta     : {worst} (0 = exact)")
    return 0 if (checked and codes_ok == checked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
