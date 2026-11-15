#!/usr/bin/env python3
"""Drop tensors from a .tmq pack — the tied head automatically, anything else on request.

Two jobs, one container walk:

  * No flags: the TIED-HEAD repair. `convert_gguf` used to write a second copy of
    the embedding table under `lm_head.weight` for every tied checkpoint
    (qwen38_4b_q6k carried 497 MB of it). A pack with an embedding table and no
    head is tied by construction and the engine's `alias_tied_lm_head()` rebuilds
    it at load, so the duplicate is pure weight. The payloads are compared by
    streaming SHA-256 first and the tool refuses unless they are identical.

  * `--drop-prefix P`: drop every tensor whose name starts with P. This is how the
    MTP head goes (`--drop-prefix mtp.`): it is opt-in, measured a LOSS (0.82x the
    plain loop), and nothing on the default decode path faults it, yet it costs
    its bytes on disk and in RSS. `convert_gguf` now omits it unless
    TM_CONVERT_MTP=1, so this mode exists to repair packs built before that.
    Dropping it disables MTP for the pack: `mtp_ready()` goes false and
    `mtp_step()` throws "this model has no MTP head" — loud, never silent.

    ./drop_tensors.py in.tmq [out.tmq] [--drop-prefix mtp.] [--force]

`--force` also drops a `lm_head.weight` that is NOT byte-identical to the
embedding — that changes the model, so it is never the default.

Exit codes: 0 dropped (or nothing to do), 1 refused, 2 malformed input.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys

MAGIC = b"TMQ1"
# bytes per block, by dtype id (see tensormark/tmq.h)
BLOCK = {0: 34, 1: 18, 4: 20, 5: 176, 6: 210, 7: 34}
# dtype id -> bytes per element, for the non-blocked ids
SCALAR = {2: 4, 3: 2}
HEAD = "lm_head.weight"
EMBED = "model.embed_tokens.weight"
CHUNK = 8 << 20


class Rec:
    __slots__ = ("name", "dtype", "dims", "nblocks", "off", "hdr", "size")

    def __init__(self, name, dtype, dims, nblocks, off, hdr):
        self.name, self.dtype, self.dims, self.nblocks = name, dtype, dims, nblocks
        self.off, self.hdr = off, hdr
        self.size = payload_size(dtype, dims, nblocks)


def payload_size(dtype, dims, nblocks):
    if dtype in BLOCK:
        return nblocks * BLOCK[dtype]
    ne = 1
    for d in dims:
        ne *= d
    if dtype in SCALAR:
        return ne * SCALAR[dtype]
    raise ValueError(f"unknown dtype {dtype}")


def read_index(f):
    """Parse every record header, leaving the file object just past the last one."""
    f.seek(0)
    if f.read(4) != MAGIC:
        raise ValueError("not a TMQ1 container")
    (count,) = struct.unpack("<I", f.read(4))
    recs = []
    for _ in range(count):
        hdr = bytearray()

        def take(n):
            b = f.read(n)
            if len(b) != n:
                raise ValueError("truncated header")
            hdr.extend(b)
            return b

        (nl,) = struct.unpack("<I", take(4))
        name = take(nl).decode("utf-8")
        dtype, rank = struct.unpack("<II", take(8))
        dims = struct.unpack("<" + "I" * rank, take(4 * rank))
        (nblocks,) = struct.unpack("<Q", take(8))
        rec = Rec(name, dtype, dims, nblocks, f.tell(), bytes(hdr))
        f.seek(rec.size, os.SEEK_CUR)          # skip the payload
        recs.append(rec)
    return recs


def digest(f, rec):
    h = hashlib.sha256()
    f.seek(rec.off)
    left = rec.size
    while left:
        b = f.read(min(CHUNK, left))
        if not b:
            raise ValueError(f"{rec.name}: truncated payload")
        h.update(b)
        left -= len(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--drop-prefix", action="append", default=[], metavar="P",
                    help="drop every tensor whose name starts with P (repeatable)")
    ap.add_argument("--force", action="store_true",
                    help="drop a lm_head.weight that is NOT byte-identical "
                         "(changes the model)")
    a = ap.parse_args(argv)
    dst = a.dst or (a.src + ".trim.tmq")

    with open(a.src, "rb") as f:
        recs = read_index(f)
        by_name = {r.name: r for r in recs}

        # Whichever mode, the head is never dropped unless it IS the embedding
        # (or --force): that is the one removal that silently changes the model.
        doomed = set()
        if a.drop_prefix:
            for r in recs:
                if any(r.name.startswith(p) for p in a.drop_prefix):
                    doomed.add(r.name)
        elif HEAD in by_name:
            doomed.add(HEAD)
        if HEAD in doomed and HEAD in by_name:
            if EMBED not in by_name:
                print(f"{a.src}: no {EMBED} to fall back on — refusing", file=sys.stderr)
                return 2
            same = digest(f, by_name[HEAD]) == digest(f, by_name[EMBED])
            print(f"{HEAD}: {by_name[HEAD].size/2**20:.1f} MB, "
                  f"{'identical to' if same else 'DIFFERENT from'} {EMBED}")
            if not same and not a.force:
                print("refusing to drop a head that is not the tied embedding "
                      "(pass --force to change the model)", file=sys.stderr)
                return 1

        kept = [r for r in recs if r.name not in doomed]
        if not doomed:
            print(f"{a.src}: nothing to drop — nothing written")
            return 0
        for r in recs:
            if r.name in doomed:
                print(f"  drop {r.size/2**20:8.2f} MB  {r.name}")

        tmp = dst + ".part"
        with open(tmp, "wb") as o:
            o.write(MAGIC)
            o.write(struct.pack("<I", len(kept)))
            for r in kept:
                o.write(r.hdr)
                f.seek(r.off)
                left = r.size
                while left:
                    b = f.read(min(CHUNK, left))
                    if not b:
                        raise ValueError(f"{r.name}: truncated payload")
                    o.write(b)
                    left -= len(b)
        os.replace(tmp, dst)

    dropped = sum(by_name[n].size for n in doomed)
    print(f"wrote {dst}  ({len(kept)} tensors, {dropped/2**20:.1f} MB smaller)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
