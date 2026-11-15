#!/usr/bin/env python3
"""Gate: the pooled-embedding path (the `--embed` CLI and the `/embed` protocol).

What the embedding IS: the mean of the final-layer hidden state over the prompt's
positions, L2-normalised. That is the same tensor whose last row feeds the LM
head, so the numerics of the hidden state itself are covered by the logits gates
transitively — but the POOLING is this path's own arithmetic, and nothing
reference-checks it. This gate therefore asserts properties, not a golden vector:

  shape        length equals config.json's hidden_size, and the vector is unit-norm
  determinism  the same text twice is bit-identical (the path is greedy-free)
  resolution   different texts give materially different vectors, so the result is
               not collapsed to a constant the checks above could not catch
  agreement    the CLI and the batch `/embed` command return the same vector, so
               the two surfaces cannot drift apart
  semantics    with a TRAINED model (not the random-weight fixture), a related pair
               scores higher cosine than an unrelated pair

The last check is the only one that can fail for lack of a trained model, and it is
skipped — not faked — when TM_TEST_MODEL points at the synthetic fixture, whose
weights have no semantics by construction. Exit 77 when no model is present, the
convention the other gates use.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tmchat import Engine  # noqa: E402

BIN = REPO / "tensormark" / "build" / "llama_chat_metal"
MODEL = Path(os.environ.get(
    "TM_TEST_MODEL", REPO / "tensormark" / "data" / "tinyllama" / "tinyllama_q40.tmq"))
TOKENIZER = Path(os.environ.get(
    "TM_TEST_TOKENIZER", REPO / "tensormark" / "data" / "tinyllama" / "tokenizer.model"))
# A synthetic fixture has random weights, so semantic checks would be meaningless
# there rather than merely unproven.
SEMANTIC = "TM_TEST_MODEL" not in os.environ

_n = 0


def ok(msg: str) -> None:
    global _n
    _n += 1
    print(f"  ok   {msg}")


def norm(vec) -> float:
    return math.sqrt(sum(x * x for x in vec))


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))


def cli_embed(text: str) -> list[float]:
    """The --embed surface, as a subprocess: a separate code path from /embed."""
    out = subprocess.run(
        [str(BIN), "--embed", "-m", str(MODEL), "-t", str(TOKENIZER), "-p", text],
        capture_output=True, text=True, errors="replace", cwd=str(REPO), check=True)
    return json.loads(out.stdout.strip())["embedding"]


def main() -> int:
    if not BIN.exists() or not MODEL.exists():
        print("SKIP: build the engine and fetch a model "
              "(./tensormark/build_llama.sh, python3 examples/fetch_model.py)")
        return 77
    hidden = json.loads((MODEL.parent / "config.json").read_text())["hidden_size"]

    vec = cli_embed("The cat sat on the mat.")
    assert len(vec) == hidden, f"--embed: {len(vec)} values, config says hidden_size {hidden}"
    ok(f"--embed: {len(vec)} values, matching config hidden_size")
    assert abs(norm(vec) - 1.0) < 1e-5, f"--embed: |v| = {norm(vec)!r}, expected 1"
    ok(f"--embed: unit-norm (|v| - 1 = {norm(vec) - 1:.2e})")

    with Engine(greedy=True) as e:
        same = e.embed("The cat sat on the mat.")
        assert same == vec, "/embed and --embed disagree on the same text"
        ok("/embed and --embed agree exactly on the same text")
        assert e.embed("The cat sat on the mat.") == same, "/embed is not deterministic"
        ok("/embed: same text twice is bit-identical")

        other = e.embed("Quantum chromodynamics is a gauge theory of the strong force.")
        similarity = cosine(same, other)
        assert similarity < 0.999, (
            f"unrelated texts give cosine {similarity:.6f}: the vector is not "
            f"resolving the input")
        ok(f"different texts differ (cosine {similarity:.4f})")

        if not SEMANTIC:
            print("  skip related-vs-unrelated cosine: TM_TEST_MODEL is the "
                  "random-weight fixture, which has no semantics to test")
        else:
            related = e.embed("A cat was sitting on a mat.")
            near, far = cosine(same, related), cosine(same, other)
            assert near > far, (
                f"a trained model should score the related pair higher: related "
                f"{near:.4f} vs unrelated {far:.4f}")
            ok(f"trained model ranks related above unrelated "
               f"(cosine {near:.4f} > {far:.4f})")

    print(f"\n{_n} checks, 0 failed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
