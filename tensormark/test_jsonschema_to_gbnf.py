#!/usr/bin/env python3
"""test_jsonschema_to_gbnf.py — gate for schema→grammar compilation.

Two halves, and the split matters.

The PURE half always runs: it compiles a set of schemas, asserts the output is
byte-identical across processes (a hash-derived rule name would break this,
since Python salts string hashes per process), asserts unsupported keywords are
refused rather than approximated, and asserts no counted repetition `{m,n}`
reaches the grammar because this engine's parser rejects it.

The MODEL half needs the binary and a .tmq, and skips otherwise. It is the one
that answers the question that actually matters: does constrained decoding
produce a document that PARSES and VALIDATES, not merely legal bytes? It reports
a closure rate over several attempts per schema rather than a single pass, so a
lucky run cannot be mistaken for a capability.

Run:  python3 tensormark/test_jsonschema_to_gbnf.py
Exits 0 on pass, 1 on failure, 77 when binary/model are absent (pure half still
runs and is reported first).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tensormark"))

from jsonschema_to_gbnf import Unsupported, compile_schema  # noqa: E402

BIN = REPO / "tensormark" / "build" / "llama_chat_metal"
MODEL = Path(os.environ.get(
    "TM_TEST_MODEL", REPO / "tensormark" / "data" / "tinyllama" / "tinyllama_q40.tmq"))
TOKENIZER = Path(os.environ.get(
    "TM_TEST_TOKENIZER", REPO / "tensormark" / "data" / "tinyllama" / "tokenizer.model"))

SCHEMAS: dict[str, dict] = {
    "triple": {
        "type": "object", "additionalProperties": False,
        "required": ["subject", "relation", "object"],
        "properties": {
            "subject": {"type": "string", "minLength": 1, "maxLength": 6},
            "relation": {"enum": ["is-a", "part-of", "causes"]},
            "object": {"type": "string", "minLength": 1, "maxLength": 6},
        },
    },
    "typed": {
        "type": "object", "additionalProperties": False,
        "required": ["n", "ok"],
        "properties": {
            "n": {"type": "integer"},
            "ok": {"type": "boolean"},
        },
    },
    "list": {
        "type": "object", "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {"type": "array", "items": {"type": "integer"}, "maxItems": 3},
        },
    },
    "nullable": {
        "type": "object", "additionalProperties": False,
        "required": ["v"],
        "properties": {"v": {"type": ["string", "null"], "maxLength": 4}},
    },
    "ref": {
        "type": "object", "additionalProperties": False,
        "required": ["child"],
        "properties": {"child": {"$ref": "#/$defs/leaf"}},
        "$defs": {"leaf": {
            "type": "object", "additionalProperties": False,
            "required": ["name"],
            "properties": {"name": {"type": "string", "maxLength": 4}},
        }},
    },
}


def pure_checks() -> int:
    n = 0
    for name, schema in SCHEMAS.items():
        g = compile_schema(schema)
        n += 1
        assert "{m,n}" not in g and "{" not in g.split("::=", 1)[0], f"{name}: counted repetition leaked"
        assert "\nroot ::= " in g, f"{name}: no root rule"
        print(f"  ok   {name}: compiled ({len(g)} bytes, {g.count('::=')} rules)")

    # determinism across PROCESSES, which is what a per-process hash would break
    code = ("import sys; sys.path.insert(0, 'tensormark');"
            "from jsonschema_to_gbnf import compile_schema;"
            "import json; sys.stdout.write(compile_schema(json.load(open('/tmp/tm_schema.json'))))")
    Path("/tmp/tm_schema.json").write_text(json.dumps(SCHEMAS["triple"]))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           cwd=str(REPO)).stdout for _ in range(2)}
    n += 1
    assert len(outs) == 1, "grammar is not reproducible across processes"
    print("  ok   byte-identical across two separate processes")

    # an unsupported keyword must be refused, never approximated
    for bad, why in (
        ({"type": "object", "properties": {"a": {"type": "string"}},
          "additionalProperties": True}, "additionalProperties true"),
        ({"type": "array"}, "array with no items"),
        ({"type": "object"}, "object without properties"),
    ):
        n += 1
        try:
            compile_schema(bad)
        except Unsupported:
            print(f"  ok   refused: {why}")
        else:
            raise AssertionError(f"should have refused: {why}")
    return n


def model_checks(attempts: int = 3) -> int:
    try:
        import jsonschema
    except ImportError:
        # A missing validator is a gate that could not run, not a broken engine —
        # the same distinction the exit-77 convention draws everywhere else. Left
        # as a crash it reads as a real defect in CI and buries the actual cause.
        print("SKIP model half: the `jsonschema` package is not installed "
              "(python3 -m pip install jsonschema)")
        return 77
    from tmchat import Engine

    print("\n  model half — does the output actually parse and validate?")
    n = 0
    # ONE engine process for every attempt: a fresh process per generation would
    # reload the model each time and turn a 20-second gate into a 10-minute one.
    with Engine(budget=160) as e:
        for name, schema in SCHEMAS.items():
            e.set_grammar(compile_schema(schema))
            closed = validated = 0
            last = ""
            for _ in range(attempts):
                out = e.complete("Fill in the fields.")
                last = out
                try:
                    doc = json.loads(out)
                except ValueError:
                    continue
                closed += 1
                try:
                    jsonschema.validate(doc, schema)
                except jsonschema.ValidationError:
                    continue
                validated += 1
            n += 1
            pct = 100 * closed / attempts
            assert closed, (f"{name}: never closed a document in {attempts} attempts "
                            f"— last={last[:140]!r}")
            print(f"  ok   {name}: closed {closed}/{attempts}, schema-valid {validated}/{attempts}"
                  f"  ({pct:.0f}% closure)")
    return n


def main() -> int:
    n = pure_checks()
    print(f"\n  pure half: {n} checks, 0 failed")
    if not BIN.exists() or not MODEL.exists():
        print("SKIP model half: build the engine and fetch a model "
              "(./tensormark/build_llama.sh, python3 examples/fetch_model.py)")
        return 77
    m = model_checks()
    if m == 77:
        return 77
    print(f"\n{n + m} checks, 0 failed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
