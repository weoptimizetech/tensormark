#!/usr/bin/env python3
"""jsonschema_to_gbnf.py — compile a JSON Schema into a CLOSED GBNF grammar.

Why this exists
---------------
A generic JSON grammar is recursive: objects nest in arrays nest in objects, and
nothing forces the document to end. Constrained decoding then guarantees legal
bytes but not a finished document — on TinyLlama-1.1B Q4_0 the model free-runs
inside a permissive grammar, repeating `"$ref"/"$title"` until the budget dies,
and `json.loads` never succeeds. Structure was never the hard part; CLOSURE is.

So this compiler does not emit a JSON grammar. It emits the *schema's* language,
with every key written out in a fixed order and every array bounded. The model's
only remaining choice is the scalar values, and there is no legal move that does
not advance toward the closing brace. Generation ends because the language is
finite, which is what makes `json.loads(output)` a reliable contract rather than
a hope.

Supported keywords
------------------
  type            object, array, string, integer, number, boolean, null, and
                  unions ("type": ["string", "null"])
  properties      emitted in declaration order; `required` first, then optional
                  ones as optional groups
  required        a key absent here is emitted as an optional group
  enum / const    alternation over the literal values (typed by JSON rendering)
  items           element schema for arrays
  minItems/maxItems   bounds; maxItems defaults to --max-items, never unbounded
  minLength/maxLength bounds on string length
  pattern         a subset: literal-anchored character classes
  $ref            inlined to --max-depth levels, then omitted if optional or
                  rejected if required

Anything unsupported raises `Unsupported` rather than quietly emitting a
permissive rule: a grammar that accepts more than the schema is worse than no
grammar, because the caller stops checking.

Usage
-----
    python3 tensormark/jsonschema_to_gbnf.py schema.json > schema.gbnf
    python3 tensormark/jsonschema_to_gbnf.py --max-items 3 schema.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Grammar rules the compiler always emits alongside the schema rules.
PREAMBLE = """ws ::= [ \\t\\n]*
string ::= "\\"" chars* "\\""
chars ::= [^"\\\\\\x00-\\x1f]
integer ::= "-"? ( "0" | [1-9] [0-9]* )
number ::= integer ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
"""


class Unsupported(Exception):
    """A schema keyword this compiler will not approximate."""


class Compiler:
    def __init__(self, schema: dict[str, Any], *, max_items: int = 4, max_depth: int = 6):
        self.root = schema
        self.max_items = max_items
        self.max_depth = max_depth
        self.rules: dict[str, str] = {}
        self.counter = 0
        self._by_body: dict[str, str] = {}       # body -> rule name, for dedup
        self.defs: dict[str, Any] = schema.get("$defs", schema.get("definitions", {}))

    # ---------------------------------------------------------------- helpers
    def fresh(self, stem: str) -> str:
        self.counter += 1
        return f"{stem}{self.counter}"

    def add(self, name: str, body: str) -> str:
        """Define a rule, deduplicating identical bodies.

        Names are assigned in first-use order, NOT from `hash(body)`: Python
        salts string hashes per process, so a hash-derived name would make the
        emitted grammar differ between two runs on the same schema. A grammar
        that is not reproducible cannot be diffed, committed, or cache-keyed.
        """
        if body in self._by_body:
            return self._by_body[body]
        rule = f"{name}_{self.counter}"
        self.counter += 1
        self.rules[rule] = body
        self._by_body[body] = rule
        return rule

    def literal(self, value: Any) -> str:
        """A GBNF literal matching the JSON *rendering* of `value`.

        The quotes in a GBNF literal are DELIMITERS, not content: `"subject"`
        matches the seven characters `subject`, not the nine of `"subject"`. So
        the JSON text has to be escaped into the literal — a key or a string
        enum member must be emitted as `"\\"subject\\""`, or the grammar accepts
        the bare word and the generated document is not JSON at all. Getting
        this backwards is invisible in the grammar text (it reads exactly like
        the JSON you wanted) and shows up only in the output.
        """
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise Unsupported(f"enum/const value not a scalar: {value!r}")
        body = json.dumps(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{body}"'

    def bounded_comma(self, item_rule: str, lo: int, hi: int) -> str:
        """The language of lo..hi comma-separated `item_rule`s.

        GBNF counted repetition `{m,n}` is rejected by this engine's parser, so
        the bound is written out as an alternation over the exact lengths. That
        is more verbose than nested optionals and much easier to get right — the
        nested-optional form is easy to write one item too wide, which shows up
        as a generated document that overruns maxItems rather than as an error.
        """
        if hi < lo:
            raise Unsupported(f"maxItems {hi} < minItems {lo}")
        alts = [' \",\" ws '.join([item_rule] * k) for k in range(max(lo, 1), hi + 1)]
        if not alts:
            return ""                              # the only legal value is []
        return " | ".join(f"( {a} )" for a in alts)

    # ------------------------------------------------------------- the walker
    def rule_for(self, schema: Any, depth: int, hint: str) -> str:
        if not isinstance(schema, dict):
            raise Unsupported(f"schema is not an object: {schema!r}")
        if depth > self.max_depth:
            if "default" in schema:
                return self.add(hint, self.literal(schema["default"]))
            raise Unsupported(f"$ref nesting exceeds max-depth={self.max_depth}")

        # $ref: inline the target so the grammar stays finite
        if "$ref" in schema:
            ref = schema["$ref"]
            if not ref.startswith("#/"):
                raise Unsupported(f"external $ref is not supported: {ref}")
            node = self._resolve(ref)
            return self.rule_for(node, depth + 1, hint)

        if "const" in schema:
            return self.add(hint, self.literal(schema["const"]))

        if "enum" in schema:
            alts = " | ".join(self.literal(v) for v in schema["enum"])
            if not alts:
                raise Unsupported("empty enum")
            return self.add(hint, alts)

        t = schema.get("type")
        if isinstance(t, list):
            alts = [self.rule_for({**schema, "type": one}, depth, hint) for one in t]
            return self.add(hint, " | ".join(alts))
        if t is None:
            # no type: fall back to whatever shape the keywords imply
            if "properties" in schema:
                t = "object"
            elif "items" in schema:
                t = "array"
            else:
                raise Unsupported("schema has neither type, enum, const nor $ref")

        if t == "null":
            return self.add(hint, '"null"')
        if t == "boolean":
            return self.add(hint, '"true" | "false"')
        if t == "integer":
            return self.add(hint, "integer")
        if t == "number":
            return self.add(hint, "number")
        if t == "string":
            return self._string_rule(schema, hint)
        if t == "array":
            return self._array_rule(schema, depth, hint)
        if t == "object":
            return self._object_rule(schema, depth, hint)
        raise Unsupported(f"type {t!r} is not supported")

    def _resolve(self, ref: str) -> Any:
        node: Any = self.root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise Unsupported(f"cannot resolve $ref {ref!r}")
        return node

    def _string_rule(self, schema: dict[str, Any], hint: str) -> str:
        maxlen = schema.get("maxLength")
        minlen = schema.get("minLength", 0)
        if minlen or maxlen is not None:
            # Bound the length so a string cannot run to the token budget. With
            # no maxLength the bound is minLength + 32 — a string of unbounded
            # length is the other way a "closed" grammar stops closing.
            hi = min(maxlen if maxlen is not None else minlen + 32, minlen + 32)
            return self.add(hint, self._length_bounded("chars", minlen, hi))
        return self.add(hint, "string")

    def _length_bounded(self, chars: str, lo: int, hi: int) -> str:
        """`"` + lo..hi units of `chars` + `"`, as an alternation over lengths."""
        if hi < lo:
            raise Unsupported("maxLength < minLength")
        alts = [" ".join([chars] * k) for k in range(lo, hi + 1)]
        body = " | ".join(f"( {a} )" for a in alts if a) or f"( {chars} )"
        return f'"\\"" ( {body} ) "\\""'

    def _array_rule(self, schema: dict[str, Any], depth: int, hint: str) -> str:
        items = schema.get("items")
        if items is None:
            raise Unsupported("array without `items` would accept anything; add items or drop the key")
        if isinstance(items, list):
            raise Unsupported("tuple-style `items` arrays are not supported")
        item_rule = self.rule_for(items, depth + 1, f"{hint}_item")
        lo = int(schema.get("minItems", 0))
        hi = int(schema.get("maxItems", self.max_items))
        if hi < lo:
            raise Unsupported("maxItems < minItems")
        body = self.bounded_comma(item_rule, lo, hi)
        # An empty body means the only legal value is `[]`; an empty group
        # `( )?` would also be a parse error, so emit nothing between brackets.
        inner = f"( {body} )" if body else ""
        return self.add(hint, f'"[" {inner} "]"')

    def _object_rule(self, schema: dict[str, Any], depth: int, hint: str) -> str:
        props: dict[str, Any] = schema.get("properties")
        if not props:
            raise Unsupported("object without `properties` would accept anything")
        if schema.get("additionalProperties", False) not in (False,):
            raise Unsupported("additionalProperties must be false (extra keys are not modelled)")
        required = list(schema.get("required", []))
        unknown = [r for r in required if r not in props]
        if unknown:
            raise Unsupported(f"required keys with no property schema: {unknown}")

        ordered = [k for k in props if k in required] + [k for k in props if k not in required]
        members = []
        for key in ordered:
            sub = self.rule_for(props[key], depth + 1, f"{hint}_{key}")
            pair = f'({self.literal(key)}) ws ":" ws {sub}'
            members.append(pair if key in required else f"( {pair} )?")

        # required first as a comma-joined head, then the optional tail. A
        # comma must precede every member except the first REQUIRED one; when
        # nothing is required the whole list is optional.
        if required:
            head = members[: len(required)]
            tail = members[len(required):]
            body = " \",\" ws ".join(head)
            for m in tail:
                body = f"{body} ( \",\" ws {m} )?"
        else:
            body = members[0]
            for m in members[1:]:
                body = f"{body} ( \",\" ws {m} )?"
        # Only make the body optional when NOTHING is required. Wrapping a
        # required body in `( ... )?` lets `{}` through, which the schema does
        # not accept — and `{}` is the shortest legal path, so a greedy model
        # finds it every time and every instance is schema-invalid.
        inner = f"( {body} )?" if not required else body
        return self.add(hint, f'"{{" {inner} "}}"')

    # ------------------------------------------------------------------ entry
    def compile(self) -> str:
        root = self.rule_for(self.root, 0, "root")
        lines = [PREAMBLE.rstrip("\n")]
        for name, body in self.rules.items():
            lines.append(f"{name} ::= {body}")
        # the engine's entry rule is always `root`
        lines.append(f"root ::= {root}")
        return "\n".join(lines) + "\n"


def compile_schema(schema: dict[str, Any], *, max_items: int = 4,
                   max_depth: int = 6) -> str:
    """Compile `schema` to a closed GBNF grammar string."""
    return Compiler(schema, max_items=max_items, max_depth=max_depth).compile()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("schema", type=Path, help="JSON Schema file, or - for stdin")
    ap.add_argument("--max-items", type=int, default=4,
                    help="bound for arrays whose schema gives no maxItems (default 4)")
    ap.add_argument("--max-depth", type=int, default=6, help="depth bound for $ref (default 6)")
    a = ap.parse_args()
    text = sys.stdin.read() if str(a.schema) == "-" else a.schema.read_text()
    try:
        sys.stdout.write(compile_schema(json.loads(text), max_items=a.max_items,
                                        max_depth=a.max_depth))
    except Unsupported as exc:
        print(f"unsupported schema: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
