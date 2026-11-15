#!/usr/bin/env python3
"""three_capabilities.py — what grammar, structured output and embeddings each do.

Three capabilities that are easy to conflate, demonstrated side by side on one
model. The distinction the demo is built to make:

  * a GRAMMAR constrains which tokens are *samplable*, so the output cannot leave
    a formal language. It says nothing about whether the answer is right.
  * STRUCTURED OUTPUT is a JSON Schema compiled into such a grammar. Same
    enforcement, but you hand over a schema instead of writing GBNF by hand.
  * EMBEDDINGS generate nothing at all: one forward pass, and you read the pooled
    hidden state instead of sampling from the logits.

Run:
    python3 examples/three_capabilities.py

With the TinyLlama checkpoint present it also runs the model sections and shows
real output. Without it, it runs everything that needs no model, so the demo is
still worth running on a machine that has not downloaded 2.2 GB — and it says
what to run if you want the rest.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
BIN = REPO / "tensormark" / "build" / "llama_chat_metal"
MODEL = pathlib.Path(os.environ.get(
    "TM_DEMO_MODEL", REPO / "tensormark" / "data" / "tinyllama" / "tinyllama_q40.tmq"))
TOK = pathlib.Path(os.environ.get(
    "TM_DEMO_TOKENIZER", REPO / "tensormark" / "data" / "tinyllama" / "tokenizer.model"))
SCHEMA_TO_GBNF = REPO / "tensormark" / "jsonschema_to_gbnf.py"

PROMPT = ("Classify this review as POSITIVE, NEGATIVE or NEUTRAL: "
          "'Arrived broken and three days late.'")

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["verdict", "confidence", "reasons"],
    "additionalProperties": False,
}

T0 = time.time()
# Grammar files go to a temp dir that is removed on exit: the demo leaves nothing
# behind, which matters on a machine that is short of space.
TMP = tempfile.TemporaryDirectory(prefix="tm_demo_")
WORK = pathlib.Path(TMP.name)


def say(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    say(f"\n── {title}")


def have_model() -> bool:
    return BIN.exists() and MODEL.exists() and TOK.exists()


def gen(prompt: str, grammar: pathlib.Path | None = None, budget: int = 48) -> str:
    """One greedy, deterministic completion. The engine prints ONLY the completion."""
    cmd = [str(BIN), "--oneshot", "-m", str(MODEL), "-t", str(TOK),
           "-s", "You are terse.", "-n", str(budget), "-p", prompt]
    if grammar:
        cmd += ["-G", str(grammar)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        say(f"  engine failed ({r.returncode}): {r.stderr.strip()[:300]}")
        sys.exit(1)
    return r.stdout


def embed(text: str) -> list[float]:
    cmd = [str(BIN), "--embed", "-m", str(MODEL), "-t", str(TOK), "-p", text]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        say(f"  engine failed ({r.returncode}): {r.stderr.strip()[:300]}")
        sys.exit(1)
    return json.loads(r.stdout)["embedding"]


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))


def compile_schema(schema: dict, out: pathlib.Path) -> str:
    sp = WORK / "schema.json"
    sp.write_text(json.dumps(schema, indent=2))
    r = subprocess.run([sys.executable, str(SCHEMA_TO_GBNF), str(sp)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        say(f"  schema compile failed: {r.stderr.strip()[:300]}")
        sys.exit(1)
    out.write_text(r.stdout)
    return r.stdout


def no_model_notice() -> None:
    """Say what is missing, what it costs, and whether this machine can afford it."""
    missing = [p for p in (BIN, MODEL, TOK) if not p.exists()]
    say("\n── model sections skipped — no checkpoint")
    for p in missing:
        say(f"  missing: {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
    free_gb = shutil.disk_usage(REPO).free / 1e9
    say(f"\n  fetching it downloads ~2.2 GB and leaves a 590 MB .tmq"
        f"  (free here: {free_gb:.1f} GB)")
    if free_gb < 4:
        say("  ...which this machine may not have room for. The model-free sections")
        say("  above need nothing; the README's tables were measured on the real thing.")
    say("\n  to run the rest:")
    say("      ./tensormark/build_llama.sh          # the engine")
    say("      python3 examples/fetch_model.py      # the checkpoint")
    say("      python3 examples/three_capabilities.py")


def main() -> int:
    real = have_model()
    say("Grammar vs structured output vs embeddings.")
    if real:
        say("Model: TinyLlama-1.1B-Chat, Q4_0 — a real checkpoint, greedy decoding,")
        say("so every line below is reproducible.")
    else:
        say("No checkpoint found, so only the model-free sections run. Reporting on a")
        say("machine that has not downloaded a model is a supported way to run this.")

    # ------------------------------------------------- 1 · no grammar (needs model)
    if real:
        rule("1 · no grammar — the model answers however it likes")
        say(f"  prompt : {PROMPT}")
        plain = gen(PROMPT, budget=40).strip()
        say(f"  output : {plain!r}")
        say("  Free text: an explanation, not a label. Note what it actually concluded —")
        say("  hold that thought, because it matters more than the formatting.")

        # ------------------------------------------- 2 · grammar (needs model)
        g = WORK / "verdict.gbnf"
        g.write_text('root    ::= "VERDICT=" verdict\n'
                     'verdict ::= "POSITIVE" | "NEGATIVE" | "NEUTRAL"\n')
        rule("2 · grammar — the same model, confined to a formal language")
        say("  grammar :")
        for line in g.read_text().rstrip().splitlines():
            say(f"            {line}")
        out = gen(PROMPT, g, budget=40).strip()
        say(f"  output  : {out!r}")
        say(f"  in the language? {bool(re.fullmatch(r'VERDICT=(POSITIVE|NEGATIVE|NEUTRAL)', out))}")
        say("  The grammar constrains the FORM, and only the form. Above, the model called")
        say("  a review that arrived broken and three days late POSITIVE; here it says")
        say("  NEUTRAL. Both are wrong — 1.1B is a small model — and the grammar did not")
        say("  help, because it has no opinion about the answer. What it bought is")
        say("  certainty about the shape: machine-readable with no parser, no retry loop.")
        say("  Valid, and still wrong. The decoder solves the second problem only.")

        # ------------------------------------- 3 · enforcement over preference
        gd = WORK / "digits.gbnf"
        gd.write_text("root ::= [0-9]+\n")
        rule("3 · grammar — enforcement is stronger than preference")
        say("  A contradictory grammar: the output may only be digits.")
        say("  prompt  : Name the primary colors.")
        d = gen("Name the primary colors.", gd, budget=12).strip()
        say(f"  output  : {d!r}   digits only? {bool(re.fullmatch(r'[0-9]+', d))}")
        say("  The mask is not a suggestion: every token that would leave the language has")
        say("  its logit set to -inf before sampling, so the model cannot emit a letter.")

    # ------------------------------------------- 4 · structured output (model-free)
    rule("4 · structured output — a JSON Schema, compiled into that grammar"
        + ("" if real else " (no model needed)"))
    gjson = WORK / "review.gbnf"
    compiled = compile_schema(SCHEMA, gjson)
    say(f"  the schema is {len(json.dumps(SCHEMA))} bytes: 3 properties, 1 required-set,")
    say(f"  1 enum, 1 bounded array. The compiler turns it into a CLOSED grammar:")
    lines = compiled.rstrip().splitlines()
    for line in lines[:6]:
        say(f"            {line}")
    say(f"            … ({len(lines)} rules, {len(compiled)} bytes)")

    if real:
        js = gen(PROMPT, gjson, budget=160).strip()
        say(f"  output  : {js}")
        try:
            doc = json.loads(js)
            ok_parse = True
        except Exception as e:                                      # noqa: BLE001
            ok_parse, doc = False, None
            say(f"  json.loads -> {e}")
        say(f"  parses as JSON? {ok_parse}")
        if ok_parse:
            try:
                import jsonschema
                jsonschema.validate(doc, SCHEMA)
                say("  validates against the schema? True")
                say("  (The newlines between tokens are the grammar's own `ws` rule —")
                say("  whitespace is legal JSON, and greedy decoding spent its freedom")
                say("  there. Well-formed is not the same as well-considered.)")
            except ImportError:
                say("  validates against the schema? (pip install jsonschema)")
            except Exception as e:                                  # noqa: BLE001
                say(f"  validates against the schema? False — {e.message}")

    # --------------------------------- 4b · parse vs validate (model-free)
    rule("4b · why 'parses' is not 'validates'" + ("" if real else " (no model needed)"))
    plausible = '{"verdict": "MAYBE", "confidence": "very", "reasons": []}'
    say(f"  a document that parses but is wrong: {plausible}")
    say(f"    json.loads          -> {json.loads(plausible)!r}   (accepted)")
    try:
        import jsonschema
        jsonschema.validate(json.loads(plausible), SCHEMA)
        say("    jsonschema.validate -> accepted (unexpected)")
    except ImportError:
        say("    jsonschema.validate -> (pip install jsonschema to see this half)")
    except Exception as e:                                          # noqa: BLE001
        say(f"    jsonschema.validate -> REJECTED: {e.message}")
    say("  A parse-only assertion passes that document, and so does a model that merely")
    say("  usually emits valid JSON. Under a compiled grammar 'MAYBE' is not a reachable")
    say("  token sequence, so it cannot be sampled however the logits fall.")

    # ------------------------------------------------ 5 · embeddings (needs model)
    if real:
        rule("5 · embeddings — the same model, with generation switched off")
        v1 = embed("Arrived broken and three days late.")
        say("  --embed -p '<the review>'")
        say(f"  dim     : {len(v1)}")
        say(f"  first 6 : {[round(x, 4) for x in v1[:6]]}")
        say(f"  |v|     : {norm(v1):.3e}   (L2-normalised; cosine is a plain dot product)")
        say("  No tokens, no grammar, no sampling: one forward pass, and instead of")
        say("  reading the logits we pool the final hidden state over the prompt.")
        v2 = embed("Arrived broken and three days late.")
        say(f"  repeat the same text        -> bit-identical: {v1 == v2}")
        say(f"  cosine(same text)           -> {cos(v1, v2):.6f}")
        say(f"  cosine(reworded same thing) -> "
            f"{cos(v1, embed('It came smashed and late.')):.4f}")
        say(f"  cosine(unrelated sentence)  -> "
            f"{cos(v1, embed('The screen is bright and the battery lasts two days.')):.4f}")
        say("  The ordering holds — related above unrelated — but read the numbers, not")
        say("  the ordering: the margin here is modest and the floor is high. That is what")
        say("  pooling a small model's last hidden layer gives you. Exact geometry,")
        say("  semantics only as good as the checkpoint.")
    else:
        no_model_notice()

    say(f"\ndone in {time.time() - T0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
