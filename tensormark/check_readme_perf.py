#!/usr/bin/env python3
"""Gate: every result in the README's Performance tables must have evidence.

The README claims its tables are "extracted from raw measurement files rather
than transcribed by hand". That claim is only meaningful if it is testable, so
this script tests it.

Three things make that harder than it looks, and each cost this script a
revision:

  * A value-based rule ("skip integers under 5 digits") cannot work. It skips
    830 tok/s, keeps 4096 out of a shape name, and cannot see 1,614 at all
    because the thousands separator splits it into invisible pieces. The rule
    here is positional: the first column is the row label, and every other cell
    is a result if either its column header or the cell itself carries a unit
    (tok/s, ms, samples/s, x, ratio, rel err).
  * Shapes leak numbers, so shape-like runs (T256, K4096, C96, 32x32) are
    stripped from labels before tokenising.
  * A single flat pool of evidence values produces FALSE PASSES. Searching one
    pool let a table's 2.36 match an unrelated kernel row's ratio_max of 2.36.
    So each table is matched against its OWN evidence block, resolved from the
    heading above it, and nothing else.

Evidence is written to docs/perf_evidence.json, which is versioned so a reader
can audit a published figure without re-running anything. tensormark/build/ is
gitignored, so an evidence file kept there is invisible to everyone but the
machine that produced it.

Exit status is non-zero when any printed result is unevidenced.

Usage:
    python3 tensormark/check_readme_perf.py [--readme PATH] [--evidence PATH]
    python3 tensormark/check_readme_perf.py --report   # exit 0, still prints
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# A cell carries a result when the header or the cell names a unit or metric.
UNIT = re.compile(
    r"(tok/s|samples/s|\bms\b|s/step|speed-?up|ratio|rel err|parity|×)",
    re.IGNORECASE)
# Shape runs glued to letters (T256, K4096, C96, 32x32) are labels, not results,
# and so are dtype/quant tags (Q4_0, Q8_0, fp16, int4) and size labels (124M,
# 1.1B). Without stripping these, "Q4_0 hybrid" yields a spurious 4 and 0 and
# every row reports twice.
SHAPE = re.compile(
    r"\b[TKNCHB]\d+\b"                       # T256, K4096, C96, B8
    r"|\d+\s*[x×]\s*\d+"                     # 32x32
    r"|\bof \d+\b"                           # of 2048
    r"|\bQ\d+(?:_\w+)?\b"                    # Q4_0, Q8_0, Q5_K
    r"|\b(?:fp|bf|int)\d+\b"                 # fp16, bf16, int4
    r"|\b\d+(?:\.\d+)?\s*[BM]\b",            # 124M, 1.1B
    re.IGNORECASE)
PARENTHETICAL = re.compile(r"\([^)]*\)")
NUMBER = re.compile(r"\d+(?:\.\d+)?(?:e-?\d+)?")   # keeps 1.2e-4 whole
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")

# Which evidence blocks each table may be checked against, keyed on the nearest
# heading above it. A table with no mapping is checked against nothing and every
# result in it is reported — better a loud unknown than a silent pass.
TABLE_EVIDENCE = (
    ("end-to-end", ("training", "inference_bench", "design_point")),
    ("final-tuning", ("finetuning",)),
    ("fine-tuning", ("finetuning",)),
    ("cross-check", ("inference_bench",)),
    ("design point", ("design_point",)),
    ("warm kv cache", ("kv_turns",)),
    ("kernels vs", ("kernels",)),
    ("per-shape kernels", ("kernels",)),
)

# The Performance section has been renamed more than once; accept both names.
SECTION_HEADS = ("\n## How it performs", "\n## Performance")


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def result_tokens(cell: str, header: str) -> list[str]:
    if not (UNIT.search(header) or UNIT.search(cell)):
        return []
    cell = PARENTHETICAL.sub(" ", cell)     # "(Q4_0 hybrid, GPU+CPU)" is a note
    # A cell that is a sentence rather than a measurement is prose, not a
    # result: "infeasible on 8 GB: 16.4 GiB weights + grads + AdamW" would
    # otherwise report the host's RAM as an unevidenced figure.
    if len(re.findall(r"[A-Za-z]{2,}", cell)) >= 3:
        return []
    cell = SHAPE.sub(" ", cell)
    cell = THOUSANDS.sub("", cell)          # 1,614 -> 1614
    return NUMBER.findall(cell)


def tables(text: str) -> list[tuple[tuple[str, ...], int, list[tuple[int, str]]]]:
    """(evidence blocks, line, [(line, token)]) for every table under a heading."""
    out = []
    heading = ""
    header: list[str] | None = None
    rows: list[tuple[int, str]] = []
    start = 0

    def flush():
        if rows:
            keys = next((k for frag, k in TABLE_EVIDENCE if frag in heading), ())
            out.append((keys, start, list(rows)))

    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("#"):
            flush()
            rows, header = [], None
            heading = line.lstrip("# ").lower()
            continue
        if not line.startswith("|"):
            if line.strip():
                flush()
                rows, header = [], None
            continue
        if set(line) <= set("|-: "):
            continue
        row = cells(line)
        if header is None:
            header = row
            start = lineno
            continue
        for idx, cell in enumerate(row):
            if idx == 0:                    # the row label column
                continue
            hdr = header[idx] if idx < len(header) else ""
            for tok in result_tokens(cell, hdr):
                rows.append((lineno, tok))
    flush()
    return out


def evidence_numbers(node) -> set[str]:
    """Every number in one evidence block."""
    found: set[str] = set()

    def walk(n):
        if isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
        elif isinstance(n, bool):
            return
        elif isinstance(n, (int, float)):
            found.add(repr(float(n)))
        elif isinstance(n, str):
            found.update(NUMBER.findall(n))

    walk(node)
    return found


def matches(tok: str, pool: set[str]) -> bool:
    """Evidenced if some recorded value rounds to the printed token.

    The README prints rounded results (1973, 3.38, 1.22x) while evidence may
    hold more precision (1972.6666), so compare at the printed token's own
    resolution: the number of decimals it shows.
    """
    tok = tok.replace(",", "")
    if tok in pool:
        return True
    printed = float(tok)
    decimals = len(tok.split(".")[1]) if "." in tok else 0
    for cand in pool:
        try:
            v = float(cand)
        except ValueError:
            continue
        if abs(v) < 1.0 and printed >= 1.0:
            continue            # a fraction cannot evidence an integer-scale result
        if round(v, decimals) == printed:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    # The README lives at docs/public/README.md in the private tree and at the
    # repository root in the public one (the export MAPs it there). A hardcoded
    # private path made this gate unrunnable in CI — it exited 2 on "no README"
    # the first time it was wired in, which is the whole reason both defaults
    # are resolved against the repo root instead of the cwd.
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--readme", default=None)
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--report", action="store_true",
                    help="always exit 0; just print the audit")
    a = ap.parse_args()

    readme = Path(a.readme) if a.readme else root / "docs" / "public" / "README.md"
    if not readme.exists():                      # the public tree's own layout
        readme = root / "README.md"
    if not readme.exists():
        print(f"no README at {readme}", file=sys.stderr)
        return 2
    text = readme.read_text()
    start = -1
    for head in SECTION_HEADS:
        start = text.find(head)
        if start >= 0:
            break
    if start < 0:
        print(f"no performance section found (looked for "
              f"{' or '.join(SECTION_HEADS)})", file=sys.stderr)
        return 2
    end = text.find("\n## ", start + 1)
    section = text[start:end if end > 0 else len(text)]
    offset = text[:start].count("\n") + 1

    ev_path = Path(a.evidence) if a.evidence else root / "docs" / "perf_evidence.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    if not ev:
        print(f"WARNING: no evidence at {ev_path} — nothing can be checked",
              file=sys.stderr)
    pools = {k: evidence_numbers(v) for k, v in ev.items()
             if k != "_meta" and isinstance(v, dict)}

    checked = 0
    unevidenced: list[tuple[int, str, str]] = []
    for keys, _s, rows in tables(section):
        pool = set().union(*(pools.get(k, set()) for k in keys)) if keys else set()
        for lineno, tok in rows:
            checked += 1
            if not matches(tok, pool):
                unevidenced.append((lineno + offset, tok,
                                    "|".join(keys) or "<unmapped>"))

    print(f"Performance tables: {checked} result numbers, "
          f"{len(unevidenced)} without evidence")
    if unevidenced:
        print("\nunevidenced (transcribed, or a stale copy of an older run):")
        for lineno, tok, key in unevidenced:
            print(f"  {readme}:{lineno}  {tok}   [{key}]")
        print("\nEither regenerate the table from the evidence, or record the "
              "number's provenance in the evidence file.")
        if not a.report:
            return 1
    else:
        print("every printed result is backed by its own evidence block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
