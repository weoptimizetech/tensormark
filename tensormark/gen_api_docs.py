#!/usr/bin/env python3
"""Generate the API reference for TensorMark's public Python surface.

Why this is generated rather than written
-----------------------------------------
The README's performance tables are generated from evidence and gated, so a stale
number cannot survive in them. An API reference is the same kind of artifact and
rots the same way: a signature that no longer matches the code is worse than no
signature at all, because it is trusted. So this page is produced from the source
and `--check` fails when the copy on disk has drifted from it.

Scope
-----
Only the symbols in PUBLIC below — the same curated surface the module docstring
of `tensormark/tensormark/torch.py` names, and nothing else. Everything else in
that module is an implementation detail, and publishing it would both mislead and
freeze internals. This layer is explicitly a *subset* of PyTorch, not a clone, so
the page states that in its own header rather than letting a reader infer parity.

Nothing here executes the module: the source is read with `ast`, so generating
documentation can never import an engine, allocate a tape, or touch a model file.

Usage
-----
    python3 tensormark/gen_api_docs.py            # write docs/API.md
    python3 tensormark/gen_api_docs.py --check    # fail if docs/API.md is stale
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tensormark" / "tensormark" / "torch.py"
TARGET = ROOT / "docs" / "API.md"

# The curated surface, grouped as it is presented. Dotted paths, resolved against
# the module: a class attribute is written `Class.member`. Adding a name here is
# the deliberate act of making it public; the drift gate then keeps it honest.
PUBLIC: list[tuple[str, list[str]]] = [
    ("Tensors", [
        "Tensor",
        "Tensor.shape",
        "Tensor.numpy",
        "Tensor.item",
        "Tensor.data",
        "Tensor.grad",
        "Tensor.backward",
        "from_numpy",
        "zeros",
        "rand",
        "randn",
    ]),
    ("Reproducibility and devices", [
        "manual_seed",
        "device",
        "no_grad",
    ]),
    ("Neural network layers", [
        "nn.Module",
        "nn.Module.parameters",
        "nn.Module.zero_grad",
        "nn.Module.to",
        "nn.Linear",
        "nn.Linear.forward",
        "nn.ReLU",
        "nn.Sequential",
        "nn.MSELoss",
    ]),
    ("Optimizers", [
        "optim.SGD",
        "optim.Adam",
    ]),
]

HEADER = """\
# TensorMark API reference

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with: python3 tensormark/gen_api_docs.py
     Checked in CI:   python3 tensormark/gen_api_docs.py --check -->

This is the reference for the Python layer's **public surface only**. It is
generated from the source, so a signature here is the signature the code has.

Two things it deliberately does not claim:

* **It is not the full PyTorch API.** `tensormark.torch` implements a subset,
  chosen so that a training loop written against PyTorch runs unchanged. Names it
  does not implement raise the ordinary `AttributeError`.
* **It is not the engine.** None of the throughput in the README travels through
  this layer; that is the C++ engine's job, and its entry points are the binaries
  `tensormark/build_llama.sh` produces.

Every symbol below is listed in `PUBLIC` in `tensormark/gen_api_docs.py`.
Anything not listed there is an implementation detail and may change without
notice.
"""


def load() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def resolve(tree: ast.Module, dotted: str) -> ast.AST | None:
    """Find `a.b.c` in the module by walking the AST, without executing it."""
    parts = dotted.split(".")
    node: ast.AST | None = tree
    for part in parts:
        found = None
        body = getattr(node, "body", None)
        if body is None:
            return None
        for item in body:
            if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == part:
                found = item
                break
        if found is None:
            return None
        node = found
    return node


def signature(node: ast.AST) -> str:
    """Render a callable's parameters, without its return annotation noise."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = node.args
    rendered: list[str] = []
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg == "self":
            continue
        rendered.append(arg.arg if default is None else
                        f"{arg.arg}={ast.unparse(default)}")
    if args.vararg:
        rendered.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        rendered.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        rendered.append(arg.arg if default is None else
                        f"{arg.arg}={ast.unparse(default)}")
    if args.kwarg:
        rendered.append("**" + args.kwarg.arg)
    return ", ".join(rendered)


def describe(node: ast.AST, dotted: str, kind: str) -> list[str]:
    """One `###` block: heading, signature, docstring."""
    # Members keep their qualifier: `data` and `shape` exist on more than one
    # class, and an unqualified heading would make the page ambiguous.
    short = dotted.split(".")[-1]
    heading = dotted if "." in dotted else short
    lines: list[str] = []
    params = signature(node)
    if kind == "class":
        call = ""
        init = next((n for n in getattr(node, "body", [])
                     if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is not None:
            call = signature(init)
        lines.append(f"### `{heading}({call})`" if call else f"### `{heading}`")
        lines.append("")
        lines.append("*class*")
    else:
        decorators = {ast.unparse(d).split("(")[0]
                      for d in getattr(node, "decorator_list", [])}
        is_property = "property" in decorators
        lines.append(f"### `{heading}`" if is_property
                     else f"### `{heading}({params})`")
        lines.append("")
        lines.append("*property*" if is_property
                     else ("*method*" if "." in dotted else "*function*"))
    lines.append("")
    doc = ast.get_docstring(node, clean=True)
    if doc:
        lines.append(doc)
    else:
        # Visible, not hidden: an undocumented public symbol should read as a gap
        # in the page, which is pressure to close it rather than a silent blank.
        lines.append("*(no docstring in the source yet)*")
    lines.append("")
    return lines


def render() -> str:
    tree = load()
    out = [HEADER]
    module_doc = ast.get_docstring(tree, clean=True)
    if module_doc:
        out.append("## What this layer promises\n")
        out.append(module_doc)
        out.append("")
    missing: list[str] = []
    for group, names in PUBLIC:
        out.append(f"## {group}\n")
        for dotted in names:
            node = resolve(tree, dotted)
            if node is None:
                raise SystemExit(
                    f"gen_api_docs: PUBLIC lists {dotted!r}, which is not in "
                    f"{SOURCE.relative_to(ROOT)} — remove it or fix the spelling")
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            out.extend(describe(node, dotted, kind))
            if not ast.get_docstring(node, clean=True):
                missing.append(dotted)
    out.append("---\n")
    out.append(f"{len(missing)} public symbols are still undocumented: "
               + (", ".join(f"`{m}`" for m in missing) if missing else "none") + ".")
    trailer = "\n".join(out).rstrip() + "\n"
    return trailer


def main() -> int:
    rendered = render()
    if "--check" in sys.argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != rendered:
            print(f"gen_api_docs: {TARGET.relative_to(ROOT)} is out of date.\n"
                  f"  Regenerate with: python3 tensormark/gen_api_docs.py",
                  file=sys.stderr)
            return 1
        print(f"gen_api_docs: {TARGET.relative_to(ROOT)} matches the source")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(rendered, encoding="utf-8")
    print(f"gen_api_docs: wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
