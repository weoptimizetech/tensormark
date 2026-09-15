#!/usr/bin/env python3
"""check_readme_links.py — the README must not point at anything that is not there.

Two failure modes, both of which have actually shipped here and are cheap to
catch mechanically:

  1. A markdown link or image targets a path that does not exist in the PUBLIC
     tree. docs/public/README.md is staged to the public repo's root, so the
     paths it writes are public paths — and a private-only file (there was a
     near miss with docs/ANE_M1_LESSONS.md, which is private while the public
     tree carries docs/ANE.md) becomes a dead link for every reader.
  2. An in-page `#anchor` points at a heading that is not in the document.

It also resolves backticked path-looking spans (`docs/BENCHMARKS.md`,
`tensormark/llama.h`) against a set of roots, because the "Where things live"
table and the inline prose are where a stale filename survives longest — those
are not links, so nothing else would ever check them.

RUN THIS AGAINST THE PUBLIC TREE. `docs/ANE_M1_LESSONS.md` is the reason: it
exists in the private tree and is absent from the public one, so a --root of the
private tree happily resolves a reference that is dead for every reader. The
public clone is what export_public.sh ships, and export_public.sh runs this
there, next to the dangling-include check it already performs.

Usage:
    python3 tensormark/check_readme_links.py [--root DIR] [--readme PATH]

Exit 0 when every reference resolves, 1 otherwise (each miss is printed).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Where a backticked path may be relative to. `tensormark/` is the C++/Python
# source dir; the repo root holds README/docs/examples/tools/scripts.
# The public README sits at the public root, so both are the same directory
# there — listing them separately keeps this correct for either layout.
MAX_REPORTED = 40

# A backticked candidate is checked only when it looks like a repository path:
# it contains a slash AND its first segment is one of these, which keeps
# `torch.nn.MSELoss` and `-t <file>` out of the check.
PATH_ROOTS = {
    "docs", "examples", "scripts", "tools", "tensormark", "tests",
    "specs", "benchmarks", "kernel", "autotune", "mnist", "profile",
}
# Extensions that make a bare filename a checkable artifact even without a
# leading directory (`llama.h`, `BENCHMARKS.md`).
ARTIFACT_EXT = {
    ".md", ".h", ".cpp", ".py", ".sh", ".json", ".png", ".svg", ".txt",
    ".yml", ".yaml", ".tmq", ".gbnf", ".toml",
}
# Backticked spans that are commands/placeholders, not paths.
SKIP_SPAN = re.compile(
    r"[\s<>=|()\[\]$*]"          # a space or shell metacharacter => not a path
    r"|^[-+]"                    # a flag
    r"|^\d"                      # a number
)

# Upstream project names that look like filenames. `llama.cpp` is the other
# project this README compares against, not a file in this repository.
PROJECT_NAMES = {"llama.cpp", "whisper.cpp", "ggml.c"}

# Names that are real but not repository files: they sit beside a downloaded
# model, or are produced by a build. Each needs a reason, or the allowlist
# becomes the place stale references go to hide.
RUNTIME_NAMES = {
    "config.json",   # written next to the .tmq by fetch_model.py
    "merges.txt",    # BPE sidecar fetched with a Qwen/LLaMA-3 tokenizer
    "special.json",  # BPE special-token sidecar, same origin
    "tokenizer.model",   # the SentencePiece file fetch_model.py downloads
}

IGNORE_DIRS = {".git", "build", "dist", ".venv", "node_modules",
               "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}

# Fallback for a tree with no .git, where `git check-ignore` cannot be asked.
# It has to name every generated directory the README legitimately refers to —
# `tensormark/build/test_gbnf` (a build output) and `tensormark/data/tinyllama/`
# (where fetch_model.py puts the weights) both shipped as false positives, each
# caught only by CI, because both exist on the machine that wrote the README.
OUTPUT_SEGMENTS = {"build", "dist", "out", "data"}

# The two charts in the README are <img src="..."> tags, not markdown images,
# so a broken src there would otherwise go unnoticed.
HTML_REF = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CODE = re.compile(r"`([^`]+)`")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)
# Fenced blocks are shell transcripts, not prose: their contents are not links
# and a path inside one is contextual. Stripping them also keeps a fence's own
# backticks from pairing with an inline span's and swallowing a real reference.
FENCE = re.compile(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", re.M | re.S)


def slug(text: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation, spaces -> dashes.

    Backticked spans inside a heading keep their text but lose the backticks;
    a heading that is entirely a code span keeps its word characters.
    """
    t = text.replace("`", "")
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)   # links keep their label
    t = t.lower()
    t = re.sub(r"[^\w\s-]", "", t)                    # drop punctuation
    t = t.strip().replace(" ", "-")
    return t


def anchors(text: str) -> set[str]:
    out: set[str] = set()
    for _, title in HEADING.findall(text):
        out.add(slug(title))
    return out


def ignored_matcher(root: Path):
    """Predicate: is this repo-relative path one that git ignores?

    A path git ignores is generated, not authored — a build output, a downloaded
    model — so its absence from a fresh checkout is not a documentation defect.
    Asking git beats a hardcoded list: the list was wrong twice.
    """
    if (root / ".git").exists():
        def via_git(rel: str) -> bool:
            p = rel.rstrip("/")
            if not p:
                return False
            # `root` is the repository top: the README's paths are relative to
            # it, and so are the ignore patterns.
            #
            # Probe twice, directory form first. A directory-only pattern
            # (`tensormark/build/`) matches a bare `tensormark/build` ONLY when
            # that directory happens to exist: git decides by stat-ing it. So the
            # same probe answers IGNORED on the machine that has built the tree
            # and not-ignored in CI's fresh clone — the difference is invisible
            # until the gate runs somewhere the directory is absent, which is
            # precisely where it matters. The trailing slash states the intent
            # instead of relying on the filesystem to guess it.
            for probe in (p + "/", p):
                r = subprocess.run(
                    ["git", "-C", str(root), "check-ignore", "-q", "--", probe],
                    capture_output=True)
                if r.returncode == 0:
                    return True
            return False
        return via_git

    def via_segments(rel: str) -> bool:
        return bool(OUTPUT_SEGMENTS & set(rel.rstrip("/").split("/")))
    return via_segments


def basename_index(root: Path) -> set[str]:
    """Every filename in the tree, by basename.

    A bare `llama.h` or `BENCHMARKS.md` names a real file, just not one at the
    repository root, so existence is settled on the basename. A name that
    appears nowhere is the defect this catches — a file that was renamed or
    never existed (`docs/` was once described by a private-only filename).
    """
    out: set[str] = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if IGNORE_DIRS & set(p.parts):
            continue
        out.add(p.name)
    return out


def check(readme: Path, root: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    text = FENCE.sub("\n", text)
    bad: list[str] = []
    have = anchors(text)
    names = basename_index(root)
    is_ignored = ignored_matcher(root)

    # ---- 1) markdown links and HTML src/href: files and in-page anchors ----
    for target in LINK.findall(text) + HTML_REF.findall(text):
        t = target.strip()
        if t.startswith(("http://", "https://", "mailto:")):
            continue
        if t.startswith("data:"):
            continue
        if t.startswith("#"):
            if slug(t[1:]) not in have:
                bad.append(f"anchor  {t}  (no such heading)")
            continue
        path = t.split("#", 1)[0]
        if not path:
            continue
        if not (root / path).exists():
            bad.append(f"link    {t}  (missing: {path})")

    # ---- 2) backticked path-looking spans (the stale-name case) ----
    for span in CODE.findall(text):
        s = span.strip().rstrip(",.;:")
        if not s or SKIP_SPAN.search(s):
            continue
        if s in PROJECT_NAMES:
            continue
        head = s.split("/", 1)[0]
        if "/" in s:
            if head not in PATH_ROOTS:
                continue
            if is_ignored(s):
                continue
            # a glob or an abbreviated listing is not a literal path
            if "*" in s or s.startswith("."):
                continue
            if not (root / s).exists():
                bad.append(f"code    `{s}`  (missing path)")
            continue
        if Path(s).suffix not in ARTIFACT_EXT:
            continue
        if s in RUNTIME_NAMES:
            continue
        if s not in names:
            bad.append(f"code    `{s}`  (no such file anywhere in the tree)")

    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", default=str(here),
                    help="tree the README's paths are relative to")
    ap.add_argument("--readme", default=None)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    readme = Path(a.readme) if a.readme else root / "docs" / "public" / "README.md"
    if not readme.exists():                      # the public tree's own layout
        readme = root / "README.md"
    if not readme.exists():
        print(f"no README at {readme}", file=sys.stderr)
        return 2

    # `relative_to` raises when the README was named from outside the tree (the
    # self-test does exactly that); a report must never die on its own label.
    def show(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except ValueError:
            return str(p)

    bad = check(readme, root)
    if bad:
        print(f"{show(readme)}: {len(bad)} unresolved reference(s)")
        for line in bad[:MAX_REPORTED]:
            print(f"  {line}")
        if len(bad) > MAX_REPORTED:
            print(f"  … and {len(bad) - MAX_REPORTED} more")
        return 1
    print(f"{show(readme)}: every link, anchor and path reference resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
