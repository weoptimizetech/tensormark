#!/usr/bin/env python3
"""fetch_models.py — get the models this engine is tested against, from wherever you keep them.

Sources are declared in examples/models.json and tried in order, so the same command
works whether the weights come from a folder on the local network, an Ollama install,
or Hugging Face. A directory is tried first by default: a share you already have costs
no bandwidth and is usually fastest.

    python3 examples/fetch_models.py               # inventory: what is here, and why it matters
    python3 examples/fetch_models.py --plan        # what is missing, and where it would come from
    python3 examples/fetch_models.py --fetch       # do it
    python3 examples/fetch_models.py --fetch gpt2  # just one

The default mode and --plan never write anything. Nothing is downloaded without
--fetch, and a fetch is refused when the bytes will not fit in the free space.

The manifest also records each artifact's ROLE, because the artifacts do not have
equal standing: `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` looks like a spare copy of
`tinyllama_q40.tmq`, but it is the instrument bench_all.sh runs llama-bench against
for the published baselines. `--list` says so instead of leaving you to guess.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "models.json"
BUILD = REPO / "tensormark" / "build"


def say(msg: str = "", end: str = "\n") -> None:
    # `end` is here for the download progress line, which redraws in place. Without
    # it the hf path raised TypeError on its first chunk — a defect that survived
    # because only the dir source had ever been exercised.
    print(msg, end=end, flush=True)


def human(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} GB"


def present(p: Path) -> bool:
    """Present means it resolves AND is non-empty, so a dangling symlink or a
    0-byte placeholder is reported missing rather than silently satisfying a gate."""
    try:
        r = p.resolve()
        return r.exists() and r.is_file() and r.stat().st_size > 0
    except OSError:
        return False


def du(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


class Manifest:
    def __init__(self, path: Path) -> None:
        self.doc = json.loads(path.read_text())
        self.models: dict[str, dict] = self.doc["models"]

    def ids(self) -> list[str]:
        return sorted(self.models)

    def dir_of(self, mid: str) -> Path:
        return REPO / self.models[mid]["dir"]

    def events(self, mid: str):                      # (name, spec, present?)
        d = self.dir_of(mid)
        for name, spec in self.models[mid]["artifacts"].items():
            yield name, spec, present(d / name)

    def missing_required(self, mid: str) -> list[str]:
        return [n for n, s, here in self.events(mid) if s.get("required") and not here]

    def sources(self, mid: str) -> list[dict]:
        return self.models[mid].get("sources", [])

    def roots(self, override: str | None) -> list[Path]:
        raw = [override] if override else self.doc["dir_roots"]
        out: list[Path] = []
        for r in raw:
            if not r:
                continue
            if "${" in r and Path(os.path.expandvars(r)) == Path(r):
                continue          # unset variable: an unexpanded ${VAR} is not a path
            p = Path(os.path.expanduser(os.path.expandvars(r)))
            if str(p) not in {str(x) for x in out}:   # an unset $VAR expands to nothing
                out.append(p)
        return out


# --------------------------------------------------------------------- sources

def probe_dir(man: Manifest, mid: str, rs: list[Path]) -> Path | None:
    """A local or network folder that already holds at least one required artifact."""
    want = [n for n, s, _ in man.events(mid) if s.get("required")]
    sub = next((s.get("sub") for s in man.sources(mid) if s["kind"] == "dir"), mid)
    for r in rs:
        for cand in ([r / sub] if sub else []) + [r]:
            if cand.is_dir() and any(present(cand / w) for w in want):
                return cand
    return None


def ollama_tags() -> list[str] | None:
    """Available tags, or None when the daemon is not reachable."""
    if not shutil.which("ollama"):
        return None
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return [line.split()[0] for line in r.stdout.splitlines()[1:] if line.split()]


def probe_ollama(src: dict, tags: list[str] | None) -> str | None:
    if tags is None:
        return None
    pat = re.compile(src.get("match", ""), re.I)
    return next((t for t in tags if pat.search(t)), None)


def hf_size(repo: str, files: list[str]) -> int:
    """HEAD the files so the plan quotes a real number, not the manifest's guess."""
    total = 0
    for f in files:
        url = f"https://huggingface.co/{repo}/resolve/main/{f}"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=20) as r:   # noqa: S310
                total += int(r.headers.get("Content-Length", 0) or 0)
        except Exception:                                        # noqa: BLE001
            return 0                                             # unknown, not zero
    return total


# --------------------------------------------------------------------- actions

def cmd_list(man: Manifest, ids: list[str]) -> int:
    for mid in ids:
        miss = man.missing_required(mid)
        say(f"{'MISS' if miss else 'ok  '} {mid:<10} {man.models[mid]['dir']}/   "
            f"[{man.models[mid]['arch']}]")
        say(f"       {man.models[mid]['role']}")
        for name, spec, here in man.events(mid):
            state = "present" if here else ("REQUIRED, MISSING" if spec.get("required") else "absent")
            say(f"         {state:<17} {name}")
            say(f"         {'':<17}   {spec['role']}")
        say(f"       {human(du(man.dir_of(mid)))} on disk")
        say()
    return 0


def cmd_plan(man: Manifest, rs: list[Path], ids: list[str]) -> int:
    tags = ollama_tags()
    need = 0
    for mid in ids:
        miss = man.missing_required(mid)
        if not miss:
            say(f"ok   {mid}: nothing to fetch")
            continue
        say(f"MISS {mid}: needs {', '.join(miss)}")
        resolved = False
        for src in man.sources(mid):
            kind = src["kind"]
            if kind == "dir":
                where = probe_dir(man, mid, rs)
                if where:
                    say(f"       → dir     {where}")
                    resolved = True
                else:
                    say(f"       ·  dir     not found under {', '.join(str(r) for r in rs)}")
            elif kind == "ollama":
                if tags is None:
                    say("       ·  ollama  unavailable — daemon not running (`ollama serve`)")
                    continue
                tag = probe_ollama(src, tags)
                if tag:
                    say(f"       → ollama  {tag}")
                    resolved = True
                else:
                    say(f"       ·  ollama  no tag matching /{src.get('match')}/ in {tags}")
            elif kind == "hf":
                n = hf_size(src["repo"], src["files"])
                say(f"       → hf      {src['repo']}  ({human(n) if n else 'size unknown'})")
                need += n
                resolved = True
            if resolved:
                break
        if not resolved:
            say("       no source resolved")
        say()
    if need:
        f = shutil.disk_usage(REPO).free
        say(f"transfer ≈ {human(need)} · free {human(f)}")
        if need > f * 0.95:
            say("  WARNING: will not fit. Free space, or point --root at a local folder.")
    return 0


def do_dir(man: Manifest, mid: str, where: Path) -> None:
    d = man.dir_of(mid)
    d.mkdir(parents=True, exist_ok=True)
    for name, _spec, here in man.events(mid):
        src, dst = where / name, d / name
        if here or not present(src):
            continue
        if shutil.which("rsync"):
            say(f"  rsync {src.name}")
            subprocess.run(["rsync", "-a", "--partial", "--progress", str(src), str(dst)],
                           check=True)
        else:
            say(f"  copy {src.name}")
            shutil.copy2(src, dst)


def get_files(repo: str, files: list[str], pdir: Path) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = pdir / Path(f).name
        if present(dst):
            say(f"  have {dst.name}")
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{f}?download=true"
        tmp = dst.with_suffix(dst.suffix + ".part")
        say(f"  GET {url}")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as out:   # noqa: S310
            total = int(r.headers.get("Content-Length", 0) or 0)
            done = 0
            while chunk := r.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    say(f"\r  {human(done)} / {human(total)} ({100 * done // total}%)", end="")
        say()
        tmp.rename(dst)


def do_hf(src: dict, pdir: Path) -> None:
    get_files(src["repo"], src["files"], pdir)
    # Sidecars that live in another repo — a GGUF mirror usually carries only the
    # weights, while config.json (the qwen35 hybrid geometry) is in the original.
    for extra in src.get("also", []):
        get_files(extra["repo"], extra["files"], pdir)


def do_ollama(tag: str, pdir: Path) -> None:
    say(f"  ollama pull {tag}")
    subprocess.run(["ollama", "pull", tag], check=True)
    show = subprocess.run(["ollama", "show", tag, "--modelfile"],
                          capture_output=True, text=True)
    blob = ""
    for line in show.stdout.splitlines():
        if line.strip().startswith("FROM "):
            blob = line.strip().split(" ", 1)[1].strip()
    if not blob or not Path(blob).exists():
        say(f"  no blob path for {tag}; leaving the weights to Ollama")
        return
    pdir.mkdir(parents=True, exist_ok=True)
    dst = pdir / (Path(blob).name + ".gguf")
    say(f"  {blob}\n    -> {dst}")
    shutil.copy2(blob, dst)


def run_recipe(man: Manifest, mid: str, recipe: list[dict]) -> None:
    d = str(man.dir_of(mid))
    for step in recipe:
        tool = BUILD / step["tool"]
        if not tool.exists():
            say(f"  ! {tool.name} not built — run ./{man.doc.get('converter_build')}")
            return
        argv = [a.replace("{dir}", d) for a in step["argv"]]
        say(f"  {tool.name} {' '.join(Path(a).name for a in argv)}")
        subprocess.run([str(tool), *argv], check=True)


def cmd_fetch(man: Manifest, rs: list[Path], ids: list[str], force: bool) -> int:
    tags = ollama_tags()
    plan: list[tuple[str, str, object]] = []
    transfer = 0
    for mid in ids:
        if not man.missing_required(mid):
            say(f"ok   {mid}: already present")
            continue
        where = probe_dir(man, mid, rs)
        if where:
            plan.append((mid, "dir", where))
            continue
        for src in man.sources(mid):
            if src["kind"] == "ollama":
                tag = probe_ollama(src, tags)
                if tag:
                    plan.append((mid, "ollama", tag))
                    break
            elif src["kind"] == "hf":
                plan.append((mid, "hf", src))
                transfer += hf_size(src["repo"], src["files"])
                break
    if not plan:
        say("nothing to do")
        return 0
    free = shutil.disk_usage(REPO).free
    if transfer and transfer > free * 0.95 and not force:
        say(f"refusing: ≈{human(transfer)} needed, {human(free)} free. "
            f"Point --root at a local folder, free space, or pass --force.")
        return 1
    for mid, kind, payload in plan:
        say(f"→ {mid} from {kind}")
        if kind == "dir":
            do_dir(man, mid, payload)                     # type: ignore[arg-type]
        elif kind == "hf":
            src = payload                                 # type: ignore[assignment]
            do_hf(src, man.dir_of(mid))
            run_recipe(man, mid, src.get("recipe", []))
        elif kind == "ollama":
            do_ollama(payload, man.dir_of(mid))           # type: ignore[arg-type]
            src = next((s for s in man.sources(mid) if s["kind"] == "hf"), None)
            if src:
                say("  (converting requires a .gguf recipe; add one to the manifest for "
                    "the Ollama path if you want it automated here)")
    say("\nre-run without arguments for the inventory.")
    return 0


def main() -> int:
    # Behave like a unix tool when piped into `head`: die on SIGPIPE instead of
    # raising BrokenPipeError out of a print() and printing a traceback over the
    # output someone was reading. Found by running `--list | head -30`.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*", help="model ids (default: all)")
    ap.add_argument("--root", help="folder to look in first (default: the manifest's dir_roots)")
    ap.add_argument("--list", action="store_true", help="inventory (the default mode)")
    ap.add_argument("--plan", action="store_true", help="show what would be fetched; writes nothing")
    ap.add_argument("--fetch", action="store_true", help="actually fetch")
    ap.add_argument("--force", action="store_true", help="fetch even if it will not fit")
    a = ap.parse_args()

    man = Manifest(MANIFEST)
    ids = a.models or man.ids()
    for mid in ids:
        if mid not in man.models:
            say(f"unknown model '{mid}'; known: {', '.join(man.ids())}")
            return 2
    rs = man.roots(a.root)

    if a.fetch:
        return cmd_fetch(man, rs, ids, a.force)
    if a.plan:
        return cmd_plan(man, rs, ids)
    return cmd_list(man, ids)


if __name__ == "__main__":
    raise SystemExit(main())
