#!/usr/bin/env python3
"""fetch_model.py — download a small Llama-family model for the engine.

Fetches the HF `safetensors` checkpoint + SentencePiece tokenizer for one of
the supported models and converts it to the engine's `.tmq` container (Q4_0
block quantization) with `tensormark/convert_tmq`. No huggingface_hub
dependency: plain HTTPS against huggingface.co resolve endpoints.

    python3 examples/fetch_model.py                 # TinyLlama 1.1B Chat -> data/tinyllama/
    python3 examples/fetch_model.py --list

This downloads ~2.2 GB for TinyLlama; disk is the only requirement. The
weights are third-party content governed by their own licenses (see NOTICE).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODELS = {
    "tinyllama": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "weights": "model.safetensors",
        "tokenizer": "tokenizer.model",
        "out": "data/tinyllama",
        "size_gb": 2.2,
    },
}


def fetch(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  {url} -> {dst}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310
        total = int(r.headers.get("Content-Length", 0)) or None
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done // total
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({pct}%)", end="", flush=True)
    print()
    tmp.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", default="tinyllama", choices=sorted(MODELS))
    ap.add_argument("--out", help="target directory (default: data/<name>/ in the repo)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for name, m in MODELS.items():
            print(f"{name}: {m['repo']} (~{m['size_gb']} GB) -> {m['out']}/")
        return
    m = MODELS[a.model]
    out = REPO / (a.out or m["out"])
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{m['repo']}/resolve/main"

    tmq = out / "tinyllama_q40.tmq"
    if tmq.exists():
        print(f"already converted: {tmq}")
        return
    weights = out / m["weights"]
    if not weights.exists():
        fetch(f"{base}/{m['weights']}?download=true", weights)
    tok = out / m["tokenizer"]
    if not tok.exists():
        fetch(f"{base}/{m['tokenizer']}?download=true", tok)

    print("converting to .tmq (Q4_0) ...")
    src = REPO / "tensormark" / "convert_tmq.cpp"
    exe = REPO / "build" / "convert_tmq"
    exe.parent.mkdir(exist_ok=True)
    if not exe.exists():
        subprocess.run(["c++", "-std=c++23", "-O2", "-Itensormark",
                        str(src), "-o", str(exe)], cwd=REPO, check=True)
    subprocess.run([str(exe), str(weights), str(tmq), "q40", "asis"], check=True)
    print(f"\ndone. try it:\n"
          f"  cd {REPO} && ./tensormark/build_llama.sh\n"
          f"  ./tensormark/build/llama_chat_metal --oneshot -m {tmq} -t {tok} "
          f"-s 'You are terse.' -n 64 -p 'Name the primary colors.'")


if __name__ == "__main__":
    main()
