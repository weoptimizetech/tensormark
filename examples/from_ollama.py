#!/usr/bin/env python3
"""from_ollama.py — run TensorMark behind an Ollama-compatible HTTP API.

If your tooling already speaks Ollama's `/api/generate` (LangChain, shell
pipelines, `curl`, editor plugins), point it here instead of re-learning a
new CLI. The server drives `build/llama_chat_metal --batch` as a subprocess
and translates its line protocol (PROMPT / EOT / <<<EOT>>>) into Ollama's
JSON-line streaming format.

    ./tensormark/build_llama.sh                 # once
    python3 examples/from_ollama.py \
        --model data/tinyllama/tinyllama_q40.tmq \
        --tokenizer data/tinyllama/tokenizer.model \
        --port 11435

    curl -s localhost:11435/api/generate -d '{"model":"tensormark",
        "prompt":"Name the primary colors.","stream":false}'
    # or streaming:
    curl -N localhost:11435/api/generate -d '{"prompt":"...", "stream":true}'

Differences from a real Ollama server (kept deliberately small and honest):
  * generation is greedy-deterministic; `options` (temperature, top_k, ...)
    are accepted but ignored;
  * each /api/generate request is stateless — the KV cache is not carried
    across requests (Ollama's `context` reuse parameter is ignored);
  * only /api/generate and /api/tags are implemented; /api/chat is not.

The model is NOT bundled: convert an HF safetensors checkpoint with
`tensormark/convert_tmq.cpp` (see examples/from_llama_cpp.md), or download
one with examples/fetch_model.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "tensormark" / "build" / "llama_chat_metal"


class LlamaChatSession:
    """One --batch subprocess; one request at a time (the engine is single-tenant)."""

    def __init__(self, model: str, tokenizer: str, system: str):
        if not BIN.exists():
            sys.exit(f"missing {BIN}\nrun ./tensormark/build_llama.sh first")
        self.proc = subprocess.Popen(
            [str(BIN), "-m", model, "-t", tokenizer, "-s", system, "--batch", "--quiet"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, cwd=str(REPO),
        )
        banner = self.proc.stdout.readline().rstrip("\n")   # "ACK ready" then "PROMPT"
        if banner != "ACK ready":
            sys.exit(f"unexpected engine banner: {banner!r}")
        ready = self.proc.stdout.readline().rstrip("\n")
        if ready != "PROMPT":
            sys.exit(f"engine did not signal readiness: {ready!r}")
        # Invariant from here: the engine has printed PROMPT and is blocked
        # reading message lines — every complete() must leave the same state.

    def complete(self, prompt: str, budget: int) -> str:
        p, q = self.proc.stdin, self.proc.stdout
        p.write(prompt.replace("\n", " ") + "\nEOT\n")  # single-line: batch protocol is line-based
        p.flush()
        reply: list[str] = []
        for line in q:
            line = line.rstrip("\n")
            if line == "<<<EOT>>>":
                break
            reply.append(line)
        else:
            sys.exit("engine stream ended unexpectedly (did the process crash?)")
        while reply and not reply[-1]:
            reply.pop()                                # engine ends the turn with a blank line
        armed = q.readline().rstrip("\n")              # re-arm: consume the next PROMPT marker
        if armed != "PROMPT":
            sys.exit(f"engine out of sync: {armed!r}")
        return "\n".join(reply)

    def close(self):
        try:
            self.proc.stdin.write("/quit\nEOT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, status=200):
            body = (json.dumps(obj) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                self._json({"models": [{"name": state["model_name"], "model": state["model_name"],
                                        "size": 0, "details": {"family": "llama"}}]})
            else:
                self._json({"error": "not found (only /api/generate and /api/tags)"}, 404)

        def do_POST(self):
            if self.path != "/api/generate":
                self._json({"error": "not found (only /api/generate)"}, 404)
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            prompt, stream = req.get("prompt", ""), bool(req.get("stream", False))
            t0 = time.time()
            text = state["session"].complete(prompt, int(req.get("budget", 256)))
            now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
            if not stream:
                self._json({"model": state["model_name"], "created_at": now(),
                            "response": text, "done": True})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for piece in (text[i:i + 8] for i in range(0, len(text), 8)):
                self.wfile.write((json.dumps({"model": state["model_name"], "created_at": now(),
                                              "response": piece, "done": False}) + "\n").encode())
                self.wfile.flush()
            self.wfile.write((json.dumps({"model": state["model_name"], "created_at": now(),
                                          "response": "", "done": True,
                                          "total_duration": int((time.time() - t0) * 1e9)}) + "\n").encode())

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="data/tinyllama/tinyllama_q40.tmq")
    ap.add_argument("--tokenizer", default="data/tinyllama/tokenizer.model")
    ap.add_argument("--system", default="You are a helpful, concise assistant.")
    ap.add_argument("--port", type=int, default=11435)
    a = ap.parse_args()
    state = {"session": LlamaChatSession(a.model, a.tokenizer, a.system),
             "model_name": Path(a.model).stem}
    print(f"tensormark ollama shim on http://127.0.0.1:{a.port} (model: {a.model})")
    try:
        ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state)).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state["session"].close()


if __name__ == "__main__":
    main()
