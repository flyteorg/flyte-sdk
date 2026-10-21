"""
A stand-in for `mlx_lm.server` with the parts of its behavior the Apple Silicon bridge depends on,
selected by FAKE_DAEMON_MODE:

- ok: serves /health, /v1/models and (streaming) completions.
- model_not_found: like mlx_lm.server given a bad --model, /health turns 503 after the load fails.
- crash: exits during startup.
- oom: serves, then SIGKILLs itself on the first completion, as the kernel does under memory pressure.
"""

import argparse
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = os.environ.get("FAKE_DAEMON_MODE", "ok")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    model: str | None = None
    healthy = True

    def log_message(self, format, *args):
        sys.stderr.write(f"fake-daemon {format % args}\n")

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200 if Handler.healthy else 503, {"status": "ok" if Handler.healthy else "unavailable"})
        elif self.path.startswith("/v1/models"):
            self._json(200, {"data": [{"id": Handler.model or "default_model"}]})
        else:
            self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if MODE == "oom" and body.get("prompt") != "Hello":
            os.kill(os.getpid(), signal.SIGKILL)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for word in ("hello", "from", "metal"):
                chunk = {"choices": [{"delta": {"content": word}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(
            200,
            {
                "model": Handler.model,
                "auth": self.headers.get("Authorization"),
                "choices": [{"message": {"content": "hello from metal"}}],
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model")
    args = parser.parse_args()
    print(f"fake mlx_lm.server starting: model={args.model}", flush=True)
    if MODE == "crash":
        print("RuntimeError: [metal::Device] Unable to load kernel", flush=True)
        sys.exit(3)
    Handler.model = args.model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if MODE == "model_not_found":
        Handler.healthy = False
        print("huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error. Repository Not Found", flush=True)
    time.sleep(0.2)
    server.serve_forever()


if __name__ == "__main__":
    main()
