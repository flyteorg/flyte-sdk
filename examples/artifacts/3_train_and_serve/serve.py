"""Serve the model from the artifact registry.

`consumes_artifacts={"model": sentiment_model}` binds the app's `model` parameter to the artifact. Two ways to
point the app at a version:

- `flyte deploy`: the app resolves the newest `sentiment_model` when it starts.
- `flyte materialize app sentiment-api --partition date=...`: builds the model for that date if it is missing,
  then redeploys the app pinned to exactly that version (`served.py` shows which one).

    flyte deploy --root-dir . serve.py api
    curl "<app url>/predict?text=love+it+works+great"
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from train import sentiment_model

import flyte
import flyte.app
from flyte.io import File

api = flyte.app.AppEnvironment(
    name="sentiment-api",
    image=flyte.Image.from_debian_base(),
    consumes_artifacts={"model": sentiment_model},
    requires_auth=False,
    port=8080,
)


@api.server
def serve(model: File) -> None:
    with open(model.path) as fh:
        weights = json.load(fh)["weights"]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            text = parse_qs(urlparse(self.path).query).get("text", [""])[0]
            score = sum(weights.get(w, 0.0) for w in text.lower().split())
            body = json.dumps({"text": text, "positive": score > 0, "score": round(score, 3)}).encode()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
