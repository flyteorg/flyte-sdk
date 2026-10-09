"""An app produces an endpoint (owner: ML Platform).

`consumes_artifacts={"model": churn_model}` is sugar for
`Parameter(name="model", value=churn_model, download=True)`: the app receives the latest churn_model at
activation and deploy writes `lineage.consumes=churn_model` on it. The backend derives
`lineage.produces=app:churn-scoring`; the SDK rejects an authored one.

    flyte deploy --root-dir . apps/scoring.py scoring
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ml.train import churn_model

import flyte
import flyte.app
from flyte.io import File

scoring = flyte.app.AppEnvironment(
    name="churn-scoring",
    image=flyte.Image.from_debian_base(),
    consumes_artifacts={"model": churn_model},  # sugar for the Parameter below
    labels={"team": "ml"},
    requires_auth=False,
    port=8080,
)

# Desugared, and the escape hatch when you need the mount path or env var:
#   flyte.app.Parameter(name="model", value=churn_model, download=True, env_var="MODEL_PATH")


def _handler(weights: dict) -> type:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"model": "churn_model", "weights": weights}).encode()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    return Handler


@scoring.server
def serve(model: File) -> None:
    """Load the model the platform resolved at activation and serve it."""
    with open(model.path) as fh:
        weights = json.load(fh)
    HTTPServer(("0.0.0.0", 8080), _handler(weights)).serve_forever()
