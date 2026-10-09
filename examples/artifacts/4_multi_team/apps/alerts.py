"""An app that reads three artifacts, one of them unpartitioned (owner: ML Platform).

`flyte materialize app churn-alerts --partition date=...` walks back from each parameter: churn_model and
daily_report for that date, and churn_thresholds, which has no partitions, at its latest version. With a range
(`date=a..b`) it builds every day and serves the newest.

    flyte deploy --root-dir . apps/alerts.py alerts
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from ml.thresholds import churn_thresholds
from ml.train import churn_model

import flyte
import flyte.app
import flyte.artifacts as artifacts
from flyte.io import File

# Analytics owns daily_report: a reference, not an import.
daily_report = artifacts.Artifact.ref("daily_report", type=File, partitions={"date": artifacts.Daily})

alerts = flyte.app.AppEnvironment(
    name="churn-alerts",
    image=flyte.Image.from_debian_base(),
    consumes_artifacts={"model": churn_model, "report": daily_report, "thresholds": churn_thresholds},
    labels={"team": "ml"},
    requires_auth=False,
    port=8080,
)


@alerts.server
def serve(model: File, report: File, thresholds: File) -> None:
    with open(thresholds.path) as fh:
        body = json.dumps(
            {"thresholds": json.load(fh), "model": os.path.basename(model.path), "report": report.path}
        ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
