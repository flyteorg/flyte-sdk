"""A dashboard that cannot import any handle (owner: Analytics).

Two edges, neither typed:

- an `AppEndpoint` parameter on churn-scoring, the app -> app edge (`lineage.consumes=app:churn-scoring`);
- a hand-written `lineage.consumes=daily_report` label: a label-only edge. It draws, answers "what breaks
  if I change this", and drives notifications, but the planner never traverses it.

    flyte deploy --root-dir . apps/dashboard.py dashboard
"""

from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import flyte
import flyte.app
from flyte.app import AppEndpoint, Parameter

dashboard = flyte.app.AppEnvironment(
    name="churn-dashboard",
    image=flyte.Image.from_debian_base(),
    parameters=[Parameter(name="scorer_url", value=AppEndpoint(app_name="churn-scoring"), env_var="SCORER_URL")],
    labels={"team": "analytics", "lineage.consumes": "daily_report"},
    requires_auth=False,
    port=8080,
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = f"<h1>Churn dashboard</h1><p>scorer: {os.environ.get('SCORER_URL', '?')}</p>".encode()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(body)


@dashboard.server
def serve() -> None:
    HTTPServer(("0.0.0.0", 8080), _Handler).serve_forever()
