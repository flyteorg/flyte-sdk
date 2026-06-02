"""
Slide deck hub: home page + two slide decks served as a Flyte AppEnvironment.

Routes
------
/                           → home page with buttons linking to both decks
/skypilot/                  → Flyte 2 + SkyPilot integration slides
/dagster/                   → Flyte 2 vs Dagster comparison slides
/health                     → readiness probe (JSON)

Deploy to the control plane:
    flyte serve examples/reference_stacks/flyres_stack/slidedeck_app.py app_env

Serve locally (no control plane needed):
    flyte serve --local examples/reference_stacks/flyres_stack/slidedeck_app.py app_env
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
# ]
# ///

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import flyte
import flyte.app

_DIR = Path(__file__).parent
_SKYPILOT_HTML = _DIR / "SKYPILOT_INTEGRATION_SLIDES.html"
_DAGSTER_HTML = _DIR / "FLYTE_VS_DAGSTER_SLIDES.html"

_PORT = 8080

_HOME_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FLYRES Stack · Slide Decks</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0b0f19; --surface: #121826; --surface-2: #1a2234;
      --border: #2a3548; --text: #e8edf7; --muted: #94a3b8;
      --flyte: #3b82f6; --dagster: #f97316; --union: #6366f1;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      min-height: 100vh; background: var(--bg); color: var(--text);
      font-family: Inter, system-ui, sans-serif;
      display: flex; align-items: center; justify-content: center;
      background:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(99,102,241,0.12), transparent),
        radial-gradient(ellipse 60% 50% at 80% 100%, rgba(59,130,246,0.1), transparent),
        var(--bg);
    }
    .shell {
      width: min(700px, 92vw);
      padding: 48px 0 64px;
      display: flex; flex-direction: column; align-items: center;
      text-align: center;
    }
    .badge {
      font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
      text-transform: uppercase; padding: 6px 14px; border-radius: 999px;
      background: rgba(99,102,241,0.15); color: #a5b4fc;
      border: 1px solid rgba(99,102,241,0.35); margin-bottom: 28px;
    }
    h1 {
      font-size: clamp(2rem, 4vw, 2.8rem); font-weight: 700;
      letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 14px;
    }
    h1 span { color: var(--accent); }
    .sub {
      font-size: 1.05rem; color: var(--muted); line-height: 1.6;
      max-width: 480px; margin-bottom: 48px;
    }
    .decks {
      display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
      width: 100%;
    }
    .deck-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 28px 26px;
      text-decoration: none;
      text-align: left;
      transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s;
      display: block;
    }
    .deck-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 16px 48px rgba(0,0,0,0.35);
    }
    .deck-card.flyte { border-color: rgba(59,130,246,0.35); }
    .deck-card.flyte:hover { border-color: var(--flyte); }
    .deck-card.dagster { border-color: rgba(249,115,22,0.35); }
    .deck-card.dagster:hover { border-color: var(--dagster); }
    .deck-icon {
      width: 40px; height: 40px; border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.3rem; margin-bottom: 14px;
    }
    .deck-card.flyte .deck-icon { background: rgba(59,130,246,0.15); }
    .deck-card.dagster .deck-icon { background: rgba(249,115,22,0.15); }
    .deck-title {
      font-size: 1.05rem; font-weight: 700; color: var(--text);
      margin-bottom: 8px;
    }
    .deck-card.flyte .deck-title { color: #93c5fd; }
    .deck-card.dagster .deck-title { color: #fdba74; }
    .deck-desc {
      font-size: 0.85rem; color: var(--muted); line-height: 1.55;
    }
    .deck-meta {
      font-size: 0.75rem; color: var(--muted); margin-top: 12px;
      opacity: 0.7;
    }
    .footer {
      margin-top: 48px; font-size: 0.78rem; color: var(--muted);
    }
    .footer a { color: var(--accent); text-decoration: none; }
    @media (max-width: 520px) { .decks { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <div class="badge">FLYRES Stack · Reference Presentations</div>
    <h1>Model Factory<br /><span>Slide Decks</span></h1>
    <p class="sub">
      Technical presentations for the FLYRES reference stack —
      Flyte 2, SkyPilot, and how Flyte compares to Dagster.
    </p>
    <div class="decks">
      <a href="skypilot/" class="deck-card flyte">
        <div class="deck-icon">🚀</div>
        <div class="deck-title">Flyte 2 + SkyPilot Integration</div>
        <div class="deck-desc">
          Layered orchestration for large-scale training — Flyte as the macro-orchestrator
          and SkyPilot as the multi-cloud GPU compute broker.
        </div>
        <div class="deck-meta">15 slides · Horizon 1 &amp; 2 · June 2026</div>
      </a>
      <a href="dagster/" class="deck-card dagster">
        <div class="deck-icon">⚖️</div>
        <div class="deck-title">Flyte 2 vs Dagster</div>
        <div class="deck-desc">
          Head-to-head comparison on the FLYRES model-factory workload — mental models,
          GPU training, data gravity, lineage, and when to choose each.
        </div>
        <div class="deck-meta">13 slides · Flyte 2.x vs Dagster 1.13+ · June 2026</div>
      </a>
    </div>
    <p class="footer">
      <a href="https://www.union.ai/docs/v2/flyte/" target="_blank">Flyte docs</a>
      &nbsp;·&nbsp;
      <a href="https://docs.dagster.io/" target="_blank">Dagster docs</a>
      &nbsp;·&nbsp;
      <code style="font-size:0.72rem;">examples/reference_stacks/flyres_stack/</code>
    </p>
  </div>
</body>
</html>
"""

app_env = flyte.app.AppEnvironment(
    name="flyres-slides",
    image=flyte.Image.from_debian_base(python_version=(3, 12)),
    port=_PORT,
    resources=flyte.Resources(cpu="0.5", memory="256Mi"),
    requires_auth=False,
    include=(
        "SKYPILOT_INTEGRATION_SLIDES.html",
        "FLYTE_VS_DAGSTER_SLIDES.html",
    ),
)


class _Handler(BaseHTTPRequestHandler):
    """Routes requests to the home page or either slide deck."""

    _skypilot: bytes = b""
    _dagster: bytes = b""
    _home: bytes = b""

    def _respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", ""):
            self._respond(200, "text/html; charset=utf-8", _Handler._home)
        elif path in ("/skypilot", "/skypilot/"):
            self._respond(200, "text/html; charset=utf-8", _Handler._skypilot)
        elif path in ("/dagster", "/dagster/"):
            self._respond(200, "text/html; charset=utf-8", _Handler._dagster)
        elif path == "/health":
            body = b'{"status": "ok"}'
            self._respond(200, "application/json", body)
        else:
            self._respond(404, "text/plain", b"not found")

    def log_message(self, _fmt, *_args) -> None:
        pass  # suppress per-request noise


@app_env.server
def serve() -> None:
    _Handler._skypilot = _SKYPILOT_HTML.read_bytes()
    _Handler._dagster = _DAGSTER_HTML.read_bytes()
    _Handler._home = _HOME_HTML.encode("utf-8")
    port = app_env.get_port().port
    server = HTTPServer(("0.0.0.0", port), _Handler)
    print(f"Serving FLYRES slide hub on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    flyte.init_from_config(root_dir=_DIR)
    app_handle = flyte.serve(app_env)
    print(f"Slide hub is ready at: {app_handle.url}")
