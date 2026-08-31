"""The GitHub integration app environment.

`GitHubAppEnvironment` is a `FastAPIAppEnvironment` that serves two purposes:

1. A **setup and management dashboard** (`/`) explaining how to configure the
   integration end to end: creating the GitHub token and webhook secrets as
   Flyte secrets, wiring a repository webhook to this app, and registering
   event handlers. `/api/status` and `/api/verify` expose machine-readable
   health information.
2. A **webhook receiver** (`/webhook` by default) that verifies the
   `X-Hub-Signature-256` HMAC, normalizes payloads into `GitHubEvent` objects,
   and dispatches them to registered handlers.

Event handlers are registered with `on_event`, and idempotent run launching is
available via `flyteplugins.github.launch_task`, so the standard pattern is:

```python
import flyte
from flyteplugins.github import GitHubAppEnvironment, launch_task

env = GitHubAppEnvironment(name="github-integration")

@env.on_event("pull_request.opened")
async def review_new_pr(event):
    import flyte.remote as remote

    task = remote.Task.get(name="review_pr_workflow", auto_version="latest")
    run = launch_task(task, key=event.dedupe_key(), repo=event.repository, number=event.number)
    return {"run": run.name}

flyte.serve(env)
```
"""

from __future__ import annotations

import html
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from flyte.app.extras import FastAPIAppEnvironment

from ._config import DEFAULT_API_BASE_URL, DEFAULT_TOKEN_ENV_VAR, DEFAULT_WEBHOOK_SECRET_ENV_VAR
from ._webhook import GitHubEvent, parse_webhook, verify_webhook_signature

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

#: Handler signature: receives a `GitHubEvent`, returns optional result JSON.
EventHandler = Callable[[GitHubEvent], Awaitable[dict[str, Any] | None]]


@dataclass(kw_only=True)
class GitHubAppEnvironment(FastAPIAppEnvironment):
    """Dashboard + webhook receiver app for the GitHub integration.

    Args:
        name: App environment name (also the app name on the platform).
        repos: Optional allowlist of repository full names (`owner/repo`).
            Events from other repositories are acknowledged but not dispatched,
            as are events carrying no repository at all — an allowlist cannot
            vouch for an event it cannot attribute.
        webhook_path: URL path of the webhook receiver.
        token_env: Environment variable holding the GitHub token (mounted from
            a Flyte secret).
        webhook_secret_env: Environment variable holding the webhook secret.
        require_signature: Reject events without a valid HMAC signature. When
            True and no webhook secret is mounted, all events are rejected with
            an explanatory error — set False for local development only.
        api_base_url: GitHub REST API base URL used by `/api/verify`.
        max_recent_events: Size of the in-memory recent-events buffer shown on
            the dashboard.
        event_handlers: Optional initial list of `(pattern, handler)` tuples;
            prefer the `on_event` decorator.
    """

    app: FastAPI | None = None
    repos: list[str] = field(default_factory=list)
    webhook_path: str = "/webhook"
    token_env: str = DEFAULT_TOKEN_ENV_VAR
    webhook_secret_env: str = DEFAULT_WEBHOOK_SECRET_ENV_VAR
    require_signature: bool = True
    api_base_url: str = DEFAULT_API_BASE_URL
    max_recent_events: int = 200
    event_handlers: list[tuple[str, EventHandler]] = field(default_factory=list)

    recent_events: deque[GitHubEvent] = field(init=False, repr=False)

    def __post_init__(self):
        self.recent_events = deque(maxlen=self.max_recent_events)
        if self.app is None:
            self.app = self._build_app()
        super().__post_init__()
        import flyte.app

        self.links = [
            flyte.app.Link(path="/", title="Setup Dashboard", is_relative=True),
            flyte.app.Link(path=self.webhook_path, title="Webhook Receiver", is_relative=True),
            *self.links,
        ]

    # ------------------------------------------------------------------
    # handler registration
    # ------------------------------------------------------------------

    def on_event(self, event_type: str = "") -> Callable[[EventHandler], EventHandler]:
        """Register an async handler for webhook events.

        Args:
            event_type: GitHub event type (`pull_request`, `issues`, ...) or
                qualified type (`pull_request.opened`). An empty string
                matches every event.

        Returns:
            A decorator that registers the handler and returns it unchanged.
        """

        def decorator(fn: EventHandler) -> EventHandler:
            self.event_handlers.append((event_type, fn))
            return fn

        return decorator

    def _matches(self, pattern: str, event: GitHubEvent) -> bool:
        if not pattern:
            return True
        return pattern == event.event_type or pattern == event.qualified_type

    # ------------------------------------------------------------------
    # FastAPI app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import HTMLResponse
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise ModuleNotFoundError(
                "fastapi is not installed. Install 'flyteplugins-github[app]' to use GitHubAppEnvironment."
            ) from exc

        app = FastAPI(
            title=f"{self.name} — GitHub integration",
            description="Setup dashboard and webhook receiver for the Flyte GitHub plugin.",
            version="1.0.0",
        )

        @app.get("/healthz")
        async def healthz() -> dict[str, str]:
            return {"status": "healthy"}

        # NOTE: routes taking `request` are registered via add_api_route with
        # concrete annotations; this module uses string annotations, which
        # FastAPI cannot resolve for closures defined inside a method.

        async def dashboard(request):  # type: ignore[no-untyped-def]
            return self._dashboard_html(str(request.base_url).rstrip("/"))

        dashboard.__annotations__ = {"request": Request}
        app.add_api_route("/", dashboard, methods=["GET"], response_class=HTMLResponse)

        @app.get("/api/status")
        async def status() -> dict[str, Any]:
            return self._status_payload()

        @app.post("/api/verify")
        async def verify() -> dict[str, Any]:
            return await self._verify_credentials()

        @app.get("/api/events")
        async def events() -> list[dict[str, Any]]:
            return [event.model_dump(mode="json", exclude={"payload"}) for event in reversed(self.recent_events)]

        async def webhook(request):  # type: ignore[no-untyped-def]
            return await self._handle_webhook(request)

        webhook.__annotations__ = {"request": Request}
        app.add_api_route(self.webhook_path, webhook, methods=["POST"])

        return app

    # ------------------------------------------------------------------
    # status and verification
    # ------------------------------------------------------------------

    def _status_payload(self) -> dict[str, Any]:
        return {
            "app": self.name,
            "token_env": self.token_env,
            "token_mounted": bool(os.environ.get(self.token_env)),
            "webhook_secret_env": self.webhook_secret_env,
            "webhook_secret_mounted": bool(os.environ.get(self.webhook_secret_env)),
            "require_signature": self.require_signature,
            "repos_allowlist": list(self.repos),
            "handlers": [pattern or "*" for pattern, _ in self.event_handlers],
            "recent_event_count": len(self.recent_events),
        }

    async def _verify_credentials(self) -> dict[str, Any]:
        import httpx

        token = os.environ.get(self.token_env)
        if not token:
            return {"ok": False, "error": f"{self.token_env} is not mounted on this app"}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    f"{self.api_base_url}/user",
                    headers={"Authorization": f"Bearer {token}", "User-Agent": "flyteplugins-github"},
                )
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach GitHub: {exc}"}
        if response.status_code != 200:
            return {"ok": False, "status_code": response.status_code, "error": response.text[:300]}
        body = response.json()
        return {
            "ok": True,
            "login": body.get("login"),
            "scopes": response.headers.get("x-oauth-scopes", ""),
        }

    # ------------------------------------------------------------------
    # webhook handling
    # ------------------------------------------------------------------

    async def _handle_webhook(self, request: Any) -> Any:
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        body = await request.body()
        headers = request.headers
        event_type = headers.get("x-github-event")

        if event_type == "ping":
            return JSONResponse({"ok": True, "ping": True})

        secret = os.environ.get(self.webhook_secret_env)
        if self.require_signature:
            if not secret:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"webhook secret {self.webhook_secret_env} is not mounted; refusing events. "
                        "Create the secret and add it to this app's secrets, or set require_signature=False "
                        "for local development."
                    ),
                )
            if not verify_webhook_signature(body, headers.get("x-hub-signature-256"), secret):
                raise HTTPException(status_code=401, detail="invalid webhook signature")

        try:
            event = parse_webhook(dict(headers), body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not parse webhook: {exc}") from exc

        self.recent_events.append(event)

        if self.repos and event.repository not in self.repos:
            return JSONResponse({"ok": True, "skipped": f"repository {event.repository} not in allowlist"})

        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for pattern, handler in self.event_handlers:
            if not self._matches(pattern, event):
                continue
            handler_name = getattr(handler, "__name__", repr(handler))
            try:
                results[handler_name] = await handler(event)
            except Exception as exc:
                logger.exception("event handler %s failed for %s", handler_name, event.qualified_type)
                errors[handler_name] = str(exc)

        return JSONResponse(
            {
                "ok": not errors,
                "event": event.qualified_type,
                "delivery_id": event.delivery_id,
                "handlers_run": list(results),
                "results": results,
                "errors": errors,
            }
        )

    # ------------------------------------------------------------------
    # dashboard HTML
    # ------------------------------------------------------------------

    def _dashboard_html(self, base_url: str) -> str:
        status = self._status_payload()
        webhook_url = f"{base_url}{self.webhook_path}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        token_badge = _badge(status["token_mounted"], f"{self.token_env} mounted", f"{self.token_env} missing")
        secret_badge = _badge(
            status["webhook_secret_mounted"],
            f"{self.webhook_secret_env} mounted",
            f"{self.webhook_secret_env} missing",
        )

        repos = (
            ", ".join(html.escape(v) for v in self.repos) if self.repos else "<em>all repositories (no allowlist)</em>"
        )
        handlers = (
            ", ".join(f"<code>{html.escape(p or '*')}</code>" for p, _ in self.event_handlers)
            or "<em>none registered</em>"
        )

        rows = []
        for event in reversed(list(self.recent_events)[-25:]):
            rows.append(
                "<tr>"
                f"<td>{html.escape(event.received_at.strftime('%m-%d %H:%M:%S'))}</td>"
                f"<td><code>{html.escape(event.qualified_type)}</code></td>"
                f"<td>{html.escape(event.repository or '')}</td>"
                f"<td>{html.escape(str(event.number or ''))}</td>"
                f"<td>{html.escape(event.sender or '')}</td>"
                "</tr>"
            )
        events_table = (
            "<table><thead><tr><th>Received</th><th>Event</th><th>Repo</th><th>#</th><th>Sender</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=5>No events received yet.</td></tr>'}</tbody></table>"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(self.name)} — GitHub integration</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; background: #f6f8fa; color: #1f2328; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin-top: 32px; }}
  .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
           padding: 16px 20px; margin: 12px 0; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 13px; margin-right: 8px; }}
  .ok {{ background: #dafbe1; color: #1a7f37; }}
  .warn {{ background: #fff8c5; color: #9a6700; }}
  code, pre {{ background: #eff2f5; border-radius: 4px; }}
  code {{ padding: 1px 5px; }}
  pre {{ padding: 12px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #f6f8fa; }}
  button {{ background: #1f883d; color: #fff; border: 0; border-radius: 6px;
            padding: 8px 14px; font-size: 14px; cursor: pointer; }}
  ol li {{ margin: 10px 0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(self.name)} — GitHub integration</h1>
  <p>Setup dashboard for the <a href="https://github.com/flyteorg/flyte-sdk">flyte-sdk</a>
     GitHub plugin. Generated {now}.</p>

  <div class="card">
    <h2 style="margin-top:0">Status</h2>
    <p>{token_badge} {secret_badge}</p>
    <p>Repositories: {repos}<br>Event handlers: {handlers}<br>
       Recent events: {len(self.recent_events)}</p>
    <button onclick="verify()">Verify GitHub credentials</button>
    <pre id="verify-result"></pre>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Setup instructions</h2>
    <ol>
      <li><strong>Create a GitHub token.</strong> A fine-grained personal access
          token (Settings → Developer settings → Fine-grained tokens) scoped to the
          target repositories with <em>Pull requests: read &amp; write</em>,
          <em>Issues: read &amp; write</em>, <em>Contents: read &amp; write</em>,
          and <em>Checks: read &amp; write</em>; or a classic token with scopes
          <code>repo</code> and <code>read:org</code>.</li>
      <li><strong>Store it as a Flyte secret</strong> and request it on the tasks
          and apps that need it:
        <pre>flyte create secret {html.escape(self.token_env)} --value &lt;token&gt;</pre>
        <pre>env = flyte.TaskEnvironment(
    name="my-workflows",
    secrets=[flyte.Secret("{html.escape(self.token_env)}", as_env_var="{html.escape(self.token_env)}")],
)</pre></li>
      <li><strong>Store the webhook secret</strong> (any random string you choose):
        <pre>flyte create secret {html.escape(self.webhook_secret_env)} --value &lt;secret&gt;</pre></li>
      <li><strong>Configure the repository webhook.</strong> In the repository
          settings (Settings → Webhooks → Add webhook):
        <ul>
          <li>Payload URL: <code>{html.escape(webhook_url)}</code></li>
          <li>Content type: <code>application/json</code></li>
          <li>Secret: the same value as the <code>{html.escape(self.webhook_secret_env)}</code> secret</li>
          <li>Events: “Let me select individual events” → Pull requests, Issues,
              Issue comments (add Pushes if you react to commits)</li>
        </ul></li>
      <li><strong>React to events.</strong> Register handlers with
          <code>env.on_event(...)</code> and launch idempotent runs with
          <code>flyteplugins.github.launch_task</code> (see the plugin README for a
          full example).</li>
      <li><strong>Expose tools to agents (optional).</strong> Deploy the MCP server
          with <code>flyteplugins.github.github_mcp_app_env()</code> so agents running
          on Flyte can read and write GitHub through the Model Context Protocol.</li>
    </ol>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Recent events</h2>
    {events_table}
  </div>
</div>
<script>
async function verify() {{
  const el = document.getElementById('verify-result');
  el.textContent = 'Checking credentials...';
  try {{
    const r = await fetch('/api/verify', {{ method: 'POST' }});
    el.textContent = JSON.stringify(await r.json(), null, 2);
  }} catch (err) {{
    el.textContent = String(err);
  }}
}}
</script>
</body>
</html>"""


def _badge(ok: bool, ok_text: str, warn_text: str) -> str:
    if ok:
        return f'<span class="badge ok">✓ {html.escape(ok_text)}</span>'
    return f'<span class="badge warn">! {html.escape(warn_text)}</span>'
