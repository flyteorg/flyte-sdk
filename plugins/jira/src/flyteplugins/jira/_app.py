"""The Jira integration app environment.

`JiraAppEnvironment` is a `FastAPIAppEnvironment` that serves two purposes:

1. A **setup and management dashboard** (`/`) explaining how to configure the
   integration end to end: creating a Jira API token, storing credentials as
   Flyte secrets, and wiring a Jira webhook to this app. `/api/status` and
   `/api/verify` expose machine-readable health information.
2. A **webhook receiver** (`/webhook` by default) that verifies a shared
   `X-Webhook-Token` header, normalizes payloads into `JiraEvent` objects, and
   dispatches them to registered handlers. Jira Cloud webhooks are not
   cryptographically signed, so this token (plus network-level protection) is
   the receiver's defense.

Event handlers are registered with `on_event`, and idempotent run launching is
available via `flyteplugins.jira.launch_task`, so the standard pattern is:

```python
import flyte
from flyteplugins.jira import JiraAppEnvironment, launch_task

env = JiraAppEnvironment(name="jira-integration")

@env.on_event("jira:issue_created")
async def triage_new_issue(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_issue", auto_version="latest")
    run = launch_task(task, key=event.dedupe_key(), issue_key=event.issue_key)
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

from ._config import (
    DEFAULT_API_PATH,
    DEFAULT_API_TOKEN_ENV_VAR,
    DEFAULT_BASE_URL_ENV_VAR,
    DEFAULT_EMAIL_ENV_VAR,
    DEFAULT_WEBHOOK_TOKEN_ENV_VAR,
)
from ._webhook import TOKEN_HEADER, JiraEvent, parse_webhook, verify_webhook_token

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

#: Handler signature: receives a `JiraEvent`, returns optional result JSON.
EventHandler = Callable[[JiraEvent], Awaitable[dict[str, Any] | None]]


@dataclass(kw_only=True)
class JiraAppEnvironment(FastAPIAppEnvironment):
    """Dashboard + webhook receiver app for the Jira integration.

    Args:
        name: App environment name (also the app name on the platform).
        project_keys: Optional allowlist of Jira project keys. Events whose
            issue belongs to another project are acknowledged but not
            dispatched. Events without a project key are always dispatched.
        webhook_path: URL path of the webhook receiver.
        base_url_env: Environment variable holding the Jira site URL (mounted
            from a Flyte secret or env var).
        email_env: Environment variable holding the account email.
        api_token_env: Environment variable holding the Jira API token.
        webhook_token_env: Environment variable holding the shared webhook
            token expected in the `X-Webhook-Token` header.
        require_webhook_token: Reject events without a matching webhook token.
            When True and no token is mounted, all events are rejected with an
            explanatory error — set False for local development only, and
            protect the endpoint at the network level.
        max_recent_events: Size of the in-memory recent-events buffer shown on
            the dashboard.
        event_handlers: Optional initial list of `(pattern, handler)` tuples;
            prefer the `on_event` decorator.
    """

    app: FastAPI | None = None
    project_keys: list[str] = field(default_factory=list)
    webhook_path: str = "/webhook"
    base_url_env: str = DEFAULT_BASE_URL_ENV_VAR
    email_env: str = DEFAULT_EMAIL_ENV_VAR
    api_token_env: str = DEFAULT_API_TOKEN_ENV_VAR
    webhook_token_env: str = DEFAULT_WEBHOOK_TOKEN_ENV_VAR
    require_webhook_token: bool = True
    max_recent_events: int = 200
    event_handlers: list[tuple[str, EventHandler]] = field(default_factory=list)

    recent_events: deque[JiraEvent] = field(init=False, repr=False)

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
            event_type: Jira webhook event name (`jira:issue_created`,
                `jira:issue_updated`, `jira:issue_deleted`,
                `comment_created`, ...). An empty string matches every event.

        Returns:
            A decorator that registers the handler and returns it unchanged.
        """

        def decorator(fn: EventHandler) -> EventHandler:
            self.event_handlers.append((event_type, fn))
            return fn

        return decorator

    def _matches(self, pattern: str, event: JiraEvent) -> bool:
        if not pattern:
            return True
        return pattern == event.webhook_event

    # ------------------------------------------------------------------
    # FastAPI app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import HTMLResponse
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise ModuleNotFoundError(
                "fastapi is not installed. Install 'flyteplugins-jira[app]' to use JiraAppEnvironment."
            ) from exc

        app = FastAPI(
            title=f"{self.name} — Jira integration",
            description="Setup dashboard and webhook receiver for the Flyte Jira plugin.",
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
            "base_url_env": self.base_url_env,
            "base_url_mounted": bool(os.environ.get(self.base_url_env)),
            "email_env": self.email_env,
            "email_mounted": bool(os.environ.get(self.email_env)),
            "api_token_env": self.api_token_env,
            "api_token_mounted": bool(os.environ.get(self.api_token_env)),
            "webhook_token_env": self.webhook_token_env,
            "webhook_token_mounted": bool(os.environ.get(self.webhook_token_env)),
            "require_webhook_token": self.require_webhook_token,
            "project_keys_allowlist": list(self.project_keys),
            "handlers": [pattern or "*" for pattern, _ in self.event_handlers],
            "recent_event_count": len(self.recent_events),
        }

    async def _verify_credentials(self) -> dict[str, Any]:
        import base64

        import httpx

        base_url = os.environ.get(self.base_url_env)
        email = os.environ.get(self.email_env)
        api_token = os.environ.get(self.api_token_env)
        missing = [
            env
            for env, value in ((self.base_url_env, base_url), (self.email_env, email), (self.api_token_env, api_token))
            if not value
        ]
        if missing:
            return {"ok": False, "error": f"missing credentials: {', '.join(missing)}"}
        credentials = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}{DEFAULT_API_PATH}/myself",
                    headers={"Authorization": f"Basic {credentials}", "Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach Jira: {exc}"}
        if response.status_code != 200:
            return {"ok": False, "status_code": response.status_code, "error": response.text[:300]}
        body = response.json()
        return {"ok": True, "display_name": body.get("displayName"), "email": body.get("emailAddress")}

    # ------------------------------------------------------------------
    # webhook handling
    # ------------------------------------------------------------------

    async def _handle_webhook(self, request: Any) -> Any:
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        body = await request.body()
        headers = request.headers

        secret = os.environ.get(self.webhook_token_env)
        if self.require_webhook_token:
            if not secret:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"webhook token {self.webhook_token_env} is not mounted; refusing events. "
                        "Create the secret and add it to this app's secrets, or set require_webhook_token=False "
                        "for local development (and protect the endpoint at the network level)."
                    ),
                )
            if not verify_webhook_token(headers.get(TOKEN_HEADER.lower()), secret):
                raise HTTPException(status_code=401, detail="invalid or missing X-Webhook-Token header")

        try:
            event = parse_webhook(dict(headers), body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not parse webhook: {exc}") from exc

        self.recent_events.append(event)

        if self.project_keys and event.project_key is not None and event.project_key not in self.project_keys:
            return JSONResponse({"ok": True, "skipped": f"project {event.project_key} not in allowlist"})

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
                "issue_key": event.issue_key,
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

        base_badge = _badge(status["base_url_mounted"], f"{self.base_url_env} mounted", f"{self.base_url_env} missing")
        email_badge = _badge(status["email_mounted"], f"{self.email_env} mounted", f"{self.email_env} missing")
        token_badge = _badge(
            status["api_token_mounted"], f"{self.api_token_env} mounted", f"{self.api_token_env} missing"
        )
        webhook_badge = _badge(
            status["webhook_token_mounted"],
            f"{self.webhook_token_env} mounted",
            f"{self.webhook_token_env} missing",
        )

        projects = ", ".join(self.project_keys) if self.project_keys else "<em>all projects (no allowlist)</em>"
        handlers = (
            ", ".join(f"<code>{html.escape(p or '*')}</code>" for p, _ in self.event_handlers)
            or "<em>none registered</em>"
        )

        rows = []
        for event in reversed(list(self.recent_events)[:25]):
            rows.append(
                "<tr>"
                f"<td>{html.escape(event.received_at.strftime('%m-%d %H:%M:%S'))}</td>"
                f"<td><code>{html.escape(event.qualified_type)}</code></td>"
                f"<td>{html.escape(event.issue_key or '')}</td>"
                f"<td>{html.escape(event.summary or '')}</td>"
                f"<td>{html.escape(event.status or '')}</td>"
                "</tr>"
            )
        events_table = (
            "<table><thead><tr><th>Received</th><th>Event</th><th>Issue</th><th>Summary</th><th>Status</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=5>No events received yet.</td></tr>'}</tbody></table>"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(self.name)} — Jira integration</title>
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
  <h1>{html.escape(self.name)} — Jira integration</h1>
  <p>Setup dashboard for the <a href="https://github.com/flyteorg/flyte-sdk">flyte-sdk</a>
     Jira plugin. Generated {now}.</p>

  <div class="card">
    <h2 style="margin-top:0">Status</h2>
    <p>{base_badge} {email_badge}<br>{token_badge} {webhook_badge}</p>
    <p>Projects: {projects}<br>Event handlers: {handlers}<br>
       Recent events: {len(self.recent_events)}</p>
    <button onclick="verify()">Verify Jira credentials</button>
    <pre id="verify-result"></pre>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Setup instructions</h2>
    <ol>
      <li><strong>Create a Jira API token.</strong> Go to
          <a href="https://id.atlassian.net/manage/profile/api-tokens">id.atlassian.net
          → API tokens</a> → <em>Create API token</em>. It pairs with your account
          email as HTTP Basic auth.</li>
      <li><strong>Store credentials as Flyte secrets</strong> and request them on
          the tasks and apps that need them:
        <pre>flyte create secret {html.escape(self.base_url_env)} --value https://&lt;site&gt;.atlassian.net
flyte create secret {html.escape(self.email_env)} --value you@example.com
flyte create secret {html.escape(self.api_token_env)} --value &lt;api-token&gt;</pre>
        <pre>env = flyte.TaskEnvironment(
    name="my-workflows",
    secrets=[
        flyte.Secret("{html.escape(self.base_url_env)}", as_env_var="{html.escape(self.base_url_env)}"),
        flyte.Secret("{html.escape(self.email_env)}", as_env_var="{html.escape(self.email_env)}"),
        flyte.Secret("{html.escape(self.api_token_env)}", as_env_var="{html.escape(self.api_token_env)}"),
    ],
)</pre></li>
      <li><strong>Create a Jira webhook.</strong> In Jira: Settings (gear) →
          Products → Webhooks (site admins) → <em>Create webhook</em>:
        <ul>
          <li>URL: <code>{html.escape(webhook_url)}</code></li>
          <li>Events: Issue created, Issue updated, Issue deleted, and Comment
              events as needed</li>
        </ul>
        Jira webhooks are <strong>not signed</strong>, so this receiver protects
        itself with a shared token. Choose a random string and store it:
        <pre>flyte create secret {html.escape(self.webhook_token_env)} --value &lt;random-string&gt;</pre>
        then deliver webhooks through a gateway or proxy that adds the
        <code>X-Webhook-Token</code> header (Jira itself cannot attach custom
        headers). If that is not possible, set
        <code>require_webhook_token=False</code> and protect the endpoint at the
        network level instead.</li>
      <li><strong>React to events.</strong> Register handlers with
          <code>env.on_event(...)</code> (names like <code>jira:issue_created</code>,
          <code>jira:issue_updated</code>, <code>comment_created</code>) and launch
          idempotent runs with <code>flyteplugins.jira.launch_task</code> (see the
          plugin README).</li>
      <li><strong>Expose tools to agents (optional).</strong> Deploy the MCP server
          with <code>flyteplugins.jira.jira_mcp_app_env()</code> so agents running
          on Flyte can read and write Jira through the Model Context Protocol.</li>
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
