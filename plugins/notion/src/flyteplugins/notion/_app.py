"""The Notion integration app environment.

Notion has no webhooks, so `NotionAppEnvironment` reacts to changes by
**polling**:

1. A **setup and management dashboard** (`/`) explains how to configure the
   integration end to end: creating a Notion internal integration, sharing
   pages/databases with it, storing the token as a Flyte secret, and wiring
   change detection either through this app's poll endpoint or a scheduled
   Flyte task. `/api/status` and `/api/verify` expose machine-readable health.
2. A **poll endpoint** (`GET /api/poll?database_id=...&since=...`) queries a
   database for pages edited since a cursor, converts them into `NotionEvent`
   objects, and dispatches them to registered handlers. Point any scheduler
   (cron, a Flyte `Trigger` with an HTTP call, or a manual request) at it.

Event handlers are registered with `on_event`, and idempotent run launching is
available via `flyteplugins.notion.launch_task`, so the standard pattern is:

```python
import flyte
from flyteplugins.notion import NotionAppEnvironment, launch_task

env = NotionAppEnvironment(name="notion-integration", databases=["db-id"])

@env.on_event("page.edited")
async def react_to_edit(event):
    import flyte.remote as remote

    task = remote.Task.get(name="handle_notion_update", auto_version="latest")
    run = await launch_task.aio(task, key=event.dedupe_key(), page_id=event.page_id)
    return {"run": run.name}

flyte.serve(env)
```

Handlers must `await launch_task.aio(...)`: the synchronous form blocks the
app's event loop, and webhook senders time deliveries out in seconds.
"""

from __future__ import annotations

import html
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from flyte.app.extras import FastAPIAppEnvironment

from ._client import NotionClient
from ._config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_NOTION_VERSION,
    DEFAULT_POLL_TOKEN_ENV_VAR,
    DEFAULT_TOKEN_ENV_VAR,
)
from ._events import NotionEvent, events_from_pages

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

#: Handler signature: receives a `NotionEvent`, returns optional result JSON.
EventHandler = Callable[[NotionEvent], Awaitable[dict[str, Any] | None]]


@dataclass(kw_only=True)
class NotionAppEnvironment(FastAPIAppEnvironment):
    """Dashboard + poll endpoint app for the Notion integration.

    Args:
        name: App environment name (also the app name on the platform).
        databases: Databases this app polls. The first entry is the default
            for `/api/poll` when no `database_id` query parameter is passed;
            when set, polling other databases is rejected.
        poll_path: URL path of the poll endpoint.
        token_env: Environment variable holding the Notion integration token
            (mounted from a Flyte secret).
        poll_token_env: Environment variable holding the shared token that
            protects the poll endpoint.
        require_poll_token: Reject poll requests without a matching
            `X-Poll-Token` header. When True and no poll token is mounted, all
            polls are rejected with an explanatory error — set False for local
            development only.
        poll_lookback_minutes: How far back to poll when no `since` cursor is
            given.
        api_base_url: Notion API base URL.
        notion_version: Value of the `Notion-Version` header.
        max_recent_events: Size of the in-memory recent-events buffer shown on
            the dashboard.
        event_handlers: Optional initial list of `(pattern, handler)` tuples;
            prefer the `on_event` decorator.
    """

    app: FastAPI | None = None
    databases: list[str] = field(default_factory=list)
    poll_path: str = "/api/poll"
    token_env: str = DEFAULT_TOKEN_ENV_VAR
    poll_token_env: str = DEFAULT_POLL_TOKEN_ENV_VAR
    require_poll_token: bool = True
    poll_lookback_minutes: int = 15
    api_base_url: str = DEFAULT_API_BASE_URL
    notion_version: str = DEFAULT_NOTION_VERSION
    max_recent_events: int = 200
    event_handlers: list[tuple[str, EventHandler]] = field(default_factory=list)

    recent_events: deque[NotionEvent] = field(init=False, repr=False)

    def __post_init__(self):
        self.recent_events = deque(maxlen=self.max_recent_events)
        if self.app is None:
            self.app = self._build_app()
        super().__post_init__()
        import flyte.app

        self.links = [
            flyte.app.Link(path="/", title="Setup Dashboard", is_relative=True),
            flyte.app.Link(path=self.poll_path, title="Poll Endpoint", is_relative=True),
            *self.links,
        ]

    # ------------------------------------------------------------------
    # handler registration
    # ------------------------------------------------------------------

    def on_event(self, event_type: str = "") -> Callable[[EventHandler], EventHandler]:
        """Register an async handler for change events.

        Args:
            event_type: Event type to match (currently `page.edited`). An
                empty string matches every event.

        Returns:
            A decorator that registers the handler and returns it unchanged.
        """

        def decorator(fn: EventHandler) -> EventHandler:
            self.event_handlers.append((event_type, fn))
            return fn

        return decorator

    def _matches(self, pattern: str, event: NotionEvent) -> bool:
        if not pattern:
            return True
        return pattern == event.event_type

    # ------------------------------------------------------------------
    # FastAPI app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        try:
            from fastapi import FastAPI, Header, Request
            from fastapi.responses import HTMLResponse
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise ModuleNotFoundError(
                "fastapi is not installed. Install 'flyteplugins-notion[app]' to use NotionAppEnvironment."
            ) from exc

        app = FastAPI(
            title=f"{self.name} — Notion integration",
            description="Setup dashboard and change-detection poll endpoint for the Flyte Notion plugin.",
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

        async def poll(
            database_id: str | None = None, since: str | None = None, x_poll_token: str | None = Header(default=None)
        ):
            return await self._handle_poll(database_id, since, x_poll_token)

        poll.__annotations__ = {
            "database_id": str | None,
            "since": str | None,
            "x_poll_token": str | None,
            "return": Any,
        }
        app.add_api_route(self.poll_path, poll, methods=["GET"])

        return app

    # ------------------------------------------------------------------
    # status and verification
    # ------------------------------------------------------------------

    def _status_payload(self) -> dict[str, Any]:
        return {
            "app": self.name,
            "token_env": self.token_env,
            "token_mounted": bool(os.environ.get(self.token_env)),
            "poll_token_env": self.poll_token_env,
            "poll_token_mounted": bool(os.environ.get(self.poll_token_env)),
            "require_poll_token": self.require_poll_token,
            "databases": list(self.databases),
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
                    f"{self.api_base_url}/users/me",
                    headers={"Authorization": f"Bearer {token}", "Notion-Version": self.notion_version},
                )
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach Notion: {exc}"}
        if response.status_code != 200:
            return {"ok": False, "status_code": response.status_code, "error": response.text[:300]}
        body = response.json()
        return {"ok": True, "name": body.get("name"), "type": body.get("type")}

    # ------------------------------------------------------------------
    # poll handling
    # ------------------------------------------------------------------

    async def _handle_poll(self, database_id: str | None, since: str | None, poll_token_header: str | None) -> Any:
        import hmac

        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        if self.require_poll_token:
            poll_token = os.environ.get(self.poll_token_env)
            if not poll_token:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"poll token {self.poll_token_env} is not mounted; refusing polls. "
                        "Create the secret and add it to this app's secrets, or set require_poll_token=False "
                        "for local development."
                    ),
                )
            # Compare as bytes: compare_digest rejects str operands containing non-ASCII, and the
            # header is attacker-controlled, so a str comparison would raise instead of returning False.
            if not poll_token_header or not hmac.compare_digest(
                poll_token_header.encode("utf-8"), poll_token.encode("utf-8")
            ):
                raise HTTPException(status_code=401, detail="invalid or missing X-Poll-Token header")

        if database_id is None:
            if self.databases:
                database_id = self.databases[0]
            else:
                raise HTTPException(status_code=400, detail="pass a database_id query parameter")
        if self.databases and database_id not in self.databases:
            raise HTTPException(status_code=403, detail=f"database {database_id} not configured on this app")

        if since is None:
            since = (datetime.now(timezone.utc) - timedelta(minutes=self.poll_lookback_minutes)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            )

        try:
            from ._config import Config

            config = Config(
                token_env=self.token_env, api_base_url=self.api_base_url, notion_version=self.notion_version
            )
            async with NotionClient(config) as client:
                pages = await client.query_database_since.aio(database_id, since)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Notion query failed: {exc}") from exc

        events = events_from_pages(pages, database_id=database_id)

        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for event in events:
            self.recent_events.append(event)
            for pattern, handler in self.event_handlers:
                if not self._matches(pattern, event):
                    continue
                handler_name = getattr(handler, "__name__", repr(handler))
                try:
                    results[f"{handler_name}:{event.page_id}"] = await handler(event)
                except Exception as exc:
                    logger.exception("event handler %s failed for page %s", handler_name, event.page_id)
                    errors[f"{handler_name}:{event.page_id}"] = str(exc)

        return JSONResponse(
            {
                "ok": not errors,
                "database_id": database_id,
                "since": since,
                "count": len(events),
                "events": [e.model_dump(mode="json", exclude={"payload"}) for e in events],
                "results": results,
                "errors": errors,
            }
        )

    # ------------------------------------------------------------------
    # dashboard HTML
    # ------------------------------------------------------------------

    def _dashboard_html(self, base_url: str) -> str:
        status = self._status_payload()
        poll_url = f"{base_url}{self.poll_path}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        token_badge = _badge(status["token_mounted"], f"{self.token_env} mounted", f"{self.token_env} missing")
        poll_badge = _badge(
            status["poll_token_mounted"],
            f"{self.poll_token_env} mounted",
            f"{self.poll_token_env} missing",
        )

        databases = ", ".join(html.escape(v) for v in self.databases) if self.databases else "<em>none configured</em>"
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
                f"<td>{html.escape(event.title or '')}</td>"
                f"<td>{html.escape(event.last_edited_time or '')}</td>"
                "</tr>"
            )
        events_table = (
            "<table><thead><tr><th>Received</th><th>Event</th><th>Title</th><th>Edited</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=4>No events received yet.</td></tr>'}</tbody></table>"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(self.name)} — Notion integration</title>
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
  <h1>{html.escape(self.name)} — Notion integration</h1>
  <p>Setup dashboard for the <a href="https://github.com/flyteorg/flyte-sdk">flyte-sdk</a>
     Notion plugin. Generated {now}.</p>

  <div class="card">
    <h2 style="margin-top:0">Status</h2>
    <p>{token_badge} {poll_badge}</p>
    <p>Databases: {databases}<br>Event handlers: {handlers}<br>
       Recent events: {len(self.recent_events)}</p>
    <button onclick="verify()">Verify Notion credentials</button>
    <pre id="verify-result"></pre>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Setup instructions</h2>
    <ol>
      <li><strong>Create a Notion internal integration.</strong> Go to
          <a href="https://www.notion.so/profile/integrations">notion.so/profile/integrations</a>
          → <em>New integration</em>, choose the workspace, and copy the
          <em>Internal Integration Secret</em>.</li>
      <li><strong>Store it as a Flyte secret</strong> and request it on the tasks
          and apps that need it:
        <pre>flyte create secret {html.escape(self.token_env)} --value ntn_...</pre>
        <pre>env = flyte.TaskEnvironment(
    name="my-workflows",
    secrets=[flyte.Secret("{html.escape(self.token_env)}", as_env_var="{html.escape(self.token_env)}")],
)</pre></li>
      <li><strong>Share pages/databases with the integration.</strong> In Notion,
          open each page or database → <code>...</code> → <em>Connections</em> →
          add your integration. The API can only see shared content.</li>
      <li><strong>Detect changes by polling</strong> — Notion has no webhooks.
          Either schedule a Flyte task with a `flyte.Trigger` that calls
          `query_database_since`, or point any scheduler at this app's poll
          endpoint:
        <pre>GET {html.escape(poll_url)}?database_id=&lt;db-id&gt;&amp;since=&lt;iso8601&gt;</pre>
          Protect it by choosing a random poll token and storing it:
        <pre>flyte create secret {html.escape(self.poll_token_env)} --value &lt;random-string&gt;</pre>
          then send it as the <code>X-Poll-Token</code> header.</li>
      <li><strong>React to events.</strong> Register handlers with
          <code>env.on_event(...)</code> and launch idempotent runs with
          <code>flyteplugins.notion.launch_task</code> (see the plugin README).</li>
      <li><strong>Expose tools to agents (optional).</strong> Deploy the MCP server
          with <code>flyteplugins.notion.notion_mcp_app_env()</code> so agents
          running on Flyte can read and write Notion through the Model Context
          Protocol.</li>
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
