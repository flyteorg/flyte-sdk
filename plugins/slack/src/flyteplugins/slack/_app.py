"""The Slack integration app environment.

`SlackAppEnvironment` is a `FastAPIAppEnvironment` that serves two purposes:

1. A **setup and management dashboard** (`/`) explaining how to configure the
   integration end to end: creating a Slack app, granting bot token scopes,
   storing the bot token and signing secret as Flyte secrets, and pointing the
   Slack Events API at this app. `/api/status` and `/api/verify` expose
   machine-readable health information.
2. An **Events API receiver** (`/events` by default) that answers Slack's
   `url_verification` challenge, verifies the `X-Slack-Signature` v0 HMAC,
   normalizes payloads into `SlackEvent` objects, and dispatches them to
   registered handlers.

Event handlers are registered with `on_event`, and idempotent run launching is
available via `flyteplugins.slack.launch_task`, so the standard pattern is:

```python
import flyte
from flyteplugins.slack import SlackAppEnvironment, launch_task

env = SlackAppEnvironment(name="slack-integration")

@env.on_event("app_mention")
async def react_to_mention(event):
    import flyte.remote as remote

    task = remote.Task.get(name="answer_mention", auto_version="latest")
    run = launch_task(task, key=event.dedupe_key(), channel=event.channel, thread_ts=event.root_ts)
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

from ._config import DEFAULT_API_BASE_URL, DEFAULT_BOT_TOKEN_ENV_VAR, DEFAULT_SIGNING_SECRET_ENV_VAR
from ._webhook import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    SlackEvent,
    parse_event,
    parse_url_verification,
    verify_event_signature,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

#: Handler signature: receives a `SlackEvent`, returns optional result JSON.
EventHandler = Callable[[SlackEvent], Awaitable[dict[str, Any] | None]]


@dataclass(kw_only=True)
class SlackAppEnvironment(FastAPIAppEnvironment):
    """Dashboard + Events API receiver app for the Slack integration.

    Args:
        name: App environment name (also the app name on the platform).
        channels: Optional allowlist of channel ids. Events from other
            channels are acknowledged but not dispatched, as are events
            carrying no channel at all — an allowlist cannot vouch for an
            event it cannot attribute.
        events_path: URL path of the Events API receiver.
        bot_token_env: Environment variable holding the Slack bot token
            (mounted from a Flyte secret).
        signing_secret_env: Environment variable holding the signing secret.
        require_signature: Reject events without a valid v0 signature. When
            True and no signing secret is mounted, all events are rejected
            with an explanatory error — set False for local development only.
        api_base_url: Slack Web API base URL used by `/api/verify`.
        max_recent_events: Size of the in-memory recent-events buffer shown on
            the dashboard.
        event_handlers: Optional initial list of `(pattern, handler)` tuples;
            prefer the `on_event` decorator.
    """

    app: FastAPI | None = None
    channels: list[str] = field(default_factory=list)
    events_path: str = "/events"
    bot_token_env: str = DEFAULT_BOT_TOKEN_ENV_VAR
    signing_secret_env: str = DEFAULT_SIGNING_SECRET_ENV_VAR
    require_signature: bool = True
    api_base_url: str = DEFAULT_API_BASE_URL
    max_recent_events: int = 200
    event_handlers: list[tuple[str, EventHandler]] = field(default_factory=list)

    recent_events: deque[SlackEvent] = field(init=False, repr=False)

    def __post_init__(self):
        self.recent_events = deque(maxlen=self.max_recent_events)
        if self.app is None:
            self.app = self._build_app()
        super().__post_init__()
        import flyte.app

        self.links = [
            flyte.app.Link(path="/", title="Setup Dashboard", is_relative=True),
            flyte.app.Link(path=self.events_path, title="Events Receiver", is_relative=True),
            *self.links,
        ]

    # ------------------------------------------------------------------
    # handler registration
    # ------------------------------------------------------------------

    def on_event(self, event_type: str = "") -> Callable[[EventHandler], EventHandler]:
        """Register an async handler for Slack events.

        Args:
            event_type: Slack event type (`message`, `app_mention`,
                `reaction_added`, ...) or qualified type (`message.channel_message`).
                An empty string matches every event.

        Returns:
            A decorator that registers the handler and returns it unchanged.
        """

        def decorator(fn: EventHandler) -> EventHandler:
            self.event_handlers.append((event_type, fn))
            return fn

        return decorator

    def _matches(self, pattern: str, event: SlackEvent) -> bool:
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
                "fastapi is not installed. Install 'flyteplugins-slack[app]' to use SlackAppEnvironment."
            ) from exc

        app = FastAPI(
            title=f"{self.name} — Slack integration",
            description="Setup dashboard and Events API receiver for the Flyte Slack plugin.",
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

        async def receiver(request):  # type: ignore[no-untyped-def]
            return await self._handle_event(request)

        receiver.__annotations__ = {"request": Request}
        app.add_api_route(self.events_path, receiver, methods=["POST"])

        return app

    # ------------------------------------------------------------------
    # status and verification
    # ------------------------------------------------------------------

    def _status_payload(self) -> dict[str, Any]:
        return {
            "app": self.name,
            "bot_token_env": self.bot_token_env,
            "bot_token_mounted": bool(os.environ.get(self.bot_token_env)),
            "signing_secret_env": self.signing_secret_env,
            "signing_secret_mounted": bool(os.environ.get(self.signing_secret_env)),
            "require_signature": self.require_signature,
            "channels_allowlist": list(self.channels),
            "handlers": [pattern or "*" for pattern, _ in self.event_handlers],
            "recent_event_count": len(self.recent_events),
        }

    async def _verify_credentials(self) -> dict[str, Any]:
        import httpx

        token = os.environ.get(self.bot_token_env)
        if not token:
            return {"ok": False, "error": f"{self.bot_token_env} is not mounted on this app"}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self.api_base_url}/auth.test",
                    headers={"Authorization": f"Bearer {token}", "User-Agent": "flyteplugins-slack"},
                )
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach Slack: {exc}"}
        data = response.json()
        if not data.get("ok", False):
            return {"ok": False, "error": data.get("error", "unknown_error")}
        return {"ok": True, "user": data.get("user"), "team": data.get("team"), "bot_id": data.get("bot_id")}

    # ------------------------------------------------------------------
    # event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, request: Any) -> Any:
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        body = await request.body()
        headers = request.headers

        # Slack's one-off URL verification handshake, sent before events flow.
        challenge = parse_url_verification(body)
        if challenge is not None:
            return JSONResponse({"challenge": challenge})

        secret = os.environ.get(self.signing_secret_env)
        if self.require_signature:
            if not secret:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"signing secret {self.signing_secret_env} is not mounted; refusing events. "
                        "Create the secret and add it to this app's secrets, or set require_signature=False "
                        "for local development."
                    ),
                )
            if not verify_event_signature(body, headers.get(TIMESTAMP_HEADER), headers.get(SIGNATURE_HEADER), secret):
                raise HTTPException(status_code=401, detail="invalid event signature")

        try:
            event = parse_event(dict(headers), body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not parse event: {exc}") from exc

        self.recent_events.append(event)

        if self.channels and event.channel not in self.channels:
            return JSONResponse({"ok": True, "skipped": f"channel {event.channel} not in allowlist"})

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
                "event_id": event.event_id,
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
        events_url = f"{base_url}{self.events_path}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        token_badge = _badge(
            status["bot_token_mounted"], f"{self.bot_token_env} mounted", f"{self.bot_token_env} missing"
        )
        secret_badge = _badge(
            status["signing_secret_mounted"],
            f"{self.signing_secret_env} mounted",
            f"{self.signing_secret_env} missing",
        )

        channels = (
            ", ".join(html.escape(v) for v in self.channels)
            if self.channels
            else "<em>all channels (no allowlist)</em>"
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
                f"<td>{html.escape(event.channel or '')}</td>"
                f"<td>{html.escape(event.user or '')}</td>"
                f"<td>{html.escape((event.text or '')[:60])}</td>"
                "</tr>"
            )
        events_table = (
            "<table><thead><tr><th>Received</th><th>Event</th><th>Channel</th><th>User</th><th>Text</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=5>No events received yet.</td></tr>'}</tbody></table>"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(self.name)} — Slack integration</title>
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
  <h1>{html.escape(self.name)} — Slack integration</h1>
  <p>Setup dashboard for the <a href="https://github.com/flyteorg/flyte-sdk">flyte-sdk</a>
     Slack plugin. Generated {now}.</p>

  <div class="card">
    <h2 style="margin-top:0">Status</h2>
    <p>{token_badge} {secret_badge}</p>
    <p>Channels: {channels}<br>Event handlers: {handlers}<br>
       Recent events: {len(self.recent_events)}</p>
    <button onclick="verify()">Verify Slack credentials</button>
    <pre id="verify-result"></pre>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Setup instructions</h2>
    <ol>
      <li><strong>Create a Slack app.</strong> Go to
          <a href="https://api.slack.com/apps">api.slack.com/apps</a> →
          <em>Create New App</em> → <em>From scratch</em>.</li>
      <li><strong>Add bot token scopes</strong> (OAuth &amp; Permissions → Scopes →
          Bot Token Scopes): <code>chat:write</code>, <code>channels:read</code>,
          <code>channels:history</code>, <code>groups:read</code>,
          <code>groups:history</code>, <code>reactions:read</code>,
          <code>reactions:write</code>, <code>users:read</code>.</li>
      <li><strong>Install the app to your workspace</strong> (OAuth &amp; Permissions →
          Install to Workspace) and copy the <em>Bot User OAuth Token</em>
          (<code>xoxb-...</code>).</li>
      <li><strong>Store credentials as Flyte secrets</strong>:
        <pre>flyte create secret {html.escape(self.bot_token_env)} --value xoxb-...
flyte create secret {html.escape(self.signing_secret_env)} --value &lt;signing-secret&gt;</pre>
        The signing secret is under <em>Basic Information → App Credentials</em>.</li>
      <li><strong>Enable the Events API</strong> (Event Subscriptions → Enable
          Events). Set the Request URL to
          <code>{html.escape(events_url)}</code>; Slack sends a
          <code>url_verification</code> challenge that this app answers
          automatically. Subscribe to bot events such as <code>message.channels</code>,
          <code>app_mention</code>, and <code>reaction_added</code>.</li>
      <li><strong>Invite the bot</strong> to the channels it should react in:
          <code>/invite @your-app</code>.</li>
      <li><strong>React to events.</strong> Register handlers with
          <code>env.on_event(...)</code> and launch idempotent runs with
          <code>flyteplugins.slack.launch_task</code> (see the plugin README).</li>
      <li><strong>Expose tools to agents (optional).</strong> Deploy the MCP server
          with <code>flyteplugins.slack.slack_mcp_app_env()</code> so agents running
          on Flyte can read and write Slack through the Model Context Protocol.</li>
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
