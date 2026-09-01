"""One app environment that receives webhooks from every supported provider.

`WebhookAppEnvironment` serves a setup dashboard at `/` and a verified receiver
at `/webhook/{provider}`. Deliveries are authenticated with the provider's own
scheme, normalized into a `WebhookEvent`, and dispatched to handlers registered
with `on_event`.

```python
import flyte
from flyte.extras import DuplicateRun, idempotent_run
from flyteplugins.webhooks import WebhookAppEnvironment, events

app_env = WebhookAppEnvironment(
    name="saas-webhooks",
    providers=["github", "slack"],
    secrets=[
        flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET"),
        flyte.Secret("SLACK_SIGNING_SECRET", as_env_var="SLACK_SIGNING_SECRET"),
    ],
)

@app_env.on_event(events.github.PullRequest.OPENED)
async def triage(event):
    import flyte.remote as remote

    task = remote.Task.get(name="github-triage.triage_pr", auto_version="latest")
    run = await idempotent_run.aio(task, key=event.dedupe_key(), repo=event.scope)
    return {"run": run.name}

flyte.serve(app_env)
```

Handlers must `await idempotent_run.aio(...)`: the blocking form stalls the
app's event loop, and webhook senders time deliveries out in seconds.
"""

from __future__ import annotations

import html
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

from flyte.app.extras import FastAPIAppEnvironment

from ._event import WebhookEvent
from ._providers import PROVIDERS, Provider, slack_url_verification

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

#: Handler signature: receives a `WebhookEvent`, returns optional result JSON.
EventHandler = Callable[[WebhookEvent], Awaitable[dict[str, Any] | None]]


@dataclass(kw_only=True)
class WebhookAppEnvironment(FastAPIAppEnvironment):
    """Dashboard plus a verified webhook receiver for one or more providers.

    Args:
        name: App environment name, and the app name on the platform.
        providers: Which providers to accept, by name (`github`, `slack`,
            `linear`, `clickup`, `jira`). Each gets a route at
            `{webhook_prefix}/{name}`; anything not listed 404s.
        scopes: Optional allowlist of repositories / channels / teams / lists /
            project keys. Events from anywhere else are acknowledged but not
            dispatched, as are events carrying no scope at all — an allowlist
            cannot vouch for an event it cannot attribute.
        webhook_prefix: URL prefix for the receiver routes.
        require_signature: Reject deliveries that fail verification. When True
            and a provider's secret is not mounted, that provider's deliveries
            are refused with an explanatory error. Set False for local
            development only.
        max_recent_events: Size of the in-memory buffer shown on the dashboard.
        event_handlers: Optional initial `(pattern, handler)` list; prefer the
            `on_event` decorator.
    """

    app: FastAPI | None = None
    providers: Sequence[str] = ("github",)
    scopes: list[str] = field(default_factory=list)
    webhook_prefix: str = "/webhook"
    require_signature: bool = True
    max_recent_events: int = 200
    event_handlers: list[tuple[str, EventHandler]] = field(default_factory=list)

    recent_events: deque[WebhookEvent] = field(init=False, repr=False)

    def __post_init__(self):
        unknown = [p for p in self.providers if p not in PROVIDERS]
        if unknown:
            raise ValueError(f"unknown provider(s) {unknown}; supported: {sorted(PROVIDERS)}")
        if not self.providers:
            raise ValueError("configure at least one provider")
        self.recent_events = deque(maxlen=self.max_recent_events)
        if self.app is None:
            self.app = self._build_app()
        super().__post_init__()
        import flyte.app

        self.links = [flyte.app.Link(path="/", title="Setup Dashboard", is_relative=True), *self.links]

    @property
    def _providers(self) -> list[Provider]:
        return [PROVIDERS[name] for name in self.providers]

    # ------------------------------------------------------------------
    # handler registration
    # ------------------------------------------------------------------

    def on_event(self, event_type: str = "") -> Callable[[EventHandler], EventHandler]:
        """Register an async handler for webhook events.

        Args:
            event_type: The event to match. Prefer the typed constants in
                `flyteplugins.webhooks.events` — `events.github.PullRequest.OPENED`
                for one action, `events.github.PullRequest.ANY` for every action
                on that type. Raw strings still work, which is the escape hatch
                for events the constants do not cover yet. An empty string
                matches every event from every configured provider.

        Returns:
            A decorator that registers the handler and returns it unchanged.
        """

        def decorator(fn: EventHandler) -> EventHandler:
            self.event_handlers.append((event_type, fn))
            return fn

        return decorator

    def _matches(self, pattern: str, event: WebhookEvent) -> bool:
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
                "fastapi is not installed. Install 'flyteplugins-webhooks[app]' to use WebhookAppEnvironment."
            ) from exc

        app = FastAPI(
            title=f"{self.name} — SaaS webhooks",
            description="Setup dashboard and verified webhook receiver.",
            version="1.0.0",
        )

        @app.get("/healthz")
        async def healthz() -> dict[str, str]:
            return {"status": "healthy"}

        @app.get("/api/status")
        async def status() -> dict[str, Any]:
            return self._status_payload()

        @app.get("/api/events")
        async def events() -> list[dict[str, Any]]:
            return [e.model_dump(mode="json", exclude={"payload"}) for e in reversed(self.recent_events)]

        # Routes taking `request` are registered via add_api_route with concrete
        # annotations; this module uses string annotations, which FastAPI cannot
        # resolve for closures defined inside a method.
        async def dashboard(request):  # type: ignore[no-untyped-def]
            return self._dashboard_html(str(request.base_url).rstrip("/"))

        dashboard.__annotations__ = {"request": Request}
        app.add_api_route("/", dashboard, methods=["GET"], response_class=HTMLResponse)

        async def receive(request):  # type: ignore[no-untyped-def]
            return await self._handle(request.path_params["provider"], request)

        receive.__annotations__ = {"request": Request}
        app.add_api_route(f"{self.webhook_prefix}/{{provider}}", receive, methods=["POST"])

        return app

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def _status_payload(self) -> dict[str, Any]:
        return {
            "app": self.name,
            "require_signature": self.require_signature,
            "scopes": list(self.scopes),
            "handlers": [pattern or "*" for pattern, _ in self.event_handlers],
            "recent_event_count": len(self.recent_events),
            "providers": [
                {
                    "name": p.name,
                    "path": f"{self.webhook_prefix}/{p.name}",
                    "secret_env": p.secret_env,
                    "secret_mounted": bool(os.environ.get(p.secret_env)),
                    "signed": p.signed,
                }
                for p in self._providers
            ],
        }

    # ------------------------------------------------------------------
    # receiving
    # ------------------------------------------------------------------

    async def _handle(self, provider_name: str, request: Any) -> Any:
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        provider = PROVIDERS.get(provider_name)
        if provider is None or provider_name not in self.providers:
            raise HTTPException(status_code=404, detail=f"provider {provider_name!r} is not configured on this app")

        body = await request.body()
        headers = dict(request.headers)

        # GitHub pings on webhook creation; Slack handshakes before events flow.
        if provider is PROVIDERS["github"] and headers.get("x-github-event") == "ping":
            return JSONResponse({"ok": True, "ping": True})
        if provider is PROVIDERS["slack"]:
            challenge = slack_url_verification(body)
            if challenge is not None:
                return JSONResponse({"challenge": challenge})

        if self.require_signature:
            secret = os.environ.get(provider.secret_env)
            if not secret:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"{provider.secret_env} is not mounted; refusing {provider.name} deliveries. "
                        "Create the secret and add it to this app's secrets, or set require_signature=False "
                        "for local development."
                    ),
                )
            if not provider.verify(body, headers, secret):
                raise HTTPException(status_code=401, detail=f"invalid {provider.name} signature")

        try:
            event = provider.parse(headers, body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not parse {provider.name} webhook: {exc}") from exc

        self.recent_events.append(event)

        if self.scopes and event.scope not in self.scopes:
            return JSONResponse({"ok": True, "skipped": f"scope {event.scope} not in allowlist"})

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
                "provider": event.provider,
                "event": event.qualified_type,
                "resource_id": event.resource_id,
                "handlers_run": list(results),
                "results": results,
                "errors": errors,
            }
        )

    # ------------------------------------------------------------------
    # dashboard
    # ------------------------------------------------------------------

    def _dashboard_html(self, base_url: str) -> str:
        status = self._status_payload()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        esc = html.escape

        provider_row_parts = []
        for entry in status["providers"]:
            secret_env = entry["secret_env"]
            badge = _badge(entry["secret_mounted"], f"{secret_env} mounted", f"{secret_env} missing")
            verification = "HMAC" if entry["signed"] else "shared token (provider does not sign)"
            provider_row_parts.append(
                "<tr>"
                f"<td><code>{esc(entry['name'])}</code></td>"
                f"<td><code>{esc(base_url)}{esc(entry['path'])}</code></td>"
                f"<td>{badge}</td>"
                f"<td>{verification}</td>"
                "</tr>"
            )
        provider_rows = "".join(provider_row_parts)

        scopes = ", ".join(esc(s) for s in self.scopes) if self.scopes else "<em>all scopes (no allowlist)</em>"
        handlers = (
            ", ".join(f"<code>{esc(p or '*')}</code>" for p, _ in self.event_handlers) or "<em>none registered</em>"
        )

        # The buffer appends on the right, so read from the end for the newest.
        rows = "".join(
            "<tr>"
            f"<td>{esc(e.received_at.strftime('%m-%d %H:%M:%S'))}</td>"
            f"<td><code>{esc(e.provider)}</code></td>"
            f"<td><code>{esc(e.qualified_type)}</code></td>"
            f"<td>{esc(e.resource_id or '')}</td>"
            f"<td>{esc((e.title or '')[:60])}</td>"
            "</tr>"
            for e in reversed(list(self.recent_events)[-25:])
        )
        events_table = (
            "<table><thead><tr><th>Received</th><th>Provider</th><th>Event</th>"
            "<th>Resource</th><th>Title</th></tr></thead>"
            f"<tbody>{rows or '<tr><td colspan=5>No events received yet.</td></tr>'}</tbody></table>"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{esc(self.name)} — SaaS webhooks</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; background: #f6f8fa; color: #1f2328; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin-top: 32px; }}
  .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
           padding: 16px 20px; margin: 12px 0; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 13px; }}
  .ok {{ background: #dafbe1; color: #1a7f37; }}
  .warn {{ background: #fff8c5; color: #9a6700; }}
  code {{ background: #eff2f5; border-radius: 4px; padding: 1px 5px; }}
  pre {{ background: #eff2f5; border-radius: 4px; padding: 12px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #f6f8fa; }}
  ol li {{ margin: 10px 0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{esc(self.name)} — SaaS webhooks</h1>
  <p>One receiver for every configured provider. Generated {now}.</p>

  <div class="card">
    <h2 style="margin-top:0">Providers</h2>
    <table><thead><tr><th>Provider</th><th>Payload URL</th><th>Secret</th><th>Verification</th></tr></thead>
    <tbody>{provider_rows}</tbody></table>
    <p style="margin-bottom:0">Scopes: {scopes}<br>Handlers: {handlers}<br>
       Recent events: {len(self.recent_events)}</p>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Setup</h2>
    <ol>
      <li><strong>Store each provider's secret</strong> and mount it on this app:
        <pre>flyte create secret &lt;SECRET_NAME&gt; --value &lt;secret&gt;</pre></li>
      <li><strong>Point each provider at its URL</strong> from the table above.
          GitHub sends a <code>ping</code> and Slack a <code>url_verification</code>
          challenge on setup; both are answered automatically, so a green check
          there means the app is reachable.</li>
      <li><strong>Register handlers</strong> with <code>on_event</code>, using the
          typed constants in <code>flyteplugins.webhooks.events</code>, and launch
          runs with <code>flyte.extras.idempotent_run</code> so redeliveries never
          launch twice.</li>
    </ol>
    <p style="margin-bottom:0">Machine-readable status is at
       <a href="/api/status"><code>/api/status</code></a>; recent events at
       <a href="/api/events"><code>/api/events</code></a>.</p>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Recent events</h2>
    {events_table}
  </div>
</div>
</body>
</html>"""


def _badge(ok: bool, ok_text: str, warn_text: str) -> str:
    css, text = ("ok", f"✓ {ok_text}") if ok else ("warn", f"! {warn_text}")
    return f'<span class="badge {css}">{html.escape(text)}</span>'
