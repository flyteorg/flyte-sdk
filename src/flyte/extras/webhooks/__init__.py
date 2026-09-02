"""Receive SaaS webhooks in Flyte, and turn them into runs.

This package holds the product-agnostic machinery, so each
`flyteplugins-<product>` plugin stays thin and consistent:

- `WebhookAppEnvironment` — one app serving a dashboard and a verified receiver
  at `/webhook/{provider}`, for whichever providers you hand it.
- `Provider` — the contract a plugin implements: which env var holds its secret
  (`default_secret_env`, which the app mounts for you), how to verify a
  delivery, how to parse one into an event.
- `WebhookEvent` — the normalized event every provider parses into, so handlers
  and dedupe keys work the same regardless of which product sent it.
- `run_once` — launch a run once per event key. Webhook senders retry on
  any non-2xx and operators re-trigger by hand; this makes that safe.
- `EventType` — base for the typed event constants each plugin ships.
- `flyte.extras.webhooks.testing` — `assert_provider_conforms`, the
  CI-enforced conformance check every plugin runs.

Serving the app needs `fastapi` and `uvicorn`, which flyte keeps as the `app`
extra rather than as runtime dependencies — importing this package never
requires them; only building the app does.

The division of labor: core owns the app, dispatch, dedupe, and the verification
primitives that are easy to get subtly wrong; a plugin owns only what is
specific to its product.

```python
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, run_once
from flyteplugins.github import GitHubProvider
from flyteplugins.github import events

app_env = WebhookAppEnvironment(name="saas-webhooks", providers=[GitHubProvider()])


@app_env.on_event(events.PullRequest.OPENED)
async def triage(event):
    import flyte.remote as remote

    task = remote.Task.get(name="github-triage.triage_pr", auto_version="latest")
    result = await run_once.aio(task, key=event.dedupe_key(), repo=event.scope)
    if not result.created:
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name}
```
"""

from ._app import EventHandler, WebhookAppEnvironment
from ._errors import SignatureError, WebhookPluginError
from ._event import WebhookEvent
from ._event_type import EventType
from ._provider import (
    HandshakeFn,
    ParseFn,
    Provider,
    VerifyFn,
    constant_time_equals,
    hex_hmac_sha256,
    json_body,
    lower_headers,
)
from ._run_once import DUPE_LABEL_KEY, RunOnceResult, blocking_run, run_once

__all__ = [
    "DUPE_LABEL_KEY",
    "EventHandler",
    "EventType",
    "HandshakeFn",
    "ParseFn",
    "Provider",
    "RunOnceResult",
    "SignatureError",
    "VerifyFn",
    "WebhookAppEnvironment",
    "WebhookEvent",
    "WebhookPluginError",
    "blocking_run",
    "constant_time_equals",
    "hex_hmac_sha256",
    "json_body",
    "lower_headers",
    "run_once",
]
