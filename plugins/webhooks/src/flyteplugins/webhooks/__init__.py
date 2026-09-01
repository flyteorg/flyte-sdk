"""Receive SaaS webhooks in Flyte, and launch runs from them.

One app environment accepts webhooks from GitHub, Slack, Linear, ClickUp, and
Jira, verifies each with that provider's own scheme, normalizes the payload into
a `WebhookEvent`, and dispatches it to handlers you register.

## Installation

```bash
pip install "flyteplugins-webhooks[app]"
```

## Receiving events

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
async def triage_new_pr(event):
    import flyte.remote as remote

    task = remote.Task.get(name="github-triage.triage_pr", auto_version="latest")
    try:
        run = await idempotent_run.aio(task, key=event.dedupe_key(), repo=event.scope)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


flyte.serve(app_env)
```

Handlers must `await idempotent_run.aio(...)`. The blocking form stalls the
app's event loop, and webhook senders time deliveries out in seconds.

## What this plugin does not do

Calling the products' APIs. Use their own maintained SDKs — `PyGithub`,
`slack_sdk`, `atlassian-python-api`, and so on — directly from your tasks. See
`examples/external_saas_integrations` for worked recipes. This plugin owns only
the part that is genuinely Flyte's: authenticating an inbound delivery and
turning it into a run.
"""

from . import events
from ._app import EventHandler, WebhookAppEnvironment
from ._errors import SignatureError, WebhookPluginError
from ._event import WebhookEvent
from ._providers import PROVIDERS, Provider

__all__ = [
    "PROVIDERS",
    "EventHandler",
    "Provider",
    "SignatureError",
    "WebhookAppEnvironment",
    "WebhookEvent",
    "WebhookPluginError",
    "events",
]
