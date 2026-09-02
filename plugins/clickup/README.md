# flyteplugins-clickup

Receive ClickUp webhooks in Flyte.

```bash
pip install "flyteplugins-clickup[app]"
```

## Using it

Hand a `ClickUpProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import DuplicateRun, WebhookAppEnvironment, run_once
from flyteplugins.clickup import ClickUpProvider, events

# ClickUpProvider.default_secret_env is mounted for you.
app_env = WebhookAppEnvironment(name="clickup-webhooks", providers=[ClickUpProvider()])


@app_env.on_event(events.Task.STATUS_UPDATED)
async def handle(event):
    import flyte.remote as remote

    task = remote.Task.get(name="my-env.my_task", auto_version="latest")
    try:
        run = await run_once.aio(task, key=event.dedupe_key(), resource=event.resource_id)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


flyte.serve(app_env)
```

Handlers must `await run_once.aio(...)`. The blocking form stalls the
app's event loop, and ClickUp times deliveries out in seconds.

One app can serve several products at once — hand it one provider per product.

## Try it

`examples/clickup_webhooks.py` runs two ways. The first needs no ClickUp account:

```bash
python examples/clickup_webhooks.py --local   # replay a real sample delivery in-process
python examples/clickup_webhooks.py           # deploy the receiver to Flyte
```

`--local` posts this plugin's `SAMPLE_DELIVERY` through the app with FastAPI's
test client, so you see a delivery verified, normalized, and dispatched — plus
an unsigned one refused with a 401, and the same delivery replayed to show the
dedupe key is stable.

## Setup

1. Store the secret and mount it on the app:
   ```bash
   flyte create secret CLICKUP_WEBHOOK_SECRET --value <secret>
   ```
2. Point ClickUp at `<app-url>/webhook/clickup`, from
   Space Settings → Integrations → Webhooks (it shows the signing secret on creation).

**Verification:** HMAC-SHA256 over the raw body (`X-Clickup-Signature`).

The list id is at the top level on list-scoped events and on the nested task for task-scoped ones; the parser reads both.

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
Raw strings still work, for events the constants do not cover yet.

## What this plugin does not do

Call the ClickUp API. Use `httpx` — ClickUp ships no Python SDK, and its API is a handful of REST calls directly from your tasks — see
`examples/external_saas_integrations`. This plugin owns only the part that is
Flyte's: authenticating an inbound delivery and turning it into a run.
