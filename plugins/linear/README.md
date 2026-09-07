# flyteplugins-linear

Receive Linear webhooks in Flyte.

```bash
pip install "flyteplugins-linear[app]"
```

## Using it

Hand a `LinearProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, run_once
from flyteplugins.linear import LinearProvider, events

# LinearProvider.default_secret_env is mounted for you.
app_env = WebhookAppEnvironment(name="linear-webhooks", providers=[LinearProvider()])


@app_env.on_event(events.Issue.CREATE)
async def handle(event):
    import flyte.remote as remote

    task = remote.Task.get(name="my-env.my_task", auto_version="latest")
    result = await run_once.aio(task, key=event.dedupe_key(), resource=event.resource_id)
    if not result.created:
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name}


flyte.serve(app_env)
```

Handlers must `await run_once.aio(...)`. The blocking form stalls the
app's event loop, and Linear times deliveries out in seconds.

One app can serve several products at once — hand it one provider per product.

## Try it

`examples/linear_webhooks.py` runs two ways. The first needs no Linear account:

```bash
python examples/linear_webhooks.py --local   # replay a real sample delivery in-process
python examples/linear_webhooks.py           # deploy the receiver to Flyte
```

`--local` posts this plugin's `SAMPLE_DELIVERY` through the app with FastAPI's
test client, so you see a delivery verified, normalized, and dispatched — plus
an unsigned one refused with a 401, and the same delivery replayed to show the
dedupe key is stable.

## Setup

1. Store the secret and mount it on the app:
   ```bash
   flyte create secret LINEAR_WEBHOOK_SECRET --value <secret>
   ```
2. Point Linear at `<app-url>/webhook/linear`, from
   Linear Settings → API → Webhooks (it shows the signing secret on creation).

**Verification:** HMAC-SHA256 over the raw body (`X-Linear-Signature`).

Comment and reaction payloads carry the team id only on the nested issue; the parser follows it, so a `scopes` allowlist can still attribute them.

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
Raw strings still work, for events the constants do not cover yet.

## What this plugin does not do

Call the Linear API. Use `gql` — Linear ships no Python SDK, and its API is a single GraphQL endpoint directly from your tasks — see
`examples/external_saas_integrations`. This plugin owns only the part that is
Flyte's: authenticating an inbound delivery and turning it into a run.
