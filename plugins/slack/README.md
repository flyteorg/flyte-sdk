# flyteplugins-slack

Receive Slack webhooks in Flyte.

```bash
pip install "flyteplugins-slack[app]"
```

## Using it

Hand a `SlackProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import DuplicateRun, WebhookAppEnvironment, idempotent_run
from flyteplugins.slack import SlackProvider, events

app_env = WebhookAppEnvironment(
    name="slack-webhooks",
    providers=[SlackProvider()],
    secrets=[flyte.Secret("SLACK_SIGNING_SECRET", as_env_var="SLACK_SIGNING_SECRET")],
)


@app_env.on_event(events.AppMention.ANY)
async def handle(event):
    import flyte.remote as remote

    task = remote.Task.get(name="my-env.my_task", auto_version="latest")
    try:
        run = await idempotent_run.aio(task, key=event.dedupe_key(), resource=event.resource_id)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


flyte.serve(app_env)
```

Handlers must `await idempotent_run.aio(...)`. The blocking form stalls the
app's event loop, and Slack times deliveries out in seconds.

One app can serve several products at once — hand it one provider per product.

## Try it

`examples/slack_webhooks.py` runs two ways. The first needs no Slack account:

```bash
python examples/slack_webhooks.py --local   # replay a real sample delivery in-process
python examples/slack_webhooks.py           # deploy the receiver to Flyte
```

`--local` posts this plugin's `SAMPLE_DELIVERY` through the app with FastAPI's
test client, so you see a delivery verified, normalized, and dispatched — plus
an unsigned one refused with a 401, and the same delivery replayed to show the
dedupe key is stable.

## Setup

1. Store the secret and mount it on the app:
   ```bash
   flyte create secret SLACK_SIGNING_SECRET --value <secret>
   ```
2. Point Slack at `<app-url>/webhook/slack`, from
   api.slack.com/apps → Event Subscriptions, then subscribe to bot events.

Slack POSTs a `url_verification` challenge before events flow; it is echoed automatically, so the Request URL field verifies itself.

**Verification:** HMAC-SHA256 over `v0:{timestamp}:{body}`, with a five-minute replay window (`X-Slack-Signature`).

Messages are keyed per message, so each one launches its own run. To collapse a whole thread onto one run, pass `event.payload["event"]["thread_ts"]` as your own key.

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
Raw strings still work, for events the constants do not cover yet.

## What this plugin does not do

Call the Slack API. Use `slack_sdk` directly from your tasks — see
`examples/external_saas_integrations`. This plugin owns only the part that is
Flyte's: authenticating an inbound delivery and turning it into a run.
