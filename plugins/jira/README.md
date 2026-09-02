# flyteplugins-jira

Receive Jira webhooks in Flyte.

```bash
pip install "flyteplugins-jira[app]"
```

## Using it

Hand a `JiraProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import DuplicateRun, WebhookAppEnvironment, run_once
from flyteplugins.jira import JiraProvider, events

# JiraProvider.default_secret_env is mounted for you.
app_env = WebhookAppEnvironment(name="jira-webhooks", providers=[JiraProvider()])


@app_env.on_event(events.Issue.CREATED)
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
app's event loop, and Jira times deliveries out in seconds.

One app can serve several products at once — hand it one provider per product.

## Try it

`examples/jira_webhooks.py` runs two ways. The first needs no Jira account:

```bash
python examples/jira_webhooks.py --local   # replay a real sample delivery in-process
python examples/jira_webhooks.py           # deploy the receiver to Flyte
```

`--local` posts this plugin's `SAMPLE_DELIVERY` through the app with FastAPI's
test client, so you see a delivery verified, normalized, and dispatched — plus
an unsigned one refused with a 401, and the same delivery replayed to show the
dedupe key is stable.

## Setup

1. Store the secret and mount it on the app:
   ```bash
   flyte create secret JIRA_WEBHOOK_TOKEN --value <secret>
   ```
2. Point Jira at `<app-url>/webhook/jira`, from
   Jira Settings → System → Webhooks.

**Verification:** **None.** Jira Cloud does not sign its webhooks.

Because there is no signature, this plugin authenticates with a shared token in `X-Webhook-Token` — which something in front of the app has to inject, since Jira cannot send custom headers. `JiraProvider` reports `signed=False`, so the dashboard says the product does not sign rather than implying a guarantee that is absent. A shared token also cannot detect body tampering, only that the sender knew the token.

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
Raw strings still work, for events the constants do not cover yet.

## What this plugin does not do

Call the Jira API. Use the `jira` package directly from your tasks — see
`examples/external_saas_integrations`. This plugin owns only the part that is
Flyte's: authenticating an inbound delivery and turning it into a run.
