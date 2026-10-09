# flyteplugins-gitlab

Receive GitLab webhooks in Flyte.

```bash
pip install "flyteplugins-gitlab[app]"
```

## Using it

Hand a `GitLabProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, run_once
from flyteplugins.gitlab import GitLabProvider, events

# GitLabProvider.default_secret_env is mounted for you.
app_env = WebhookAppEnvironment(name="gitlab-webhooks", providers=[GitLabProvider()])


@app_env.on_event(events.MergeRequest.OPEN)
async def handle(event):
    import flyte.remote as remote

    task = remote.Task.get(name="my-env.my_task", auto_version="latest")
    result = await run_once.aio(task, key=event.dedupe_key(), project=event.scope)
    if not result.created:
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name}


flyte.serve(app_env)
```

Handlers must `await run_once.aio(...)`. The blocking form stalls the
app's event loop, and GitLab times deliveries out in seconds.

One app can serve several products at once — hand it one provider per product.

## Try it

`examples/gitlab_webhooks.py` runs two ways. The first needs no GitLab account:

```bash
python examples/gitlab_webhooks.py --local   # replay a real sample delivery in-process
python examples/gitlab_webhooks.py           # deploy the receiver to Flyte
```

`--local` posts this plugin's `SAMPLE_DELIVERY` through the app with FastAPI's
test client, so you see a delivery verified, normalized, and dispatched — plus
one with the wrong token refused, and the same delivery replayed to show the
dedupe key is stable.

## Setup

1. Store the secret and mount it on the app:
   ```bash
   flyte create secret GITLAB_WEBHOOK_TOKEN --value <token>
   ```
2. Point GitLab at `<app-url>/webhook/gitlab`, from
   Project Settings → Webhooks → Add webhook.

**Verification:** none, in the sense of a signature. GitLab's webhook *Secret
token* is a static shared token, not an HMAC; GitLab sends it verbatim in
`X-Gitlab-Token`, which this plugin compares in constant time. The dashboard
reports `signed=False` so you can tell this apart from a product that actually
signs the body.

(GitLab's newer *Signing token* mode does sign the body, following the Standard
Webhooks spec: `webhook-signature` carries an HMAC-SHA256 over
`webhook-id`.`webhook-timestamp`.body. This plugin authenticates the classic
secret-token mode and does not cover that one.)

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
`MergeRequest`, `Issue`, and `Release` split the action (`merge_request.open`);
`Pipeline` and `Deployment` split on GitLab's `status` instead
(`pipeline.success`, `deployment.failed`); `Push`, `TagPush`, and `Note` carry
only `ANY` because GitLab sends them no action. Raw strings still work, for
events the constants do not cover yet.

`payloads` adds typed views (`payloads.merge_request(event)`, `payloads.note(event)`,
`payloads.push(event)`) of the common payload fields for autocomplete inside
handlers.

## What this plugin does not do

Call the GitLab API. Use `python-gitlab` from your tasks — install
`flyteplugins-gitlab[gitlab]` for it (the plugin does not import it) — to open,
comment on, approve, or merge MRs once a webhook has launched a run. This plugin
owns only the part that is Flyte's: authenticating an inbound delivery and
turning it into a run.
