# flyteplugins-slack

Receive Slack webhooks in Flyte: Events API callbacks, interactivity payloads
(Block Kit actions, shortcuts, modals), and slash commands — one route serves
all three.

```bash
pip install "flyteplugins-slack[app]"
```

## Using it

Hand a `SlackProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, run_once
from flyteplugins.slack import SlackProvider, events

# SlackProvider.default_secret_env is mounted for you.
app_env = WebhookAppEnvironment(name="slack-webhooks", providers=[SlackProvider()])


@app_env.on_event(events.AppMention.ANY)
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
app's event loop, and Slack times deliveries out in seconds.

One app can serve several products at once — hand it one provider per product.

## Try it

Two examples, each runnable two ways — `--local` needs no Slack account:

```bash
python examples/slack_webhooks.py --local      # Events API: replay a real sample delivery
python examples/slack_interactions.py --local  # buttons + slash commands, signed and replayed
python examples/slack_webhooks.py              # deploy the receiver to Flyte
```

`--local` posts signed deliveries through the app with FastAPI's test client,
so you see each one verified, normalized, and dispatched — plus an unsigned one
refused with a 401. `slack_webhooks.py` covers the Events API and stable dedupe
keys; `slack_interactions.py` covers a Block Kit button (`block_actions.<action_id>`),
a slash command (`command.<name>`), and the `ssl_check` probe.

## Setup

1. Store the secret and mount it on the app:
   ```bash
   flyte create secret SLACK_SIGNING_SECRET --value <secret>
   ```
2. Point Slack at `<app-url>/webhook/slack` — the same URL in every place your
   app uses, at api.slack.com/apps:
   - **Event Subscriptions** → Request URL, then subscribe to bot events;
   - **Interactivity & Shortcuts** → Request URL, for Block Kit buttons,
     shortcuts, and modals;
   - **Slash Commands** → each command's Request URL.

Slack POSTs a `url_verification` challenge before events flow and an `ssl_check`
probe to interactivity and slash-command URLs; both are answered automatically,
so the Request URL fields verify themselves.

**Verification:** HMAC-SHA256 over `v0:{timestamp}:{body}`, with a five-minute replay window (`X-Slack-Signature`). The same scheme signs all three delivery shapes.

Messages are keyed per message, so each one launches its own run. To collapse a whole thread onto one run, pass `event.payload["event"]["thread_ts"]` as your own key.

## Interactivity and slash commands

An interaction's action is its `action_id` (or `callback_id`), and a slash
command's is its name, so one button or one command registers as a raw string:

```python
@app_env.on_event("block_actions.approve_reply")
async def approve(event):
    # event.payload is Slack's full JSON: actions, container, message, response_url.
    channel, ts = event.payload["container"]["channel_id"], event.payload["container"]["message_ts"]
    ...


@app_env.on_event("command.deploy")  # /deploy
async def deploy(event):
    text = event.payload["text"]
    ...
```

`events.Interaction.BLOCK_ACTIONS` and `events.Command.ANY` match whole
categories. Slack shows the user an error unless the delivery is answered
within 3 seconds, so handlers for these must do nothing slower than
`run_once.aio` — post progress back via `slack_sdk` from the launched task.

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
Raw strings still work, for events the constants do not cover yet.

## What this plugin does not do

Call the Slack API. Use `slack_sdk` directly from your tasks — see
`examples/external_saas_integrations`. This plugin owns only the part that is
Flyte's: authenticating an inbound delivery and turning it into a run.
