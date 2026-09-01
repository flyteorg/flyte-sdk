# Flyte Slack Plugin

Read and write Slack from Flyte tasks, react to Slack Events API events with
an app environment, and expose everything as an MCP server for agents running
on Flyte.

## Installation

```bash
pip install "flyteplugins-slack"            # client only
pip install "flyteplugins-slack[app]"       # + FastAPI app environment
pip install "flyteplugins-slack[mcp]"       # + MCP server
```

## Setup

The plugin reads credentials from environment variables, which on Flyte are
populated by mounting secrets:

```bash
flyte create secret SLACK_BOT_TOKEN --value xoxb-...
flyte create secret SLACK_SIGNING_SECRET --value <signing-secret>   # only for events
```

Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps) (From
scratch), add the bot token scopes you need (`chat:write`, `channels:read`,
`channels:history`, `groups:history`, `reactions:read`, `reactions:write`,
`users:read` cover this plugin), install it to your workspace, and copy the
Bot User OAuth Token. The signing secret lives under Basic Information → App
Credentials.

Request the secrets on any task or app environment that needs them:

```python
env = flyte.TaskEnvironment(
    name="slack-demo",
    secrets=[flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN")],
)
```

## Read/write from tasks

```python
import flyte
from flyteplugins.slack import SlackClient, events

@env.task
async def notify(channel: str, message: str) -> str:
    async with SlackClient() as client:
        result = await client.post_message.aio(channel, message)
    return result.get("permalink", "")
```

The client covers messages (post/update/thread replies/permalinks), channel
listing/info/history, threads, users, reactions, and channel creation — see
`flyteplugins.slack.SlackClient`. Slack's `ok: false` responses are raised as
`SlackAPIError`; 429 rate limits are retried automatically.

### Both call forms

Every client method is available two ways. `await client.post_message.aio(...)` is the
async form — use it in `async def` tasks and anywhere on an app's event loop.
`client.post_message(...)` is the blocking form, for plain `def` tasks and scripts:

```python
@env.task
def summarize(...) -> str:
    with SlackClient() as client:          # note: `with`, not `async with`
        result = client.post_message(channel, message)
    ...
```

The blocking form parks the calling thread until the call returns, so never
reach for it inside an `async def` task or a webhook handler — it would stall
the event loop and everything else waiting on it.

## React to Slack events

`SlackAppEnvironment` serves a **setup dashboard** (`/`) and an **Events API
receiver** (`/events`). The dashboard walks through Slack app creation, bot
token scopes, secret creation, and Events API configuration; `/api/status` and
`/api/verify` expose machine-readable health. The receiver answers Slack's
`url_verification` challenge automatically.

```python
import flyte
from flyteplugins.slack import SlackAppEnvironment, events, launch_task

app_env = SlackAppEnvironment(
    name="slack-integration",
    secrets=[
        flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN"),
        flyte.Secret("SLACK_SIGNING_SECRET", as_env_var="SLACK_SIGNING_SECRET"),
    ],
)

@app_env.on_event(events.AppMention.ANY)
async def answer_mention(event):
    import flyte.remote as remote

    task = remote.Task.get(name="answer_mention", auto_version="latest")
    run = await launch_task.aio(task, key=event.dedupe_key(), channel=event.channel, thread_ts=event.root_ts)
    return {"run": run.name}

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(app_env)
```

Events are verified against `SLACK_SIGNING_SECRET` (`X-Slack-Signature` v0
HMAC with a five-minute replay window), normalized into `SlackEvent` objects,
matched against the optional `channels` allowlist, and dispatched to handlers
registered with `on_event` (event types like `message`, `app_mention`,
`reaction_added`; an empty pattern matches everything).

`launch_task` launches runs **idempotently**: every run carries a `dedupe`
label derived from the event, and a second delivery of the same event raises
`DuplicateRun` instead of launching a second run. Identity lives entirely on
that label — run names are left to the control plane. Failed or aborted runs
never block, so re-triggering after a failure is a retry.

`dedupe_key()` defaults to `scope="thread"`: every message in a thread maps to
one key, so a thread launches one run and later replies raise `DuplicateRun`.
That fits "open one ticket per thread". For "answer every question asked", use
`event.dedupe_key("message")` — each message then gets its own key while
redeliveries of one message still dedupe. The key is just a string, so pass
your own if neither scope fits.

Always `await launch_task.aio(...)` inside a handler. The synchronous
`launch_task(...)` form is for scripts: it blocks the calling thread, which on
the app's event loop stalls every other in-flight request.


Set the Events API Request URL to the app's public URL + `/events`; Slack's
verification challenge is answered automatically. Then invite the bot to the
channels it should react in.

## MCP server for agents

The read/write surface doubles as MCP tools, so agents running on Flyte can
use Slack through the Model Context Protocol:

```python
import flyte
from flyteplugins.slack import events, slack_mcp_app_env

mcp_env = slack_mcp_app_env(
    "slack-mcp",
    secrets=[flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN")],
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(mcp_env)
```

The server is **read-only by default** (channels, history, threads, users).
Pass `read_only=False` to include posting, reactions, and channel creation.
Tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`) are set
from the tool registry. Reacting to events is intentionally *not* an MCP tool
— that is the app environment's job.

Connect an agent running on Flyte:

```python
from flyte.ai.agents import Agent, MCPServerSpec

agent = Agent(
    name="slack-agent",
    mcp_servers=[MCPServerSpec(name="slack", url="https://<app>/mcp/mcp")],
)
```

## Configuration

`flyteplugins.slack.Config` controls token/signing-secret env var names, the
API base URL, timeouts, and retries. The module exports `default_config`; pass
a custom `Config` to `SlackClient`, `build_mcp_server`, or the app environment
when you need it.

## Testing

An end-to-end pass against a real Slack workspace. Use a scratch channel —
step 4 posts real messages.

**1. Create the Slack app.** At <https://api.slack.com/apps> → *Create New
App* → *From scratch*. Then:

- *OAuth & Permissions* → Bot Token Scopes: `chat:write`, `channels:read`,
  `channels:history`, `reactions:write`, `app_mentions:read`
- *Install to Workspace*, and copy the Bot User OAuth Token (`xoxb-…`)
- *Basic Information* → copy the Signing Secret

```bash
flyte create secret SLACK_BOT_TOKEN --value xoxb-...
flyte create secret SLACK_SIGNING_SECRET --value <signing-secret>
```

Invite the bot to your test channel: `/invite @your-app`.

**2. Check the client works before involving the platform:**

```bash
export SLACK_BOT_TOKEN=xoxb-...
python -c "
from flyteplugins.slack import SlackClient
with SlackClient() as c:
    print(c.post_message('<channel-id>', 'hello from the flyte slack plugin'))
"
```

A `not_in_channel` error here means the bot was never invited; `missing_scope`
names the scope to add and reinstall for.

**3. Deploy the task the receiver will launch.**

```bash
flyte deploy plugins/slack/examples/notify_channel.py env
```

`react_to_slack_events.py` looks this task up by name (`answer_mention`), so it
has to exist before the app can launch it.

**4. Deploy the events app.**

```bash
python plugins/slack/examples/react_to_slack_events.py
```

It prints the app URL. Open it: the dashboard should show both secrets mounted,
and *Verify Slack credentials* should return the bot's identity.

**5. Point Slack at the app.** In *Event Subscriptions*:

- Request URL: `<app-url>/events`

Slack immediately POSTs a `url_verification` challenge; the receiver echoes it
back automatically, so the field should go green on its own. If it does not,
the app is not reachable — check the URL before going further.

- Subscribe to bot events: `app_mention`, `reaction_added`
- Save, then reinstall the app if Slack prompts you to

**6. Trigger a real event.** In your test channel, `@`-mention the bot. Then
check, in order:

- `<app-url>/api/events` — the normalized event.
- `flyte get runs` — a run whose `dedupe` label matches.
- The thread — the launched task replies in it.

**7. Confirm the dedupe scope.** Mention the bot *again in the same thread*.
By default `dedupe_key()` uses `scope="thread"`, so this is treated as a
duplicate and no second run launches — one run per thread. That is right for
"open one ticket per thread" and wrong for "answer every question", so if you
want the latter, switch the handler to `event.dedupe_key("message")` and repeat:
each mention should now get its own run.

**8. Optional — the MCP server.**

```bash
python plugins/slack/examples/slack_mcp_server.py
claude mcp add --transport http slack-mcp <app-url>/mcp/mcp
```

Ask an agent to summarize recent messages in the channel. The default surface
is read-only.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| Request URL never verifies | The app is unreachable, or `/events` was omitted from the URL. |
| Events return 401 | The Signing Secret does not match `SLACK_SIGNING_SECRET`. |
| Events return 401 only sometimes | Clock skew — signatures older than five minutes are rejected as replays. |
| Events return 503 | `SLACK_SIGNING_SECRET` is not mounted; check `/api/status`. |
| 200 but no run | No handler matched, or the thread's key already has a run — see step 7. |
| `not_in_channel` from the task | The bot was never invited to the channel. |
