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
from flyteplugins.slack import SlackClient

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
from flyteplugins.slack import SlackAppEnvironment, launch_task

app_env = SlackAppEnvironment(
    name="slack-integration",
    secrets=[
        flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN"),
        flyte.Secret("SLACK_SIGNING_SECRET", as_env_var="SLACK_SIGNING_SECRET"),
    ],
)

@app_env.on_event("app_mention")
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
from flyteplugins.slack import slack_mcp_app_env

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
