# Flyte ClickUp Plugin

Read and write ClickUp tasks from Flyte tasks, react to ClickUp webhook events
with an app environment, and expose everything as an MCP server for agents
running on Flyte.

## Installation

```bash
pip install "flyteplugins-clickup"            # client only
pip install "flyteplugins-clickup[app]"       # + FastAPI app environment
pip install "flyteplugins-clickup[mcp]"       # + MCP server
```

## Setup

The plugin reads credentials from environment variables, which on Flyte are
populated by mounting secrets:

```bash
flyte create secret CLICKUP_TOKEN --value <token>
flyte create secret CLICKUP_WEBHOOK_SECRET --value <signing-secret>   # only for webhooks
```

Generate a personal API token in ClickUp under avatar → Settings → Apps → API
Token. The webhook signing secret is shown by ClickUp when you create a
webhook.

Request the secrets on any task or app environment that needs them:

```python
env = flyte.TaskEnvironment(
    name="clickup-demo",
    secrets=[flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN")],
)
```

## Read/write from tasks

```python
import flyte
from flyteplugins.clickup import ClickUpClient

@env.task
async def open_ticket(list_id: str, name: str) -> str:
    async with ClickUpClient() as client:
        task = await client.create_task(list_id, name)
    return task["url"]
```

The client covers workspaces, spaces, folders, lists, list statuses, tasks,
and comments — see `flyteplugins.clickup.ClickUpClient`. Errors are raised as
`ClickUpAPIError`; 429 rate limits are retried.

### Status pre-check before updates

ClickUp rejects transitions to statuses a list does not define, and the
failure surfaces as an opaque 400. Validate first:

```python
@env.task
async def close_ticket(task_id: str) -> str:
    async with ClickUpClient() as client:
        task = await client.get_task(task_id)
        valid = await client.list_statuses(task["list_id"])
        if "done" not in valid:
            raise ValueError(f"'done' is not valid here; choose from {valid}")
        await client.update_task(task_id, status="done")
    return task_id
```

## React to ClickUp events

`ClickUpAppEnvironment` serves a **setup dashboard** (`/`) and a **webhook
receiver** (`/webhook`). The dashboard walks through token creation, secret
creation, and ClickUp webhook configuration; `/api/status` and `/api/verify`
expose machine-readable health.

```python
import flyte
from flyteplugins.clickup import ClickUpAppEnvironment, launch_task

app_env = ClickUpAppEnvironment(
    name="clickup-integration",
    secrets=[
        flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN"),
        flyte.Secret("CLICKUP_WEBHOOK_SECRET", as_env_var="CLICKUP_WEBHOOK_SECRET"),
    ],
)

@app_env.on_event("taskCreated")
async def triage_new_task(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_task", auto_version="latest")
    run = launch_task(task, key=event.dedupe_key(), task_id=event.task_id)
    return {"run": run.name}

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(app_env)
```

Webhook payloads are HMAC-verified against `CLICKUP_WEBHOOK_SECRET`
(`x-clickup-signature`), normalized into `ClickUpEvent` objects, matched
against the optional `list_ids` allowlist, and dispatched to handlers
registered with `on_event` (names like `taskCreated`, `taskStatusUpdated`,
`taskCommented`; an empty pattern matches everything).

`launch_task` launches runs **idempotently**: every run carries a `dedupe`
label derived from the event (event name + task id + ClickUp's event
timestamp, so retries dedupe but later updates to the same task produce new
keys), and a second delivery of the same event raises `DuplicateRun` instead
of launching a second run. Failed or aborted runs never block, so
re-triggering after a failure is a retry.

Create the webhook in ClickUp (space or list → Settings → Webhooks) pointing
at the app's public URL + `/webhook`.

## MCP server for agents

The read/write surface doubles as MCP tools, so agents running on Flyte can
use ClickUp through the Model Context Protocol:

```python
import flyte
from flyteplugins.clickup import clickup_mcp_app_env

mcp_env = clickup_mcp_app_env(
    "clickup-mcp",
    secrets=[flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN")],
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(mcp_env)
```

The server is **read-only by default**. Pass `read_only=False` to include task
creation, updates, and commenting, and `include_destructive=True` to
additionally expose `delete_task`. Tool annotations (`readOnlyHint`,
`destructiveHint`, `idempotentHint`) are set from the tool registry. Reacting
to events is intentionally *not* an MCP tool — that is the app environment's
job.

Connect an agent running on Flyte:

```python
from flyte.ai.agents import Agent, MCPServerSpec

agent = Agent(
    name="clickup-agent",
    mcp_servers=[MCPServerSpec(name="clickup", url="https://<app>/mcp/mcp")],
)
```

## Configuration

`flyteplugins.clickup.Config` controls token/webhook-secret env var names, the
API base URL, timeouts, and retries. The module exports `default_config`; pass
a custom `Config` to `ClickUpClient`, `build_mcp_server`, or the app
environment when you need it.
