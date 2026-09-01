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
from flyteplugins.clickup import ClickUpClient, events

@env.task
async def open_ticket(list_id: str, name: str) -> str:
    async with ClickUpClient() as client:
        task = await client.create_task.aio(list_id, name)
    return task["url"]
```

The client covers workspaces, spaces, folders, lists, list statuses, tasks,
and comments — see `flyteplugins.clickup.ClickUpClient`. Errors are raised as
`ClickUpAPIError`; 429 rate limits are retried.

### Both call forms

Every client method is available two ways. `await client.get_task.aio(...)` is the
async form — use it in `async def` tasks and anywhere on an app's event loop.
`client.get_task(...)` is the blocking form, for plain `def` tasks and scripts:

```python
@env.task
def summarize(...) -> str:
    with ClickUpClient() as client:          # note: `with`, not `async with`
        task = client.get_task(task_id)
    ...
```

The blocking form parks the calling thread until the call returns, so never
reach for it inside an `async def` task or a webhook handler — it would stall
the event loop and everything else waiting on it.

### Status pre-check before updates

ClickUp rejects transitions to statuses a list does not define, and the
failure surfaces as an opaque 400. Validate first:

```python
@env.task
async def close_ticket(task_id: str) -> str:
    async with ClickUpClient() as client:
        task = await client.get_task.aio(task_id)
        valid = await client.list_statuses.aio(task["list_id"])
        if "done" not in valid:
            raise ValueError(f"'done' is not valid here; choose from {valid}")
        await client.update_task.aio(task_id, status="done")
    return task_id
```

## React to ClickUp events

`ClickUpAppEnvironment` serves a **setup dashboard** (`/`) and a **webhook
receiver** (`/webhook`). The dashboard walks through token creation, secret
creation, and ClickUp webhook configuration; `/api/status` and `/api/verify`
expose machine-readable health.

```python
import flyte
from flyteplugins.clickup import ClickUpAppEnvironment, events, launch_task

app_env = ClickUpAppEnvironment(
    name="clickup-integration",
    secrets=[
        flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN"),
        flyte.Secret("CLICKUP_WEBHOOK_SECRET", as_env_var="CLICKUP_WEBHOOK_SECRET"),
    ],
)

@app_env.on_event(events.Task.CREATED)
async def triage_new_task(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_task", auto_version="latest")
    run = await launch_task.aio(task, key=event.dedupe_key(), task_id=event.task_id)
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
re-triggering after a failure is a retry. Identity lives entirely on that label — run names are left to the
control plane. The key is just a string: pass your own to choose a different
idempotency scope.

Always `await launch_task.aio(...)` inside a handler. The synchronous
`launch_task(...)` form is for scripts: it blocks the calling thread, which on
the app's event loop stalls every other in-flight request.


Create the webhook in ClickUp (space or list → Settings → Webhooks) pointing
at the app's public URL + `/webhook`.

## MCP server for agents

The read/write surface doubles as MCP tools, so agents running on Flyte can
use ClickUp through the Model Context Protocol:

```python
import flyte
from flyteplugins.clickup import clickup_mcp_app_env, events

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

## Testing

An end-to-end pass against a real ClickUp workspace. Use a scratch list —
step 4 creates and comments on real tasks.

**1. Create the credentials.** A personal API token from ClickUp → Settings →
Apps → *Generate*:

```bash
flyte create secret CLICKUP_TOKEN --value pk_...
```

**2. Check the client works before involving the platform**, and find the list
id you will test against:

```bash
export CLICKUP_TOKEN=pk_...
python -c "
from flyteplugins.clickup import ClickUpClient
with ClickUpClient() as c:
    ws = c.list_workspaces()[0]
    print('workspace', ws['id'], ws['name'])
    for s in c.list_spaces(ws['id']):
        for lst in c.list_lists(space_id=s['id']):
            print('  list', lst['id'], lst['name'])
"
```

**3. Deploy the task the webhook will launch.**

```bash
flyte deploy plugins/clickup/examples/manage_ticket.py env
```

`react_to_clickup_events.py` looks this task up by name (`triage_task`), so it
has to exist before the app can launch it.

**4. Run a task directly**, to confirm writes land before any webhook is
involved:

```bash
flyte run plugins/clickup/examples/manage_ticket.py open_ticket \
    --list_id <list-id> --name "Flyte test ticket" --description "created by the plugin test"
```

**5. Deploy the webhook app.**

```bash
python plugins/clickup/examples/react_to_clickup_events.py
```

It prints the app URL. Open it: the dashboard should show the token mounted,
and *Verify ClickUp credentials* should return your user.

**6. Point ClickUp at the app.** Space or workspace Settings → Integrations →
Webhooks → *Create Webhook*:

- Endpoint: `<app-url>/webhook`
- Events: *taskCreated* and *taskStatusUpdated*

ClickUp shows a signing secret on creation. Store it and redeploy so it is
mounted:

```bash
flyte create secret CLICKUP_WEBHOOK_SECRET --value <signing-secret>
```

**7. Trigger a real event.** Create a task in the watched list. Then check, in
order:

- `<app-url>/api/events` — the normalized event, `qualified_type` of
  `taskCreated`.
- `flyte get runs` — a run whose `dedupe` label matches.
- The ticket — the triage task's comment.

**8. Confirm later updates get their own runs.** Change the task's status. The
dedupe key folds in ClickUp's own event timestamp, so this is a new key and
launches a second run, while a redelivery of the *same* event does not.

**9. Optional — the allowlist.** Redeploy with `list_ids=["<list-id>"]` and
create a task in a different list. The receiver should answer 200 with a
`skipped` message. The allowlist fails closed, so an event carrying no list id
is skipped too.

**10. Optional — the MCP server.**

```bash
python plugins/clickup/examples/clickup_mcp_server.py
claude mcp add --transport http clickup-mcp <app-url>/mcp/mcp
```

Ask an agent to summarize the list's open tasks. The default surface is
read-only.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| Webhook delivery returns 401 | `CLICKUP_WEBHOOK_SECRET` does not match the secret ClickUp generated. |
| Delivery returns 503 | `CLICKUP_WEBHOOK_SECRET` is not mounted; check `/api/status`. |
| 200 but no run | No handler matched, or the allowlist skipped it — the response body says which. |
| A status transition fails from a task | ClickUp rejects statuses the list does not define; `close_ticket` calls `list_statuses` first for exactly this reason. |
