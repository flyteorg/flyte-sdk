# Flyte Notion Plugin

Read and write Notion from Flyte tasks, detect Notion changes by polling
(Notion has no webhooks) through an app environment, and expose everything as
an MCP server for agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-notion"            # client only
pip install "flyteplugins-notion[app]"       # + FastAPI app environment
pip install "flyteplugins-notion[mcp]"       # + MCP server
```

## Setup

The plugin reads credentials from environment variables, which on Flyte are
populated by mounting secrets:

```bash
flyte create secret NOTION_TOKEN --value ntn_...
flyte create secret NOTION_POLL_TOKEN --value <random-string>   # only for the poll endpoint
```

Create an internal integration at
[notion.so/profile/integrations](https://www.notion.so/profile/integrations)
and copy its Internal Integration Secret. **Then share every page or database
the plugin should touch with the integration** (page → `...` → Connections →
add your integration) — the Notion API can only see shared content.

Request the secrets on any task or app environment that needs them:

```python
env = flyte.TaskEnvironment(
    name="notion-demo",
    secrets=[flyte.Secret("NOTION_TOKEN", as_env_var="NOTION_TOKEN")],
)
```

## Read/write from tasks

```python
import flyte
from flyteplugins.notion import NotionClient, events, select_property, title_property

@env.task
async def add_row(database_id: str, name: str, status: str) -> str:
    async with NotionClient() as client:
        page = await client.create_database_page.aio(
            database_id,
            {"Name": title_property(name), "Status": select_property(status)},
        )
    return page["url"]
```

The client covers search, pages, databases, queries, blocks, page creation
(in databases and as child pages), updates, and archiving — see
`flyteplugins.notion.NotionClient`. Property values and blocks are built with
the exported helpers (`title_property`, `select_property`, `paragraph_block`,
...); errors are raised as `NotionAPIError` carrying Notion's error code, and
429 rate limits are retried.

### Both call forms

Every client method is available two ways. `await client.get_page.aio(...)` is the
async form — use it in `async def` tasks and anywhere on an app's event loop.
`client.get_page(...)` is the blocking form, for plain `def` tasks and scripts:

```python
@env.task
def summarize(...) -> str:
    with NotionClient() as client:          # note: `with`, not `async with`
        page = client.get_page(page_id)
    ...
```

The blocking form parks the calling thread until the call returns, so never
reach for it inside an `async def` task or a webhook handler — it would stall
the event loop and everything else waiting on it.

## React to Notion changes (polling)

Notion has no webhooks, so `NotionAppEnvironment` detects changes by polling:
it queries a database for pages edited since a cursor, converts them into
`NotionEvent` objects, and dispatches them to handlers. Trigger the poll from
any scheduler — cron, a Flyte `Trigger`, or manually.

```python
import flyte
from flyteplugins.notion import NotionAppEnvironment, events, launch_task

app_env = NotionAppEnvironment(
    name="notion-integration",
    databases=["<database-id>"],
    secrets=[
        flyte.Secret("NOTION_TOKEN", as_env_var="NOTION_TOKEN"),
        flyte.Secret("NOTION_POLL_TOKEN", as_env_var="NOTION_POLL_TOKEN"),
    ],
)

@app_env.on_event(events.Page.EDITED)
async def react_to_edit(event):
    import flyte.remote as remote

    task = remote.Task.get(name="handle_notion_update", auto_version="latest")
    run = await launch_task.aio(task, key=event.dedupe_key(), page_id=event.page_id)
    return {"run": run.name}

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(app_env)
```

Poll with:

```bash
curl -H "X-Poll-Token: $NOTION_POLL_TOKEN" \
  "https://<app>/api/poll?database_id=<db-id>&since=2024-06-01T00:00:00.000Z"
```

Without `database_id` the first configured database is used; without `since`
the lookback defaults to `poll_lookback_minutes` (15). The dashboard (`/`)
walks through integration creation, page sharing, secret creation, and these
polling options; `/api/status` and `/api/verify` expose machine-readable
health.

Alternatively, skip the app and poll from a scheduled task —
`examples/poll_for_updates.py` shows a `flyte.Trigger` calling
`query_database_since` directly.

`launch_task` launches runs **idempotently**: every run carries a `dedupe`
label derived from the event (page id + edit timestamp), so overlapping polls
never launch duplicate runs; a later edit of the same page produces a new key.
Failed or aborted runs never block, so re-triggering after a failure is a
retry. Identity lives entirely on that label — run names are left to the
control plane. The key is just a string: pass your own to choose a different
idempotency scope.

Always `await launch_task.aio(...)` inside a handler. The synchronous
`launch_task(...)` form is for scripts: it blocks the calling thread, which on
the app's event loop stalls every other in-flight request.


## MCP server for agents

The read/write surface doubles as MCP tools, so agents running on Flyte can
use Notion through the Model Context Protocol:

```python
import flyte
from flyteplugins.notion import events, notion_mcp_app_env

mcp_env = notion_mcp_app_env(
    "notion-mcp",
    secrets=[flyte.Secret("NOTION_TOKEN", as_env_var="NOTION_TOKEN")],
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(mcp_env)
```

The server is **read-only by default** (search, pages, databases, queries,
blocks). Pass `read_only=False` to include page creation/updates and block
appending, and `include_destructive=True` to additionally expose
`archive_page`. Tool annotations (`readOnlyHint`, `destructiveHint`,
`idempotentHint`) are set from the tool registry. Change detection is
intentionally *not* an MCP tool — that is the app environment's (or a
scheduled task's) job.

Connect an agent running on Flyte:

```python
from flyte.ai.agents import Agent, MCPServerSpec

agent = Agent(
    name="notion-agent",
    mcp_servers=[MCPServerSpec(name="notion", url="https://<app>/mcp/mcp")],
)
```

## Configuration

`flyteplugins.notion.Config` controls token env var names, the API base URL,
the `Notion-Version` header, timeouts, and retries. The module exports
`default_config`; pass a custom `Config` to `NotionClient` or
`build_mcp_server` when you need it.
