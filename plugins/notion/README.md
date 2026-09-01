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

## Testing

An end-to-end pass against a real Notion workspace. Notion has no webhooks, so
change detection is polling — that makes this the easiest of the integrations
to test, since you drive the polls yourself.

**1. Create the integration.** At
<https://www.notion.so/profile/integrations> → *New integration*, with
*Read content*, *Update content*, and *Insert content* capabilities. Copy the
Internal Integration Secret:

```bash
flyte create secret NOTION_TOKEN     --value ntn_...
flyte create secret NOTION_POLL_TOKEN --value <random-string>
```

`NOTION_POLL_TOKEN` is a value you invent; it protects the poll endpoint, which
would otherwise be an open trigger for anyone who finds the URL.

**2. Share a database with the integration.** This is the step people miss —
a token alone grants nothing. Open the target database in Notion → *…* →
*Connections* → add your integration. Then copy the database id from its URL:
the 32-character hex segment before the `?`.

**3. Check the client works before involving the platform:**

```bash
export NOTION_TOKEN=ntn_...
python -c "
from flyteplugins.notion import NotionClient
with NotionClient() as c:
    print(c.get_me())
    print([p['title'] for p in c.query_database('<database-id>')])
"
```

An `object_not_found` here means step 2 was skipped — the database is not
shared with the integration.

**4. Run a task directly**, to confirm writes land:

```bash
flyte run plugins/notion/examples/write_to_notion.py add_row \
    --database_id <database-id> --name "Flyte test row" --status "Not started"
```

`Status` must be an existing select option on the database, or Notion rejects
the write.

**5. Deploy the poll app.**

```bash
python plugins/notion/examples/react_to_notion_changes.py
```

It prints the app URL. Open it: the dashboard should show both secrets mounted,
and *Verify Notion credentials* should return the integration's name.

`poll_for_updates.py` is the alternative shape — a scheduled `flyte.Trigger`
that calls `query_database_since` directly, with no app at all. If that is what
you plan to run, deploy it instead and skip to step 8; steps 6 and 7 below
exercise the app's endpoint.

**6. Poll by hand.** Edit a page in the shared database, then:

```bash
curl -H 'X-Poll-Token: <the value you chose>' \
     '<app-url>/api/poll?database_id=<database-id>'
```

The response lists the edited pages as normalized events, plus the run each
handler launched. With no `since` parameter the endpoint looks back
`poll_lookback_minutes` (15 by default), so a page you just edited is in range.

**7. Confirm overlapping polls are safe.** Run the exact same curl again
immediately. The same page is still within the lookback window, so it is
reported again — but the dedupe key folds in the page's `last_edited_time`, so
no second run launches. This is the property that makes a polling schedule with
overlap safe to run, and it is worth seeing once by hand.

Now edit the page again and re-poll: `last_edited_time` has advanced, so this
*does* launch a new run.

**8. Optional — the MCP server.**

```bash
python plugins/notion/examples/notion_mcp_server.py
claude mcp add --transport http notion-mcp <app-url>/mcp/mcp
```

Ask an agent to summarize the database's rows. The default surface is
read-only.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| Poll returns 401 | The `X-Poll-Token` header is missing or does not match `NOTION_POLL_TOKEN`. |
| Poll returns 503 | `NOTION_POLL_TOKEN` is not mounted; check `/api/status`. |
| Poll returns 403 | The app has a `databases` allowlist and the requested id is not on it. |
| Poll returns 502 | The Notion query itself failed — the detail carries Notion's own message. |
| `object_not_found` | The database was never shared with the integration (step 2). |
| A page edit produces no run | Its `last_edited_time` is older than the lookback window; pass `&since=<ISO-8601>` to widen it. |
