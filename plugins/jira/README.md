# Flyte Jira Plugin

Read and write Jira Cloud issues from Flyte tasks, react to Jira webhook
events with an app environment, and expose everything as an MCP server for
agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-jira"            # client only
pip install "flyteplugins-jira[app]"       # + FastAPI app environment
pip install "flyteplugins-jira[mcp]"       # + MCP server
```

## Setup

Jira Cloud authenticates with your account email plus an API token created at
[id.atlassian.net/manage/profile/api-tokens](https://id.atlassian.net/manage/profile/api-tokens).
Store all three credentials as Flyte secrets:

```bash
flyte create secret JIRA_BASE_URL --value https://<site>.atlassian.net
flyte create secret JIRA_EMAIL --value you@example.com
flyte create secret JIRA_API_TOKEN --value <api-token>
flyte create secret JIRA_WEBHOOK_TOKEN --value <random-string>   # only for webhooks
```

Request the secrets on any task or app environment that needs them:

```python
env = flyte.TaskEnvironment(
    name="jira-demo",
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
    ],
)
```

## Read/write from tasks

```python
import flyte
from flyteplugins.jira import JiraClient

@env.task
async def open_ticket(project_key: str, summary: str) -> str:
    async with JiraClient() as client:
        issue = await client.create_issue(project_key, summary)
    return issue["url"]
```

The client covers projects, issues, JQL search, comments, and workflow
transitions — see `flyteplugins.jira.JiraClient`. Plain-text descriptions and
comments are converted to Jira's Atlassian Document Format automatically, and
issue descriptions are converted back to plain text on read. Errors are raised
as `JiraAPIError`; 429 rate limits are retried.

## React to Jira events

`JiraAppEnvironment` serves a **setup dashboard** (`/`) and a **webhook
receiver** (`/webhook`). The dashboard walks through API token creation,
secret creation, and Jira webhook configuration; `/api/status` and
`/api/verify` expose machine-readable health.

```python
import flyte
from flyteplugins.jira import JiraAppEnvironment, launch_task

app_env = JiraAppEnvironment(
    name="jira-integration",
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
        flyte.Secret("JIRA_WEBHOOK_TOKEN", as_env_var="JIRA_WEBHOOK_TOKEN"),
    ],
)

@app_env.on_event("jira:issue_created")
async def triage_new_issue(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_issue", auto_version="latest")
    run = launch_task(task, key=event.dedupe_key(), issue_key=event.issue_key)
    return {"run": run.name}

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(app_env)
```

**Jira webhooks are not signed.** The receiver protects itself with a shared
token: choose a random string, store it as `JIRA_WEBHOOK_TOKEN`, and deliver
webhooks through a gateway or proxy that adds the `X-Webhook-Token` header
(Jira itself cannot attach custom headers). If that is not possible, set
`require_webhook_token=False` and protect the endpoint at the network level —
the dashboard explains both options.

Events are normalized into `JiraEvent` objects, matched against the optional
`project_keys` allowlist, and dispatched to handlers registered with
`on_event` (names like `jira:issue_created`, `jira:issue_updated`,
`comment_created`; an empty pattern matches everything).

`launch_task` launches runs **idempotently**: every run carries a `dedupe`
label derived from the event, and a second delivery of the same event raises
`DuplicateRun` instead of launching a second run. Failed or aborted runs never
block, so re-triggering after a failure is a retry.

Create the webhook in Jira (gear → Products → Webhooks, site admins) pointing
at the app's public URL + `/webhook`.

## MCP server for agents

The read/write surface doubles as MCP tools, so agents running on Flyte can
use Jira through the Model Context Protocol:

```python
import flyte
from flyteplugins.jira import jira_mcp_app_env

mcp_env = jira_mcp_app_env(
    "jira-mcp",
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
    ],
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(mcp_env)
```

The server is **read-only by default** (projects, issues, search, comments,
transitions). Pass `read_only=False` to include issue creation, updates,
commenting, and transitions, and `include_destructive=True` to additionally
expose `delete_issue`. Tool annotations (`readOnlyHint`, `destructiveHint`,
`idempotentHint`) are set from the tool registry. Reacting to events is
intentionally *not* an MCP tool — that is the app environment's job.

Connect an agent running on Flyte:

```python
from flyte.ai.agents import Agent, MCPServerSpec

agent = Agent(
    name="jira-agent",
    mcp_servers=[MCPServerSpec(name="jira", url="https://<app>/mcp/mcp")],
)
```

## Configuration

`flyteplugins.jira.Config` controls the credential env var names, the REST API
path, timeouts, and retries. The module exports `default_config`; pass a custom
`Config` to `JiraClient`, `build_mcp_server`, or the app environment when you
need it.
