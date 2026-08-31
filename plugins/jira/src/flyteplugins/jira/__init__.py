"""Jira Cloud integration for Flyte.

Read and write Jira issues from Flyte tasks, react to Jira webhook events
through an app environment, and expose the operations as an MCP server for
agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-jira[app,mcp]"
```

## Setup

Jira Cloud authenticates with your account email plus an API token created at
[id.atlassian.net](https://id.atlassian.net/manage/profile/api-tokens), and
needs your site URL:

```python
env = flyte.TaskEnvironment(
    name="jira-demo",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-jira"),
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
    ],
)
```

## Read/write from tasks

```python
from flyteplugins.jira import JiraClient

@env.task
async def open_ticket(project_key: str, summary: str, description: str) -> str:
    async with JiraClient() as client:
        issue = await client.create_issue(project_key, summary, description=description)
    return issue["url"]
```

## React to Jira events

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
    run = await launch_task.aio(task, key=event.dedupe_key(), issue_key=event.issue_key)
    return {"run": run.name}

flyte.serve(app_env)
```

Handlers must `await launch_task.aio(...)`: the synchronous form blocks the
app's event loop, and webhook senders time deliveries out in seconds.

Jira webhooks are not signed; the receiver protects itself with a shared
`X-Webhook-Token` header. The app's dashboard (`/`) walks through token
creation, secret creation, webhook configuration, and the token delivery
options.

## MCP server for agents

```python
import flyte
from flyteplugins.jira import jira_mcp_app_env

mcp_env = jira_mcp_app_env("jira-mcp")  # read-only by default
flyte.serve(mcp_env)
```
"""

from ._app import JiraAppEnvironment
from ._client import JiraClient
from ._config import (
    DEFAULT_API_PATH,
    DEFAULT_API_TOKEN_ENV_VAR,
    DEFAULT_BASE_URL_ENV_VAR,
    DEFAULT_EMAIL_ENV_VAR,
    DEFAULT_WEBHOOK_TOKEN_ENV_VAR,
    Config,
    default_config,
)
from ._dispatch import DUPE_LABEL_KEY, DuplicateRun, blocking_run, launch_task
from ._errors import JiraAPIError, JiraPluginError, MissingCredentialsError, WebhookSignatureError
from ._mcp import build_mcp_server, jira_mcp_app_env
from ._tools import TOOL_GROUPS, TOOL_REGISTRY, ToolInfo, build_tool_functions
from ._webhook import JiraEvent, issue_from_payload, parse_webhook, verify_webhook_token

__all__ = [
    "DEFAULT_API_PATH",
    "DEFAULT_API_TOKEN_ENV_VAR",
    "DEFAULT_BASE_URL_ENV_VAR",
    "DEFAULT_EMAIL_ENV_VAR",
    "DEFAULT_WEBHOOK_TOKEN_ENV_VAR",
    "DUPE_LABEL_KEY",
    "TOOL_GROUPS",
    "TOOL_REGISTRY",
    "Config",
    "DuplicateRun",
    "JiraAPIError",
    "JiraAppEnvironment",
    "JiraClient",
    "JiraEvent",
    "JiraPluginError",
    "MissingCredentialsError",
    "ToolInfo",
    "WebhookSignatureError",
    "blocking_run",
    "build_mcp_server",
    "build_tool_functions",
    "default_config",
    "issue_from_payload",
    "jira_mcp_app_env",
    "launch_task",
    "parse_webhook",
    "verify_webhook_token",
]
