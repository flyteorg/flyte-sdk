"""Linear integration for Flyte.

Read and write Linear issues from Flyte tasks, react to Linear webhook events
through an app environment, and expose the operations as an MCP server for
agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-linear[app,mcp]"
```

## Read/write from tasks

```python
import flyte
from flyteplugins.linear import LinearClient

env = flyte.TaskEnvironment(
    name="linear-demo",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-linear"),
    secrets=[flyte.Secret("LINEAR_API_KEY", as_env_var="LINEAR_API_KEY")],
)

@env.task
async def open_bug(team_id: str, title: str, description: str) -> str:
    async with LinearClient() as client:
        issue = await client.create_issue(team_id, title, description=description)
    return issue["url"]
```

## React to Linear events

```python
import flyte
from flyteplugins.linear import LinearAppEnvironment, launch_task

app_env = LinearAppEnvironment(
    name="linear-integration",
    secrets=[
        flyte.Secret("LINEAR_API_KEY", as_env_var="LINEAR_API_KEY"),
        flyte.Secret("LINEAR_WEBHOOK_SECRET", as_env_var="LINEAR_WEBHOOK_SECRET"),
    ],
)

@app_env.on_event("Issue.create")
async def triage_new_issue(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_issue", auto_version="latest")
    run = await launch_task.aio(task, key=event.dedupe_key(), issue_id=event.entity_id)
    return {"run": run.name}

flyte.serve(app_env)
```

Handlers must `await launch_task.aio(...)`: the synchronous form blocks the
app's event loop, and webhook senders time deliveries out in seconds.

The app's dashboard (`/`) walks through API key creation, Flyte secret
creation, and Linear webhook configuration.

## MCP server for agents

```python
import flyte
from flyteplugins.linear import linear_mcp_app_env

mcp_env = linear_mcp_app_env("linear-mcp")  # read-only by default
flyte.serve(mcp_env)
```
"""

from ._app import LinearAppEnvironment
from ._client import LinearClient
from ._config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY_ENV_VAR,
    DEFAULT_WEBHOOK_SECRET_ENV_VAR,
    Config,
    default_config,
)
from ._dispatch import DUPE_LABEL_KEY, DuplicateRun, blocking_run, launch_task
from ._errors import LinearAPIError, LinearPluginError, MissingCredentialsError, WebhookSignatureError
from ._mcp import build_mcp_server, linear_mcp_app_env
from ._tools import TOOL_GROUPS, TOOL_REGISTRY, ToolInfo, build_tool_functions
from ._webhook import LinearEvent, parse_webhook, verify_webhook_signature

__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_API_KEY_ENV_VAR",
    "DEFAULT_WEBHOOK_SECRET_ENV_VAR",
    "DUPE_LABEL_KEY",
    "TOOL_GROUPS",
    "TOOL_REGISTRY",
    "Config",
    "DuplicateRun",
    "LinearAPIError",
    "LinearAppEnvironment",
    "LinearClient",
    "LinearEvent",
    "LinearPluginError",
    "MissingCredentialsError",
    "ToolInfo",
    "WebhookSignatureError",
    "blocking_run",
    "build_mcp_server",
    "build_tool_functions",
    "default_config",
    "launch_task",
    "linear_mcp_app_env",
    "parse_webhook",
    "verify_webhook_signature",
]
