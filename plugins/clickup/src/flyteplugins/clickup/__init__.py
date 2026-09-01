"""ClickUp integration for Flyte.

Read and write ClickUp tasks from Flyte tasks, react to ClickUp webhook events
through an app environment, and expose the operations as an MCP server for
agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-clickup[app,mcp]"
```

## Read/write from tasks

```python
import flyte
from flyteplugins.clickup import ClickUpClient, events

env = flyte.TaskEnvironment(
    name="clickup-demo",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-clickup"),
    secrets=[flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN")],
)

@env.task
async def open_ticket(list_id: str, name: str, description: str) -> str:
    async with ClickUpClient() as client:
        task = await client.create_task.aio(list_id, name, description=description)
    return task["url"]
```

Every client method has two call forms: `await client.get_task.aio(...)` for
async tasks and app handlers, and `client.get_task(...)` (under a plain `with`)
for sync tasks and scripts. The blocking form stalls the calling thread, so never
use it on an event loop.

## Status pre-check before updating

ClickUp rejects transitions to statuses a list does not define, so validate
first:

```python
@env.task
async def close_ticket(task_id: str) -> str:
    async with ClickUpClient() as client:
        task = await client.get_task.aio(task_id)
        valid = await client.list_statuses.aio(task["list_id"])
        if "done" not in valid:
            raise ValueError(f"'done' is not a valid status; choose from {valid}")
        await client.update_task.aio(task_id, status="done")
    return task_id
```

## React to ClickUp events

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

flyte.serve(app_env)
```

Handlers must `await launch_task.aio(...)`: the synchronous form blocks the
app's event loop, and webhook senders time deliveries out in seconds.

The app's dashboard (`/`) walks through token creation, Flyte secret creation,
and ClickUp webhook configuration.

## MCP server for agents

```python
import flyte
from flyteplugins.clickup import clickup_mcp_app_env, events

mcp_env = clickup_mcp_app_env("clickup-mcp")  # read-only by default
flyte.serve(mcp_env)
```
"""

from . import events
from ._app import ClickUpAppEnvironment
from ._client import ClickUpClient
from ._config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_TOKEN_ENV_VAR,
    DEFAULT_WEBHOOK_SECRET_ENV_VAR,
    Config,
    default_config,
)
from ._dispatch import DUPE_LABEL_KEY, DuplicateRun, blocking_run, launch_task
from ._errors import ClickUpAPIError, ClickUpPluginError, MissingCredentialsError, WebhookSignatureError
from ._mcp import build_mcp_server, clickup_mcp_app_env
from ._tools import TOOL_GROUPS, TOOL_REGISTRY, ToolInfo, build_tool_functions
from ._webhook import ClickUpEvent, parse_webhook, verify_webhook_signature

__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_TOKEN_ENV_VAR",
    "DEFAULT_WEBHOOK_SECRET_ENV_VAR",
    "DUPE_LABEL_KEY",
    "TOOL_GROUPS",
    "TOOL_REGISTRY",
    "ClickUpAPIError",
    "ClickUpAppEnvironment",
    "ClickUpClient",
    "ClickUpEvent",
    "ClickUpPluginError",
    "Config",
    "DuplicateRun",
    "MissingCredentialsError",
    "ToolInfo",
    "WebhookSignatureError",
    "blocking_run",
    "build_mcp_server",
    "build_tool_functions",
    "clickup_mcp_app_env",
    "default_config",
    "events",
    "launch_task",
    "parse_webhook",
    "verify_webhook_signature",
]
