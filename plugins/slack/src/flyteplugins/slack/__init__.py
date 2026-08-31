"""Slack integration for Flyte.

Read and write Slack from Flyte tasks, react to Slack Events API events
through an app environment, and expose the operations as an MCP server for
agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-slack[app,mcp]"
```

## Read/write from tasks

```python
import flyte
from flyteplugins.slack import SlackClient

env = flyte.TaskEnvironment(
    name="slack-demo",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slack"),
    secrets=[flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN")],
)

@env.task
async def notify(channel: str, message: str) -> str:
    async with SlackClient() as client:
        result = await client.post_message(channel, message)
    return result.get("permalink", result["ts"])
```

## React to Slack events

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

flyte.serve(app_env)
```

Handlers must `await launch_task.aio(...)`: the synchronous form blocks the
app's event loop, and webhook senders time deliveries out in seconds.

The app's dashboard (`/`) walks through Slack app creation, bot token scopes,
Flyte secret creation, and Events API configuration (including the automatic
`url_verification` challenge handshake).

## MCP server for agents

```python
import flyte
from flyteplugins.slack import slack_mcp_app_env

mcp_env = slack_mcp_app_env("slack-mcp")  # read-only by default
flyte.serve(mcp_env)
```
"""

from ._app import SlackAppEnvironment
from ._client import SlackClient
from ._config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_BOT_TOKEN_ENV_VAR,
    DEFAULT_SIGNING_SECRET_ENV_VAR,
    Config,
    default_config,
)
from ._dispatch import DUPE_LABEL_KEY, DuplicateRun, blocking_run, launch_task
from ._errors import EventSignatureError, MissingCredentialsError, SlackAPIError, SlackPluginError
from ._mcp import build_mcp_server, slack_mcp_app_env
from ._tools import TOOL_GROUPS, TOOL_REGISTRY, ToolInfo, build_tool_functions
from ._webhook import DedupeScope, SlackEvent, parse_event, parse_url_verification, verify_event_signature

__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_BOT_TOKEN_ENV_VAR",
    "DEFAULT_SIGNING_SECRET_ENV_VAR",
    "DUPE_LABEL_KEY",
    "TOOL_GROUPS",
    "TOOL_REGISTRY",
    "Config",
    "DedupeScope",
    "DuplicateRun",
    "EventSignatureError",
    "MissingCredentialsError",
    "SlackAPIError",
    "SlackAppEnvironment",
    "SlackClient",
    "SlackEvent",
    "SlackPluginError",
    "ToolInfo",
    "blocking_run",
    "build_mcp_server",
    "build_tool_functions",
    "default_config",
    "launch_task",
    "parse_event",
    "parse_url_verification",
    "slack_mcp_app_env",
    "verify_event_signature",
]
