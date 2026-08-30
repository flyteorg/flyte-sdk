"""Notion integration for Flyte.

Read and write Notion from Flyte tasks, detect Notion changes by polling
(Notion has no webhooks) through an app environment, and expose the operations
as an MCP server for agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-notion[app,mcp]"
```

## Read/write from tasks

```python
import flyte
from flyteplugins.notion import NotionClient, title_property, select_property

env = flyte.TaskEnvironment(
    name="notion-demo",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-notion"),
    secrets=[flyte.Secret("NOTION_TOKEN", as_env_var="NOTION_TOKEN")],
)

@env.task
async def add_row(database_id: str, name: str, status: str) -> str:
    async with NotionClient() as client:
        page = await client.create_database_page(
            database_id,
            {"Name": title_property(name), "Status": select_property(status)},
        )
    return page["url"]
```

## React to Notion changes (polling)

Notion has no webhooks, so change detection is polling-based. Either schedule
a Flyte task with `flyte.Trigger` that calls `query_database_since`, or use
the app environment's poll endpoint:

```python
import flyte
from flyteplugins.notion import NotionAppEnvironment, launch_task

app_env = NotionAppEnvironment(
    name="notion-integration",
    databases=["<database-id>"],
    secrets=[
        flyte.Secret("NOTION_TOKEN", as_env_var="NOTION_TOKEN"),
        flyte.Secret("NOTION_POLL_TOKEN", as_env_var="NOTION_POLL_TOKEN"),
    ],
)

@app_env.on_event("page.edited")
async def react_to_edit(event):
    import flyte.remote as remote

    task = remote.Task.get(name="handle_notion_update", auto_version="latest")
    run = launch_task(task, key=event.dedupe_key(), page_id=event.page_id)
    return {"run": run.name}

flyte.serve(app_env)
```

Then point any scheduler at `GET <app-url>/api/poll` with an `X-Poll-Token`
header. The dashboard (`/`) walks through integration creation, page sharing,
secret creation, and the polling options.

## MCP server for agents

```python
import flyte
from flyteplugins.notion import notion_mcp_app_env

mcp_env = notion_mcp_app_env("notion-mcp")  # read-only by default
flyte.serve(mcp_env)
```
"""

from ._app import NotionAppEnvironment
from ._client import NotionClient
from ._config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_NOTION_VERSION,
    DEFAULT_POLL_TOKEN_ENV_VAR,
    DEFAULT_TOKEN_ENV_VAR,
    Config,
    default_config,
)
from ._dispatch import DUPE_LABEL_KEY, DuplicateRun, blocking_run, launch_task, run_name_for
from ._errors import MissingCredentialsError, NotionAPIError, NotionPluginError
from ._events import NotionEvent, events_from_pages
from ._helpers import (
    bulleted_block,
    checkbox_property,
    date_property,
    email_property,
    extract_rich_text,
    extract_title,
    heading_block,
    multi_select_property,
    number_property,
    paragraph_block,
    rich_text_property,
    select_property,
    title_property,
    to_do_block,
    url_property,
)
from ._mcp import build_mcp_server, notion_mcp_app_env
from ._tools import TOOL_GROUPS, TOOL_REGISTRY, ToolInfo, build_tool_functions

__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_NOTION_VERSION",
    "DEFAULT_POLL_TOKEN_ENV_VAR",
    "DEFAULT_TOKEN_ENV_VAR",
    "DUPE_LABEL_KEY",
    "TOOL_GROUPS",
    "TOOL_REGISTRY",
    "Config",
    "DuplicateRun",
    "MissingCredentialsError",
    "NotionAPIError",
    "NotionAppEnvironment",
    "NotionClient",
    "NotionEvent",
    "NotionPluginError",
    "ToolInfo",
    "blocking_run",
    "build_mcp_server",
    "build_tool_functions",
    "bulleted_block",
    "checkbox_property",
    "date_property",
    "default_config",
    "email_property",
    "events_from_pages",
    "extract_rich_text",
    "extract_title",
    "heading_block",
    "launch_task",
    "multi_select_property",
    "notion_mcp_app_env",
    "number_property",
    "paragraph_block",
    "rich_text_property",
    "run_name_for",
    "select_property",
    "title_property",
    "to_do_block",
    "url_property",
]
