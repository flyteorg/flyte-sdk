"""MCP tool registry for the Notion plugin.

The plugin's read/write operations double as MCP tools so agents running on
Flyte can use them through a deployed MCP server. Each entry in `TOOL_REGISTRY`
maps a tool name to its metadata (group, title, and behavior hints), and
`build_tool_functions` produces the async callables that back the tools.

Change detection (polling) is deliberately *not* a tool: reacting to Notion
changes is the job of the `NotionAppEnvironment` poll endpoint or a scheduled
Flyte task, not of an agent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ._client import NotionClient
from ._config import Config, default_config

ToolFn = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ToolInfo:
    """Metadata for one MCP tool."""

    title: str
    group: str  # "read" | "write"
    read_only: bool
    destructive: bool = False
    idempotent: bool = True


#: Registry of every tool the plugin can expose, keyed by tool name.
TOOL_REGISTRY: dict[str, ToolInfo] = {
    # -- read ---------------------------------------------------------------
    "get_me": ToolInfo("Get integration user", "read", read_only=True),
    "search": ToolInfo("Search pages and databases", "read", read_only=True),
    "get_page": ToolInfo("Get page", "read", read_only=True),
    "get_database": ToolInfo("Get database schema", "read", read_only=True),
    "query_database": ToolInfo("Query database", "read", read_only=True),
    "list_block_children": ToolInfo("List block children", "read", read_only=True),
    # -- write (non-destructive) ---------------------------------------------
    "create_database_page": ToolInfo("Create database page", "write", read_only=False, idempotent=False),
    "create_page": ToolInfo("Create page", "write", read_only=False, idempotent=False),
    "update_page": ToolInfo("Update page", "write", read_only=False),
    "append_blocks": ToolInfo("Append blocks", "write", read_only=False, idempotent=False),
    # -- write (destructive) --------------------------------------------------
    "archive_page": ToolInfo("Archive page", "write", read_only=False, destructive=True),
}

#: Tool groups exposed by `build_tool_functions` and the MCP server builder.
TOOL_GROUPS = ("read", "write")


def build_tool_functions(
    config: Config | None = None,
    *,
    token: str | None = None,
    groups: list[str] | None = None,
    read_only: bool = True,
    include_destructive: bool = False,
) -> dict[str, ToolFn]:
    """Build the async tool callables selected by the given filters.

    Each callable creates its own `NotionClient` per invocation, so tools are
    safe to call concurrently from an MCP server.

    Args:
        config: Plugin configuration; defaults to the module-level config.
        token: Optional explicit token, forwarded to the client.
        groups: Tool groups to include (`read`, `write`). Defaults to all.
        read_only: When True, only read-only tools are returned regardless of
            `groups`.
        include_destructive: Destructive tools (e.g. `archive_page`) are
            excluded unless this is True.

    Returns:
        Mapping of tool name to async callable.
    """
    cfg = config or default_config
    selected: dict[str, ToolFn] = {}
    for name, info in TOOL_REGISTRY.items():
        if info.read_only is False and read_only:
            continue
        if groups is not None and info.group not in groups:
            continue
        if info.destructive and not include_destructive:
            continue
        selected[name] = _make_tool(name, cfg, token)
    return selected


def _make_tool(name: str, config: Config, token: str | None) -> ToolFn:
    method = getattr(NotionClient, name)
    sig = inspect.signature(method)
    params = [p for pname, p in sig.parameters.items() if pname != "self"]

    async def tool(*args: Any, **kwargs: Any) -> Any:
        async with NotionClient(config, token=token) as client:
            return await getattr(client, name).aio(*args, **kwargs)

    tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = method.__doc__ or TOOL_REGISTRY[name].title
    tool.__annotations__ = {k: v for k, v in method.__annotations__.items() if k != "return"} | {
        "return": method.__annotations__.get("return", Any)
    }
    return tool
