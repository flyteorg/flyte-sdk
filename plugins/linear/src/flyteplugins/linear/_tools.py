"""MCP tool registry for the Linear plugin.

The plugin's read/write operations double as MCP tools so agents running on
Flyte can use them through a deployed MCP server. Each entry in `TOOL_REGISTRY`
maps a tool name to its metadata (group, title, and behavior hints), and
`build_tool_functions` produces the async callables that back the tools.

Event ingestion is deliberately *not* a tool: reacting to Linear events is the
job of the `LinearAppEnvironment` webhook receiver, not of an agent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ._client import LinearClient
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
    "get_viewer": ToolInfo("Get authenticated user", "read", read_only=True),
    "list_teams": ToolInfo("List teams", "read", read_only=True),
    "list_workflow_states": ToolInfo("List workflow states", "read", read_only=True),
    "list_issues": ToolInfo("List issues", "read", read_only=True),
    "get_issue": ToolInfo("Get issue", "read", read_only=True),
    "list_comments": ToolInfo("List comments", "read", read_only=True),
    # -- write (non-destructive) ---------------------------------------------
    "create_issue": ToolInfo("Create issue", "write", read_only=False, idempotent=False),
    "update_issue": ToolInfo("Update issue", "write", read_only=False),
    "add_comment": ToolInfo("Add comment", "write", read_only=False, idempotent=False),
}

#: Tool groups exposed by `build_tool_functions` and the MCP server builder.
TOOL_GROUPS = ("read", "write")


def build_tool_functions(
    config: Config | None = None,
    *,
    api_key: str | None = None,
    groups: list[str] | None = None,
    read_only: bool = True,
    include_destructive: bool = False,
) -> dict[str, ToolFn]:
    """Build the async tool callables selected by the given filters.

    Each callable creates its own `LinearClient` per invocation, so tools are
    safe to call concurrently from an MCP server.

    Args:
        config: Plugin configuration; defaults to the module-level config.
        api_key: Optional explicit API key, forwarded to the client.
        groups: Tool groups to include (`read`, `write`). Defaults to all.
        read_only: When True, only read-only tools are returned regardless of
            `groups`.
        include_destructive: Destructive tools are excluded unless this is
            True. Linear has no destructive tools today; the flag keeps the
            interface consistent across plugins.

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
        selected[name] = _make_tool(name, cfg, api_key)
    return selected


def _make_tool(name: str, config: Config, api_key: str | None) -> ToolFn:
    method = getattr(LinearClient, name)
    sig = inspect.signature(method)
    params = [p for pname, p in sig.parameters.items() if pname != "self"]

    async def tool(*args: Any, **kwargs: Any) -> Any:
        async with LinearClient(config, api_key=api_key) as client:
            return await getattr(client, name).aio(*args, **kwargs)

    tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = method.__doc__ or TOOL_REGISTRY[name].title
    tool.__annotations__ = {k: v for k, v in method.__annotations__.items() if k != "return"} | {
        "return": method.__annotations__.get("return", Any)
    }
    return tool
