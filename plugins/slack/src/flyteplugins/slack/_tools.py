"""MCP tool registry for the Slack plugin.

The plugin's read/write operations double as MCP tools so agents running on
Flyte can use them through a deployed MCP server. Each entry in `TOOL_REGISTRY`
maps a tool name to its metadata (group, title, and behavior hints), and
`build_tool_functions` produces the async callables that back the tools.

Event ingestion is deliberately *not* a tool: reacting to Slack events is the
job of the `SlackAppEnvironment` receiver, not of an agent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ._client import SlackClient
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
    "list_channels": ToolInfo("List channels", "read", read_only=True),
    "get_channel": ToolInfo("Get channel", "read", read_only=True),
    "get_channel_history": ToolInfo("Read channel history", "read", read_only=True),
    "get_thread": ToolInfo("Read thread", "read", read_only=True),
    "get_message_permalink": ToolInfo("Get message permalink", "read", read_only=True),
    "get_user": ToolInfo("Get user", "read", read_only=True),
    # -- write (non-destructive) ---------------------------------------------
    "post_message": ToolInfo("Post message", "write", read_only=False, idempotent=False),
    "reply_in_thread": ToolInfo("Reply in thread", "write", read_only=False, idempotent=False),
    "update_message": ToolInfo("Update message", "write", read_only=False),
    "add_reaction": ToolInfo("Add reaction", "write", read_only=False),
    "remove_reaction": ToolInfo("Remove reaction", "write", read_only=False),
    "create_channel": ToolInfo("Create channel", "write", read_only=False, idempotent=False),
}

#: Tool groups exposed by `build_tool_functions` and the MCP server builder.
TOOL_GROUPS = ("read", "write")


def build_tool_functions(
    config: Config | None = None,
    *,
    bot_token: str | None = None,
    groups: list[str] | None = None,
    read_only: bool = True,
    include_destructive: bool = False,
) -> dict[str, ToolFn]:
    """Build the async tool callables selected by the given filters.

    Each callable creates its own `SlackClient` per invocation, so tools are
    safe to call concurrently from an MCP server.

    Args:
        config: Plugin configuration; defaults to the module-level config.
        bot_token: Optional explicit bot token, forwarded to the client.
        groups: Tool groups to include (`read`, `write`). Defaults to all.
        read_only: When True, only read-only tools are returned regardless of
            `groups`.
        include_destructive: Destructive tools are excluded unless this is
            True. Slack has no destructive tools today; the flag keeps the
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
        selected[name] = _make_tool(name, cfg, bot_token)
    return selected


def _make_tool(name: str, config: Config, bot_token: str | None) -> ToolFn:
    method = getattr(SlackClient, name)
    sig = inspect.signature(method)
    params = [p for pname, p in sig.parameters.items() if pname != "self"]

    async def tool(*args: Any, **kwargs: Any) -> Any:
        async with SlackClient(config, bot_token=bot_token) as client:
            return await getattr(client, name).aio(*args, **kwargs)

    tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = method.__doc__ or TOOL_REGISTRY[name].title
    tool.__annotations__ = {k: v for k, v in method.__annotations__.items() if k != "return"} | {
        "return": method.__annotations__.get("return", Any)
    }
    return tool
