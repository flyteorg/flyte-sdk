"""MCP tool registry for the GitHub plugin.

The plugin's read/write operations double as MCP tools so agents running on
Flyte can use them through a deployed MCP server. Each entry in `TOOL_REGISTRY`
maps a tool name to its metadata (group, title, and behavior hints), and
`build_tool_functions` produces the async callables that back the tools.

Event ingestion is deliberately *not* a tool: reacting to GitHub events is the
job of the `GitHubAppEnvironment` webhook receiver, not of an agent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ._client import GitHubClient
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
    "get_user": ToolInfo("Get authenticated user", "read", read_only=True),
    "get_repository": ToolInfo("Get repository", "read", read_only=True),
    "list_repositories": ToolInfo("List repositories", "read", read_only=True),
    "list_repository_files": ToolInfo("List repository files", "read", read_only=True),
    "get_file_contents": ToolInfo("Read file contents", "read", read_only=True),
    "list_commits": ToolInfo("List commits", "read", read_only=True),
    "list_issues": ToolInfo("List issues", "read", read_only=True),
    "get_issue": ToolInfo("Get issue", "read", read_only=True),
    "list_issue_comments": ToolInfo("List issue comments", "read", read_only=True),
    "list_pull_requests": ToolInfo("List pull requests", "read", read_only=True),
    "get_pull_request": ToolInfo("Get pull request", "read", read_only=True),
    "get_pull_request_files": ToolInfo("Get pull request files", "read", read_only=True),
    "get_pull_request_reviews": ToolInfo("Get pull request reviews", "read", read_only=True),
    # -- write (non-destructive) --------------------------------------------
    "create_issue": ToolInfo("Create issue", "write", read_only=False, idempotent=False),
    "create_issue_comment": ToolInfo("Comment on issue or PR", "write", read_only=False, idempotent=False),
    "update_issue": ToolInfo("Update issue", "write", read_only=False),
    "add_labels": ToolInfo("Add labels", "write", read_only=False),
    "create_pull_request": ToolInfo("Create pull request", "write", read_only=False, idempotent=False),
    "update_pull_request": ToolInfo("Update pull request", "write", read_only=False),
    "create_pull_request_review": ToolInfo("Submit PR review", "write", read_only=False, idempotent=False),
    "create_branch": ToolInfo("Create branch", "write", read_only=False),
    "create_or_update_file": ToolInfo("Create or update file", "write", read_only=False),
    "create_check_run": ToolInfo("Create check run", "write", read_only=False),
    # -- write (destructive) --------------------------------------------------
    "merge_pull_request": ToolInfo("Merge pull request", "write", read_only=False, destructive=True, idempotent=False),
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

    Each callable creates its own `GitHubClient` per invocation, so tools are
    safe to call concurrently from an MCP server.

    Args:
        config: Plugin configuration; defaults to the module-level config.
        token: Optional explicit token, forwarded to the client.
        groups: Tool groups to include (`read`, `write`). Defaults to all.
        read_only: When True, only read-only tools are returned regardless of
            `groups`.
        include_destructive: Destructive tools (e.g. `merge_pull_request`) are
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
    method = getattr(GitHubClient, name)
    sig = inspect.signature(method)
    params = [p for pname, p in sig.parameters.items() if pname != "self"]

    async def tool(*args: Any, **kwargs: Any) -> Any:
        async with GitHubClient(config, token=token) as client:
            return await getattr(client, name).aio(*args, **kwargs)

    tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = method.__doc__ or TOOL_REGISTRY[name].title
    tool.__annotations__ = {k: v for k, v in method.__annotations__.items() if k != "return"} | {
        "return": method.__annotations__.get("return", Any)
    }
    return tool
