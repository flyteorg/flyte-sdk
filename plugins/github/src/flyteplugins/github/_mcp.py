"""MCP server builders for the GitHub plugin.

`build_mcp_server` turns the plugin's tool registry into a FastMCP server, and
`github_mcp_app_env` wraps it in a Flyte `MCPAppEnvironment` that can be
deployed with `flyte.serve` so agents running on Flyte (or any MCP client) can
call the tools.

The default surface is read-only: agents can investigate issues and PRs but
cannot change anything. Pass `read_only=False` (and, for merge,
`include_destructive=True`) to widen the surface deliberately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._config import Config, default_config
from ._tools import TOOL_REGISTRY, build_tool_functions

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

DEFAULT_INSTRUCTIONS = (
    "GitHub integration tools. Read tools fetch repositories, issues, pull "
    "requests, files, and reviews. Write tools create or update issues, PRs, "
    "branches, and check runs. Reacting to GitHub events (webhooks) is handled "
    "by the GitHub app environment, not by this server."
)


def build_mcp_server(
    config: Config | None = None,
    *,
    token: str | None = None,
    name: str = "github",
    instructions: str | None = None,
    read_only: bool = True,
    groups: list[str] | None = None,
    include_destructive: bool = False,
) -> FastMCP:
    """Build a FastMCP server exposing the plugin's GitHub tools.

    Args:
        config: Plugin configuration; defaults to the module-level config.
        token: Optional explicit token; otherwise read from the environment.
        name: Server name advertised to MCP clients.
        instructions: Server instructions shown to clients; defaults to a
            description of the read/write split.
        read_only: Only expose read tools (default True).
        groups: Optional explicit tool-group filter (`read`, `write`).
        include_destructive: Include destructive write tools like
            `merge_pull_request` (requires `read_only=False`).

    Returns:
        A configured FastMCP server. Use `MCPAppEnvironment` or
        `github_mcp_app_env` to deploy it.
    """
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(
            "mcp is not installed. Install 'flyteplugins-github[mcp]' (or 'mcp') to build the MCP server."
        ) from exc

    mcp = FastMCP(name=name, instructions=instructions or DEFAULT_INSTRUCTIONS)
    for tool_name, fn in build_tool_functions(
        config or default_config,
        token=token,
        groups=groups,
        read_only=read_only,
        include_destructive=include_destructive,
    ).items():
        info = TOOL_REGISTRY[tool_name]
        mcp.add_tool(
            fn,
            name=tool_name,
            title=info.title,
            annotations=ToolAnnotations(
                readOnlyHint=info.read_only,
                destructiveHint=info.destructive,
                idempotentHint=info.idempotent,
                openWorldHint=True,
            ),
        )
    return mcp


def github_mcp_app_env(
    name: str = "github-mcp",
    *,
    config: Config | None = None,
    token: str | None = None,
    read_only: bool = True,
    include_destructive: bool = False,
    **app_kwargs: Any,
) -> Any:
    """Create a Flyte `MCPAppEnvironment` serving the GitHub MCP server.

    Example:

    ```python
    import flyte
    from flyteplugins.github import github_mcp_app_env

    env = github_mcp_app_env("github-mcp")
    flyte.serve(env)
    ```

    Args:
        name: App environment name.
        config: Plugin configuration.
        token: Optional explicit token (prefer mounting a secret instead).
        read_only: Only expose read tools (default True).
        include_destructive: Include destructive tools like
            `merge_pull_request`.
        **app_kwargs: Forwarded to `MCPAppEnvironment` (image, resources,
            secrets, env_vars, ...).

    Returns:
        A `flyte.ai.mcp.MCPAppEnvironment` instance.
    """
    from flyte.ai.mcp import MCPAppEnvironment

    mcp = build_mcp_server(
        config,
        token=token,
        read_only=read_only,
        include_destructive=include_destructive,
    )
    return MCPAppEnvironment(name=name, mcp=mcp, **app_kwargs)
