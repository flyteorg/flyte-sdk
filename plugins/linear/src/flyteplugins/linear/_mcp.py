"""MCP server builders for the Linear plugin.

`build_mcp_server` turns the plugin's tool registry into a FastMCP server, and
`linear_mcp_app_env` wraps it in a Flyte `MCPAppEnvironment` that can be
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
    "Linear integration tools. Read tools fetch teams, workflow states, "
    "issues, and comments. Write tools create or update issues and add "
    "comments. Reacting to Linear webhooks is handled by the Linear app "
    "environment, not by this server."
)


def build_mcp_server(
    config: Config | None = None,
    *,
    api_key: str | None = None,
    name: str = "linear",
    instructions: str | None = None,
    read_only: bool = True,
    groups: list[str] | None = None,
    include_destructive: bool = False,
) -> FastMCP:
    """Build a FastMCP server exposing the plugin's Linear tools.

    Args:
        config: Plugin configuration; defaults to the module-level config.
        api_key: Optional explicit API key; otherwise read from the environment.
        name: Server name advertised to MCP clients.
        instructions: Server instructions shown to clients; defaults to a
            description of the read/write split.
        read_only: Only expose read tools (default True).
        groups: Optional explicit tool-group filter (`read`, `write`).
        include_destructive: Include destructive write tools like
            `merge_pull_request` (requires `read_only=False`).

    Returns:
        A configured FastMCP server. Use `MCPAppEnvironment` or
        `linear_mcp_app_env` to deploy it.
    """
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(
            "mcp is not installed. Install 'flyteplugins-linear[mcp]' (or 'mcp') to build the MCP server."
        ) from exc

    mcp = FastMCP(name=name, instructions=instructions or DEFAULT_INSTRUCTIONS)
    for tool_name, fn in build_tool_functions(
        config or default_config,
        api_key=api_key,
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


def linear_mcp_app_env(
    name: str = "linear-mcp",
    *,
    config: Config | None = None,
    api_key: str | None = None,
    read_only: bool = True,
    include_destructive: bool = False,
    **app_kwargs: Any,
) -> Any:
    """Create a Flyte `MCPAppEnvironment` serving the Linear MCP server.

    Example:

    ```python
    import flyte
    from flyteplugins.linear import linear_mcp_app_env

    env = linear_mcp_app_env("linear-mcp")
    flyte.serve(env)
    ```

    Args:
        name: App environment name.
        config: Plugin configuration.
        api_key: Optional explicit API key (prefer mounting a secret instead).
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
        api_key=api_key,
        read_only=read_only,
        include_destructive=include_destructive,
    )
    return MCPAppEnvironment(name=name, mcp=mcp, **app_kwargs)
