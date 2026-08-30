"""MCP server builders for the Jira plugin.

`build_mcp_server` turns the plugin's tool registry into a FastMCP server, and
`jira_mcp_app_env` wraps it in a Flyte `MCPAppEnvironment` that can be
deployed with `flyte.serve` so agents running on Flyte (or any MCP client) can
call the tools.

The default surface is read-only: agents can investigate projects, issues,
and comments but cannot change anything. Pass `read_only=False` (and
`include_destructive=True` for `delete_issue`) to widen the surface
deliberately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._config import Config, default_config
from ._tools import TOOL_REGISTRY, build_tool_functions

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

DEFAULT_INSTRUCTIONS = (
    "Jira integration tools. Read tools fetch projects, issues, comments, "
    "and transitions. Write tools create or update issues, comment, and "
    "transition. Reacting to Jira webhooks is handled by the Jira app "
    "environment, not by this server."
)


def build_mcp_server(
    config: Config | None = None,
    *,
    base_url: str | None = None,
    email: str | None = None,
    api_token: str | None = None,
    name: str = "jira",
    instructions: str | None = None,
    read_only: bool = True,
    groups: list[str] | None = None,
    include_destructive: bool = False,
) -> FastMCP:
    """Build a FastMCP server exposing the plugin's Jira tools.

    Args:
        config: Plugin configuration; defaults to the module-level config.
        base_url: Optional explicit site URL; otherwise read from the environment.
        email: Optional explicit account email; otherwise read from the environment.
        api_token: Optional explicit API token; otherwise read from the environment.
        name: Server name advertised to MCP clients.
        instructions: Server instructions shown to clients; defaults to a
            description of the read/write split.
        read_only: Only expose read tools (default True).
        groups: Optional explicit tool-group filter (`read`, `write`).
        include_destructive: Include destructive write tools like
            `delete_issue` (requires `read_only=False`).

    Returns:
        A configured FastMCP server. Use `MCPAppEnvironment` or
        `jira_mcp_app_env` to deploy it.
    """
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(
            "mcp is not installed. Install 'flyteplugins-jira[mcp]' (or 'mcp') to build the MCP server."
        ) from exc

    mcp = FastMCP(name=name, instructions=instructions or DEFAULT_INSTRUCTIONS)
    for tool_name, fn in build_tool_functions(
        config or default_config,
        base_url=base_url,
        email=email,
        api_token=api_token,
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


def jira_mcp_app_env(
    name: str = "jira-mcp",
    *,
    config: Config | None = None,
    base_url: str | None = None,
    email: str | None = None,
    api_token: str | None = None,
    read_only: bool = True,
    include_destructive: bool = False,
    **app_kwargs: Any,
) -> Any:
    """Create a Flyte `MCPAppEnvironment` serving the Jira MCP server.

    Example:

    ```python
    import flyte
    from flyteplugins.jira import jira_mcp_app_env

    env = jira_mcp_app_env("jira-mcp")
    flyte.serve(env)
    ```

    Args:
        name: App environment name.
        config: Plugin configuration.
        base_url: Optional explicit site URL (prefer mounting a secret instead).
        email: Optional explicit account email (prefer mounting a secret instead).
        api_token: Optional explicit API token (prefer mounting a secret instead).
        read_only: Only expose read tools (default True).
        include_destructive: Include destructive tools like
            `delete_issue`.
        **app_kwargs: Forwarded to `MCPAppEnvironment` (image, resources,
            secrets, env_vars, ...).

    Returns:
        A `flyte.ai.mcp.MCPAppEnvironment` instance.
    """
    from flyte.ai.mcp import MCPAppEnvironment

    mcp = build_mcp_server(
        config,
        base_url=base_url,
        email=email,
        api_token=api_token,
        read_only=read_only,
        include_destructive=include_destructive,
    )
    return MCPAppEnvironment(name=name, mcp=mcp, **app_kwargs)
