from __future__ import annotations

import click

import flyte


def _comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",")]
    items = [v for v in items if v]
    return items or None


def _csv_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> list[str] | None:
    return _comma_list(value)


@click.command(help="Serve a Flyte MCP server over HTTP (FastMCP + Starlette).")
@click.option("--name", default="flyte-mcp-server", show_default=True, help="App name.")
@click.option("--title", default=None, help="Optional MCP server title (defaults to --name).")
@click.option("--instructions", default=None, help="Optional MCP server instructions string.")
@click.option("--port", type=int, default=8080, show_default=True, help="HTTP port to bind.")
@click.option("--mcp-mount-path", default="/flyte-mcp", show_default=True, help="Mount path for MCP endpoint.")
@click.option(
    "--tool-groups",
    default=None,
    callback=_csv_callback,
    help="Comma-separated tool groups to enable (mutually exclusive with --tools).",
)
@click.option(
    "--tools",
    default=None,
    callback=_csv_callback,
    help="Comma-separated individual tools to enable (mutually exclusive with --tool-groups).",
)
@click.option("--sdk-examples-path", default=None, help="Path for search_flyte_sdk_examples.")
@click.option("--docs-examples-path", default=None, help="Path for search_flyte_docs_examples.")
@click.option("--full-docs-path", default=None, help="Path for search_full_docs.")
@click.option(
    "--init-from-config/--no-init-from-config",
    default=True,
    show_default=True,
    help="Initialize Flyte config before serving.",
)
def main(
    name: str,
    title: str | None,
    instructions: str | None,
    port: int,
    mcp_mount_path: str,
    tool_groups: list[str] | None,
    tools: list[str] | None,
    sdk_examples_path: str | None,
    docs_examples_path: str | None,
    full_docs_path: str | None,
    init_from_config: bool,
) -> None:
    if init_from_config:
        flyte.init_from_config()

    from flyte.ai.mcp import FlyteMCPAppEnvironment

    env = FlyteMCPAppEnvironment(
        name=name,
        title=title,
        instructions=instructions,
        port=port,
        mcp_mount_path=mcp_mount_path,
        tool_groups=tool_groups,
        tools=tools,
        sdk_examples_path=sdk_examples_path,
        docs_examples_path=docs_examples_path,
        full_docs_path=full_docs_path,
    )
    flyte.serve(env)
