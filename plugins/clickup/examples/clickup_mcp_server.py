"""Serve the ClickUp integration as an MCP server for agents on Flyte.

The plugin's read/write surface doubles as MCP tools. By default the server is
read-only: agents can browse workspaces, lists, statuses, tasks, and comments
but cannot change anything. Set `read_only=False` to expose task creation,
updates, and commenting, and `include_destructive=True` to also expose
`delete_task`.

Requirements:
    pip install "flyteplugins-clickup[mcp]"

Setup:
    flyte create secret CLICKUP_TOKEN --value <token>

Usage:
    python plugins/clickup/examples/clickup_mcp_server.py

    Connect an MCP client (streamable-http session URL is `/mcp/mcp`):

    $ claude mcp add --transport http clickup-mcp https://<app>/mcp/mcp

    Or from an agent running on Flyte:

    ```python
    from flyte.ai.agents import Agent, MCPServerSpec

    agent = Agent(
        name="clickup-agent",
        mcp_servers=[MCPServerSpec(name="clickup", url="https://<app>/mcp/mcp")],
    )
    ```
"""

import flyte

from flyteplugins.clickup import clickup_mcp_app_env

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-clickup[mcp]")

mcp_env = clickup_mcp_app_env(
    "clickup-mcp",
    image=image,
    secrets=[flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN")],
    # read_only=True is the default; widen deliberately:
    # read_only=False, include_destructive=False,
)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(mcp_env)
    handle.activate(wait=True)
    print(f"MCP server ready at {handle.endpoint}/mcp/mcp")
