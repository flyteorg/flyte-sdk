"""Serve the Linear integration as an MCP server for agents on Flyte.

The plugin's read/write surface doubles as MCP tools. By default the server is
read-only: agents can read teams, workflow states, issues, and comments but
cannot change anything. Set `read_only=False` to expose issue creation,
updates, and commenting.

Requirements:
    pip install "flyteplugins-linear[mcp]"

Setup:
    flyte create secret LINEAR_API_KEY --value <api-key>

Usage:
    python plugins/linear/examples/linear_mcp_server.py

    Connect an MCP client (streamable-http session URL is `/mcp/mcp`):

    $ claude mcp add --transport http linear-mcp https://<app>/mcp/mcp

    Or from an agent running on Flyte:

    ```python
    from flyte.ai.agents import Agent, MCPServerSpec

    agent = Agent(
        name="linear-agent",
        mcp_servers=[MCPServerSpec(name="linear", url="https://<app>/mcp/mcp")],
    )
    ```
"""

import flyte

from flyteplugins.linear import linear_mcp_app_env

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-linear[mcp]")

mcp_env = linear_mcp_app_env(
    "linear-mcp",
    image=image,
    secrets=[flyte.Secret("LINEAR_API_KEY", as_env_var="LINEAR_API_KEY")],
    # read_only=True is the default; widen deliberately:
    # read_only=False,
)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(mcp_env)
    handle.activate(wait=True)
    print(f"MCP server ready at {handle.endpoint}/mcp/mcp")
