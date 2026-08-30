"""Serve the Notion integration as an MCP server for agents on Flyte.

The plugin's read/write surface doubles as MCP tools. By default the server is
read-only: agents can search and read pages, databases, and blocks but cannot
change anything. Set `read_only=False` to expose page creation, updates, and
block appending, and `include_destructive=True` to also expose `archive_page`.

Requirements:
    pip install "flyteplugins-notion[mcp]"

Setup:
    flyte create secret NOTION_TOKEN --value ntn_...

Usage:
    python plugins/notion/examples/notion_mcp_server.py

    Connect an MCP client (streamable-http session URL is `/mcp/mcp`):

    $ claude mcp add --transport http notion-mcp https://<app>/mcp/mcp

    Or from an agent running on Flyte:

    ```python
    from flyte.ai.agents import Agent, MCPServerSpec

    agent = Agent(
        name="notion-agent",
        mcp_servers=[MCPServerSpec(name="notion", url="https://<app>/mcp/mcp")],
    )
    ```
"""

import flyte

from flyteplugins.notion import notion_mcp_app_env

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-notion[mcp]")

mcp_env = notion_mcp_app_env(
    "notion-mcp",
    image=image,
    secrets=[flyte.Secret("NOTION_TOKEN", as_env_var="NOTION_TOKEN")],
    # read_only=True is the default; widen deliberately:
    # read_only=False, include_destructive=False,
)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(mcp_env)
    handle.activate(wait=True)
    print(f"MCP server ready at {handle.endpoint}/mcp/mcp")
