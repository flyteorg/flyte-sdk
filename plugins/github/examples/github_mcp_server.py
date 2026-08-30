"""Serve the GitHub integration as an MCP server for agents on Flyte.

The plugin's read/write surface doubles as MCP tools. By default the server is
read-only: agents can investigate repositories, issues, and pull requests but
cannot change anything. Set `read_only=False` to expose write tools, and
`include_destructive=True` to also expose `merge_pull_request`.

Requirements:
    pip install "flyteplugins-github[mcp]"

Setup:
    flyte create secret GITHUB_TOKEN --value <token>

Usage:
    python plugins/github/examples/github_mcp_server.py

    Connect an MCP client (streamable-http session URL is `/mcp/mcp`):

    $ claude mcp add --transport http github-mcp https://<app>/mcp/mcp

    Or from an agent running on Flyte:

    ```python
    from flyte.ai.agents import Agent, MCPServerSpec

    agent = Agent(
        name="github-agent",
        mcp_servers=[MCPServerSpec(name="github", url="https://<app>/mcp/mcp")],
    )
    ```
"""

import flyte

from flyteplugins.github import github_mcp_app_env

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-github[mcp]")

mcp_env = github_mcp_app_env(
    "github-mcp",
    image=image,
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
    # read_only=True is the default; widen deliberately:
    # read_only=False, include_destructive=False,
)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(mcp_env)
    handle.activate(wait=True)
    print(f"MCP server ready at {handle.endpoint}/mcp/mcp")
