"""Serve the Slack integration as an MCP server for agents on Flyte.

The plugin's read/write surface doubles as MCP tools. By default the server is
read-only: agents can read channels, history, threads, and users but cannot
post anything. Set `read_only=False` to expose posting, reactions, and channel
creation.

Requirements:
    pip install "flyteplugins-slack[mcp]"

Setup:
    flyte create secret SLACK_BOT_TOKEN --value xoxb-...

Usage:
    python plugins/slack/examples/slack_mcp_server.py

    Connect an MCP client (streamable-http session URL is `/mcp/mcp`):

    $ claude mcp add --transport http slack-mcp https://<app>/mcp/mcp

    Or from an agent running on Flyte:

    ```python
    from flyte.ai.agents import Agent, MCPServerSpec

    agent = Agent(
        name="slack-agent",
        mcp_servers=[MCPServerSpec(name="slack", url="https://<app>/mcp/mcp")],
    )
    ```
"""

import flyte

from flyteplugins.slack import slack_mcp_app_env

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-slack[mcp]")

mcp_env = slack_mcp_app_env(
    "slack-mcp",
    image=image,
    secrets=[flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN")],
    # read_only=True is the default; widen deliberately:
    # read_only=False,
)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(mcp_env)
    handle.activate(wait=True)
    print(f"MCP server ready at {handle.endpoint}/mcp/mcp")
