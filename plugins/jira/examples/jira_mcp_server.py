"""Serve the Jira integration as an MCP server for agents on Flyte.

The plugin's read/write surface doubles as MCP tools. By default the server is
read-only: agents can browse projects, issues, comments, and transitions but
cannot change anything. Set `read_only=False` to expose issue creation,
updates, commenting, and transitions, and `include_destructive=True` to also
expose `delete_issue`.

Requirements:
    pip install "flyteplugins-jira[mcp]"

Setup:
    flyte create secret JIRA_BASE_URL --value https://<site>.atlassian.net
    flyte create secret JIRA_EMAIL --value you@example.com
    flyte create secret JIRA_API_TOKEN --value <api-token>

Usage:
    python plugins/jira/examples/jira_mcp_server.py

    Connect an MCP client (streamable-http session URL is `/mcp/mcp`):

    $ claude mcp add --transport http jira-mcp https://<app>/mcp/mcp

    Or from an agent running on Flyte:

    ```python
    from flyte.ai.agents import Agent, MCPServerSpec

    agent = Agent(
        name="jira-agent",
        mcp_servers=[MCPServerSpec(name="jira", url="https://<app>/mcp/mcp")],
    )
    ```
"""

import flyte

from flyteplugins.jira import jira_mcp_app_env

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-jira[mcp]")

mcp_env = jira_mcp_app_env(
    "jira-mcp",
    image=image,
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
    ],
    # read_only=True is the default; widen deliberately:
    # read_only=False, include_destructive=False,
)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(mcp_env)
    handle.activate(wait=True)
    print(f"MCP server ready at {handle.endpoint}/mcp/mcp")
