"""Live deployment of the GitHub MCP server (github_mcp_server.py equivalent)."""

import flyte

from _livetest.common import GH_SECRET, image
from flyteplugins.github import github_mcp_app_env

mcp_env = github_mcp_app_env(
    "github-mcp",
    image=image("mcp>=1.26.0,<2"),
    secrets=[GH_SECRET],
    requires_auth=False,
)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(mcp_env)
    handle.activate(wait=True)
    print("ENDPOINT:", handle.endpoint)
