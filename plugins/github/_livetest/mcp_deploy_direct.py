"""MCP server deployed by constructing MCPAppEnvironment at module scope.

Contrast with mcp_deploy.py, which uses the plugin's github_mcp_app_env()
factory. The factory builds the env inside the plugin module, so the SDK's
caller-frame resolution cannot find it as a module-level variable and emits a
container command with no --resolver args (fserve then exits 1).

Separately, FastMCP enables DNS-rebinding protection with an empty
allowed_hosts list, which rejects every request arriving through the app's
public hostname with 421 "Invalid Host header". Protection stays on here; the
app's assigned hostname is allowlisted instead.
"""

import flyte
from flyte.ai.mcp import MCPAppEnvironment
from mcp.server.transport_security import TransportSecuritySettings

from _livetest.common import image
from flyteplugins.github import build_mcp_server

APP_HOST = "empty-moon-34922.apps.demo.hosted.unionai.cloud"

_server = build_mcp_server(read_only=True)
_server.settings.transport_security = TransportSecuritySettings(
    allowed_hosts=[APP_HOST, f"{APP_HOST}:*"],
    allowed_origins=[f"https://{APP_HOST}"],
)

mcp_env = MCPAppEnvironment(
    name="github-mcp",
    mcp=_server,
    image=image("mcp>=1.26.0,<2"),
    requires_auth=False,
)


if __name__ == "__main__":
    flyte.init_from_config(project="niels", domain="development")
    handle = flyte.serve(mcp_env)
    handle.activate(wait=True)
    print("ENDPOINT:", handle.endpoint)
