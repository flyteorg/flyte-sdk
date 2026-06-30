"""Deploy any FastMCP server as a Flyte app using ``MCPAppEnvironment``.

This example builds a tiny FastMCP instance with a single tool and serves it
with the same HTTP layout as other MCP app environments (health check + mounted
MCP ASGI app).

Requirements:
    pip install 'flyte[mcp]'

Usage (from repo root):

    $ python examples/mcp/generic_mcp_app.py
"""

from __future__ import annotations

import flyte
from flyte.ai.mcp import MCPAppEnvironment


def main() -> None:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(name="demo-generic-mcp")

    @mcp.tool()
    def ping() -> str:
        """Health-style echo for demos."""
        return "pong"

    env = MCPAppEnvironment(
        name="generic-mcp-demo",
        mcp=mcp,
        transport="streamable-http",
        mcp_mount_path="/mcp",
    )

    flyte.init_from_config()
    handle = flyte.serve(env)
    handle.activate(wait=True)
    print(f"App is ready at {handle.endpoint}")


if __name__ == "__main__":
    main()
