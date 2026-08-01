"""A Flyte MCP server app that exposes Flyte operations as MCP tools.

This example deploys an MCP (Model Context Protocol) server that allows AI
assistants and LLM-based clients to interact with the Flyte control plane
using the standardized MCP protocol.

The server exposes tools for running tasks, monitoring runs, managing apps
and triggers, and searching Flyte SDK/docs examples.

Requirements:
    pip install 'flyte[mcp]'

Usage:

    Deploy all tools

    $ python examples/mcp/flyte_mcp_app.py

    Or serve locally for development (recommended: `uvx`). This binds
    http://localhost:8080/flyte-mcp/mcp; pass `--transport stdio` instead when an
    MCP client launches the server as a subprocess.

    $ uvx --from "flyte[mcp]" flyte-mcp

    If you're running from this repo checkout (note the `[mcp]` extra -- without it
    the server exits with "mcp is not installed"):
    $ uvx --from ".[mcp]" flyte-mcp

    ------------------------------
    Connect from Claude Code
    ------------------------------

    Some agent harnesses can't reach `localhost` URLs. For local usage, prefer
    configuring Claude Code to launch the server via `uvx` (process-based setup).

    Add as a local stdio MCP server:
    $ claude mcp add --transport stdio flyte-mcp -- \
      uvx --from "flyte[mcp]" flyte-mcp --transport stdio

    If you deploy this app remotely (so it has a public base URL), use that URL instead.
    With default ``transport="streamable-http"`` and ``mcp_mount_path="/flyte-mcp"``, the MCP
    session URL is ``/flyte-mcp/mcp`` (Mount ``/flyte-mcp`` plus the FastMCP streamable path).

    $ claude mcp add --transport http flyte-mcp-remote https://<YOUR_HOST>/flyte-mcp/mcp

    If your remote deployment requires auth, add headers (example):
    $ claude mcp add --transport http \
      --header "Authorization: Bearer $TOKEN" \
      flyte-mcp-remote https://<YOUR_HOST>/flyte-mcp/mcp

    ------------------------------
    Connect from OpenCode
    ------------------------------

    For local usage (no `localhost` required), configure OpenCode to launch the
    server as a local MCP process:

    {
      "$schema": "https://opencode.ai/config.json",
      "mcp": {
        "flyte-mcp": {
          "type": "local",
          "command": ["uvx", "--from", "flyte[mcp]", "flyte-mcp", "--transport", "stdio"],
          "enabled": true
        }
      }
    }

    For a remote deployment:

    {
      "$schema": "https://opencode.ai/config.json",
      "mcp": {
      "flyte-mcp": {
        "type": "remote",
        "url": "https://<YOUR_HOST>/flyte-mcp/mcp",
        "enabled": true,
        "headers": {
          "Authorization": "Bearer $YOUR_TOKEN"
        }
      }
    }
"""

import flyte
from flyte.ai.mcp import FlyteMCPAppEnvironment

image = flyte.Image.from_debian_base(name="flyte-mcp-server-image").with_pip_packages("mcp", "starlette", "uvicorn")

# Deploy an MCP server with all tools enabled
mcp_env = FlyteMCPAppEnvironment(
    name="flyte-mcp-server",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    transport="streamable-http",
    instructions=(
        "This MCP server provides tools to interact with the Flyte control plane. "
        "Use the available tools to run tasks, monitor runs, manage apps and triggers, "
        "and search SDK/docs examples."
    ),
)

if __name__ == "__main__":
    flyte.init_from_config()
    app_handle = flyte.serve(mcp_env)
    print(f"{app_handle.url}")
