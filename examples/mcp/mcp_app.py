"""A Flyte MCP server app that exposes Flyte operations as MCP tools.

This example deploys an MCP (Model Context Protocol) server that allows AI
assistants and LLM-based clients to interact with the Flyte control plane
using the standardized MCP protocol.

The server exposes tools for running tasks, monitoring runs, managing apps
and triggers, building container images, building and running UV scripts
remotely, and searching Flyte SDK/docs examples.

Requirements:
    pip install 'flyte[mcp]'

Usage:
    # Deploy all tools
    python examples/mcp/mcp_app.py

    # Or serve locally for development
    flyte serve examples/mcp/mcp_app.py
"""

import flyte
from flyte.ai.mcp import FlyteMCPAppEnvironment

image = flyte.Image.from_debian_base().with_pip_packages("mcp", "starlette", "uvicorn")

# Deploy an MCP server with all tools enabled
mcp_env = FlyteMCPAppEnvironment(
    name="flyte-mcp-server",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    instructions=(
        "This MCP server provides tools to interact with the Flyte control plane. "
        "Use the available tools to run tasks, monitor runs, manage apps, build images, "
        "build and run UV scripts remotely, and search SDK/docs examples."
    ),
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(mcp_env)
