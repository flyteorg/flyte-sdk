"""A Flyte MCP server with filtered tools, scripting, and search.

This example shows how to deploy a more restricted MCP server that exposes
task, run, script, and search tools, with a task allowlist to restrict which
tasks can be accessed and configurable search paths for documentation.

Requirements:
    pip install 'flyte[mcp]'
"""

import flyte
from flyte.ai.mcp import FlyteMCPAppEnvironment

image = flyte.Image.from_debian_base().with_pip_packages("mcp", "starlette", "uvicorn")

mcp_env = FlyteMCPAppEnvironment(
    name="restricted-mcp",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    tool_groups=["task", "run", "script", "search"],
    task_allowlist=["my-project/my-task", "another-task"],
    sdk_examples_path="/root/flyte-sdk/examples",
    docs_examples_path="/root/docs-examples",
    full_docs_path="/root/full-docs.txt",
    instructions=(
        "This MCP server provides tools to run and monitor specific Flyte tasks, "
        "build and run UV scripts remotely, and search Flyte SDK/docs examples. "
        "Only allowlisted tasks can be accessed."
    ),
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(mcp_env)
