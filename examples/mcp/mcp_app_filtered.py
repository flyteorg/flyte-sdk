"""A Flyte MCP server with filtered tools, scripting, and search.

This example shows how to deploy a more restricted MCP server that exposes
task, run, script, and search tools, with a task allowlist to restrict which
tasks can be accessed and configurable search paths for documentation.

**Search paths (`FlyteMCPAppEnvironment`)**

The ``search`` tool group reads from up to three filesystem locations. For a
remote deployment the image must contain those paths. This example bakes them
in at build time:

- ``sdk_examples_path``: ``/root/flyte-sdk/examples`` — clone
  [flyteorg/flyte-sdk](https://github.com/flyteorg/flyte-sdk) into
  ``/root/flyte-sdk``.
- ``docs_examples_path``: ``/root/unionai-examples/v2`` — clone
  [unionai/unionai-examples](https://github.com/unionai/unionai-examples) into
  ``/root/unionai-examples``.
- ``full_docs_path``: ``/root/llms.txt`` — fetch
  [Union docs llms.txt](https://www.union.ai/docs/v2/union/llms.txt) to that
  path (e.g. ``curl``).

Image build runs ``git clone`` and ``curl``; you need network access during the
image build (and ``git`` / ``curl`` installed in the image).

Requirements:
    pip install 'flyte[mcp]'

Usage:

    From the repo root:

    $ python examples/mcp/mcp_app_filtered.py

    For a generic MCP server without editing this file, use the packaged CLI and flags
    (see ``flyte-mcp --help``); allowlists are only configured via code as in this example.

    ------------------------------
    Connect from Claude Code
    ------------------------------
    Some agent harnesses can't reach `localhost` URLs. For local usage, prefer
    configuring Claude Code to launch the server via `uvx` (process-based setup).

    Add as a local stdio MCP server:
    $ claude mcp add --transport stdio flyte-mcp-filtered -- uvx --with "flyte[mcp]" flyte-mcp

    If you deploy this app remotely (so it has a public base URL), use that URL instead.
    With default ``transport="streamable-http"`` and ``mcp_mount_path="/flyte-mcp"``, use the
    MCP session URL ``https://<YOUR_HOST>/flyte-mcp/mcp``.

    $ claude mcp add --transport http flyte-mcp-filtered-remote https://<YOUR_HOST>/flyte-mcp/mcp

    If your remote deployment requires auth, add headers (example):
    $ claude mcp add --transport http \
      --header "Authorization: Bearer $TOKEN" \
      flyte-mcp-filtered-remote https://<YOUR_HOST>/flyte-mcp/mcp

    ------------------------------
    Connect from OpenCode
    ------------------------------

    For local usage (no `localhost` required), configure OpenCode to launch the
    server as a local MCP process:

    {
      "$schema": "https://opencode.ai/config.json",
      "mcp": {
        "flyte-mcp-filtered": {
          "type": "local",
          "command": ["uvx", "--with", "flyte[mcp]", "flyte-mcp", "--tool-groups", "task,run,script,search"],
          "enabled": true
        }
      }
    }

    For a remote deployment:

    {
      "$schema": "https://opencode.ai/config.json",
      "mcp": {
        "flyte-mcp-filtered": {
          "type": "remote",
          "url": "https://<YOUR_HOST>/flyte-mcp/mcp",
          "enabled": true,
          "headers": {
            "Authorization": "Bearer YOUR_TOKEN"
          }
        }
      }
    }
"""

import flyte
from flyte.ai.mcp import FlyteMCPAppEnvironment

# Bake search corpora into the app image (paths must match sdk_examples_path, docs_examples_path, full_docs_path).
image = (
    flyte.Image.from_debian_base()
    .with_apt_packages("ca-certificates", "git", "curl")
    .with_pip_packages("mcp", "starlette", "uvicorn")
    .with_commands(
        [
            "git clone --depth 1 https://github.com/flyteorg/flyte-sdk.git /root/flyte-sdk",
            "git clone --depth 1 https://github.com/unionai/unionai-examples.git /root/unionai-examples",
            "curl -fsSL https://www.union.ai/docs/v2/union/llms.txt -o /root/llms.txt",
        ]
    )
)

mcp_env = FlyteMCPAppEnvironment(
    name="restricted-mcp",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    transport="streamable-http",
    tool_groups=["task", "run", "script", "search"],
    task_allowlist=["my-project/my-task", "another-task"],
    # Search roots (see module docstring): clone/fetch above must populate these paths.
    sdk_examples_path="/root/flyte-sdk/examples",
    docs_examples_path="/root/unionai-examples/v2",
    full_docs_path="/root/llms.txt",
    instructions=(
        "This MCP server provides tools to run and monitor specific Flyte tasks, "
        "build and run UV scripts remotely, and search Flyte SDK/docs examples. "
        "Only allowlisted tasks can be accessed."
    ),
)

if __name__ == "__main__":
    flyte.init_from_config()
    app_handle = flyte.serve(mcp_env)
    app_handle.activate(wait=True)
    print(f"App is ready at {app_handle.endpoint}")
