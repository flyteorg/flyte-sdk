"""Internal helper :class:`flyte.TaskEnvironment` for the Flyte MCP server.

This module defines two tasks consumed by
:class:`flyte.ai.mcp.FlyteMCPAppEnvironment`'s remote UV-script tools:

- ``flyte_mcp_tasks.build_image`` — runs the user's UV script with ``--build``
  to kick off a remote image build.
- ``flyte_mcp_tasks.run_task`` — runs the user's UV script (no ``--build``)
  to dispatch the script's ``main`` task on the remote Flyte cluster.

This module lives inside the ``flyte`` package itself so no extra plugin needs
to be installed in the helper-task image — installing ``flyte`` is enough.

Deploy once to your Flyte cluster before serving the MCP app::

    python -m flyte.ai.mcp.tasks

Or programmatically::

    import flyte
    from flyte.ai.mcp.tasks import env

    flyte.init_from_config()
    deployments = flyte.deploy(env)
    print(deployments[0].table_repr())

The task names and version produced by this deployment are the defaults that
:class:`FlyteMCPAppEnvironment` already looks up
(``uv_script_build_task_name="flyte_mcp_tasks.build_image"``,
``uv_script_run_task_name="flyte_mcp_tasks.run_task"``), so no extra wiring is
required on the MCP server side.

Auth: each invocation receives the caller's ``Authorization`` header via a
per-request Flyte secret mounted as ``FLYTE_PASSTHROUGH_API_KEY``. The script
template uses that env var to authenticate back to the Flyte cluster.
"""

from __future__ import annotations

import flyte

from ._uv_script_utils import (
    RunResult,
    build_script_image_,
    run_script_remote_,
)

__all__ = ["RunResult", "build_image", "env", "run_task"]


env = flyte.TaskEnvironment(
    name="_flyte_mcp_app_env_tasks",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=(
        flyte.Image.from_debian_base(
            flyte_version=flyte.version(),
            name="flyte-mcp-tasks-image",
        ).with_apt_packages("ca-certificates", "git")
    ),
    reusable=flyte.ReusePolicy(
        replicas=(0, 4),
        concurrency=4,
        idle_ttl=30,
    ),
)


@env.task(cache="auto", retries=3)
async def build_image(script: str, tail: int = 50) -> RunResult:
    """Build the container image for a user-supplied Flyte UV script.

    Writes ``script`` to disk inside this task's container and invokes it with
    ``--build``. The script template (see ``flyte.ai.mcp._flyte_mcp_app.UV_SCRIPT_FORMAT``)
    handles ``flyte.init_passthrough`` and ``flyte.build`` itself.
    """
    return await build_script_image_(script, tail=tail)


@env.task(cache="auto", retries=3)
async def run_task(script: str, tail: int = 50) -> RunResult:
    """Execute a user-supplied Flyte UV script and return its stdout/stderr.

    Writes ``script`` to disk inside this task's container and invokes it with
    no extra args, so the script's ``main`` task is dispatched remotely.
    """
    return await run_script_remote_(script, tail=tail)
