"""Error handling and diagnostics — what to use when something goes wrong.

The reliable surface for "I want to see what the script printed" is
**declared output collectors** — :class:`Stdout` and :class:`Stderr`. They
turn the script's streams into typed outputs that flow through Flyte's
data plane like any other value, so they work in every execution mode
(local or remote, workflow or standalone).

When an output collector can't find its expected file (e.g. the script
failed before writing), the resulting error carries the captured
``stderr`` and ``returncode`` — useful for the common "the script ran but
produced nothing" failure mode. You'll see this in the task's failure
message in the Flyte UI / logs.

``debug=True`` on ``shell.create()`` makes the container print the
rendered bash to stderr *before* running. Combined with a declared
:class:`Stderr` collector, you can pipe that dump into your workflow for
post-mortem inspection.

Run locally::

    uv run python 09_error_handling.py
"""

import sys
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.extras import shell
from flyte.extras.shell import Stderr, Stdout

# Captures both streams as typed outputs — the canonical pattern for
# making a tool's output visible to downstream workflow tasks.
echo_with_diagnostics = shell.create(
    name="echo_with_diagnostics",
    image="debian:12-slim",
    inputs={"msg": str},
    outputs={
        "out": Stdout(type=str),
        "err": Stderr(type=str),
    },
    script=r"""
        echo "stdout: {inputs.msg}"
        echo "stderr: also writing some logs" >&2
    """,
)


# Same shape but with debug=True. The container prints the rendered script
# to its own stderr before running, and the declared Stderr() captures it.
echo_with_debug_dump = shell.create(
    name="echo_with_debug_dump",
    image="debian:12-slim",
    inputs={"msg": str},
    outputs={
        "out": Stdout(type=str),
        "err": Stderr(type=str),
    },
    script=r"""
        echo "running: {inputs.msg}"
    """,
    debug=True,
    cache="disable",
)


env = flyte.TaskEnvironment(
    name="shell_error_handling",
    depends_on=[echo_with_diagnostics.env, echo_with_debug_dump.env],
    image=(
        flyte.Image.from_debian_base().clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent.parent / "dist",
                package_name="flyte",
            ),
            name="shell-basic",
        )
    ),
)


@env.task
async def inspect_streams(msg: str) -> tuple[str, str]:
    """Surface stdout and stderr as workflow values."""
    return await echo_with_diagnostics(msg=msg)


@env.task
async def inspect_debug_dump(msg: str) -> tuple[str, str]:
    """``debug=True`` adds the rendered script to stderr — visible via the
    declared :class:`Stderr` collector."""
    return await echo_with_debug_dump(msg=msg)


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    print("\n--- inspect_streams ---")
    run = flyte.with_runcontext(mode=mode).run(inspect_streams, "hello")
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")

    print("\n--- inspect_debug_dump ---")
    run2 = flyte.with_runcontext(mode=mode).run(inspect_debug_dump, "hello")
    print(run2.url if mode == "remote" else run2)
    print(f"Output: {run2.outputs()}")
