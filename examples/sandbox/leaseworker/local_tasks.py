"""
Orchestrator that calls tasks defined in the same file
======================================================

``flyte.sandbox.orchestrator`` can call tasks defined locally with
``@env.task``, not only remote references. Nothing has to be deployed first::

    python local_tasks.py

When the orchestrator is run, the SDK builds the image and code bundle of the
local tasks, as it does for any ``flyte.run``, and ships their specs inside
the orchestrator's template. The orchestrator itself still starts without a
pod: a sandbox leaseworker runs its source and launches each task call as a
child action.

``mixed_pipeline`` combines both kinds. Its remote task must be deployed::

    flyte deploy math_tasks.py env

``queue`` must route to a sandbox leaseworker, and ``child_queue`` to a worker
that runs container tasks.
"""

from typing import Any, Callable

import flyte
import flyte.remote
import flyte.sandbox

env = flyte.TaskEnvironment(name="sandbox-local")


@env.task
def square(x: int) -> int:
    return x * x


@env.task
def describe(label: str, value: int) -> str:
    return f"{label} = {value}"


# A remote reference, resolved to a concrete version when the orchestrator runs.
add: Callable[..., Any] = flyte.remote.Task.get("sandbox-math.add", auto_version="latest")


@flyte.sandbox.orchestrator(queue="rust-1", child_queue="testcluster")
def local_pipeline(x: int) -> str:
    # Both calls are local tasks. Inside the sandbox each returns its result
    # directly, with no await.
    squared = square(x)
    return describe("square", squared)


@flyte.sandbox.orchestrator(queue="rust-1", child_queue="testcluster")
def mixed_pipeline(x: int, y: int) -> str:
    # `add` is deployed and fetched by the worker; `square` and `describe`
    # travel with the orchestrator.
    total = add(square(x), square(y))
    return describe("sum of squares", total)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(local_pipeline, x=7)
    print(run.url)
    run.wait()
    print(run.outputs())
