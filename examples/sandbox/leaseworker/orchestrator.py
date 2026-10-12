"""
Orchestrator that runs on a sandbox leaseworker
===============================================

``flyte.sandbox.orchestrator`` ships this function's *source* in the task
template. A sandbox leaseworker runs it directly -- no image is built and no
pod starts for the orchestrator -- and launches each task call as a child
action.

Deploy the worker tasks first, then run::

    flyte deploy math_tasks.py env
    python orchestrator.py

The scheduler sends the orchestrator to a sandbox leaseworker by its task
type, on whatever queue the run uses, and the tasks it calls run on the same
queue on the cluster's usual leaseworker.

The tasks here are remote references, so nothing is built and the run is
submitted straight away. ``local_tasks.py`` shows an orchestrator that calls
tasks defined in the same file.
"""

from typing import Any, Callable

import flyte
import flyte.remote
import flyte.sandbox

# Lazy handles, resolved to a concrete version when the orchestrator is run.
# Inside the sandbox each call returns the task's result directly (no await).
add: Callable[..., Any] = flyte.remote.Task.get("sandbox-math.add", auto_version="latest")
multiply: Callable[..., Any] = flyte.remote.Task.get("sandbox-math.multiply", auto_version="latest")
fail_if_negative: Callable[..., Any] = flyte.remote.Task.get("sandbox-math.fail_if_negative", auto_version="latest")


@flyte.sandbox.orchestrator
def pipeline(x: int, y: int) -> int:
    total = add(x, y)
    return multiply(total, 2)


@flyte.sandbox.orchestrator
def sum_of_squares(numbers: list[int]) -> int:
    total = 0
    for n in numbers:
        total = add(total, multiply(n, n))
    return total


@flyte.sandbox.orchestrator
def tolerant(x: int) -> str:
    # A failed task raises in the sandbox, where it can be handled.
    try:
        checked = fail_if_negative(x)
        return "ok " + str(checked)
    except RuntimeError as e:
        return "recovered: " + str(e)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pipeline, x=3, y=4)
    print(run.url)
    run.wait()
    print(run.outputs())
