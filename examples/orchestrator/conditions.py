"""
Orchestrator that waits for a person
====================================

``flyte.orchestrator.task`` can pause until someone supplies a value from the
UI or the API, with ``flyte.new_condition`` and the condition's ``wait()``.
While it waits, nothing runs and no pod is held: a sandbox leaseworker keeps
only the orchestrator's paused interpreter, and a restarted worker picks the
wait up again from stored state.

Deploy the worker tasks first, then run::

    flyte deploy math_tasks.py env
    python conditions.py

The run pauses at each condition. Signal it from the run's page in the UI, or
from the CLI with the condition's action name, which the UI shows::

    flyte signal condition <run-name> <action-name> true
    flyte signal condition <run-name> <action-name> 3

or from Python::

    condition = flyte.remote.Condition.get("approve", run_name=run.name)
    condition.signal(True)
"""

from typing import Any, Callable

import flyte
import flyte.orchestrator
import flyte.remote

add: Callable[..., Any] = flyte.remote.Task.get("sandbox-math.add", auto_version="latest")
multiply: Callable[..., Any] = flyte.remote.Task.get("sandbox-math.multiply", auto_version="latest")


@flyte.orchestrator.task
def approved_total(x: int, y: int) -> int:
    total = add(x, y)
    approval = flyte.new_condition(
        "approve",
        prompt=f"The total is **{total}**. Scale it before returning?",
        prompt_type="markdown",
        timeout=24 * 60 * 60,
    )
    if not approval.wait():
        return total
    factor = flyte.new_condition("factor", prompt="Scale by how much?", data_type=int)
    return multiply(total, factor.wait())


@flyte.orchestrator.task
def approval_or_default(x: int) -> str:
    # Only the function's source is shipped, so it imports what it uses.
    from datetime import timedelta

    approval = flyte.new_condition("approve", prompt=f"Use {x}?", timeout=timedelta(minutes=5))
    try:
        return f"approved: {approval.wait()}"
    except flyte.errors.ConditionTimedoutError:
        # Raised when nobody signals the condition in time.
        return "no answer, used the default"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(approved_total, x=3, y=4)
    print(run.url)
    run.wait()
    print("approved_total:", run.outputs())
