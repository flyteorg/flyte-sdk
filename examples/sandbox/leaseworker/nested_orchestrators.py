"""
Orchestrators that call only other orchestrators
================================================

Every function here is a ``flyte.sandbox.orchestrator``, so the whole run
happens inside sandbox leaseworkers: no task container starts, no image is
built and nothing is deployed first. Each call to another orchestrator is
still a separate child action, with its own entry in the UI::

    python nested_orchestrators.py

The leaves do plain Python. ``report`` calls ``sum_of_squares``, which calls
``square`` once per number, so ``report(numbers=[1, 2, 3])`` makes five
actions in three levels. An orchestrator has to be defined before one that
calls it, because the calls are found when the caller is decorated.

``queue`` must route to a sandbox leaseworker. Child orchestrators are sent
back the same way.
"""

import flyte
import flyte.sandbox


@flyte.sandbox.orchestrator(queue="rust-1")
def square(x: int) -> int:
    return x * x


@flyte.sandbox.orchestrator(queue="rust-1")
def sum_of_squares(numbers: list[int]) -> int:
    total = 0
    for n in numbers:
        total = total + square(n)
    return total


@flyte.sandbox.orchestrator(queue="rust-1")
def report(numbers: list[int]) -> str:
    return f"the squares of {numbers} add up to {sum_of_squares(numbers)}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(report, numbers=[1, 2, 3])
    print(run.url)
    run.wait()
    print(run.outputs())
