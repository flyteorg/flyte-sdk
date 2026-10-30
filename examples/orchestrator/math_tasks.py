"""
Worker tasks for the leaseworker orchestrator example
=====================================================

Ordinary container tasks. Deploy them once::

    flyte deploy math_tasks.py env

``orchestrator.py`` references them with ``flyte.remote.Task.get()``.
"""

import flyte

env = flyte.TaskEnvironment(name="sandbox-math")


@env.task
def add(a: int, b: int) -> int:
    return a + b


@env.task
def multiply(a: int, b: int) -> int:
    return a * b


@env.task
def fail_if_negative(x: int) -> int:
    if x < 0:
        raise ValueError(f"{x} is negative")
    return x
