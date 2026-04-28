"""Helper tasks used by notebook examples."""

import flyte

env = flyte.TaskEnvironment("tasks")


@env.task
def double(n: int) -> int:
    return n * 2


@env.task
def add(a: float, b: float) -> float:
    return a + b


@env.task
async def async_double(n: int) -> int:
    return n * 2


@env.task
async def async_add(a: float, b: float) -> float:
    return a + b
