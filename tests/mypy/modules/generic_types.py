"""Generic and TypeVar test."""

from typing import TypeVar

from flyte import TaskEnvironment

env = TaskEnvironment(name="generic_env")

T = TypeVar("T")


@env.task
async def identity_task(x: T) -> T:
    """Identity task with concrete types."""
    return x


@env.task
async def passthrough_list(items: list[T]) -> list[T]:
    """Passthrough for lists."""
    return items


@env.task
async def transform_list(items: list[T]) -> list[str]:
    """Transform list from one type to another."""
    return [str(x) for x in items]


@env.task
async def call_functions(x: T) -> tuple[T, list[T], list[str]]:
    o0 = await identity_task(x)
    o1 = await passthrough_list([x, x, x])
    o2 = await transform_list([x, x, x])
    return o0, o1, o2


@env.task
async def main() -> tuple[int, list[int], list[str]]:
    return await call_functions(1)
