"""Generic and TypeVar test."""
from typing import TypeVar

from flyte import TaskEnvironment

env = TaskEnvironment(name="generic_env")

T = TypeVar("T")


@env.task
async def identity_task(x: int) -> int:
    """Identity task with concrete types."""
    return x


@env.task
async def passthrough_list(items: list[str]) -> list[str]:
    """Passthrough for lists."""
    return items


@env.task
async def transform_list(items: list[int]) -> list[str]:
    """Transform list from one type to another."""
    return [str(x) for x in items]
