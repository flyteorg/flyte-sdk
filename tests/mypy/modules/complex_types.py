"""Complex type annotations test."""
from typing import Any, Dict, List, Optional

from flyte import TaskEnvironment

env = TaskEnvironment(name="complex_types_env")


@env.task
async def list_task(items: list[int]) -> list[int]:
    """Task with list types."""
    return [x * 2 for x in items]


@env.task
async def dict_task(data: dict[str, int]) -> dict[str, int]:
    """Task with dict types."""
    return {k: v * 2 for k, v in data.items()}


@env.task
async def optional_task(x: int, y: int | None = None) -> int:
    """Task with optional parameters."""
    if y is None:
        return x
    return x + y


@env.task
async def nested_types_task(
    data: dict[str, list[int]]
) -> list[dict[str, int]]:
    """Task with nested complex types."""
    return [{k: sum(v)} for k, v in data.items()]


@env.task
async def any_type_task(data: Any) -> Any:
    """Task with Any types."""
    return data

