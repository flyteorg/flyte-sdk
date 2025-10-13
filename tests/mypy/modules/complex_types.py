"""Complex type annotations test."""

from typing import Any

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
async def nested_types_task(data: dict[str, list[int]]) -> list[dict[str, int]]:
    """Task with nested complex types."""
    return [{k: sum(v)} for k, v in data.items()]


@env.task
async def any_type_task(data: Any) -> Any:
    """Task with Any types."""
    return data

@env.task
async def main() -> tuple[list[int], dict[str, int], int, list[dict[str, int]], Any]:
    o0 = await list_task([1, 2, 3])
    o1 = await dict_task({"a": 1, "b": 2})
    o2 = await optional_task(1, 2)
    o3 = await nested_types_task({"a": [1, 2, 3], "b": [4, 5, 6]})
    o4 = await any_type_task("hello")
    return o0, o1, o2, o3, o4
