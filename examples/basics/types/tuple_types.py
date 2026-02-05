"""
Example demonstrating typed tuple inputs and outputs in Flyte tasks.

Typed tuples like tuple[int, str, float] are now supported in Flyte and can be used
as task inputs and outputs. The underlying implementation uses Pydantic to serialize
and deserialize the tuple data.

Note: Untyped tuples (bare `tuple` without type parameters) are still not supported.
"""

from dataclasses import dataclass

import flyte


env = flyte.TaskEnvironment(
    name="tuple_types_example",
    image=flyte.Image.from_debian_base(),
)


@env.task
async def create_tuple() -> tuple[int, str, float]:
    """Create and return a typed tuple."""
    return (42, "hello", 3.14)


@env.task
async def process_tuple(data: tuple[int, str, float]) -> str:
    """Process a typed tuple input."""
    num, text, decimal = data
    return f"Number: {num}, Text: {text}, Decimal: {decimal}"


@env.task
async def nested_tuple() -> tuple[tuple[int, int], str]:
    """Create a nested tuple."""
    return ((1, 2), "nested")


@env.task
async def process_nested_tuple(data: tuple[tuple[int, int], str]) -> int:
    """Process a nested tuple and return the sum of the inner tuple."""
    inner, _ = data
    return inner[0] + inner[1]


@dataclass
class Point:
    """A simple 2D point."""

    x: float
    y: float


@env.task
async def tuple_with_dataclass() -> tuple[Point, Point, str]:
    """Create a tuple containing dataclass instances."""
    return (Point(x=0.0, y=0.0), Point(x=1.0, y=1.0), "line segment")


@env.task
async def process_tuple_with_dataclass(data: tuple[Point, Point, str]) -> float:
    """Calculate the distance between two points in a tuple."""
    p1, p2, _ = data
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


@env.task
async def tuple_workflow() -> tuple[
    tuple[int, str, float],
    str,
    tuple[tuple[int, int], str],
    int,
    tuple[Point, Point, str],
    float,
]:
    """Workflow demonstrating tuple type usage."""
    # Simple tuple
    t = await create_tuple()
    result = await process_tuple(data=t)
    print(f"Simple tuple result: {result}")

    # Nested tuple
    nt = await nested_tuple()
    nested_result = await process_nested_tuple(data=nt)
    print(f"Nested tuple sum: {nested_result}")

    # Tuple with dataclass
    dc_tuple = await tuple_with_dataclass()
    distance = await process_tuple_with_dataclass(data=dc_tuple)
    print(f"Distance between points: {distance}")

    return t, result, nt, nested_result, dc_tuple, distance


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running tuple workflow...")
    run = flyte.run(tuple_workflow)
    print(f"Run URL: {run.url}")
    run.wait()
    print("Tuple workflow completed!")
    outputs = run.outputs()
    print(f"Outputs: {outputs}")
