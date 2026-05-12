"""Test data file for tuple and NamedTuple CLI input tests."""

from dataclasses import dataclass
from typing import List, NamedTuple

import flyte

env = flyte.TaskEnvironment(name="tuple_inputs")


# ============================================================================
# Simple tuple types
# ============================================================================


@env.task
async def process_simple_tuple(data: tuple[int, str, float]) -> str:
    """Task that takes a simple tuple input."""
    num, text, decimal = data
    return f"Number: {num}, Text: {text}, Decimal: {decimal}"


@env.task
async def process_nested_tuple(data: tuple[tuple[int, int], str]) -> int:
    """Task that takes a nested tuple input."""
    inner, _ = data
    return inner[0] + inner[1]


# ============================================================================
# Tuple with complex types
# ============================================================================


@dataclass
class Point:
    """A simple 2D point."""

    x: float
    y: float


@env.task
async def process_tuple_with_dataclass(data: tuple[Point, Point, str]) -> float:
    """Task that takes a tuple with dataclass elements."""
    p1, p2, _ = data
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


@env.task
async def process_tuple_with_list(data: tuple[List[int], str]) -> int:
    """Task that takes a tuple containing a list."""
    numbers, _ = data
    return sum(numbers)


# ============================================================================
# NamedTuple types
# ============================================================================


class Coordinates(NamedTuple):
    """A geographic coordinate."""

    latitude: float
    longitude: float
    altitude: float = 0.0


class PersonInfo(NamedTuple):
    """Information about a person."""

    name: str
    age: int
    email: str


class ModelMetrics(NamedTuple):
    """Machine learning model metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float


@env.task
async def process_coordinates(coords: Coordinates) -> str:
    """Task that takes a Coordinates NamedTuple input."""
    return f"Location: ({coords.latitude}, {coords.longitude}) at {coords.altitude}m elevation"


@env.task
async def process_person(person: PersonInfo) -> str:
    """Task that takes a PersonInfo NamedTuple input."""
    return f"Hello, {person.name}! You are {person.age} years old. Email: {person.email}"


@env.task
async def process_metrics(metrics: ModelMetrics) -> str:
    """Task that takes a ModelMetrics NamedTuple input."""
    return (
        f"Accuracy: {metrics.accuracy:.2%}, "
        f"Precision: {metrics.precision:.2%}, "
        f"Recall: {metrics.recall:.2%}, "
        f"F1: {metrics.f1_score:.2%}"
    )


# ============================================================================
# Nested NamedTuple types
# ============================================================================


@dataclass
class Address:
    """A physical address."""

    street: str
    city: str
    country: str


class Employee(NamedTuple):
    """Employee information with nested types."""

    info: PersonInfo
    department: str
    address: Address


@env.task
async def process_employee(emp: Employee) -> str:
    """Task that takes a nested NamedTuple with dataclass."""
    return f"{emp.info.name} works in {emp.department} at {emp.address.city}, {emp.address.country}"
