"""Test data file for TypedDict CLI input tests."""

from dataclasses import dataclass
from typing import List, TypedDict

import flyte

env = flyte.TaskEnvironment(name="typeddict_inputs")


# ============================================================================
# Simple TypedDict types
# ============================================================================


class Coordinates(TypedDict):
    """A geographic coordinate."""

    latitude: float
    longitude: float
    altitude: float


class PersonInfo(TypedDict):
    """Information about a person."""

    name: str
    age: int
    email: str


class ModelMetrics(TypedDict):
    """Machine learning model metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float


@env.task
async def process_coordinates(coords: Coordinates) -> str:
    """Task that takes a Coordinates TypedDict input."""
    return f"Location: ({coords['latitude']}, {coords['longitude']}) at {coords['altitude']}m elevation"


@env.task
async def process_person(person: PersonInfo) -> str:
    """Task that takes a PersonInfo TypedDict input."""
    return f"Hello, {person['name']}! You are {person['age']} years old. Email: {person['email']}"


@env.task
async def process_metrics(metrics: ModelMetrics) -> str:
    """Task that takes a ModelMetrics TypedDict input."""
    return (
        f"Accuracy: {metrics['accuracy']:.2%}, "
        f"Precision: {metrics['precision']:.2%}, "
        f"Recall: {metrics['recall']:.2%}, "
        f"F1: {metrics['f1_score']:.2%}"
    )


# ============================================================================
# TypedDict with complex nested types
# ============================================================================


@dataclass
class Address:
    """A physical address."""

    street: str
    city: str
    country: str


class EmployeeInfo(TypedDict):
    """Employee information with nested types."""

    name: str
    age: int
    department: str
    address: Address


@env.task
async def process_employee(emp: EmployeeInfo) -> str:
    """Task that takes an EmployeeInfo TypedDict with nested dataclass."""
    return f"{emp['name']} works in {emp['department']} at {emp['address'].city}, {emp['address'].country}"


# ============================================================================
# TypedDict with list fields
# ============================================================================


class TeamInfo(TypedDict):
    """Team information with list of members."""

    team_name: str
    members: List[str]
    scores: List[float]


@env.task
async def process_team(team: TeamInfo) -> str:
    """Task that takes a TeamInfo TypedDict with list fields."""
    avg_score = sum(team["scores"]) / len(team["scores"]) if team["scores"] else 0
    return f"Team {team['team_name']} has {len(team['members'])} members with avg score {avg_score:.2f}"


# ============================================================================
# Nested TypedDict types
# ============================================================================


class InnerConfig(TypedDict):
    """Inner configuration."""

    enabled: bool
    threshold: float


class OuterConfig(TypedDict):
    """Outer configuration with nested TypedDict."""

    name: str
    inner: InnerConfig


@env.task
async def process_nested_typeddict(config: OuterConfig) -> str:
    """Task that takes a nested TypedDict input."""
    status = "enabled" if config["inner"]["enabled"] else "disabled"
    return f"Config '{config['name']}' is {status} with threshold {config['inner']['threshold']}"


# ============================================================================
# TypedDict with dict fields
# ============================================================================


class ConfigWithDict(TypedDict):
    """Configuration with dict fields."""

    name: str
    settings: dict
    labels: dict


@env.task
async def process_typeddict_with_dict(config: ConfigWithDict) -> str:
    """Task that takes a TypedDict with dict fields."""
    return f"Config '{config['name']}' has {len(config['settings'])} settings and {len(config['labels'])} labels"
