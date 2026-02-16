"""Test data file for TypedDict CLI input tests."""

import sys
from dataclasses import dataclass
from typing import List

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing_extensions import NotRequired

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


# ============================================================================
# TypedDict with NotRequired fields
# Tests that NotRequired[T] is properly unwrapped for Pydantic validation
# and that optional fields not provided are absent from output (not None)
# ============================================================================


class ToolCall(TypedDict):
    """A tool call made by an AI assistant."""

    name: str
    args: dict


class AIResponse(TypedDict):
    """Response from an AI assistant with optional tool_calls.

    The tool_calls field uses NotRequired to mark it as optional.
    This tests that NotRequired[T] is properly unwrapped when passed to Pydantic.
    """

    content: str
    role: str
    tool_calls: NotRequired[List[ToolCall]]


@env.task
async def process_ai_response(response: AIResponse) -> str:
    """Task that takes an AIResponse TypedDict with NotRequired field.

    Tests:
    1. NotRequired[T] is properly unwrapped (no PydanticSchemaGenerationError)
    2. Optional fields not provided are absent (not None)
    """
    # This check verifies that tool_calls is absent when not provided, not set to None
    if "tool_calls" in response:
        tool_names = [tc["name"] for tc in response["tool_calls"]]
        return f"{response['role']}: {response['content']} [tools: {', '.join(tool_names)}]"
    else:
        return f"{response['role']}: {response['content']}"


class UserProfile(TypedDict):
    """User profile with multiple NotRequired fields."""

    username: str
    email: str
    display_name: NotRequired[str]
    bio: NotRequired[str]
    age: NotRequired[int]


@env.task
async def process_user_profile(profile: UserProfile) -> str:
    """Task that takes a UserProfile with multiple NotRequired fields."""
    parts = [f"User: {profile['username']} ({profile['email']})"]

    # Check each optional field - should be absent, not None
    if "display_name" in profile:
        parts.append(f"Display: {profile['display_name']}")
    if "bio" in profile:
        parts.append(f"Bio: {profile['bio']}")
    if "age" in profile:
        parts.append(f"Age: {profile['age']}")

    return " | ".join(parts)
