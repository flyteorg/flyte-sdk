"""
Example demonstrating TypedDict inputs and outputs in Flyte tasks.

TypedDicts are now supported in Flyte and can be used as task inputs and outputs.
The underlying implementation uses Pydantic to serialize and deserialize the TypedDict data.

TypedDicts provide a convenient way to define typed dictionary structures with
type safety and IDE autocompletion support.
"""

from dataclasses import dataclass
from typing import List, TypedDict

import flyte

env = flyte.TaskEnvironment(name="typeddict_types_example")


# Define TypedDicts using the class syntax (recommended)
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
async def create_coordinates() -> Coordinates:
    """Create and return a Coordinates TypedDict."""
    return Coordinates(latitude=37.7749, longitude=-122.4194, altitude=16.0)


@env.task
async def process_coordinates(coords: Coordinates) -> str:
    """Process a Coordinates TypedDict input."""
    return f"Location: ({coords['latitude']}, {coords['longitude']}) at {coords['altitude']}m elevation"


@env.task
async def create_person() -> PersonInfo:
    """Create and return a PersonInfo TypedDict."""
    return PersonInfo(name="Alice", age=30, email="alice@example.com")


@env.task
async def greet_person(person: PersonInfo) -> str:
    """Generate a greeting for a person."""
    return f"Hello, {person['name']}! You are {person['age']} years old."


@env.task
async def calculate_metrics(predictions: List[int], labels: List[int]) -> ModelMetrics:
    """Calculate model metrics from predictions and labels."""
    # Simple metric calculation for demonstration
    correct = sum(1 for p, label in zip(predictions, labels) if p == label)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    # Simplified metrics (in practice, use sklearn or similar)
    return ModelMetrics(
        accuracy=accuracy,
        precision=accuracy * 0.95,  # Simplified
        recall=accuracy * 0.9,  # Simplified
        f1_score=accuracy * 0.92,  # Simplified
    )


@env.task
async def report_metrics(metrics: ModelMetrics) -> str:
    """Generate a report from model metrics."""
    return (
        f"Model Performance:\n"
        f"  Accuracy: {metrics['accuracy']:.2%}\n"
        f"  Precision: {metrics['precision']:.2%}\n"
        f"  Recall: {metrics['recall']:.2%}\n"
        f"  F1 Score: {metrics['f1_score']:.2%}"
    )


# TypedDict with complex nested types
@dataclass
class Address:
    """A physical address."""

    street: str
    city: str
    country: str


class Employee(TypedDict):
    """Employee information with nested types."""

    info: PersonInfo
    department: str
    address: Address


class Outputs(TypedDict):
    """Outputs for a workflow."""

    employee: Employee
    location: str
    person: str
    metrics: str
    description: str


@env.task
async def create_employee() -> Employee:
    """Create an employee with nested TypedDict and dataclass."""
    return Employee(
        info=PersonInfo(name="Bob", age=35, email="bob@company.com"),
        department="Engineering",
        address=Address(street="123 Main St", city="San Francisco", country="USA"),
    )


@env.task
async def describe_employee(emp: Employee) -> str:
    """Generate a description of an employee."""
    return f"{emp['info']['name']} works in {emp['department']} at {emp['address'].city}, {emp['address'].country}"


@env.task
async def typeddict_workflow() -> Outputs:
    """Workflow demonstrating TypedDict type usage."""
    # Simple TypedDict
    coords = await create_coordinates()
    location_str = await process_coordinates(coords=coords)
    print(f"Coordinates: {location_str}")

    # Person TypedDict
    person = await create_person()
    greeting = await greet_person(person=person)
    print(f"Person: {greeting}")

    # Model metrics
    predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    labels = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    metrics = await calculate_metrics(predictions=predictions, labels=labels)
    report = await report_metrics(metrics=metrics)
    print(f"Metrics:\n{report}")

    # Nested TypedDict with dataclass
    employee = await create_employee()
    description = await describe_employee(emp=employee)
    print(f"Employee: {description}")

    return Outputs(employee=employee, location=location_str, person=greeting, metrics=report, description=description)


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running TypedDict workflow...")
    run = flyte.run(typeddict_workflow)
    print(f"Run URL: {run.url}")
    run.wait()
    print("TypedDict workflow completed!")
    outputs = run.outputs()
    print(f"Outputs: {outputs}")
