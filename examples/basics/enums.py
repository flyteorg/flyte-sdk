import enum
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

import flyte

env = flyte.TaskEnvironment(name="enums")


# -- Enum definitions ----------------------------------------------------------


class Color(str, enum.Enum):
    RED = "red-value"
    GREEN = "green-value"
    BLUE = "blue-value"


class Size(StrEnum):
    SMALL = "sm-value"
    MEDIUM = "md-value"
    LARGE = "lg-value"
    EXTRA_LARGE = "xl-value"


class Priority(str, enum.Enum):
    LOW = "low-value"
    MEDIUM = "medium-value"
    HIGH = "high-value"
    CRITICAL = "critical-value"


# -- Pydantic models with enums -----------------------------------------------


class ShirtOrder(BaseModel):
    color: Color
    size: Size
    quantity: int


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class Customer(BaseModel):
    name: str
    priority: Priority
    address: Address


class FullOrder(BaseModel):
    customer: Customer
    items: list[ShirtOrder]
    notes: Optional[str] = None


# -- Tasks ---------------------------------------------------------------------


@env.task
async def standalone_enum_echo(color: Color, size: Size) -> str:
    """Standalone enums as direct task inputs."""
    return f"color={color.name}({color.value}), size={size.name}({size.value})"


@env.task
async def simple_pydantic_enum(order: ShirtOrder) -> str:
    """Enum fields inside a flat Pydantic model."""
    return f"{order.quantity}x {order.color.name} shirt, size {order.size.name}"


@env.task
async def nested_pydantic_enum(order: FullOrder) -> str:
    """Enums inside nested Pydantic models."""
    items_desc = ", ".join(f"{item.quantity}x {item.color.name}-{item.size.name}" for item in order.items)
    return f"Order for {order.customer.name} (priority={order.customer.priority.name}): {items_desc}"


@env.task
async def main() -> list[str]:
    results = []

    # 1. Standalone enums
    r = await standalone_enum_echo(color=Color.RED, size=Size.LARGE)
    results.append(r)

    # 2. Simple pydantic with enums
    r = await simple_pydantic_enum(order=ShirtOrder(color=Color.BLUE, size=Size.MEDIUM, quantity=3))
    results.append(r)

    # 3. Nested pydantic with enums
    order = FullOrder(
        customer=Customer(
            name="Alice",
            priority=Priority.HIGH,
            address=Address(street="123 Main St", city="Springfield", zip_code="62704"),
        ),
        items=[
            ShirtOrder(color=Color.RED, size=Size.SMALL, quantity=2),
            ShirtOrder(color=Color.GREEN, size=Size.EXTRA_LARGE, quantity=1),
        ],
        notes="Gift wrap please",
    )
    r = await nested_pydantic_enum(order=order)
    results.append(r)

    for line in results:
        print(line)

    return results


if __name__ == "__main__":
    flyte.init_from_config()
    # run = flyte.run(main)
    run = flyte.run(standalone_enum_echo, color=Color.RED, size=Size.LARGE)
    print(run.url)
