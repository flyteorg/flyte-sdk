"""Pydantic v2 discriminated (tagged) unions as task inputs.

``Annotated[Union[A, B], Field(discriminator=...)]`` is a common Pydantic v2
pattern that emits a JSON schema using ``oneOf`` plus a top-level
``discriminator`` object (no top-level ``type`` and no ``anyOf``). This
example exercises that pattern end-to-end through ``flyte.run`` so the
type engine has to roundtrip the union through serialization and reconstruct
the right variant on the receiving side.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

import flyte

env = flyte.TaskEnvironment(
    name="inputs_pydantic_union_types",
    image=flyte.Image.from_debian_base(),
)


class Circle(BaseModel):
    kind: Literal["circle"] = "circle"
    color: str = ""
    radius: float = 0.0


class Rectangle(BaseModel):
    kind: Literal["rectangle"] = "rectangle"
    color: str = ""
    width: float = 0.0
    height: float = 0.0


Shape = Annotated[Union[Circle, Rectangle], Field(discriminator="kind")]


class Properties(BaseModel):
    shape: Shape


class ShapeReport(BaseModel):
    kind: str
    color: str
    area: float


@env.task
def describe_shape(props: Properties) -> ShapeReport:
    """Receive a Pydantic model whose field is a discriminated union."""
    shape = props.shape
    if isinstance(shape, Circle):
        area = 3.141592653589793 * shape.radius * shape.radius
    else:
        area = shape.width * shape.height
    return ShapeReport(kind=shape.kind, color=shape.color, area=area)


@env.task
def main(props: Properties = Properties(shape=Rectangle(color="blue", width=4.0, height=2.5))) -> ShapeReport:
    return describe_shape(props=props)


if __name__ == "__main__":
    flyte.init_from_config()

    # Run with the default (Rectangle) shape.
    print("Testing with Rectangle:")
    r1 = flyte.run(main)
    print(r1.name)
    print(r1.url)
    r1.wait()

    # Run with a Circle to exercise the other variant of the discriminated union.
    print("\nTesting with Circle:")
    r2 = flyte.run(main, props=Properties(shape=Circle(color="red", radius=2.0)))
    print(r2.name)
    print(r2.url)
    r2.wait()
