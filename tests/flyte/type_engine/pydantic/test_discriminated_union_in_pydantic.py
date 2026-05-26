"""Regression tests for Pydantic v2 tagged (discriminated) unions.

Pydantic v2 emits JSON schemas for ``Annotated[Union[A, B], Field(discriminator=...)]``
using ``oneOf`` and a top-level ``discriminator`` object instead of ``anyOf``. The
type engine previously assumed any non-``$ref`` / non-``anyOf`` / non-``enum``
property had a top-level ``type`` key, which crashed with ``KeyError: 'type'`` on
discriminated unions and made tasks with such inputs uninvocable.
"""

from __future__ import annotations

import typing
from typing import Annotated, List, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from flyte.types import TypeEngine
from flyte.types._type_engine import (
    _create_pydantic_model_from_schema,
    _DiscriminatedUnion,
    _get_element_type,
    _get_pydantic_element_type,
    convert_mashumaro_json_schema_to_python_class,
    generate_attribute_list_from_dataclass_json_mixin,
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


class OptionalShapeProperties(BaseModel):
    shape: Optional[Shape] = None


class ListOfShapes(BaseModel):
    shapes: List[Shape] = Field(default_factory=list)


def test_oneof_schema_does_not_raise_key_error():
    """Reproducer for the KeyError reported by customers using discriminated unions."""
    schema = Properties.model_json_schema()
    attribute_list, _ = generate_attribute_list_from_dataclass_json_mixin(schema, "Properties")
    assert any(entry[0] == "shape" for entry in attribute_list)


def test_oneof_resolves_to_union_type():
    """The shape field should resolve to a Python Union of the variant classes."""
    schema = Properties.model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "Properties")

    import dataclasses

    shape_field = next(f for f in dataclasses.fields(cls) if f.name == "shape")
    origin = typing.get_origin(shape_field.type)
    assert origin is Union
    variants = typing.get_args(shape_field.type)
    assert len(variants) == 2
    variant_names = {v.__name__ for v in variants}
    assert variant_names == {"Circle", "Rectangle"}


def test_oneof_dispatches_on_discriminator():
    """Constructing the generated dataclass from a dict should pick the right variant."""
    schema = Properties.model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "Properties")

    circle_instance = cls(shape={"kind": "circle", "color": "red", "radius": 1.5})
    assert type(circle_instance.shape).__name__ == "Circle"
    assert circle_instance.shape.radius == 1.5

    rect_instance = cls(shape={"kind": "rectangle", "color": "blue", "width": 4.0, "height": 2.0})
    assert type(rect_instance.shape).__name__ == "Rectangle"
    assert rect_instance.shape.width == 4.0
    assert rect_instance.shape.height == 2.0


def test_discriminator_metadata_captured():
    """The internal nested-type descriptor should carry the discriminator property name and mapping."""
    schema = Properties.model_json_schema()
    _, nested_types = generate_attribute_list_from_dataclass_json_mixin(schema, "Properties")
    descriptor = nested_types["shape"]
    assert isinstance(descriptor, _DiscriminatedUnion)
    assert descriptor.discriminator_property == "kind"
    assert set(descriptor.mapping.keys()) == {"circle", "rectangle"}
    assert {cls.__name__ for cls in descriptor.variants} == {"Circle", "Rectangle"}


def test_get_element_type_handles_oneof():
    """_get_element_type should resolve oneOf to a Union[...] type."""
    schema = Properties.model_json_schema()
    prop = schema["properties"]["shape"]
    result = _get_element_type(prop, schema)
    assert typing.get_origin(result) is Union
    assert len(typing.get_args(result)) == 2


def test_get_pydantic_element_type_handles_oneof():
    """_get_pydantic_element_type should also resolve oneOf to a Union[...] type."""
    schema = Properties.model_json_schema()
    prop = schema["properties"]["shape"]
    result = _get_pydantic_element_type(prop, schema)
    assert typing.get_origin(result) is Union


def test_dynamic_pydantic_model_validates_discriminated_union():
    """The dynamic Pydantic model created from a schema should validate discriminated input."""
    schema = Properties.model_json_schema()
    dynamic_model = _create_pydantic_model_from_schema(schema)

    inst = dynamic_model.model_validate({"shape": {"kind": "circle", "color": "red", "radius": 1.5}})
    assert type(inst.shape).__name__ == "Circle"

    inst = dynamic_model.model_validate({"shape": {"kind": "rectangle", "width": 2.0, "height": 1.0}})
    assert type(inst.shape).__name__ == "Rectangle"


@pytest.mark.asyncio
async def test_discriminated_union_roundtrip_via_type_engine():
    """A model with a discriminated union should roundtrip through to_literal / to_python_value."""
    input_val = Properties(shape=Circle(color="red", radius=1.5))
    lit = TypeEngine.to_literal_type(Properties)
    lv = await TypeEngine.to_literal(input_val, python_type=Properties, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert issubclass(guessed, BaseModel)

    pv = await TypeEngine.to_python_value(lv, Properties)
    assert pv == input_val


@pytest.mark.asyncio
async def test_optional_discriminated_union_roundtrip():
    """Optional[Annotated[Union[..., ...], Field(discriminator=...)]] should roundtrip cleanly."""
    lit = TypeEngine.to_literal_type(OptionalShapeProperties)
    guessed = TypeEngine.guess_python_type(lit)
    assert issubclass(guessed, BaseModel)

    input_val = OptionalShapeProperties(shape=Rectangle(width=3.0, height=4.0))
    lv = await TypeEngine.to_literal(input_val, python_type=OptionalShapeProperties, expected=lit)
    pv = await TypeEngine.to_python_value(lv, OptionalShapeProperties)
    assert pv == input_val

    input_none = OptionalShapeProperties(shape=None)
    lv_none = await TypeEngine.to_literal(input_none, python_type=OptionalShapeProperties, expected=lit)
    pv_none = await TypeEngine.to_python_value(lv_none, OptionalShapeProperties)
    assert pv_none == input_none


@pytest.mark.asyncio
async def test_list_of_discriminated_union_roundtrip():
    """List[Annotated[Union[...], discriminator]] should roundtrip cleanly."""
    input_val = ListOfShapes(shapes=[Circle(radius=1.0), Rectangle(width=2.0, height=3.0)])
    lit = TypeEngine.to_literal_type(ListOfShapes)
    lv = await TypeEngine.to_literal(input_val, python_type=ListOfShapes, expected=lit)
    pv = await TypeEngine.to_python_value(lv, ListOfShapes)
    assert pv == input_val
