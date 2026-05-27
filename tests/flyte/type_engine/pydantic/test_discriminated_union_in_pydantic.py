"""Regression tests for Pydantic v2 tagged (discriminated) unions.

Pydantic v2 emits JSON schemas for ``Annotated[Union[A, B], Field(discriminator=...)]``
using ``oneOf`` and a top-level ``discriminator`` object instead of ``anyOf``. The
type engine previously assumed any non-``$ref`` / non-``anyOf`` / non-``enum``
property had a top-level ``type`` key, which crashed with ``KeyError: 'type'`` on
discriminated unions and made tasks with such inputs uninvocable.
"""

from __future__ import annotations

import typing
from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from flyte.types import TypeEngine
from flyte.types._type_engine import (
    _create_pydantic_model_from_schema,
    _DiscriminatedUnion,
    _get_element_type,
    _get_pydantic_element_type,
    _normalize_discriminator_value,
    _select_unambiguous_variant,
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


# ---------------------------------------------------------------------------
# Regression tests
# - Don't silently pick the wrong variant when discriminator dispatch fails.
# - Enum-typed discriminator fields should still dispatch correctly.
# ---------------------------------------------------------------------------


class _OverlapA(BaseModel):
    kind: Literal["a"] = "a"
    shared: int = 0


class _OverlapB(BaseModel):
    kind: Literal["b"] = "b"
    shared: int = 0


_OverlapShape = Annotated[Union[_OverlapA, _OverlapB], Field(discriminator="kind")]


class OverlapProperties(BaseModel):
    shape: _OverlapShape


def test_unknown_discriminator_value_raises_clear_error():
    """An unknown discriminator value must raise ValueError instead of silently dispatching.

    Two variants with overlapping fields previously caused the first-matching variant to
    silently win when the discriminator lookup missed; we now fail loudly.
    """
    schema = OverlapProperties.model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "OverlapProperties")

    with pytest.raises(ValueError, match="does not match any known variant"):
        cls(shape={"kind": "c", "shared": 1})


def test_missing_discriminator_field_raises_clear_error():
    """When the discriminator property is missing from the input we surface a clear error."""
    schema = OverlapProperties.model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "OverlapProperties")

    with pytest.raises(ValueError, match="missing the discriminator property"):
        cls(shape={"shared": 1})


def test_overlapping_variants_dispatch_correctly_by_discriminator():
    """Sanity check that overlapping-field variants dispatch via discriminator alone."""
    schema = OverlapProperties.model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "OverlapProperties")

    a_inst = cls(shape={"kind": "a", "shared": 1})
    assert type(a_inst.shape).__name__ == "_OverlapA"

    b_inst = cls(shape={"kind": "b", "shared": 2})
    assert type(b_inst.shape).__name__ == "_OverlapB"


# ---- Enum discriminator field --------------------------------------------------


class _ShapeKind(str, Enum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"


class _EnumCircle(BaseModel):
    kind: Literal[_ShapeKind.CIRCLE] = _ShapeKind.CIRCLE
    radius: float = 0.0


class _EnumRectangle(BaseModel):
    kind: Literal[_ShapeKind.RECTANGLE] = _ShapeKind.RECTANGLE
    width: float = 0.0
    height: float = 0.0


_EnumShape = Annotated[Union[_EnumCircle, _EnumRectangle], Field(discriminator="kind")]


class EnumDiscriminatorProperties(BaseModel):
    shape: _EnumShape


def test_enum_str_discriminator_dispatches_with_string_value():
    """A ``str, Enum``-typed discriminator dispatches when the dict carries the string."""
    schema = EnumDiscriminatorProperties.model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "EnumDiscriminatorProperties")

    inst = cls(shape={"kind": "circle", "radius": 1.5})
    assert type(inst.shape).__name__ == "_EnumCircle"
    assert inst.shape.radius == 1.5


def test_enum_discriminator_dispatches_with_enum_member_value():
    """An ``Enum`` member used as the discriminator value should still resolve.

    ``pydantic.BaseModel.model_dump()`` on a model with a non-``str`` enum returns the
    enum member rather than its underlying value, so dict-to-object construction must
    unwrap it before performing the mapping lookup.
    """

    class NonStrShapeKind(Enum):
        CIRCLE = "circle"
        RECTANGLE = "rectangle"

    class NonStrCircle(BaseModel):
        kind: Literal[NonStrShapeKind.CIRCLE] = NonStrShapeKind.CIRCLE
        radius: float = 0.0

    class NonStrRectangle(BaseModel):
        kind: Literal[NonStrShapeKind.RECTANGLE] = NonStrShapeKind.RECTANGLE
        width: float = 0.0
        height: float = 0.0

    NonStrShape = Annotated[Union[NonStrCircle, NonStrRectangle], Field(discriminator="kind")]

    class NonStrProperties(BaseModel):
        shape: NonStrShape

    schema = NonStrProperties.model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "NonStrProperties")

    inst = cls(shape={"kind": NonStrShapeKind.RECTANGLE, "width": 1.0, "height": 2.0})
    assert type(inst.shape).__name__ == "NonStrRectangle"
    assert inst.shape.width == 1.0


# ---- Mapping backfill when schema omits discriminator.mapping ----------------


def test_mapping_is_backfilled_when_schema_omits_explicit_mapping():
    """If ``discriminator.mapping`` is absent the variants' ``const`` values are used."""
    schema = Properties.model_json_schema()
    # Drop the explicit mapping to simulate a JSON Schema generator that only supplies
    # ``propertyName`` (which is what the OpenAPI/JSON Schema specs technically require).
    del schema["properties"]["shape"]["discriminator"]["mapping"]

    cls = convert_mashumaro_json_schema_to_python_class(schema, "Properties")
    _, nested_types = generate_attribute_list_from_dataclass_json_mixin(schema, "Properties")
    descriptor = nested_types["shape"]

    assert isinstance(descriptor, _DiscriminatedUnion)
    assert set(descriptor.mapping.keys()) == {"circle", "rectangle"}

    circle_inst = cls(shape={"kind": "circle", "color": "red", "radius": 1.5})
    assert type(circle_inst.shape).__name__ == "Circle"


# ---- Helper unit tests --------------------------------------------------------


def test_normalize_discriminator_value_unwraps_enums():
    class _K(str, Enum):
        A = "a"

    class _K2(Enum):
        B = 1

    assert _normalize_discriminator_value(_K.A) == "a"
    assert _normalize_discriminator_value(_K2.B) == 1
    assert _normalize_discriminator_value("plain") == "plain"
    assert _normalize_discriminator_value(7) == 7


def test_select_unambiguous_variant_returns_none_on_ambiguity():
    import dataclasses as _dc

    @_dc.dataclass
    class _A:
        x: int = 0
        y: int = 0

    @_dc.dataclass
    class _B:
        x: int = 0
        y: int = 0

    assert _select_unambiguous_variant([_A, _B], {"x": 1, "y": 2}) is None


def test_select_unambiguous_variant_returns_single_match():
    import dataclasses as _dc

    @_dc.dataclass
    class _A:
        x: int = 0

    @_dc.dataclass
    class _B:
        y: int = 0

    matched = _select_unambiguous_variant([_A, _B], {"x": 1})
    assert matched is _A
