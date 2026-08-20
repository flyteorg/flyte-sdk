"""Optional fields inside a list-element model.

A model reached as a list element is reconstructed through the dataclass path
(``convert_mashumaro_json_schema_to_python_class``), not the pydantic path used for the
top-level model. Its ``anyOf`` handling used to break on ``X | None`` fields:

- ``Model | None`` recursed on the bare ``{"$ref": ...}`` variant -> KeyError: 'properties'
- ``Enum | None`` resolved to an enum definition with no properties -> KeyError: 'properties'
- ``list[int] | None`` read ``items`` from the property instead of the variant -> KeyError: 'items'
- the rebuilt annotation dropped the Optional wrap, so decoding a ``None`` value failed
  pydantic validation even once the crashes were fixed.
"""

import dataclasses
import enum
import typing

import pytest
from pydantic import BaseModel

from flyte.types import TypeEngine


class Color(str, enum.Enum):
    RED = "red"
    BLUE = "blue"


class Inner(BaseModel):
    val: int


class Experiment(BaseModel):
    url: str
    inner: Inner | None = None  # nested optional ref, exercises $defs propagation


class Elem(BaseModel):
    name: str
    experiment: Experiment | None = None
    color: Color | None = None
    tags: dict[str, str] | None = None
    nums: list[int] | None = None
    note: str | None = None


class Wrap(BaseModel):
    items: list[Elem]


def _guessed_elem_type() -> type:
    guessed = TypeEngine.guess_python_type(TypeEngine.to_literal_type(Wrap))
    (elem_type,) = typing.get_args(guessed.model_fields["items"].annotation)
    return elem_type


def _field_types(cls: type) -> dict[str, typing.Any]:
    return {f.name: f.type for f in dataclasses.fields(cls)}


def _optional_inner(field_type: typing.Any) -> typing.Any:
    """Assert ``field_type`` is Optional[X] and return X."""
    assert typing.get_origin(field_type) is typing.Union
    args = typing.get_args(field_type)
    assert type(None) in args
    (inner,) = [a for a in args if a is not type(None)]
    return inner


def test_guess_python_type_with_optional_nested_model_in_list_element():
    # This is the call that used to raise KeyError: 'properties'.
    elem_type = _guessed_elem_type()
    fields = _field_types(elem_type)

    nested = _optional_inner(fields["experiment"])
    assert dataclasses.is_dataclass(nested)
    nested_fields = _field_types(nested)
    assert nested_fields["url"] is str
    # The doubly-nested optional ref must resolve too (requires $defs propagation).
    assert dataclasses.is_dataclass(_optional_inner(nested_fields["inner"]))


def test_guess_python_type_with_optional_enum_in_list_element():
    fields = _field_types(_guessed_elem_type())
    # Enums reconstruct as str on this path; the Optional wrap must survive.
    assert _optional_inner(fields["color"]) is str


def test_guess_python_type_with_optional_list_in_list_element():
    # This is the shape that used to raise KeyError: 'items'.
    fields = _field_types(_guessed_elem_type())
    assert _optional_inner(fields["nums"]) == typing.List[int]


def test_guess_python_type_with_optional_dict_in_list_element():
    fields = _field_types(_guessed_elem_type())
    assert _optional_inner(fields["tags"]) == typing.Dict[str, str]


def test_defaulted_fields_are_not_dropped_from_list_element():
    # Pre-2.4.1 the reconstruction iterated only ``required``, silently dropping every
    # defaulted field from the rebuilt class.
    fields = _field_types(_guessed_elem_type())
    assert set(fields) == {"name", "experiment", "color", "tags", "nums", "note"}


@pytest.mark.asyncio
async def test_roundtrip_through_guessed_type_preserves_optional_values():
    original = Wrap(
        items=[
            Elem(
                name="a",
                experiment=Experiment(url="http://x", inner=Inner(val=3)),
                color=Color.RED,
                tags={"k": "v"},
                nums=[1, 2],
                note="hi",
            ),
            Elem(name="b"),  # every optional field stays None
        ]
    )
    lit = TypeEngine.to_literal_type(Wrap)
    lv = await TypeEngine.to_literal(original, Wrap, lit)

    guessed = TypeEngine.guess_python_type(lit)
    decoded = await TypeEngine.to_python_value(lv, guessed)

    dumped = decoded.model_dump()
    assert dumped["items"][0]["experiment"] == {"url": "http://x", "inner": {"val": 3}}
    assert dumped["items"][0]["color"] == "red"
    assert dumped["items"][0]["tags"] == {"k": "v"}
    assert dumped["items"][0]["nums"] == [1, 2]
    assert dumped["items"][1]["name"] == "b"
    # None values must decode as None, not fail validation against a bare annotation.
    assert dumped["items"][1]["experiment"] is None
    assert dumped["items"][1]["color"] is None
    assert dumped["items"][1]["tags"] is None
    assert dumped["items"][1]["nums"] is None
    assert dumped["items"][1]["note"] is None
