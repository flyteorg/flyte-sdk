"""Tests for JSON-schema-driven dataclass field defaults in the type engine."""

from __future__ import annotations

import dataclasses
import itertools
import json

import pydantic
import pytest
from mashumaro.codecs.json import JSONDecoder
from pydantic import Field

from flyte.types._type_engine import (
    _mutable_schema_default_factory,
    convert_mashumaro_json_schema_to_python_class,
)


def test_mutable_schema_default_factory_returns_independent_copies():
    factory = _mutable_schema_default_factory(["x", "y"])
    first = factory()
    second = factory()
    first.append("z")
    assert second == ["x", "y"]


def test_convert_schema_honors_list_field_defaults():
    schema = {
        "type": "object",
        "title": "_ModelWithListDefault",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["a", "b"],
            },
        },
    }
    cls = convert_mashumaro_json_schema_to_python_class(schema, "_ModelWithListDefault")
    decoded = JSONDecoder(cls).decode("{}")
    assert decoded.tags == ["a", "b"]


def test_convert_schema_partial_input_applies_missing_field_defaults():
    schema = pydantic.create_model(
        "_Partial",
        message=(str, Field(default="hello, flyte")),
        font=(str, Field(default="standard")),
    ).model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "_Partial")
    decoded = JSONDecoder(cls).decode(json.dumps({"message": "hello, niels"}))
    assert decoded.message == "hello, niels"
    assert decoded.font == "standard"
    assert dataclasses.is_dataclass(cls)


def _default_factory_schema(property_order):
    """Schema for a model with a required field, a plain-default field, and a default_factory field.

    Pydantic omits ``default`` from the JSON schema for ``default_factory`` fields and leaves them
    out of ``required``, so ``b`` here mimics ``b: list[int] = Field(default_factory=list)``.
    ``property_order`` lets us simulate the unordered protobuf Struct that carries the schema.
    """
    props = {
        "a": {"type": "integer"},
        "c": {"default": 5, "type": "integer"},
        "b": {"type": "array", "items": {"type": "integer"}},
    }
    return {
        "type": "object",
        "title": "_DefaultFactoryModel",
        "properties": {name: props[name] for name in property_order},
        "required": ["a"],
    }


@pytest.mark.parametrize("property_order", list(itertools.permutations(["a", "b", "c"])))
def test_convert_schema_default_factory_field_does_not_break_ordering(property_order):
    # Regression: a default_factory field (no "default" key, not in "required") used to be emitted
    # without a dataclass default. Whenever a plain-default field preceded it in the (unordered)
    # schema, make_dataclass raised "non-default argument 'b' follows default argument".
    schema = _default_factory_schema(property_order)
    cls = convert_mashumaro_json_schema_to_python_class(schema, "_DefaultFactoryModel")
    field_names = {f.name for f in dataclasses.fields(cls)}
    # All fields survive reconstruction (the non-required ones must not be dropped).
    assert field_names == {"a", "b", "c"}
    # Present values decode through; absent defaulted fields fall back to their defaults. A list
    # default_factory field rebuilds an empty list (matching the producer's default_factory=list),
    # not None.
    decoded = JSONDecoder(cls).decode(json.dumps({"a": 1, "b": [7, 8]}))
    assert (decoded.a, decoded.b) == (1, [7, 8])
    partial = JSONDecoder(cls).decode(json.dumps({"a": 1}))
    assert partial.c == 5
    assert partial.b == []


def test_convert_schema_all_optional_no_required_key_does_not_break_ordering():
    # Pydantic omits the "required" key entirely when every field is optional. A plain-default field
    # ordered before a default_factory field still triggered the make_dataclass ordering crash.
    schema = {
        "type": "object",
        "title": "_AllOptional",
        "properties": {
            "c": {"default": 5, "type": "integer"},
            "b": {"type": "array", "items": {"type": "integer"}},
        },
    }
    cls = convert_mashumaro_json_schema_to_python_class(schema, "_AllOptional")
    decoded = JSONDecoder(cls).decode("{}")
    assert decoded.c == 5
    assert decoded.b == []


def test_convert_schema_default_factory_fields_rebuild_empty_collections():
    # default_factory list/dict fields (no "default" key in the schema) should reconstruct with empty
    # collection defaults -- []/{} -- matching the producer's default_factory=list / =dict, rather
    # than None (which would also violate the declared list/dict field type).
    schema = {
        "type": "object",
        "title": "_Collections",
        "properties": {
            "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["name"],
    }
    cls = convert_mashumaro_json_schema_to_python_class(schema, "_Collections")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    assert fields["tags"].default_factory() == []
    assert fields["meta"].default_factory() == {}
    decoded = JSONDecoder(cls).decode(json.dumps({"name": "x"}))
    assert decoded.tags == []
    assert decoded.meta == {}


def test_convert_schema_nested_default_factory_model_rebuilds_instance():
    # A nested-model default_factory field (e.g. nested: Nested = Field(default_factory=Nested))
    # reconstructs by constructing the reconstructed nested class -- not None -- so an absent value
    # rebuilds the producer's default. The nested model's own default_factory fields rebuild too.
    class Nested(pydantic.BaseModel):
        val: int = 0
        labels: list[str] = Field(default_factory=list)

    class Outer(pydantic.BaseModel):
        name: str
        nested: Nested = Field(default_factory=Nested)

    cls = convert_mashumaro_json_schema_to_python_class(Outer.model_json_schema(), "Outer")
    nested_field = {f.name: f for f in dataclasses.fields(cls)}["nested"]
    built = nested_field.default_factory()
    assert built.val == 0
    assert built.labels == []
    # Absent in the payload -> rebuilt; present -> decoded through.
    partial = JSONDecoder(cls).decode(json.dumps({"name": "x"}))
    assert partial.nested.val == 0 and partial.nested.labels == []
    full = JSONDecoder(cls).decode(json.dumps({"name": "x", "nested": {"val": 7, "labels": ["a"]}}))
    assert full.nested.val == 7 and full.nested.labels == ["a"]


def test_convert_schema_nested_model_with_required_field_falls_back_to_none():
    # A nested model that is NOT no-arg constructible (has a required field with no default) can't be
    # rebuilt as a default, so the field falls back to Optional[...] = None rather than crashing.
    class NestedReq(pydantic.BaseModel):
        required_val: int  # no default -> NestedReq() would raise

    class Outer(pydantic.BaseModel):
        name: str
        nested: NestedReq = Field(default_factory=lambda: NestedReq(required_val=1))

    cls = convert_mashumaro_json_schema_to_python_class(Outer.model_json_schema(), "Outer")
    nested_field = {f.name: f for f in dataclasses.fields(cls)}["nested"]
    assert nested_field.default is None
    assert nested_field.default_factory is dataclasses.MISSING
    decoded = JSONDecoder(cls).decode(json.dumps({"name": "x"}))
    assert decoded.nested is None


def test_convert_schema_reconstructs_pydantic_default_factory_model_fully():
    # End-to-end mirror of the customer's report: a model mixing a required field, a literal default,
    # two collection default_factory fields, and a nested-model default_factory field must
    # reconstruct all 5 fields without the make_dataclass ordering crash, regardless of schema order.
    class Nested(pydantic.BaseModel):
        val: int = 0

    class ExampleOutput(pydantic.BaseModel):
        name: str
        status: str = "ok"
        tags: list[str] = Field(default_factory=list)
        meta: dict = Field(default_factory=dict)
        nested: Nested = Field(default_factory=Nested)

    cls = convert_mashumaro_json_schema_to_python_class(ExampleOutput.model_json_schema(), "ExampleOutput")
    assert {f.name for f in dataclasses.fields(cls)} == {"name", "status", "tags", "meta", "nested"}
    partial = JSONDecoder(cls).decode(json.dumps({"name": "x"}))
    assert partial.status == "ok"
    assert partial.tags == []
    assert partial.meta == {}
    assert partial.nested.val == 0
