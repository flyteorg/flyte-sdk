"""Tests for JSON-schema-driven dataclass field defaults in the type engine."""

from __future__ import annotations

import dataclasses
import json

import pydantic
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
