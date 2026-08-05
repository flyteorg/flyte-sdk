"""Tests for ``flyte._json_schema.literal_type_to_json_schema``."""

from __future__ import annotations

import pydantic
from pydantic import Field

from flyte._json_schema import literal_type_to_json_schema
from flyte.types import TypeEngine


class _InputsWithDefaults(pydantic.BaseModel):
    message: str = Field(default="hello, flyte")
    font: str = Field(default="standard")


def test_literal_type_to_json_schema_omits_defaulted_fields_from_required():
    lt = TypeEngine.to_literal_type(_InputsWithDefaults)
    schema = literal_type_to_json_schema(lt)

    assert schema["type"] == "object"
    assert schema.get("required") == []
    assert schema["properties"]["message"]["default"] == "hello, flyte"
    assert schema["properties"]["font"]["default"] == "standard"


def test_native_interface_json_schema_omits_defaulted_pydantic_input_fields():
    """Task-level ``NativeInterface.json_schema`` matches partial-input semantics."""
    import flyte

    env = flyte.TaskEnvironment(name="test_json_schema_defaults")

    @env.task
    def task(inputs: _InputsWithDefaults = _InputsWithDefaults()) -> str:
        return inputs.message

    schema = task.native_interface.json_schema
    assert schema["properties"]["inputs"]["required"] == []
    assert schema["properties"]["inputs"]["properties"]["font"]["default"] == "standard"
