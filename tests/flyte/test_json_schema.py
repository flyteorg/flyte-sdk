"""Tests for ``flyte._json_schema.literal_type_to_json_schema``."""

from __future__ import annotations

import asyncio
import enum
from typing import Annotated, Optional, Union

import pydantic
from annotated_types import Gt
from pydantic import Field, StringConstraints

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


# --- pydantic field metadata on Annotated inputs ---------------------------------------------------------------


class _Model(pydantic.BaseModel):
    a: int


class _Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


def _schema(t) -> dict:
    return literal_type_to_json_schema(TypeEngine.to_literal_type(t))


def test_annotated_field_constraints_are_preserved():
    assert _schema(Annotated[str, Field(pattern=r"^s3://")]) == {"type": "string", "pattern": "^s3://"}
    assert _schema(Annotated[float, Field(ge=0, le=1)]) == {
        "type": "number",
        "format": "float",
        "minimum": 0,
        "maximum": 1,
    }
    assert _schema(Annotated[int, Gt(0)]) == {"type": "integer", "exclusiveMinimum": 0}
    assert _schema(Annotated[str, StringConstraints(min_length=2)]) == {"type": "string", "minLength": 2}


def test_annotated_field_documentation_is_preserved():
    schema = _schema(Annotated[int, Field(title="Count", description="How many", examples=[1, 2])])
    assert schema == {"type": "integer", "title": "Count", "description": "How many", "examples": [1, 2]}

    schema = _schema(Annotated[str, Field(json_schema_extra={"format": "uri", "x-custom": "yes"})])
    assert schema == {"type": "string", "format": "uri", "x-custom": "yes"}


def test_annotated_field_default_is_not_part_of_the_schema():
    """Whether an input is required comes from the signature, not from Field(default=...)."""
    assert _schema(Annotated[int, Field(default=3, ge=0)]) == {"type": "integer", "minimum": 0}


def test_annotated_field_on_containers():
    assert _schema(Annotated[list[str], Field(min_length=1, max_length=5)]) == {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
    }
    assert _schema(list[Annotated[str, Field(pattern="^a")]]) == {
        "type": "array",
        "items": {"type": "string", "pattern": "^a"},
    }
    assert _schema(Annotated[dict[str, int], Field(description="mapped")]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "description": "mapped",
    }
    assert _schema(Annotated[dict, Field(description="free")]) == {"type": "object", "description": "free"}


def test_annotated_field_on_struct_and_enum():
    schema = _schema(Annotated[_Model, Field(description="a model")])
    assert schema["description"] == "a model"
    assert schema["properties"] == {"a": {"title": "A", "type": "integer"}}

    assert _schema(Annotated[_Color, Field(description="pick one")]) == {
        "type": "string",
        "enum": ["RED", "BLUE"],
        "description": "pick one",
    }


def test_annotated_field_on_optional_and_union():
    # pydantic attaches the constraint to the non-None branch; Optional[X] flattens to X's schema
    assert _schema(Annotated[Optional[str], Field(pattern="x", description="opt")]) == {
        "type": "string",
        "pattern": "x",
        "description": "opt",
    }
    assert _schema(Optional[Annotated[str, Field(pattern="x")]]) == {"type": "string", "pattern": "x"}

    schema = _schema(Annotated[Union[int, str], Field(description="either")])
    assert schema["description"] == "either"
    assert schema["format"] == "union"
    assert [v["type"] for v in schema["oneOf"]] == ["integer", "string"]
    assert all("description" not in v for v in schema["oneOf"])


def test_non_pydantic_annotations_leave_the_schema_alone():
    assert _schema(Annotated[str, "just a note"]) == {"type": "string"}
    assert _schema(str) == {"type": "string"}


def test_annotated_field_round_trips_values():
    async def round_trip(t, value):
        lt = TypeEngine.to_literal_type(t)
        lit = await TypeEngine.to_literal(value, t, lt)
        return await TypeEngine.to_python_value(lit, t)

    assert asyncio.run(round_trip(Annotated[str, Field(pattern=r"^s3://")], "s3://bucket")) == "s3://bucket"
    assert asyncio.run(round_trip(Annotated[float, Field(ge=0, le=1)], 0.5)) == 0.5
    assert asyncio.run(round_trip(Annotated[_Color, Field(description="c")], _Color.RED)) is _Color.RED
    assert asyncio.run(round_trip(Annotated[Optional[str], Field(pattern="x")], None)) is None


def test_task_json_schema_keeps_annotated_field_metadata():
    import flyte

    env = flyte.TaskEnvironment(name="test_json_schema_annotated")

    @env.task
    def task(
        my_s3_uri: Annotated[str, Field(pattern=r"^s3://", description="Where to read from")],
        my_float: Annotated[float, Field(ge=0, le=1)] = 0.5,
    ) -> str:
        return my_s3_uri

    schema = task.native_interface.json_schema
    assert schema["properties"]["my_s3_uri"] == {
        "type": "string",
        "pattern": "^s3://",
        "description": "Where to read from",
    }
    assert schema["properties"]["my_float"] == {"type": "number", "format": "float", "minimum": 0, "maximum": 1}
    assert schema["required"] == ["my_s3_uri"]
