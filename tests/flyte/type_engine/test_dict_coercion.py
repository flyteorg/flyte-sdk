"""Tests for auto-coercing a plain dict into a BaseModel/dataclass at the type-engine layer.

This lets callers (e.g. ``flyte.run`` used as an API-service entrypoint) pass JSON-like dicts for
BaseModel/dataclass parameters instead of constructing the model. Missing fields are filled from the
model's defaults; only missing *required* fields error. See PydanticTransformer.to_literal /
DataclassTransformer.to_literal in flyte/types/_type_engine.py.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import pytest
from pydantic import BaseModel

from flyte.types import TypeEngine
from flyte.types._type_engine import DataclassTransformer, PydanticTransformer, TypeTransformerFailedError


# ---------------------------------------------------------------------------
# Pydantic BaseModel
# ---------------------------------------------------------------------------
class MyModel(BaseModel):
    x: int
    y: float = 4.1
    z: str = "foo"


class Nested(BaseModel):
    inner: MyModel
    name: str


@pytest.mark.asyncio
async def test_dict_to_basemodel_full():
    lit = TypeEngine.to_literal_type(MyModel)
    from_dict = await TypeEngine.to_literal({"x": 5, "y": 1.5, "z": "bar"}, MyModel, lit)
    from_model = await TypeEngine.to_literal(MyModel(x=5, y=1.5, z="bar"), MyModel, lit)
    assert from_dict == from_model

    # round-trips back to the model
    back = await TypeEngine.to_python_value(from_dict, MyModel)
    assert back == MyModel(x=5, y=1.5, z="bar")


@pytest.mark.asyncio
async def test_dict_to_basemodel_partial_uses_defaults():
    lit = TypeEngine.to_literal_type(MyModel)
    lv = await TypeEngine.to_literal({"x": 5}, MyModel, lit)
    back = await TypeEngine.to_python_value(lv, MyModel)
    assert back == MyModel(x=5, y=4.1, z="foo")


@pytest.mark.asyncio
async def test_dict_to_optional_basemodel():
    lit = TypeEngine.to_literal_type(Optional[MyModel])
    lv = await TypeEngine.to_literal({"x": 7}, Optional[MyModel], lit)
    back = await TypeEngine.to_python_value(lv, Optional[MyModel])
    assert back == MyModel(x=7, y=4.1, z="foo")

    none_lv = await TypeEngine.to_literal(None, Optional[MyModel], lit)
    assert await TypeEngine.to_python_value(none_lv, Optional[MyModel]) is None


@pytest.mark.asyncio
async def test_nested_dict_to_basemodel():
    lit = TypeEngine.to_literal_type(Nested)
    lv = await TypeEngine.to_literal({"inner": {"x": 1}, "name": "n"}, Nested, lit)
    back = await TypeEngine.to_python_value(lv, Nested)
    assert back == Nested(inner=MyModel(x=1), name="n")


@pytest.mark.asyncio
async def test_dict_to_basemodel_missing_required_raises():
    lit = TypeEngine.to_literal_type(MyModel)
    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal({"y": 1.0}, MyModel, lit)


# ---------------------------------------------------------------------------
# Decoupled remote case: the client does NOT have the original model class, so
# the type is reconstructed from the JSON schema via guess_python_type. The
# reconstructed model must still carry every field (including defaulted ones)
# so a minimal dict can be filled in from defaults.
# ---------------------------------------------------------------------------
class QueryInfo(BaseModel):
    query: str  # required, no default
    name: str = "default"
    tags: Optional[dict] = None


@pytest.mark.asyncio
async def test_reconstructed_model_keeps_defaulted_fields():
    lit = TypeEngine.to_literal_type(QueryInfo)
    reconstructed = PydanticTransformer().guess_python_type(lit)
    # All fields present, required field first (definition order preserved for cache consistency).
    assert set(reconstructed.model_fields.keys()) == {"query", "name", "tags"}


@pytest.mark.asyncio
async def test_decoupled_dict_minimal_fills_defaults():
    lit = TypeEngine.to_literal_type(QueryInfo)
    reconstructed = PydanticTransformer().guess_python_type(lit)

    lv = await TypeEngine.to_literal({"query": "select 1"}, reconstructed, lit)
    back = await TypeEngine.to_python_value(lv, reconstructed)
    assert back.query == "select 1"
    assert back.name == "default"
    assert back.tags is None


@pytest.mark.asyncio
async def test_decoupled_dict_overrides_defaults():
    lit = TypeEngine.to_literal_type(QueryInfo)
    reconstructed = PydanticTransformer().guess_python_type(lit)

    lv = await TypeEngine.to_literal({"query": "q", "name": "n", "tags": {"a": "b"}}, reconstructed, lit)
    back = await TypeEngine.to_python_value(lv, reconstructed)
    assert (back.query, back.name, back.tags) == ("q", "n", {"a": "b"})


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class MyDC:
    x: int
    y: float = 4.1
    z: str = "foo"


@pytest.mark.asyncio
async def test_dict_to_dataclass_full():
    lit = TypeEngine.to_literal_type(MyDC)
    from_dict = await TypeEngine.to_literal({"x": 5, "y": 1.5, "z": "bar"}, MyDC, lit)
    back = await TypeEngine.to_python_value(from_dict, MyDC)
    assert back == MyDC(x=5, y=1.5, z="bar")


@pytest.mark.asyncio
async def test_dict_to_dataclass_partial_uses_defaults():
    lit = TypeEngine.to_literal_type(MyDC)
    lv = await TypeEngine.to_literal({"x": 5}, MyDC, lit)
    back = await TypeEngine.to_python_value(lv, MyDC)
    assert back == MyDC(x=5, y=4.1, z="foo")


@pytest.mark.asyncio
async def test_dict_to_dataclass_missing_required_raises():
    lit = TypeEngine.to_literal_type(MyDC)
    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal({"y": 1.0}, MyDC, lit)


# ---------------------------------------------------------------------------
# Dataclass with `dict[str, str] | None` — the exact type that triggered the
# customer's original KeyError on the decoupled (reconstructed) path. Covers
# both the direct (class available) and decoupled (reconstructed) cases.
# ---------------------------------------------------------------------------
@dataclass
class TaskInput:
    query: str  # required, no default
    name: str = "default"
    tags: Optional[Dict[str, str]] = None


@pytest.mark.asyncio
async def test_dataclass_dict_optional_direct():
    lit = TypeEngine.to_literal_type(TaskInput)
    lv = await TypeEngine.to_literal({"query": "q", "tags": {"a": "b"}}, TaskInput, lit)
    back = await TypeEngine.to_python_value(lv, TaskInput)
    assert back == TaskInput(query="q", name="default", tags={"a": "b"})


@pytest.mark.asyncio
async def test_dataclass_dict_optional_decoupled_reconstruction():
    lit = TypeEngine.to_literal_type(TaskInput)
    # Reconstruct from schema only (client without the original class). This used to raise
    # KeyError: 'title' on the dict[str, str] | None field.
    reconstructed = DataclassTransformer().guess_python_type(lit)

    import dataclasses as _dc

    assert {f.name for f in _dc.fields(reconstructed)} == {"query", "name", "tags"}

    # minimal input -> defaults filled
    lv = await TypeEngine.to_literal({"query": "q"}, reconstructed, lit)
    back = await TypeEngine.to_python_value(lv, reconstructed)
    assert back.query == "q"
    assert back.name == "default"
    assert back.tags is None

    # dict value provided
    lv2 = await TypeEngine.to_literal({"query": "q", "tags": {"a": "b"}}, reconstructed, lit)
    back2 = await TypeEngine.to_python_value(lv2, reconstructed)
    assert back2.tags == {"a": "b"}
