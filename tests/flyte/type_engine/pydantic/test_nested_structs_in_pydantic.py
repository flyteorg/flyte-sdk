import dataclasses
import typing
from enum import Enum

import pytest
from pydantic import BaseModel

from flyte.types import TypeEngine
from flyte.types._type_engine import _get_element_type

# -- Models --


class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class Inner(BaseModel):
    name: str
    value: int


class ListOfLists(BaseModel):
    matrix: typing.List[typing.List[int]]


class ListOfDicts(BaseModel):
    records: typing.List[typing.Dict[str, int]]


class DictOfDicts(BaseModel):
    nested_map: typing.Dict[str, typing.Dict[str, int]]


class ListOfListOfModels(BaseModel):
    nested: typing.List[typing.List[Inner]]


class ListOfDictOfModels(BaseModel):
    records: typing.List[typing.Dict[str, Inner]]


class DictOfDictOfModels(BaseModel):
    nested_map: typing.Dict[str, typing.Dict[str, Inner]]


class ModelWithEnum(BaseModel):
    name: str
    status: Status


class ListOfEnumModels(BaseModel):
    jobs: typing.List[ModelWithEnum]


class DictOfEnumModels(BaseModel):
    jobs_by_id: typing.Dict[str, ModelWithEnum]


class ComplexModel(BaseModel):
    """Combines lists, dicts, enums, and nested models all together."""

    nested_list: typing.List[typing.List[Inner]]
    dict_of_models: typing.Dict[str, Inner]
    list_of_dicts: typing.List[typing.Dict[str, int]]
    enum_models: typing.List[ModelWithEnum]
    optional_inner: typing.Optional[Inner] = None


class ListOfOptionalModels(BaseModel):
    items: typing.List[typing.Optional[Inner]]


# -- Nested arrays --


def test_list_of_list_of_int_schema():
    """List[List[int]] should resolve the inner element to List[int], not str."""
    schema = ListOfLists.model_json_schema()
    items = schema["properties"]["matrix"]["items"]
    result = _get_element_type(items, schema)
    assert result == typing.List[int]


@pytest.mark.asyncio
async def test_list_of_list_of_int_roundtrip():
    """List[List[int]] should roundtrip through to_literal / to_python_value."""
    input_val = ListOfLists(matrix=[[1, 2], [3, 4]])
    lit = TypeEngine.to_literal_type(ListOfLists)
    lv = await TypeEngine.to_literal(input_val, python_type=ListOfLists, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)
    v = guessed(matrix=[[1, 2], [3, 4]])
    assert v.matrix == [[1, 2], [3, 4]]

    pv = await TypeEngine.to_python_value(lv, ListOfLists)
    assert pv == input_val


@pytest.mark.asyncio
async def test_list_of_list_of_models_roundtrip():
    """List[List[Inner]] should resolve the $ref inside the nested array."""
    input_val = ListOfListOfModels(nested=[[Inner(name="a", value=1)], [Inner(name="b", value=2)]])
    lit = TypeEngine.to_literal_type(ListOfListOfModels)
    lv = await TypeEngine.to_literal(input_val, python_type=ListOfListOfModels, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    pv = await TypeEngine.to_python_value(lv, ListOfListOfModels)
    assert pv == input_val


# -- Nested objects  --


def test_list_of_dict_schema():
    """List[Dict[str, int]] should resolve an inner element to Dict[str, int], not str."""
    schema = ListOfDicts.model_json_schema()
    items = schema["properties"]["records"]["items"]
    result = _get_element_type(items, schema)
    assert result == typing.Dict[str, int]


def test_dict_of_dict_schema():
    """Dict[str, Dict[str, int]] should resolve an inner element to Dict[str, int]."""
    schema = DictOfDicts.model_json_schema()
    additional = schema["properties"]["nested_map"]["additionalProperties"]
    result = _get_element_type(additional, schema)
    assert result == typing.Dict[str, int]


@pytest.mark.asyncio
async def test_list_of_dict_roundtrip():
    """List[Dict[str, int]] should roundtrip through to_literal / to_python_value."""
    input_val = ListOfDicts(records=[{"a": 1, "b": 2}, {"c": 3}])
    lit = TypeEngine.to_literal_type(ListOfDicts)
    lv = await TypeEngine.to_literal(input_val, python_type=ListOfDicts, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)
    v = guessed(records=[{"a": 1}, {"b": 2}])
    assert v.records == [{"a": 1}, {"b": 2}]

    pv = await TypeEngine.to_python_value(lv, ListOfDicts)
    assert pv == input_val


@pytest.mark.asyncio
async def test_list_of_dict_of_models_roundtrip():
    """List[Dict[str, Inner]] should resolve the $ref inside the nested dict."""
    input_val = ListOfDictOfModels(records=[{"x": Inner(name="a", value=1)}])
    lit = TypeEngine.to_literal_type(ListOfDictOfModels)
    lv = await TypeEngine.to_literal(input_val, python_type=ListOfDictOfModels, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    pv = await TypeEngine.to_python_value(lv, ListOfDictOfModels)
    assert pv == input_val


@pytest.mark.asyncio
async def test_dict_of_dict_of_models_roundtrip():
    """Dict[str, Dict[str, Inner]] should resolve the $ref inside nested dicts."""
    input_val = DictOfDictOfModels(nested_map={"outer": {"inner": Inner(name="a", value=1)}})
    lit = TypeEngine.to_literal_type(DictOfDictOfModels)
    lv = await TypeEngine.to_literal(input_val, python_type=DictOfDictOfModels, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    pv = await TypeEngine.to_python_value(lv, DictOfDictOfModels)
    assert pv == input_val


# -- anyOf with $ref  --


def test_list_of_optional_model_no_keyerror():
    """List[Optional[Inner]] should not raise KeyError on the $ref variant."""
    schema = ListOfOptionalModels.model_json_schema()
    items = schema["properties"]["items"]["items"]
    # items is {"anyOf": [{"$ref": "#/$defs/Inner"}, {"type": "null"}]}
    # This used to crash with KeyError: 'type'
    result = _get_element_type(items, schema)
    # Should be Optional[<some dataclass>]
    assert hasattr(result, "__origin__")  # typing.Optional creates a Union


def test_list_of_optional_primitive():
    """List[Optional[str]] should still work (no $ref, just type keys)."""

    class ListOfOptionalStr(BaseModel):
        values: typing.List[typing.Optional[str]]

    schema = ListOfOptionalStr.model_json_schema()
    items = schema["properties"]["values"]["items"]
    result = _get_element_type(items, schema)
    assert result == typing.Optional[str]


@pytest.mark.asyncio
async def test_list_of_optional_model_roundtrip():
    """List[Optional[Inner]] should produce a usable guessed type."""
    lit = TypeEngine.to_literal_type(ListOfOptionalModels)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)


# -- null type --


def test_null_type_returns_nonetype():
    """An element with type 'null' should resolve to NoneType, not str."""
    result = _get_element_type({"type": "null"})
    assert result is type(None)
    assert result is not str


# -- Enum in nested containers --


@pytest.mark.asyncio
async def test_list_of_enum_models_roundtrip():
    """List[ModelWithEnum] where model has an enum field should roundtrip."""
    input_val = ListOfEnumModels(
        jobs=[
            ModelWithEnum(name="job1", status=Status.ACTIVE),
            ModelWithEnum(name="job2", status=Status.INACTIVE),
        ]
    )
    lit = TypeEngine.to_literal_type(ListOfEnumModels)
    lv = await TypeEngine.to_literal(input_val, python_type=ListOfEnumModels, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    pv = await TypeEngine.to_python_value(lv, ListOfEnumModels)
    assert pv == input_val


@pytest.mark.asyncio
async def test_dict_of_enum_models_roundtrip():
    """Dict[str, ModelWithEnum] where model has an enum field should roundtrip."""
    input_val = DictOfEnumModels(
        jobs_by_id={
            "j1": ModelWithEnum(name="job1", status=Status.ACTIVE),
        }
    )
    lit = TypeEngine.to_literal_type(DictOfEnumModels)
    lv = await TypeEngine.to_literal(input_val, python_type=DictOfEnumModels, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    pv = await TypeEngine.to_python_value(lv, DictOfEnumModels)
    assert pv == input_val


# -- Combined: lists + dicts + enums + models --


@pytest.mark.asyncio
async def test_complex_model_roundtrip():
    """A model combining nested lists, dicts, enums, and optional models should roundtrip."""
    input_val = ComplexModel(
        nested_list=[[Inner(name="a", value=1)], [Inner(name="b", value=2)]],
        dict_of_models={"x": Inner(name="c", value=3)},
        list_of_dicts=[{"k1": 10, "k2": 20}],
        enum_models=[ModelWithEnum(name="job1", status=Status.ACTIVE)],
        optional_inner=Inner(name="d", value=4),
    )
    lit = TypeEngine.to_literal_type(ComplexModel)
    lv = await TypeEngine.to_literal(input_val, python_type=ComplexModel, expected=lit)
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    pv = await TypeEngine.to_python_value(lv, ComplexModel)
    assert pv == input_val


@pytest.mark.asyncio
async def test_complex_model_with_none_optional():
    """The complex model with optional_inner=None should roundtrip."""
    input_val = ComplexModel(
        nested_list=[],
        dict_of_models={},
        list_of_dicts=[],
        enum_models=[],
        optional_inner=None,
    )
    lit = TypeEngine.to_literal_type(ComplexModel)
    lv = await TypeEngine.to_literal(input_val, python_type=ComplexModel, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ComplexModel)
    assert pv.optional_inner is None
    assert pv.nested_list == []
    assert pv.dict_of_models == {}
