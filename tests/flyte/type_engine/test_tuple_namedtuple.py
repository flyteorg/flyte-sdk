"""
Tests for tuple and namedtuple type transformers with nested combinations
of Flyte-supported container types: dict, list, dataclass, pydantic BaseModel,
flyte.io.File, flyte.io.Dir, and flyte.io.DataFrame.
"""

import os
import tempfile
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import pytest
from mashumaro.mixins.json import DataClassJSONMixin
from pydantic import BaseModel, Field

import flyte
from flyte.io import DataFrame, Dir, File
from flyte.types._type_engine import (
    NamedTupleTransformer,
    TupleTransformer,
    TypeEngine,
)


# =====================================================
# Test Fixtures
# =====================================================


@pytest.fixture
def local_dummy_txt_file():
    fd, path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write("Hello World")
        yield path
    finally:
        os.remove(path)


@pytest.fixture
def local_dummy_directory():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        with open(os.path.join(temp_dir.name, "file"), "w") as tmp:
            tmp.write("Hello world")
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


# =====================================================
# Helper Types for Tests
# =====================================================


class Status(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class SimpleDataclass:
    x: int
    y: str


@dataclass
class DataclassWithList:
    items: List[int]
    name: str


@dataclass
class DataclassJSONMixin(DataClassJSONMixin):
    a: int
    b: str


class SimplePydanticModel(BaseModel):
    x: int
    y: str


class PydanticModelWithList(BaseModel):
    items: List[int] = Field(default_factory=list)
    name: str = ""


class SimpleNamedTuple(NamedTuple):
    x: int
    y: str


class NestedNamedTuple(NamedTuple):
    inner: SimpleNamedTuple
    z: float


# =====================================================
# BASIC TUPLE TESTS
# =====================================================


class TestBasicTuples:
    """Test basic tuple type transformations."""

    @pytest.mark.asyncio
    async def test_tuple_of_primitives(self):
        """Test tuple with primitive types."""
        pt = tuple[int, str, float, bool]
        lt = TypeEngine.to_literal_type(pt)
        value = (42, "hello", 3.14, True)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value
        assert isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_tuple_of_two_elements(self):
        """Test simple two-element tuple."""
        pt = tuple[int, str]
        lt = TypeEngine.to_literal_type(pt)
        value = (123, "world")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_with_none(self):
        """Test tuple containing optional values."""
        pt = tuple[Optional[int], Optional[str]]
        lt = TypeEngine.to_literal_type(pt)
        value = (None, "test")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_all_none(self):
        """Test tuple with all None values."""
        pt = tuple[Optional[int], Optional[str], Optional[float]]
        lt = TypeEngine.to_literal_type(pt)
        value = (None, None, None)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value


# =====================================================
# BASIC NAMEDTUPLE TESTS
# =====================================================


class TestBasicNamedTuples:
    """Test basic NamedTuple type transformations."""

    @pytest.mark.asyncio
    async def test_simple_namedtuple(self):
        """Test simple NamedTuple with primitive types."""
        value = SimpleNamedTuple(x=42, y="hello")
        lt = TypeEngine.to_literal_type(SimpleNamedTuple)

        lv = await TypeEngine.to_literal(value, SimpleNamedTuple, lt)
        result = await TypeEngine.to_python_value(lv, SimpleNamedTuple)

        assert result.x == value.x
        assert result.y == value.y

    @pytest.mark.asyncio
    async def test_namedtuple_with_optionals(self):
        """Test NamedTuple with optional fields."""

        class OptionalNamedTuple(NamedTuple):
            x: Optional[int]
            y: Optional[str]

        value = OptionalNamedTuple(x=None, y="test")
        lt = TypeEngine.to_literal_type(OptionalNamedTuple)

        lv = await TypeEngine.to_literal(value, OptionalNamedTuple, lt)
        result = await TypeEngine.to_python_value(lv, OptionalNamedTuple)

        assert result.x == value.x
        assert result.y == value.y

    @pytest.mark.asyncio
    async def test_nested_namedtuple(self):
        """Test NamedTuple containing another NamedTuple."""
        inner = SimpleNamedTuple(x=10, y="inner")
        value = NestedNamedTuple(inner=inner, z=2.5)
        lt = TypeEngine.to_literal_type(NestedNamedTuple)

        lv = await TypeEngine.to_literal(value, NestedNamedTuple, lt)
        result = await TypeEngine.to_python_value(lv, NestedNamedTuple)

        assert result.inner.x == value.inner.x
        assert result.inner.y == value.inner.y
        assert result.z == value.z


# =====================================================
# TUPLE WITH CONTAINER TYPES
# =====================================================


class TestTupleWithContainers:
    """Test tuples containing dict, list, and other containers."""

    @pytest.mark.asyncio
    async def test_tuple_with_list(self):
        """Test tuple containing a list."""
        pt = tuple[List[int], str]
        lt = TypeEngine.to_literal_type(pt)
        value = ([1, 2, 3, 4, 5], "test")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_with_dict(self):
        """Test tuple containing a dict."""
        pt = tuple[Dict[str, int], str]
        lt = TypeEngine.to_literal_type(pt)
        value = ({"a": 1, "b": 2, "c": 3}, "test")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_with_nested_list(self):
        """Test tuple containing nested lists."""
        pt = tuple[List[List[int]], str]
        lt = TypeEngine.to_literal_type(pt)
        value = ([[1, 2], [3, 4], [5, 6]], "nested")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_with_nested_dict(self):
        """Test tuple containing nested dicts."""
        pt = tuple[Dict[str, Dict[str, int]], int]
        lt = TypeEngine.to_literal_type(pt)
        value = ({"outer": {"inner1": 1, "inner2": 2}}, 42)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_with_list_of_dicts(self):
        """Test tuple containing a list of dicts."""
        pt = tuple[List[Dict[str, int]], str]
        lt = TypeEngine.to_literal_type(pt)
        value = ([{"a": 1}, {"b": 2}, {"c": 3}], "list_of_dicts")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_with_dict_of_lists(self):
        """Test tuple containing a dict of lists."""
        pt = tuple[Dict[str, List[int]], int]
        lt = TypeEngine.to_literal_type(pt)
        value = ({"evens": [2, 4, 6], "odds": [1, 3, 5]}, 100)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value


# =====================================================
# TUPLE WITH DATACLASS
# =====================================================


class TestTupleWithDataclass:
    """Test tuples containing dataclasses."""

    @pytest.mark.asyncio
    async def test_tuple_with_dataclass(self):
        """Test tuple containing a dataclass."""
        pt = tuple[SimpleDataclass, int]
        lt = TypeEngine.to_literal_type(pt)
        dc = SimpleDataclass(x=42, y="hello")
        value = (dc, 100)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0].x == dc.x
        assert result[0].y == dc.y
        assert result[1] == 100

    @pytest.mark.asyncio
    async def test_tuple_with_dataclass_containing_list(self):
        """Test tuple containing a dataclass with a list field."""
        pt = tuple[DataclassWithList, str]
        lt = TypeEngine.to_literal_type(pt)
        dc = DataclassWithList(items=[1, 2, 3], name="test")
        value = (dc, "extra")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0].items == dc.items
        assert result[0].name == dc.name
        assert result[1] == "extra"

    @pytest.mark.asyncio
    async def test_tuple_with_list_of_dataclasses(self):
        """Test tuple containing a list of dataclasses."""
        pt = tuple[List[SimpleDataclass], int]
        lt = TypeEngine.to_literal_type(pt)
        dcs = [SimpleDataclass(x=1, y="a"), SimpleDataclass(x=2, y="b")]
        value = (dcs, 42)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert len(result[0]) == 2
        assert result[0][0].x == 1
        assert result[0][1].x == 2
        assert result[1] == 42

    @pytest.mark.asyncio
    async def test_tuple_with_dict_of_dataclasses(self):
        """Test tuple containing a dict with dataclass values."""
        pt = tuple[Dict[str, SimpleDataclass], str]
        lt = TypeEngine.to_literal_type(pt)
        dcs = {
            "first": SimpleDataclass(x=1, y="a"),
            "second": SimpleDataclass(x=2, y="b"),
        }
        value = (dcs, "dict_of_dc")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0]["first"].x == 1
        assert result[0]["second"].x == 2
        assert result[1] == "dict_of_dc"

    @pytest.mark.asyncio
    async def test_tuple_with_mashumaro_dataclass(self):
        """Test tuple containing a DataClassJSONMixin dataclass."""
        pt = tuple[DataclassJSONMixin, float]
        lt = TypeEngine.to_literal_type(pt)
        dc = DataclassJSONMixin(a=99, b="mashumaro")
        value = (dc, 3.14)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0].a == dc.a
        assert result[0].b == dc.b
        assert result[1] == pytest.approx(3.14)


# =====================================================
# TUPLE WITH PYDANTIC BASEMODEL
# =====================================================


class TestTupleWithPydantic:
    """Test tuples containing Pydantic BaseModel."""

    @pytest.mark.asyncio
    async def test_tuple_with_pydantic_model(self):
        """Test tuple containing a Pydantic BaseModel."""
        pt = tuple[SimplePydanticModel, int]
        lt = TypeEngine.to_literal_type(pt)
        model = SimplePydanticModel(x=42, y="pydantic")
        value = (model, 100)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0].x == model.x
        assert result[0].y == model.y
        assert result[1] == 100

    @pytest.mark.asyncio
    async def test_tuple_with_pydantic_model_containing_list(self):
        """Test tuple containing a Pydantic model with a list field."""
        pt = tuple[PydanticModelWithList, str]
        lt = TypeEngine.to_literal_type(pt)
        model = PydanticModelWithList(items=[1, 2, 3], name="test")
        value = (model, "extra")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0].items == model.items
        assert result[0].name == model.name
        assert result[1] == "extra"

    @pytest.mark.asyncio
    async def test_tuple_with_list_of_pydantic_models(self):
        """Test tuple containing a list of Pydantic models."""
        pt = tuple[List[SimplePydanticModel], int]
        lt = TypeEngine.to_literal_type(pt)
        models = [SimplePydanticModel(x=1, y="a"), SimplePydanticModel(x=2, y="b")]
        value = (models, 42)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert len(result[0]) == 2
        assert result[0][0].x == 1
        assert result[0][1].x == 2
        assert result[1] == 42

    @pytest.mark.asyncio
    async def test_tuple_with_dict_of_pydantic_models(self):
        """Test tuple containing a dict with Pydantic model values."""
        pt = tuple[Dict[str, SimplePydanticModel], str]
        lt = TypeEngine.to_literal_type(pt)
        models = {
            "first": SimplePydanticModel(x=1, y="a"),
            "second": SimplePydanticModel(x=2, y="b"),
        }
        value = (models, "dict_of_models")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0]["first"].x == 1
        assert result[0]["second"].x == 2
        assert result[1] == "dict_of_models"

    @pytest.mark.asyncio
    async def test_tuple_with_nested_pydantic_model(self):
        """Test tuple containing nested Pydantic models."""

        class InnerModel(BaseModel):
            value: int

        class OuterModel(BaseModel):
            inner: InnerModel
            name: str

        pt = tuple[OuterModel, int]
        lt = TypeEngine.to_literal_type(pt)
        model = OuterModel(inner=InnerModel(value=42), name="outer")
        value = (model, 100)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0].inner.value == 42
        assert result[0].name == "outer"
        assert result[1] == 100


# =====================================================
# NAMEDTUPLE WITH CONTAINER TYPES
# =====================================================


class TestNamedTupleWithContainers:
    """Test NamedTuples containing dict, list, and other containers."""

    @pytest.mark.asyncio
    async def test_namedtuple_with_list(self):
        """Test NamedTuple containing a list."""

        class NamedTupleWithList(NamedTuple):
            items: List[int]
            name: str

        value = NamedTupleWithList(items=[1, 2, 3, 4, 5], name="test")
        lt = TypeEngine.to_literal_type(NamedTupleWithList)

        lv = await TypeEngine.to_literal(value, NamedTupleWithList, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithList)

        assert result.items == value.items
        assert result.name == value.name

    @pytest.mark.asyncio
    async def test_namedtuple_with_dict(self):
        """Test NamedTuple containing a dict."""

        class NamedTupleWithDict(NamedTuple):
            mapping: Dict[str, int]
            label: str

        value = NamedTupleWithDict(mapping={"a": 1, "b": 2, "c": 3}, label="test")
        lt = TypeEngine.to_literal_type(NamedTupleWithDict)

        lv = await TypeEngine.to_literal(value, NamedTupleWithDict, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithDict)

        assert result.mapping == value.mapping
        assert result.label == value.label

    @pytest.mark.asyncio
    async def test_namedtuple_with_nested_list(self):
        """Test NamedTuple containing nested lists."""

        class NamedTupleWithNestedList(NamedTuple):
            matrix: List[List[int]]
            dims: int

        value = NamedTupleWithNestedList(matrix=[[1, 2], [3, 4], [5, 6]], dims=3)
        lt = TypeEngine.to_literal_type(NamedTupleWithNestedList)

        lv = await TypeEngine.to_literal(value, NamedTupleWithNestedList, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithNestedList)

        assert result.matrix == value.matrix
        assert result.dims == value.dims

    @pytest.mark.asyncio
    async def test_namedtuple_with_dict_of_lists(self):
        """Test NamedTuple containing a dict of lists."""

        class NamedTupleWithDictOfLists(NamedTuple):
            data: Dict[str, List[int]]
            count: int

        value = NamedTupleWithDictOfLists(
            data={"evens": [2, 4, 6], "odds": [1, 3, 5]},
            count=6,
        )
        lt = TypeEngine.to_literal_type(NamedTupleWithDictOfLists)

        lv = await TypeEngine.to_literal(value, NamedTupleWithDictOfLists, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithDictOfLists)

        assert result.data == value.data
        assert result.count == value.count


# =====================================================
# NAMEDTUPLE WITH DATACLASS
# =====================================================


class TestNamedTupleWithDataclass:
    """Test NamedTuples containing dataclasses."""

    @pytest.mark.asyncio
    async def test_namedtuple_with_dataclass(self):
        """Test NamedTuple containing a dataclass."""

        class NamedTupleWithDataclass(NamedTuple):
            dc: SimpleDataclass
            extra: int

        dc = SimpleDataclass(x=42, y="hello")
        value = NamedTupleWithDataclass(dc=dc, extra=100)
        lt = TypeEngine.to_literal_type(NamedTupleWithDataclass)

        lv = await TypeEngine.to_literal(value, NamedTupleWithDataclass, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithDataclass)

        assert result.dc.x == dc.x
        assert result.dc.y == dc.y
        assert result.extra == 100

    @pytest.mark.asyncio
    async def test_namedtuple_with_list_of_dataclasses(self):
        """Test NamedTuple containing a list of dataclasses."""

        class NamedTupleWithListOfDC(NamedTuple):
            items: List[SimpleDataclass]
            count: int

        dcs = [SimpleDataclass(x=1, y="a"), SimpleDataclass(x=2, y="b")]
        value = NamedTupleWithListOfDC(items=dcs, count=2)
        lt = TypeEngine.to_literal_type(NamedTupleWithListOfDC)

        lv = await TypeEngine.to_literal(value, NamedTupleWithListOfDC, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithListOfDC)

        assert len(result.items) == 2
        assert result.items[0].x == 1
        assert result.items[1].x == 2
        assert result.count == 2

    @pytest.mark.asyncio
    async def test_namedtuple_with_dict_of_dataclasses(self):
        """Test NamedTuple containing a dict with dataclass values."""

        class NamedTupleWithDictOfDC(NamedTuple):
            mapping: Dict[str, SimpleDataclass]
            label: str

        dcs = {
            "first": SimpleDataclass(x=1, y="a"),
            "second": SimpleDataclass(x=2, y="b"),
        }
        value = NamedTupleWithDictOfDC(mapping=dcs, label="test")
        lt = TypeEngine.to_literal_type(NamedTupleWithDictOfDC)

        lv = await TypeEngine.to_literal(value, NamedTupleWithDictOfDC, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithDictOfDC)

        assert result.mapping["first"].x == 1
        assert result.mapping["second"].x == 2
        assert result.label == "test"


# =====================================================
# NAMEDTUPLE WITH PYDANTIC BASEMODEL
# =====================================================


class TestNamedTupleWithPydantic:
    """Test NamedTuples containing Pydantic BaseModel."""

    @pytest.mark.asyncio
    async def test_namedtuple_with_pydantic_model(self):
        """Test NamedTuple containing a Pydantic BaseModel."""

        class NamedTupleWithPydantic(NamedTuple):
            model: SimplePydanticModel
            extra: int

        model = SimplePydanticModel(x=42, y="pydantic")
        value = NamedTupleWithPydantic(model=model, extra=100)
        lt = TypeEngine.to_literal_type(NamedTupleWithPydantic)

        lv = await TypeEngine.to_literal(value, NamedTupleWithPydantic, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithPydantic)

        assert result.model.x == model.x
        assert result.model.y == model.y
        assert result.extra == 100

    @pytest.mark.asyncio
    async def test_namedtuple_with_list_of_pydantic_models(self):
        """Test NamedTuple containing a list of Pydantic models."""

        class NamedTupleWithListOfPydantic(NamedTuple):
            items: List[SimplePydanticModel]
            count: int

        models = [SimplePydanticModel(x=1, y="a"), SimplePydanticModel(x=2, y="b")]
        value = NamedTupleWithListOfPydantic(items=models, count=2)
        lt = TypeEngine.to_literal_type(NamedTupleWithListOfPydantic)

        lv = await TypeEngine.to_literal(value, NamedTupleWithListOfPydantic, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithListOfPydantic)

        assert len(result.items) == 2
        assert result.items[0].x == 1
        assert result.items[1].x == 2
        assert result.count == 2


# =====================================================
# TUPLE WITH FLYTE IO TYPES (File, Dir)
# =====================================================


class TestTupleWithFlyteIOTypes:
    """Test tuples containing Flyte IO types: File, Dir."""

    @pytest.mark.asyncio
    async def test_tuple_with_file(self, local_dummy_txt_file):
        """Test tuple containing a Flyte File."""
        pt = tuple[File, str]
        lt = TypeEngine.to_literal_type(pt)
        file = File(path=local_dummy_txt_file)
        value = (file, "test")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert isinstance(result[0], File)
        assert result[1] == "test"

    @pytest.mark.asyncio
    async def test_tuple_with_dir(self, local_dummy_directory):
        """Test tuple containing a Flyte Dir."""
        pt = tuple[Dir, str]
        lt = TypeEngine.to_literal_type(pt)
        directory = Dir(path=local_dummy_directory)
        value = (directory, "test")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert isinstance(result[0], Dir)
        assert result[1] == "test"

    @pytest.mark.asyncio
    async def test_tuple_with_list_of_files(self, local_dummy_txt_file):
        """Test tuple containing a list of Flyte Files."""
        pt = tuple[List[File], int]
        lt = TypeEngine.to_literal_type(pt)
        files = [File(path=local_dummy_txt_file), File(path=local_dummy_txt_file)]
        value = (files, 2)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert len(result[0]) == 2
        assert all(isinstance(f, File) for f in result[0])
        assert result[1] == 2

    @pytest.mark.asyncio
    async def test_tuple_with_dict_of_files(self, local_dummy_txt_file):
        """Test tuple containing a dict with File values."""
        pt = tuple[Dict[str, File], str]
        lt = TypeEngine.to_literal_type(pt)
        files = {
            "file1": File(path=local_dummy_txt_file),
            "file2": File(path=local_dummy_txt_file),
        }
        value = (files, "dict_of_files")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert len(result[0]) == 2
        assert all(isinstance(f, File) for f in result[0].values())
        assert result[1] == "dict_of_files"

    @pytest.mark.asyncio
    async def test_tuple_with_file_and_dir(self, local_dummy_txt_file, local_dummy_directory):
        """Test tuple containing both File and Dir."""
        pt = tuple[File, Dir, str]
        lt = TypeEngine.to_literal_type(pt)
        file = File(path=local_dummy_txt_file)
        directory = Dir(path=local_dummy_directory)
        value = (file, directory, "both")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert isinstance(result[0], File)
        assert isinstance(result[1], Dir)
        assert result[2] == "both"


# =====================================================
# NAMEDTUPLE WITH FLYTE IO TYPES (File, Dir)
# =====================================================


class TestNamedTupleWithFlyteIOTypes:
    """Test NamedTuples containing Flyte IO types: File, Dir."""

    @pytest.mark.asyncio
    async def test_namedtuple_with_file(self, local_dummy_txt_file):
        """Test NamedTuple containing a Flyte File."""

        class NamedTupleWithFile(NamedTuple):
            file: File
            label: str

        file = File(path=local_dummy_txt_file)
        value = NamedTupleWithFile(file=file, label="test")
        lt = TypeEngine.to_literal_type(NamedTupleWithFile)

        lv = await TypeEngine.to_literal(value, NamedTupleWithFile, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithFile)

        assert isinstance(result.file, File)
        assert result.label == "test"

    @pytest.mark.asyncio
    async def test_namedtuple_with_dir(self, local_dummy_directory):
        """Test NamedTuple containing a Flyte Dir."""

        class NamedTupleWithDir(NamedTuple):
            directory: Dir
            label: str

        directory = Dir(path=local_dummy_directory)
        value = NamedTupleWithDir(directory=directory, label="test")
        lt = TypeEngine.to_literal_type(NamedTupleWithDir)

        lv = await TypeEngine.to_literal(value, NamedTupleWithDir, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithDir)

        assert isinstance(result.directory, Dir)
        assert result.label == "test"

    @pytest.mark.asyncio
    async def test_namedtuple_with_list_of_files(self, local_dummy_txt_file):
        """Test NamedTuple containing a list of Flyte Files."""

        class NamedTupleWithFileList(NamedTuple):
            files: List[File]
            count: int

        files = [File(path=local_dummy_txt_file), File(path=local_dummy_txt_file)]
        value = NamedTupleWithFileList(files=files, count=2)
        lt = TypeEngine.to_literal_type(NamedTupleWithFileList)

        lv = await TypeEngine.to_literal(value, NamedTupleWithFileList, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithFileList)

        assert len(result.files) == 2
        assert all(isinstance(f, File) for f in result.files)
        assert result.count == 2


# =====================================================
# TUPLE WITH DATAFRAME (requires pandas)
# =====================================================


@pytest.mark.skipif(
    "pandas" not in __import__("sys").modules,
    reason="Pandas is not installed",
)
class TestTupleWithDataFrame:
    """Test tuples containing Flyte DataFrame."""

    @pytest.mark.asyncio
    async def test_tuple_with_dataframe(self, ctx_with_test_raw_data_path):
        """Test tuple containing a Flyte DataFrame."""
        import pandas as pd

        pt = tuple[DataFrame, str]
        lt = TypeEngine.to_literal_type(pt)
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        sd = DataFrame.from_df(val=df)
        value = (sd, "test")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert isinstance(result[0], DataFrame)
        assert result[1] == "test"

    @pytest.mark.asyncio
    async def test_tuple_with_list_of_dataframes(self, ctx_with_test_raw_data_path):
        """Test tuple containing a list of Flyte DataFrames."""
        import pandas as pd

        pt = tuple[List[DataFrame], int]
        lt = TypeEngine.to_literal_type(pt)
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pd.DataFrame({"c": [3, 4], "d": ["z", "w"]})
        sds = [DataFrame.from_df(val=df1), DataFrame.from_df(val=df2)]
        value = (sds, 2)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert len(result[0]) == 2
        assert all(isinstance(sd, DataFrame) for sd in result[0])
        assert result[1] == 2


# =====================================================
# COMPLEX NESTED COMBINATIONS
# =====================================================


class TestComplexNestedCombinations:
    """Test complex nested combinations of tuples, namedtuples, and other types."""

    @pytest.mark.asyncio
    async def test_tuple_of_tuples(self):
        """Test tuple containing other tuples."""
        pt = tuple[tuple[int, str], tuple[float, bool]]
        lt = TypeEngine.to_literal_type(pt)
        value = ((42, "hello"), (3.14, True))

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_with_dataclass_containing_pydantic(self):
        """Test tuple containing a dataclass that contains a Pydantic model."""

        class InnerPydantic(BaseModel):
            value: int

        @dataclass
        class OuterDataclass:
            model: InnerPydantic
            name: str

        pt = tuple[OuterDataclass, int]
        lt = TypeEngine.to_literal_type(pt)
        dc = OuterDataclass(model=InnerPydantic(value=42), name="test")
        value = (dc, 100)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0].model.value == 42
        assert result[0].name == "test"
        assert result[1] == 100

    @pytest.mark.asyncio
    async def test_tuple_with_union(self):
        """Test tuple containing a union type."""
        pt = tuple[Union[int, str], Union[float, bool]]
        lt = TypeEngine.to_literal_type(pt)
        value = (42, 3.14)

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_tuple_with_enum(self):
        """Test tuple containing an enum type."""
        pt = tuple[Status, str]
        lt = TypeEngine.to_literal_type(pt)
        value = (Status.APPROVED, "test")

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result[0] == Status.APPROVED
        assert result[1] == "test"

    @pytest.mark.asyncio
    async def test_namedtuple_with_tuple(self):
        """Test NamedTuple containing a tuple."""

        class NamedTupleWithTuple(NamedTuple):
            inner_tuple: tuple[int, str]
            extra: float

        inner = (42, "hello")
        value = NamedTupleWithTuple(inner_tuple=inner, extra=3.14)
        lt = TypeEngine.to_literal_type(NamedTupleWithTuple)

        lv = await TypeEngine.to_literal(value, NamedTupleWithTuple, lt)
        result = await TypeEngine.to_python_value(lv, NamedTupleWithTuple)

        assert result.inner_tuple == inner
        assert result.extra == pytest.approx(3.14)

    @pytest.mark.asyncio
    async def test_list_of_tuples(self):
        """Test list containing tuples."""
        pt = List[tuple[int, str]]
        lt = TypeEngine.to_literal_type(pt)
        value = [(1, "a"), (2, "b"), (3, "c")]

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_dict_with_tuple_values(self):
        """Test dict with tuple values."""
        pt = Dict[str, tuple[int, float]]
        lt = TypeEngine.to_literal_type(pt)
        value = {"a": (1, 1.1), "b": (2, 2.2), "c": (3, 3.3)}

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert result == value

    @pytest.mark.asyncio
    async def test_list_of_namedtuples(self):
        """Test list containing NamedTuples."""
        pt = List[SimpleNamedTuple]
        lt = TypeEngine.to_literal_type(pt)
        value = [
            SimpleNamedTuple(x=1, y="a"),
            SimpleNamedTuple(x=2, y="b"),
            SimpleNamedTuple(x=3, y="c"),
        ]

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert len(result) == 3
        assert result[0].x == 1
        assert result[1].x == 2
        assert result[2].x == 3

    @pytest.mark.asyncio
    async def test_dict_with_namedtuple_values(self):
        """Test dict with NamedTuple values."""
        pt = Dict[str, SimpleNamedTuple]
        lt = TypeEngine.to_literal_type(pt)
        value = {
            "first": SimpleNamedTuple(x=1, y="a"),
            "second": SimpleNamedTuple(x=2, y="b"),
        }

        lv = await TypeEngine.to_literal(value, pt, lt)
        result = await TypeEngine.to_python_value(lv, pt)

        assert len(result) == 2
        assert result["first"].x == 1
        assert result["second"].x == 2


# =====================================================
# FLYTE TASK INTEGRATION TESTS
# =====================================================


class TestFlyteTaskIntegration:
    """Test tuple and namedtuple types in Flyte tasks."""

    @pytest.mark.asyncio
    async def test_task_with_tuple_input_output(self):
        """Test Flyte task with tuple input and output."""
        env = flyte.TaskEnvironment(name="test-tuple-task")

        @env.task
        async def process_tuple(data: tuple[int, str]) -> tuple[str, int]:
            return (data[1], data[0])

        result = await process_tuple(data=(42, "hello"))
        assert result == ("hello", 42)

    @pytest.mark.asyncio
    async def test_task_with_namedtuple_input_output(self):
        """Test Flyte task with NamedTuple input and output."""
        env = flyte.TaskEnvironment(name="test-namedtuple-task")

        @env.task
        async def process_namedtuple(data: SimpleNamedTuple) -> SimpleNamedTuple:
            return SimpleNamedTuple(x=data.x * 2, y=data.y.upper())

        result = await process_namedtuple(data=SimpleNamedTuple(x=21, y="hello"))
        assert result.x == 42
        assert result.y == "HELLO"

    @pytest.mark.asyncio
    async def test_task_with_tuple_containing_dataclass(self):
        """Test Flyte task with tuple containing dataclass."""
        env = flyte.TaskEnvironment(name="test-tuple-dc-task")

        @env.task
        async def process_tuple_dc(
            data: tuple[SimpleDataclass, int]
        ) -> tuple[int, str]:
            return (data[0].x + data[1], data[0].y)

        dc = SimpleDataclass(x=10, y="test")
        result = await process_tuple_dc(data=(dc, 32))
        assert result == (42, "test")

    @pytest.mark.asyncio
    async def test_task_with_tuple_containing_pydantic(self):
        """Test Flyte task with tuple containing Pydantic model."""
        env = flyte.TaskEnvironment(name="test-tuple-pydantic-task")

        @env.task
        async def process_tuple_pydantic(
            data: tuple[SimplePydanticModel, int]
        ) -> tuple[int, str]:
            return (data[0].x + data[1], data[0].y)

        model = SimplePydanticModel(x=10, y="test")
        result = await process_tuple_pydantic(data=(model, 32))
        assert result == (42, "test")

    @pytest.mark.asyncio
    async def test_task_with_namedtuple_containing_list(self):
        """Test Flyte task with NamedTuple containing list."""
        env = flyte.TaskEnvironment(name="test-namedtuple-list-task")

        class DataWithList(NamedTuple):
            items: List[int]
            name: str

        @env.task
        async def sum_items(data: DataWithList) -> int:
            return sum(data.items)

        result = await sum_items(data=DataWithList(items=[1, 2, 3, 4, 5], name="test"))
        assert result == 15

    @pytest.mark.asyncio
    async def test_workflow_with_tuple_passing(self):
        """Test passing tuples between tasks in a workflow."""
        env = flyte.TaskEnvironment(name="test-tuple-workflow")

        @env.task
        async def create_tuple() -> tuple[int, str]:
            return (42, "hello")

        @env.task
        async def transform_tuple(data: tuple[int, str]) -> tuple[str, int]:
            return (data[1].upper(), data[0] * 2)

        @env.task
        async def workflow() -> tuple[str, int]:
            t = await create_tuple()
            return await transform_tuple(data=t)

        result = await workflow()
        assert result == ("HELLO", 84)

    @pytest.mark.asyncio
    async def test_workflow_with_namedtuple_passing(self):
        """Test passing NamedTuples between tasks in a workflow."""
        env = flyte.TaskEnvironment(name="test-namedtuple-workflow")

        @env.task
        async def create_namedtuple() -> SimpleNamedTuple:
            return SimpleNamedTuple(x=42, y="hello")

        @env.task
        async def transform_namedtuple(data: SimpleNamedTuple) -> SimpleNamedTuple:
            return SimpleNamedTuple(x=data.x * 2, y=data.y.upper())

        @env.task
        async def workflow() -> SimpleNamedTuple:
            nt = await create_namedtuple()
            return await transform_namedtuple(data=nt)

        result = await workflow()
        assert result.x == 84
        assert result.y == "HELLO"


# =====================================================
# FLYTE RUN INTEGRATION TESTS
# =====================================================


class TestFlyteRunIntegration:
    """Test tuple and namedtuple types with flyte.run()."""

    def test_flyte_run_with_tuple(self):
        """Test flyte.run() with tuple input and output."""
        flyte.init()
        env = flyte.TaskEnvironment(name="test-run-tuple")

        @env.task
        async def double_tuple(data: tuple[int, str]) -> tuple[int, str]:
            return (data[0] * 2, data[1] * 2)

        run = flyte.run(double_tuple, data=(21, "hi"))
        # When a task returns a tuple, outputs() returns the elements unpacked
        result = run.outputs()
        assert result[0] == 42
        assert result[1] == "hihi"

    def test_flyte_run_with_namedtuple(self):
        """Test flyte.run() with NamedTuple input and output."""
        flyte.init()
        env = flyte.TaskEnvironment(name="test-run-namedtuple")

        @env.task
        async def double_namedtuple(data: SimpleNamedTuple) -> SimpleNamedTuple:
            return SimpleNamedTuple(x=data.x * 2, y=data.y * 2)

        run = flyte.run(double_namedtuple, data=SimpleNamedTuple(x=21, y="hi"))
        # When a task returns a NamedTuple, outputs() returns the elements unpacked
        result = run.outputs()
        assert result[0] == 42
        assert result[1] == "hihi"

    def test_flyte_run_with_tuple_containing_dataclass(self):
        """Test flyte.run() with tuple containing dataclass."""
        flyte.init()
        env = flyte.TaskEnvironment(name="test-run-tuple-dc")

        @env.task
        async def extract_from_tuple(data: tuple[SimpleDataclass, int]) -> int:
            return data[0].x + data[1]

        dc = SimpleDataclass(x=10, y="test")
        run = flyte.run(extract_from_tuple, data=(dc, 32))
        result = run.outputs()[0]
        assert result == 42

    def test_flyte_run_with_namedtuple_containing_dict(self):
        """Test flyte.run() with NamedTuple containing dict."""
        flyte.init()
        env = flyte.TaskEnvironment(name="test-run-namedtuple-dict")

        class DataWithDict(NamedTuple):
            mapping: Dict[str, int]
            label: str

        @env.task
        async def sum_values(data: DataWithDict) -> int:
            return sum(data.mapping.values())

        run = flyte.run(
            sum_values,
            data=DataWithDict(mapping={"a": 10, "b": 20, "c": 12}, label="test"),
        )
        result = run.outputs()[0]
        assert result == 42


# =====================================================
# TRANSFORMER SPECIFIC TESTS
# =====================================================


class TestTransformers:
    """Test TupleTransformer and NamedTupleTransformer directly."""

    def test_tuple_transformer_type_resolution(self):
        """Test that TupleTransformer is correctly resolved."""
        transformer = TypeEngine.get_transformer(tuple[int, str])
        assert isinstance(transformer, TupleTransformer)

    def test_namedtuple_transformer_type_resolution(self):
        """Test that NamedTupleTransformer is correctly resolved."""
        transformer = TypeEngine.get_transformer(SimpleNamedTuple)
        assert isinstance(transformer, NamedTupleTransformer)

    def test_tuple_literal_type(self):
        """Test that tuple literal type is correctly generated."""
        lt = TypeEngine.to_literal_type(tuple[int, str, float])
        assert lt.simple == 9  # STRUCT

    def test_namedtuple_literal_type(self):
        """Test that NamedTuple literal type is correctly generated."""
        lt = TypeEngine.to_literal_type(SimpleNamedTuple)
        assert lt.simple == 9  # STRUCT
