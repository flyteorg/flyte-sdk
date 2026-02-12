"""Tests for custom TypeTransformer schema_match in nested Pydantic models.

Verifies that when a custom type with a registered TypeTransformer (implementing
schema_match) is nested inside a Pydantic BaseModel, guess_python_type correctly
reconstructs the custom type instead of building a generic dataclass.
"""

import dataclasses
import typing

import pytest
from flyteidl2.core.literals_pb2 import Literal
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from pydantic import BaseModel

from flyte.io import Dir, File
from flyte.types import TypeEngine
from flyte.types._type_engine import TypeTransformer, _match_registered_type_from_schema


# -- Custom type and transformer --


class Coordinate(BaseModel):
    x: float
    y: float


class CoordinateTransformer(TypeTransformer[Coordinate]):
    """A transformer for Coordinate —
    Coordinate is a BaseModel and the default auto-matches."""

    def __init__(self):
        super().__init__("Coordinate", Coordinate)

    def get_literal_type(self, t=None) -> LiteralType:
        return LiteralType(simple=SimpleType.STRUCT)

    async def to_literal(self, python_val, python_type, expected) -> Literal:
        raise NotImplementedError

    async def to_python_value(self, lv, expected_python_type):
        raise NotImplementedError


# -- Models using Coordinate --


class ModelWithCoord(BaseModel):
    label: str
    coord: Coordinate


class ModelWithListOfCoords(BaseModel):
    coords: typing.List[Coordinate]


class ModelWithDictOfCoords(BaseModel):
    coord_map: typing.Dict[str, Coordinate]


class ModelWithOptionalCoord(BaseModel):
    coord: typing.Optional[Coordinate] = None


class ModelWithNestedListOfCoords(BaseModel):
    nested: typing.List[typing.List[Coordinate]]


class ModelWithMixed(BaseModel):
    coords: typing.List[Coordinate]
    file: File
    dir_map: typing.Dict[str, Dir]


# -- Fixtures --


@pytest.fixture(autouse=True)
def register_coordinate_transformer():
    """Register the custom transformer for each test, then clean up."""
    transformer = CoordinateTransformer()
    TypeEngine.register(transformer)
    yield
    TypeEngine._REGISTRY.pop(Coordinate, None)


# -- Unit tests for _match_registered_type_from_schema --


def test_match_returns_coordinate_for_matching_schema():
    schema = Coordinate.model_json_schema()
    result = _match_registered_type_from_schema(schema)
    assert result is Coordinate


def test_match_returns_none_for_unmatched_schema():
    schema = {"type": "object", "title": "Unknown", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    result = _match_registered_type_from_schema(schema)
    assert result is None


def test_match_returns_file_for_file_schema():
    schema = File.model_json_schema()
    result = _match_registered_type_from_schema(schema)
    assert result is File


def test_match_returns_dir_for_dir_schema():
    schema = Dir.model_json_schema()
    result = _match_registered_type_from_schema(schema)
    assert result is Dir


# -- guess_python_type structure verification --


def test_coord_in_model_guess_type():
    """guess_python_type should reconstruct coord as Coordinate, not a generic dataclass."""
    lit = TypeEngine.to_literal_type(ModelWithCoord)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)
    assert "coord" in hints
    assert hints["coord"] is Coordinate


def test_list_of_coords_guess_type():
    """guess_python_type should reconstruct List[Coordinate] with Coordinate as inner type."""
    lit = TypeEngine.to_literal_type(ModelWithListOfCoords)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)
    coords_type = hints["coords"]
    assert typing.get_origin(coords_type) is list
    inner = typing.get_args(coords_type)[0]
    assert inner is Coordinate


def test_dict_of_coords_guess_type():
    """guess_python_type should reconstruct Dict[str, Coordinate] with Coordinate as value type."""
    lit = TypeEngine.to_literal_type(ModelWithDictOfCoords)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)
    map_type = hints["coord_map"]
    assert typing.get_origin(map_type) is dict
    key_type, val_type = typing.get_args(map_type)
    assert key_type is str
    assert val_type is Coordinate


def test_nested_list_of_coords_guess_type():
    """guess_python_type should reconstruct List[List[Coordinate]]."""
    lit = TypeEngine.to_literal_type(ModelWithNestedListOfCoords)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)
    nested_type = hints["nested"]
    assert typing.get_origin(nested_type) is list

    inner_list = typing.get_args(nested_type)[0]
    assert typing.get_origin(inner_list) is list

    innermost = typing.get_args(inner_list)[0]
    assert innermost is Coordinate


def test_mixed_model_guess_type():
    """guess_python_type should reconstruct a model mixing custom type, File, and Dir."""
    lit = TypeEngine.to_literal_type(ModelWithMixed)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)

    # coords: List[Coordinate]
    coords_type = hints["coords"]
    assert typing.get_origin(coords_type) is list
    assert typing.get_args(coords_type)[0] is Coordinate

    # file: File
    assert hints["file"] is File

    # dir_map: Dict[str, Dir]
    dir_map_type = hints["dir_map"]
    assert typing.get_origin(dir_map_type) is dict
    _, dir_val = typing.get_args(dir_map_type)
    assert dir_val is Dir


# -- schema_match default behavior --


def test_base_transformer_schema_match_returns_false():
    """The default schema_match on TypeTransformer should return False."""

    class DummyTransformer(TypeTransformer[str]):
        def __init__(self):
            super().__init__("Dummy", str)

        def get_literal_type(self, t=None):
            return LiteralType(simple=SimpleType.STRING)

        async def to_literal(self, val, typ, expected):
            raise NotImplementedError

        async def to_python_value(self, lv, typ):
            raise NotImplementedError

    t = DummyTransformer()
    assert t.schema_match({"type": "string"}) is False
    assert t.schema_match({}) is False