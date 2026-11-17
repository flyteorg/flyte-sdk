"""Tests for flyte.types._json_coercion."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from flyte.io import DataFrame, Dir, File
from flyte.types._json_coercion import (
    coerce_json_value_sync,
    json_dict_to_literal,
    literal_to_json_dict,
    serialize_json_value_sync,
)
from flyte.types._type_engine import TypeEngine


def test_json_dict_to_literal_file():
    lt = TypeEngine.to_literal_type(File)
    lit = json_dict_to_literal({"uri": "s3://bucket/x.csv", "format": "csv", "hash": "abc"}, lt)
    assert lit.scalar.blob.uri == "s3://bucket/x.csv"
    assert lit.scalar.blob.metadata.type.format == "csv"
    assert lit.hash == "abc"


def test_literal_to_json_dict_file_roundtrip():
    lt = TypeEngine.to_literal_type(File)
    lit = json_dict_to_literal({"uri": "s3://bucket/x.csv", "format": "csv"}, lt)
    out = literal_to_json_dict(lit, lt)
    assert out["uri"] == "s3://bucket/x.csv"
    assert out["format"] == "csv"
    assert out["dimensionality"] == "SINGLE"


def test_coerce_json_value_file_from_uri_dict():
    out = coerce_json_value_sync({"uri": "s3://bucket/data.csv", "format": "csv"}, File)
    assert isinstance(out, File)
    assert out.path == "s3://bucket/data.csv"
    assert out.format == "csv"


def test_coerce_json_value_dir_from_uri_dict():
    out = coerce_json_value_sync({"uri": "s3://bucket/output/", "format": ""}, Dir)
    assert isinstance(out, Dir)
    assert out.path == "s3://bucket/output/"


def test_coerce_json_value_dataframe_from_uri_dict():
    out = coerce_json_value_sync({"uri": "s3://bucket/table.parquet", "format": "parquet"}, DataFrame)
    assert isinstance(out, DataFrame)
    assert out.uri == "s3://bucket/table.parquet"


def test_serialize_json_value_file():
    f = File(path="s3://bucket/a.txt", format="txt", hash="abc")
    payload = serialize_json_value_sync(f, File)
    assert payload == {
        "uri": "s3://bucket/a.txt",
        "format": "txt",
        "dimensionality": "SINGLE",
        "name": "a.txt",
        "hash": "abc",
    }


@pytest.mark.asyncio
async def test_type_engine_roundtrip_file():
    original = File(path="s3://bucket/a.txt", format="txt")
    lt = TypeEngine.to_literal_type(File)
    lit = await TypeEngine.to_literal(original, File, lt)
    restored = await TypeEngine.to_python_value(lit, File)
    assert restored.path == original.path
    assert restored.format == original.format


def test_stringify_via_agents_helper():
    from flyte._utils.asyn import run_sync
    from flyte.ai.agents._tools import _stringify_tool_result

    f = File(path="s3://bucket/report.txt", format="txt")
    parsed = json.loads(run_sync(_stringify_tool_result, f))
    assert parsed["uri"] == "s3://bucket/report.txt"
    assert parsed["dimensionality"] == "SINGLE"


@pytest.mark.asyncio
async def test_coerces_the_scalars_json_cannot_express():
    import datetime
    import enum

    from flyte.types._json_coercion import coerce_json_value

    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    Guessed = enum.Enum("DynamicEnum", {"RED": "RED", "BLUE": "BLUE"})

    assert await coerce_json_value(3, float) == 3.0 and isinstance(await coerce_json_value(3, float), float)
    assert await coerce_json_value(True, float) is True, "a bool is not an int for float coercion"
    assert await coerce_json_value("2026-11-12", datetime.date) == datetime.date(2026, 11, 12)
    assert await coerce_json_value("2026-11-12T10:30:00", datetime.datetime) == datetime.datetime(2026, 11, 12, 10, 30)
    assert await coerce_json_value("blue", Color) is Color.BLUE, "by member name, any case"
    assert await coerce_json_value(Color.RED, Guessed) is Guessed.RED, "a member of a structurally identical enum"
    assert await coerce_json_value([1, 2], list[float]) == [1.0, 2.0]
    assert await coerce_json_value({"a": "red"}, dict[str, Color]) == {"a": Color.RED}
    with pytest.raises(ValueError, match="not one of RED, BLUE"):
        await coerce_json_value("green", Color)
    with pytest.raises(ValueError, match="not an ISO date"):
        await coerce_json_value("tomorrow", datetime.date)


@dataclass
class GeometryConfig:
    coordinates: tuple[float, float]
    tags: set[str]
    frozen_ids: frozenset[int]


@pytest.mark.asyncio
async def test_coerce_json_value_tuple():
    from flyte.types._json_coercion import coerce_json_value

    # Fixed-length tuple
    t1 = await coerce_json_value([1, "hello"], tuple[int, str])
    assert t1 == (1, "hello")
    assert isinstance(t1, tuple)

    # Fixed-length tuple with scalar coercion (int to float)
    t2 = await coerce_json_value([1, 2], tuple[float, float])
    assert t2 == (1.0, 2.0)
    assert isinstance(t2, tuple)

    # Variable-length tuple
    t3 = await coerce_json_value([1, 2, 3], tuple[int, ...])
    assert t3 == (1, 2, 3)
    assert isinstance(t3, tuple)

    # Bare tuple
    t4 = await coerce_json_value([1, "a"], tuple)
    assert t4 == (1, "a")
    assert isinstance(t4, tuple)

    # Passthrough existing tuple
    t5 = await coerce_json_value((1, 2), tuple[int, int])
    assert t5 == (1, 2)
    assert isinstance(t5, tuple)

    # typing.Tuple alias
    import typing

    t6 = await coerce_json_value([1, "b"], typing.Tuple[int, str])
    assert t6 == (1, "b")
    assert isinstance(t6, tuple)

    # Length mismatch raises ValueError
    with pytest.raises(ValueError, match=r"Expected 2 elements for tuple\[int, str\], got 3"):
        await coerce_json_value([1, "a", "extra"], tuple[int, str])
    with pytest.raises(ValueError, match=r"Expected 2 elements for tuple\[int, str\], got 1"):
        await coerce_json_value([1], tuple[int, str])


@pytest.mark.asyncio
async def test_coerce_json_value_set_and_frozenset():
    from flyte.types._json_coercion import coerce_json_value

    # Subscripted set
    s1 = await coerce_json_value(["a", "b"], set[str])
    assert s1 == {"a", "b"}
    assert isinstance(s1, set)

    # Bare set
    s2 = await coerce_json_value([1, 2], set)
    assert s2 == {1, 2}
    assert isinstance(s2, set)

    # Passthrough existing set
    s3 = await coerce_json_value({"x", "y"}, set[str])
    assert s3 == {"x", "y"}
    assert isinstance(s3, set)

    # Subscripted frozenset
    fs1 = await coerce_json_value([1, 2], frozenset[int])
    assert fs1 == frozenset({1, 2})
    assert isinstance(fs1, frozenset)

    # Bare frozenset
    fs2 = await coerce_json_value([10, 20], frozenset)
    assert fs2 == frozenset({10, 20})
    assert isinstance(fs2, frozenset)

    # typing.Set and typing.FrozenSet
    import typing

    s4 = await coerce_json_value(["z"], typing.Set[str])
    assert s4 == {"z"}
    assert isinstance(s4, set)

    fs3 = await coerce_json_value([5], typing.FrozenSet[int])
    assert fs3 == frozenset({5})
    assert isinstance(fs3, frozenset)


@pytest.mark.asyncio
async def test_coerce_and_serialize_dataclass_with_collections():
    from flyte.types._json_coercion import coerce_json_value, serialize_json_value

    payload = {
        "coordinates": [10.5, 20.5],
        "tags": ["geo", "v1"],
        "frozen_ids": [1, 2, 3],
    }
    config = await coerce_json_value(payload, GeometryConfig)
    assert isinstance(config, GeometryConfig)
    assert config.coordinates == (10.5, 20.5)
    assert config.tags == {"geo", "v1"}
    assert config.frozen_ids == frozenset({1, 2, 3})

    serialized = await serialize_json_value(config)
    assert serialized["coordinates"] == [10.5, 20.5]
    assert set(serialized["tags"]) == {"geo", "v1"}
    assert set(serialized["frozen_ids"]) == {1, 2, 3}

    # Verify JSON serialization works without error
    dumped = json.dumps(serialized)
    assert '"coordinates": [10.5, 20.5]' in dumped


@pytest.mark.asyncio
async def test_serialize_json_value_tuple_and_set():
    from flyte.types._json_coercion import serialize_json_value

    # Tuple serialization to list
    assert await serialize_json_value((1, 2)) == [1, 2]
    assert await serialize_json_value((1, "a"), tuple[int, str]) == [1, "a"]
    assert await serialize_json_value((1, 2, 3), tuple[int, ...]) == [1, 2, 3]

    # Tuple with Flyte IO
    files = (File(path="s3://b/1.csv"), File(path="s3://b/2.csv"))
    res = await serialize_json_value(files, tuple[File, File])
    assert isinstance(res, list)
    assert res[0]["uri"] == "s3://b/1.csv"
    assert res[1]["uri"] == "s3://b/2.csv"

    # Set and frozenset serialization to list
    s_res = await serialize_json_value({"apple", "banana"})
    assert isinstance(s_res, list)
    assert set(s_res) == {"apple", "banana"}

    fs_res = await serialize_json_value(frozenset({10, 20}))
    assert isinstance(fs_res, list)
    assert set(fs_res) == {10, 20}


@pytest.mark.asyncio
async def test_coerce_and_serialize_nested_collections():
    from flyte.types._json_coercion import (
        coerce_json_args,
        coerce_json_value,
        coerce_json_value_sync,
        serialize_json_value,
        serialize_json_value_sync,
    )

    # Nested: list of tuples
    nested_t = await coerce_json_value([[1, "a"], [2, "b"]], list[tuple[int, str]])
    assert nested_t == [(1, "a"), (2, "b")]
    assert isinstance(nested_t[0], tuple)

    # Nested: dict of sets
    nested_s = await coerce_json_value({"odd": [1, 3], "even": [2, 4]}, dict[str, set[int]])
    assert nested_s == {"odd": {1, 3}, "even": {2, 4}}
    assert isinstance(nested_s["odd"], set)

    # Serialize back
    s_t = await serialize_json_value(nested_t, list[tuple[int, str]])
    assert s_t == [[1, "a"], [2, "b"]]

    s_s = await serialize_json_value(nested_s, dict[str, set[int]])
    assert set(s_s["odd"]) == {1, 3}
    assert set(s_s["even"]) == {2, 4}

    # Synchronous wrappers
    sync_t = coerce_json_value_sync([10, "sync"], tuple[int, str])
    assert sync_t == (10, "sync")
    assert serialize_json_value_sync(sync_t, tuple[int, str]) == [10, "sync"]

    # coerce_json_args
    args = {"point": [3.5, 4.5], "tags": ["alpha", "beta"]}
    inputs = {"point": (tuple[float, float], None), "tags": (set[str], None)}
    coerced_args = await coerce_json_args(args, inputs)
    assert coerced_args["point"] == (3.5, 4.5)
    assert coerced_args["tags"] == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_coerce_json_value_generic_passthrough():
    import typing

    from flyte.types._json_coercion import coerce_json_value

    _T = typing.TypeVar("_T")

    class CustomBox(typing.Generic[_T]):
        def __init__(self, value: _T):
            self.value = value

    box = CustomBox[int](42)
    # Parameterized generic where origin is CustomBox
    res = await coerce_json_value(box, CustomBox[int])
    assert res is box

    # Any and None passthrough
    assert await coerce_json_value(123, typing.Any) == 123
    assert await coerce_json_value("foo", None) == "foo"
