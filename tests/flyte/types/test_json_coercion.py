"""Tests for flyte.types._json_coercion."""

from __future__ import annotations

import datetime
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
class JobConfig:
    timeout: datetime.timedelta


@pytest.mark.asyncio
async def test_coerce_and_serialize_timedelta():
    from flyte.types._json_coercion import coerce_json_value

    # String ISO duration
    td = await coerce_json_value("PT1H", datetime.timedelta)
    assert td == datetime.timedelta(hours=1)

    # Weeks and fractions
    td_week = await coerce_json_value("P1W", datetime.timedelta)
    assert td_week == datetime.timedelta(weeks=1)

    td_frac = await coerce_json_value("PT0.5S", datetime.timedelta)
    assert td_frac == datetime.timedelta(seconds=0.5)

    # Int/float seconds
    assert await coerce_json_value(3600, datetime.timedelta) == datetime.timedelta(hours=1)
    assert await coerce_json_value(0.5, datetime.timedelta) == datetime.timedelta(seconds=0.5)

    # Dict
    assert await coerce_json_value({"hours": 1, "minutes": 30}, datetime.timedelta) == datetime.timedelta(
        hours=1, minutes=30
    )

    # Passthrough
    orig = datetime.timedelta(minutes=15)
    assert await coerce_json_value(orig, datetime.timedelta) is orig

    # Collections
    assert await coerce_json_value(["PT1H", "PT30M"], list[datetime.timedelta]) == [
        datetime.timedelta(hours=1),
        datetime.timedelta(minutes=30),
    ]
    assert await coerce_json_value({"t": "PT1H"}, dict[str, datetime.timedelta]) == {"t": datetime.timedelta(hours=1)}

    # Dataclass
    cfg = await coerce_json_value({"timeout": "PT1H"}, JobConfig)
    assert isinstance(cfg.timeout, datetime.timedelta)
    assert cfg.timeout == datetime.timedelta(hours=1)

    # Invalid string raises ValueError
    with pytest.raises(ValueError, match="not a valid duration"):
        await coerce_json_value("invalid", datetime.timedelta)

    # Unsupported type raises ValueError
    with pytest.raises(ValueError, match="Cannot coerce bool to duration"):
        await coerce_json_value(True, datetime.timedelta)

    # Invalid dict arguments raise ValueError
    with pytest.raises(ValueError, match="Invalid duration arguments"):
        await coerce_json_value({"days": 10**10}, datetime.timedelta)

    # Large numeric duration causing OverflowError raises ValueError
    with pytest.raises(ValueError, match="is not a valid duration"):
        await coerce_json_value(10**20, datetime.timedelta)


@pytest.mark.asyncio
async def test_serialize_json_value_scalars():
    from flyte.types._json_coercion import serialize_json_value

    td = datetime.timedelta(hours=1, minutes=30)
    config = JobConfig(timeout=td)

    payload = await serialize_json_value(config)
    assert payload == {"timeout": "PT1H30M"}
    # Verify that payload can be dumped to json without error
    assert json.dumps(payload) == '{"timeout": "PT1H30M"}'

    # Direct scalar serialization of timedelta variations
    assert await serialize_json_value(datetime.timedelta(0)) == "PT0S"
    assert await serialize_json_value(datetime.timedelta(days=1)) == "P1D"
    assert await serialize_json_value(datetime.timedelta(days=2, hours=1, seconds=45)) == "P2DT1H45S"
    assert await serialize_json_value(datetime.timedelta(seconds=1.5)) == "PT1.5S"
    assert await serialize_json_value(-datetime.timedelta(hours=1)) == "-PT1H"
