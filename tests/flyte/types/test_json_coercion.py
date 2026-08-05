"""Tests for flyte.types._json_coercion."""

from __future__ import annotations

import json

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
