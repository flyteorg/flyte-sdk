"""Tests for flyte.io File / Dir / DataFrame support in agent tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from flyte import TaskEnvironment
from flyte.ai.agents._tools import (
    _coerce_tool_args,
    _json_schema_for_callable,
    _make_callable_tool,
    _stringify_tool_result,
)
from flyte.io import DataFrame, Dir, File
from flyte.models import NativeInterface
from flyte.types._json_coercion import coerce_json_value_sync, serialize_json_value_sync


def test_json_schema_includes_file_dir_dataframe():
    def process(input_file: File, output_dir: Dir, table: DataFrame, label: str) -> str:
        """Process IO types."""
        return label

    schema = _json_schema_for_callable(process)
    assert schema["properties"]["input_file"]["format"] == "blob"
    assert schema["properties"]["input_file"]["properties"]["dimensionality"]["default"] == "SINGLE"
    assert schema["properties"]["output_dir"]["format"] == "blob"
    assert schema["properties"]["output_dir"]["properties"]["dimensionality"]["default"] == "MULTIPART"
    assert schema["properties"]["table"]["format"] == "structured-dataset"
    assert schema["properties"]["label"] == {"type": "string"}


def test_coerce_uri_dict_to_file():
    value = {"uri": "s3://bucket/data.csv", "format": "csv", "dimensionality": "SINGLE"}
    out = coerce_json_value_sync(value, File)
    assert isinstance(out, File)
    assert out.path == "s3://bucket/data.csv"
    assert out.format == "csv"


def test_coerce_uri_dict_to_dir():
    value = {"uri": "s3://bucket/output/", "format": "", "dimensionality": "MULTIPART"}
    out = coerce_json_value_sync(value, Dir)
    assert isinstance(out, Dir)
    assert out.path == "s3://bucket/output/"


def test_coerce_uri_dict_to_dataframe():
    value = {"uri": "s3://bucket/table.parquet", "format": "parquet"}
    out = coerce_json_value_sync(value, DataFrame)
    assert isinstance(out, DataFrame)
    assert out.uri == "s3://bucket/table.parquet"
    assert out.format == "parquet"


def test_coerce_string_path_to_file():
    out = coerce_json_value_sync("/tmp/example.csv", File)
    assert isinstance(out, File)
    assert out.path == "/tmp/example.csv"


def test_serialize_file_for_llm():
    f = File(path="s3://bucket/a.txt", format="txt", hash="abc")
    payload = serialize_json_value_sync(f, File)
    assert payload == {
        "uri": "s3://bucket/a.txt",
        "format": "txt",
        "dimensionality": "SINGLE",
        "name": "a.txt",
        "hash": "abc",
    }


def test_serialize_dir_for_llm():
    d = Dir(path="s3://bucket/out/", format="")
    payload = serialize_json_value_sync(d, Dir)
    assert payload["uri"] == "s3://bucket/out/"
    assert payload["dimensionality"] == "MULTIPART"


def test_stringify_file_result():
    from flyte._utils.asyn import run_sync

    f = File(path="s3://bucket/report.txt", format="txt")
    out = run_sync(_stringify_tool_result, f)
    parsed = json.loads(out)
    assert parsed["uri"] == "s3://bucket/report.txt"
    assert parsed["dimensionality"] == "SINGLE"


@pytest.mark.asyncio
async def test_callable_tool_coerces_file_input():
    seen: dict[str, object] = {}

    def inspect_file(f: File) -> str:
        """Return the file path."""
        seen["path"] = f.path
        return f.path

    tool = _make_callable_tool(inspect_file)
    result = await tool.execute({"f": {"uri": "s3://bucket/x.csv", "format": "csv"}})
    assert result == "s3://bucket/x.csv"
    assert seen["path"] == "s3://bucket/x.csv"


@pytest.mark.asyncio
async def test_task_tool_coerces_dataframe_input():
    env = TaskEnvironment(name="io_tool_env", image="auto")

    @env.task
    async def row_count(df: DataFrame) -> int:
        """Count rows in a dataframe reference."""
        return len(df.uri or "")

    assert NativeInterface.from_callable(row_count.func)

    coerced = await _coerce_tool_args(
        row_count,
        {"df": {"uri": "s3://bucket/table.parquet", "format": "parquet"}},
    )
    assert isinstance(coerced["df"], DataFrame)
    assert coerced["df"].uri == "s3://bucket/table.parquet"


@pytest.mark.asyncio
async def test_agent_chains_file_through_mock_llm():
    """Regression: agent loop must run multiple task tools when the LLM requests them."""
    from flyte.ai.agents import Agent, LLMMessage

    executed: list[str] = []

    async def write_scores_csv(rows: list[dict[str, str | int]]) -> File:
        executed.append("write_scores_csv")
        return File(path="s3://bucket/scores.csv", format="csv")

    async def describe_file(file: File) -> dict[str, str]:
        executed.append("describe_file")
        return {"path": file.path, "format": file.format}

    file_json = {"uri": "s3://bucket/scores.csv", "format": "csv", "dimensionality": "SINGLE"}

    llm = AsyncMock(
        side_effect=[
            LLMMessage(
                content=None,
                tool_calls=[
                    {"id": "c1", "name": "write_scores_csv", "arguments": {"rows": [{"name": "Ada", "score": 98}]}}
                ],
            ),
            LLMMessage(
                content=None,
                tool_calls=[{"id": "c2", "name": "describe_file", "arguments": {"file": file_json}}],
            ),
            LLMMessage(content="CSV has 1 row at s3://bucket/scores.csv", tool_calls=[]),
        ]
    )

    agent = Agent(
        name="chain-test",
        instructions="Chain tools.",
        tools=[write_scores_csv, describe_file],
        call_llm=llm,
        max_turns=5,
    )
    result = await agent.run.aio("create and describe csv", [])
    assert result.error == ""
    assert executed == ["write_scores_csv", "describe_file"]
    assert result.attempts == 3


@pytest.mark.asyncio
async def test_system_prompt_includes_io_handoff_hint():
    from flyte.ai.agents import Agent

    async def describe_file(file: File) -> dict[str, str]:
        return {"path": file.path}

    agent = Agent(name="t", instructions="Do work.", tools=[describe_file])
    assert "complete" in agent.system_prompt.lower()
    assert "`uri`" in agent.system_prompt
