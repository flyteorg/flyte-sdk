"""Tests for the tool bridge — the seam that lets the harness call Flyte tasks.

These run the published shim as a real subprocess, exactly as the harness's bash
tool would, so the whole path is exercised: shim -> unix socket -> bridge ->
``task.aio`` -> result text on stdout. No harness runtime and no network needed.
"""

import asyncio
import json
import os
import stat
import subprocess
from unittest.mock import AsyncMock, patch

import flyte
import pytest

from flyteplugins.agents.deepseek import TOOLS_DIRNAME, ToolBridge, tool


def _weather_tool(env_name):
    env = flyte.TaskEnvironment(env_name)

    @tool
    @env.task
    async def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    return get_weather


async def _call_shim(workspace, name, args, timeout=30):
    """Run a published shim the way the harness's bash tool would."""
    shim = workspace / TOOLS_DIRNAME / name
    return await asyncio.to_thread(
        subprocess.run,
        [str(shim), json.dumps(args)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.asyncio
async def test_start_publishes_an_executable_shim_per_tool(tmp_path):
    bridge = ToolBridge([_weather_tool("ds_bridge_a")])
    try:
        await bridge.start(tmp_path)
        shim = tmp_path / TOOLS_DIRNAME / "get_weather"
        assert shim.is_file()
        assert shim.stat().st_mode & stat.S_IXUSR  # bash has to be able to run it
        assert os.path.exists(bridge.socket_path)
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_shim_subprocess_round_trips_to_the_flyte_task(tmp_path):
    """The end-to-end path the model actually takes."""
    weather = _weather_tool("ds_bridge_b")
    bridge = ToolBridge([weather])
    try:
        await bridge.start(tmp_path)
        with patch.object(weather.task, "aio", new_callable=AsyncMock, return_value="sunny in Paris") as mock_aio:
            done = await _call_shim(tmp_path, "get_weather", {"city": "Paris"})
    finally:
        await bridge.stop()

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "sunny in Paris"
    # The call reached the task through ``aio`` — i.e. as a durable child action.
    mock_aio.assert_awaited_once_with(city="Paris")


@pytest.mark.asyncio
async def test_concurrent_tool_calls_are_served_together(tmp_path):
    """Each call is its own connection, so the harness can fan out."""
    weather = _weather_tool("ds_bridge_c")
    bridge = ToolBridge([weather])
    try:
        await bridge.start(tmp_path)
        with patch.object(weather.task, "aio", new_callable=AsyncMock, side_effect=lambda city: f"sunny in {city}"):
            results = await asyncio.gather(
                _call_shim(tmp_path, "get_weather", {"city": "Paris"}),
                _call_shim(tmp_path, "get_weather", {"city": "Tokyo"}),
                _call_shim(tmp_path, "get_weather", {"city": "Lima"}),
            )
    finally:
        await bridge.stop()

    assert [r.returncode for r in results] == [0, 0, 0]
    assert sorted(r.stdout.strip() for r in results) == ["sunny in Lima", "sunny in Paris", "sunny in Tokyo"]


@pytest.mark.asyncio
async def test_a_failing_task_is_reported_to_the_model_not_raised(tmp_path):
    """A tool error becomes a non-zero exit + stderr, so the agent can react."""
    weather = _weather_tool("ds_bridge_d")
    bridge = ToolBridge([weather])
    try:
        await bridge.start(tmp_path)
        boom = RuntimeError("upstream is down")
        with patch.object(weather.task, "aio", new_callable=AsyncMock, side_effect=boom):
            done = await _call_shim(tmp_path, "get_weather", {"city": "Paris"})
    finally:
        await bridge.stop()

    assert done.returncode == 1
    assert "RuntimeError: upstream is down" in done.stderr


@pytest.mark.asyncio
async def test_invalid_json_arguments_are_rejected_by_the_shim(tmp_path):
    bridge = ToolBridge([_weather_tool("ds_bridge_e")])
    try:
        await bridge.start(tmp_path)
        shim = tmp_path / TOOLS_DIRNAME / "get_weather"
        done = await asyncio.to_thread(
            subprocess.run, [str(shim), "not json"], capture_output=True, text=True, timeout=30
        )
    finally:
        await bridge.stop()

    assert done.returncode == 2
    assert "must be a JSON object" in done.stderr


@pytest.mark.asyncio
async def test_unknown_tool_is_answered_with_the_available_names(tmp_path):
    bridge = ToolBridge([_weather_tool("ds_bridge_f")])
    try:
        await bridge.start(tmp_path)
        response = await bridge._dispatch({"tool": "nope", "args": {}})
    finally:
        await bridge.stop()

    assert response["ok"] is False
    assert "unknown tool 'nope'" in response["error"]
    assert "get_weather" in response["error"]


@pytest.mark.asyncio
async def test_stop_removes_the_socket(tmp_path):
    bridge = ToolBridge([_weather_tool("ds_bridge_g")])
    await bridge.start(tmp_path)
    socket_path = bridge.socket_path
    await bridge.stop()
    assert not os.path.exists(socket_path)


@pytest.mark.asyncio
async def test_no_tools_means_no_socket_and_no_instructions(tmp_path):
    bridge = ToolBridge([])
    try:
        await bridge.start(tmp_path)
        assert bridge.socket_path is None
        assert bridge.instructions() == ""
        assert not (tmp_path / TOOLS_DIRNAME).exists()
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_instructions_tell_the_model_how_to_call_each_tool(tmp_path):
    """The harness has no tool-declaration channel, so this prose is the contract."""
    env = flyte.TaskEnvironment("ds_bridge_h")

    @tool
    @env.task
    async def book(city: str, nights: int) -> str:
        """Book a stay."""
        return city

    bridge = ToolBridge([book])
    try:
        await bridge.start(tmp_path)
        manual = bridge.instructions()
    finally:
        await bridge.stop()

    assert "book(city: string, nights: integer) — Book a stay." in manual
    assert f'{TOOLS_DIRNAME}/book \'{{"city": "...", "nights": 0}}\'' in manual


@pytest.mark.asyncio
async def test_tool_calls_are_recorded_on_the_timeline(tmp_path):
    weather = _weather_tool("ds_bridge_i")
    rows = []

    class RecordingTimeline:
        def row(self, **kwargs):
            rows.append(kwargs)

    bridge = ToolBridge([weather], timeline=RecordingTimeline())
    try:
        await bridge.start(tmp_path)
        with patch.object(weather.task, "aio", new_callable=AsyncMock, return_value="sunny in Paris"):
            await bridge._dispatch({"tool": "get_weather", "args": {"city": "Paris"}})
    finally:
        await bridge.stop()

    assert [r["meta"] for r in rows] == ["tool", "tool result"]
    assert all(r["label"] == "get_weather" for r in rows)
