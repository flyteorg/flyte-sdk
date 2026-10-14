"""Tests for flyte.map / flyte.map.aio in local mode.

Regression tests for https://github.com/flyteorg/flyte-sdk/issues/1359:
in local mode the map local branch yielded bare, un-awaited coroutines
instead of resolved task results.
"""

import inspect

import pytest

import flyte

env = flyte.TaskEnvironment(name="map_local_test")


@env.task
async def echo(x: str) -> str:
    return f"echo:{x}"


@env.task
def echo_sync(x: str) -> str:
    return f"echo:{x}"


@env.task
async def aio_parent() -> list[str]:
    out = []
    async for r in flyte.map.aio(echo, ["a", "b", "c"], concurrency=0, return_exceptions=False):
        assert not inspect.iscoroutine(r), f"map.aio yielded an un-awaited coroutine: {r!r}"
        out.append(r)
    return out


@env.task
def sync_parent() -> list[str]:
    out = []
    for r in flyte.map(echo_sync, ["a", "b", "c"], return_exceptions=False):
        assert not inspect.iscoroutine(r), f"map yielded an un-awaited coroutine: {r!r}"
        out.append(r)
    return out


@pytest.mark.asyncio
async def test_map_aio_local_yields_results():
    """flyte.map.aio inside a local-mode parent task must yield awaited results."""
    await flyte.init.aio()
    result = await flyte.with_runcontext(mode="local").run.aio(aio_parent)
    assert result.outputs()[0] == ["echo:a", "echo:b", "echo:c"]


def test_map_sync_local_yields_results():
    """flyte.map inside a local-mode parent task must yield awaited results."""
    flyte.init()
    result = flyte.with_runcontext(mode="local").run(sync_parent)
    assert result.outputs()[0] == ["echo:a", "echo:b", "echo:c"]


@env.task
def sync_parent_async_child() -> list[str]:
    out = []
    for r in flyte.map(echo, ["a", "b", "c"], return_exceptions=False):
        assert not inspect.iscoroutine(r), f"map yielded an un-awaited coroutine: {r!r}"
        out.append(r)
    return out


def test_map_sync_local_async_child_yields_results():
    """Blocking flyte.map over an async task in local mode must yield awaited results."""
    flyte.init()
    result = flyte.with_runcontext(mode="local").run(sync_parent_async_child)
    assert result.outputs()[0] == ["echo:a", "echo:b", "echo:c"]


@env.task
async def fail_on_b(x: str) -> str:
    if x == "b":
        raise ValueError(f"boom:{x}")
    return f"echo:{x}"


@env.task
async def aio_parent_return_exceptions() -> list[str]:
    out = []
    async for r in flyte.map.aio(fail_on_b, ["a", "b", "c"], return_exceptions=True):
        assert not inspect.iscoroutine(r), f"map.aio yielded an un-awaited coroutine: {r!r}"
        out.append(f"error:{r}" if isinstance(r, Exception) else r)
    return out


@pytest.mark.asyncio
async def test_map_aio_local_return_exceptions():
    """With return_exceptions=True, local-mode map.aio must yield the exception, not a coroutine."""
    await flyte.init.aio()
    result = await flyte.with_runcontext(mode="local").run.aio(aio_parent_return_exceptions)
    outputs = result.outputs()[0]
    assert outputs[0] == "echo:a"
    assert outputs[1].startswith("error:")
    assert "boom:b" in outputs[1]
    assert outputs[2] == "echo:c"
