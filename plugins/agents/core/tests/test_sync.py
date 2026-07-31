"""Tests for the sync bridge (run_coro_sync / sync_variant)."""

import asyncio
import contextvars
import inspect

import pytest

from flyteplugins.agents.core import run_coro_sync, sync_variant

_var: contextvars.ContextVar[str] = contextvars.ContextVar("agents_sync_test", default="unset")


async def _echo(value: str, *, suffix: str = "") -> str:
    await asyncio.sleep(0)
    return value + suffix


def test_run_coro_sync_returns_value():
    assert run_coro_sync(_echo("hello")) == "hello"


def test_run_coro_sync_propagates_exceptions():
    async def _boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_coro_sync(_boom())


def test_run_coro_sync_copies_contextvars():
    async def _read() -> str:
        return _var.get()

    token = _var.set("task-context")
    try:
        assert run_coro_sync(_read()) == "task-context"
    finally:
        _var.reset(token)


def test_run_coro_sync_inside_running_loop_thread():
    # A Flyte sync task body executes on a thread that already has a running
    # event loop (run_sync_with_loop), where asyncio.run() would raise. The
    # bridge must still work from that shape.
    async def _outer() -> str:
        return run_coro_sync(_echo("nested"))

    assert asyncio.run(_outer()) == "nested"


def test_run_coro_sync_reuses_loop_across_calls():
    # Loop persistence is what keeps SDK-cached async resources (HTTP clients,
    # pools) usable across repeated run_agent_sync calls from the same thread.
    async def _get_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    assert run_coro_sync(_get_loop()) is run_coro_sync(_get_loop())


def test_sync_variant_wraps_and_preserves_signature():
    echo_sync = sync_variant(_echo)
    assert echo_sync("a", suffix="b") == "ab"
    assert not inspect.iscoroutinefunction(echo_sync)
    assert echo_sync.__name__ == "_echo_sync"
    assert "suffix" in inspect.signature(echo_sync).parameters
