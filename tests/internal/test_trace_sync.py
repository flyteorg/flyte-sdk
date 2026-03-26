"""Tests for sync trace module-level proxy functions (_fetch_action_outputs, _record_trace_action).

Covers: context propagation, syncify non-blocking, correctness, error propagation.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from flyte._internal.controllers._trace import TraceInfo
from flyte._trace import _fetch_action_outputs, _record_trace_action
from flyte.models import ActionID, NativeInterface
from flyte.syncify import syncify


def _make_trace_info(**overrides):
    """Helper to build a TraceInfo with sensible defaults."""
    defaults = {
        "name": "test_func",
        "action": ActionID(name="test-action", run_name="test-run"),
        "interface": MagicMock(spec=NativeInterface),
        "inputs_path": "/tmp/inputs",
    }
    defaults.update(overrides)
    return TraceInfo(**defaults)


# ---------------------------------------------------------------------------
# 1. Correctness - ok=True (cached) and ok=False (execute) paths
# ---------------------------------------------------------------------------


def test_fetch_action_outputs_returns_ok_true():
    """_fetch_action_outputs correctly returns (TraceInfo, True) from controller."""
    info = _make_trace_info(output=42)
    controller = MagicMock()
    controller.get_action_outputs = AsyncMock(return_value=(info, True))
    iface = MagicMock(spec=NativeInterface)

    def sample_func(x: int) -> int:
        return x

    result_info, ok = _fetch_action_outputs(controller, iface, sample_func, 5)

    assert ok is True
    assert result_info is info
    assert result_info.output == 42
    controller.get_action_outputs.assert_awaited_once_with(iface, sample_func, 5)


def test_fetch_action_outputs_returns_ok_false():
    """_fetch_action_outputs returns (TraceInfo, False) when no cached result."""
    info = _make_trace_info()
    controller = MagicMock()
    controller.get_action_outputs = AsyncMock(return_value=(info, False))
    iface = MagicMock(spec=NativeInterface)

    def sample_func(x: int) -> int:
        return x

    result_info, ok = _fetch_action_outputs(controller, iface, sample_func, 7)

    assert ok is False
    assert result_info is info
    controller.get_action_outputs.assert_awaited_once_with(iface, sample_func, 7)


def test_record_trace_action_calls_controller():
    """_record_trace_action calls controller.record_trace with the given info."""
    info = _make_trace_info()
    controller = MagicMock()
    controller.record_trace = AsyncMock(return_value=None)

    _record_trace_action(controller, info)

    controller.record_trace.assert_awaited_once_with(info)


# ---------------------------------------------------------------------------
# 2. Error propagation
# ---------------------------------------------------------------------------


def test_fetch_action_outputs_propagates_error():
    """Exceptions from controller.get_action_outputs propagate through the proxy."""
    controller = MagicMock()
    controller.get_action_outputs = AsyncMock(side_effect=RuntimeError("fetch failed"))
    iface = MagicMock(spec=NativeInterface)

    def sample_func():
        pass

    with pytest.raises(RuntimeError, match="fetch failed"):
        _fetch_action_outputs(controller, iface, sample_func)


def test_record_trace_action_propagates_error():
    """Exceptions from controller.record_trace propagate through the proxy."""
    info = _make_trace_info()
    controller = MagicMock()
    controller.record_trace = AsyncMock(side_effect=ValueError("record failed"))

    with pytest.raises(ValueError, match="record failed"):
        _record_trace_action(controller, info)


# ---------------------------------------------------------------------------
# 3. Syncify non-blocking - global syncify loop stays responsive
# ---------------------------------------------------------------------------


def test_sync_trace_proxy_does_not_block_syncify_loop():
    """The global syncify loop remains responsive while trace proxy calls execute.

    Strategy: run a slow trace proxy call in a background thread, then verify that
    a *separate* @syncify call on the global loop completes promptly.
    """

    @syncify
    async def ping():
        """A trivial function on the global syncify loop."""
        return "pong"

    # Simulate a slow controller.get_action_outputs
    slow_info = _make_trace_info()

    async def _slow_get(iface, func, *args, **kwargs):
        await asyncio.sleep(0.5)
        return slow_info, False

    slow_controller = MagicMock()
    slow_controller.get_action_outputs = _slow_get
    iface = MagicMock(spec=NativeInterface)

    def sample_func():
        pass

    # Start slow trace proxy call in a background thread
    trace_result = {}

    def run_slow_trace():
        try:
            result = _fetch_action_outputs(slow_controller, iface, sample_func)
            trace_result["ok"] = result
        except Exception as e:
            trace_result["error"] = e

    slow_thread = threading.Thread(target=run_slow_trace)
    slow_thread.start()

    # Give the slow call a moment to start
    time.sleep(0.1)

    # Now call ping() on the global syncify loop - it should complete quickly
    start = time.monotonic()
    result = ping()
    elapsed = time.monotonic() - start

    assert result == "pong"
    # ping() should finish well within 2 seconds even though the trace call takes 0.5s
    assert elapsed < 2.0, f"Global syncify loop was blocked for {elapsed:.2f}s"

    slow_thread.join(timeout=5)
    assert "error" not in trace_result, f"Trace proxy raised: {trace_result.get('error')}"


# ---------------------------------------------------------------------------
# 4. Context propagation - controller sees args faithfully
# ---------------------------------------------------------------------------


def test_fetch_action_outputs_passes_args_and_kwargs():
    """All positional and keyword args are forwarded to the controller."""
    captured = {}

    async def _capture_get(iface, func, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _make_trace_info(), True

    controller = MagicMock()
    controller.get_action_outputs = _capture_get
    iface = MagicMock(spec=NativeInterface)

    def sample_func(a, b, key=None):
        pass

    _fetch_action_outputs(controller, iface, sample_func, 1, 2, key="val")

    assert captured["args"] == (1, 2)
    assert captured["kwargs"] == {"key": "val"}
