"""Tests for ActionDetails.watch resilience to transient stream interruptions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.workflow import run_service_pb2

from flyte.remote._action import ActionDetails, _is_transient_watch_error

ACTION_ID = identifier_pb2.ActionIdentifier(name="a0")


def _resp(phase: int) -> run_service_pb2.WatchActionDetailsResponse:
    r = run_service_pb2.WatchActionDetailsResponse()
    r.details.status.phase = phase
    return r


RUNNING = _resp(phase_pb2.ACTION_PHASE_RUNNING)
SUCCEEDED = _resp(phase_pb2.ACTION_PHASE_SUCCEEDED)


def _stream(items):
    """An async iterator that yields responses and raises any exception it encounters."""

    async def gen():
        for item in items:
            if isinstance(item, BaseException):
                raise item
            yield item

    return gen()


def _client_with_streams(streams):
    """A fake client whose watch_action_details returns each stream in order."""
    it = iter(streams)
    client = MagicMock()
    client.run_service.watch_action_details = MagicMock(side_effect=lambda request: _stream(next(it)))
    return client


async def _collect():
    phases = []
    async for ad in ActionDetails.watch.aio(ACTION_ID):
        phases.append(ad.raw_phase)
    return phases


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr("flyte.remote._action._WATCH_RECONNECT_INITIAL_BACKOFF_SECS", 0.001)
    monkeypatch.setattr("flyte.remote._action._WATCH_RECONNECT_MAX_BACKOFF_SECS", 0.002)


@pytest.mark.asyncio
async def test_reconnects_after_transient_error(fast_backoff):
    client = _client_with_streams(
        [
            [RUNNING, ConnectError(Code.CANCELED, "stream reset")],
            [SUCCEEDED],
        ]
    )
    with patch("flyte.remote._action.get_client", return_value=client), patch("flyte.remote._action.ensure_client"):
        phases = await _collect()
    assert phases == [phase_pb2.ACTION_PHASE_RUNNING, phase_pb2.ACTION_PHASE_SUCCEEDED]
    assert client.run_service.watch_action_details.call_count == 2


@pytest.mark.asyncio
async def test_reconnects_after_clean_close_before_terminal(fast_backoff):
    client = _client_with_streams(
        [
            [RUNNING],  # stream ends cleanly while still running
            [SUCCEEDED],
        ]
    )
    with patch("flyte.remote._action.get_client", return_value=client), patch("flyte.remote._action.ensure_client"):
        phases = await _collect()
    assert phases == [phase_pb2.ACTION_PHASE_RUNNING, phase_pb2.ACTION_PHASE_SUCCEEDED]
    assert client.run_service.watch_action_details.call_count == 2


@pytest.mark.asyncio
async def test_non_transient_error_raises(fast_backoff):
    client = _client_with_streams([[RUNNING, ConnectError(Code.INTERNAL, "boom")]])
    with patch("flyte.remote._action.get_client", return_value=client), patch("flyte.remote._action.ensure_client"):
        with pytest.raises(ConnectError):
            await _collect()


@pytest.mark.asyncio
async def test_gives_up_after_max_consecutive_failures(fast_backoff):
    from flyte.remote._action import _WATCH_RECONNECT_MAX_ATTEMPTS

    err = ConnectError(Code.UNAVAILABLE, "down")
    client = _client_with_streams([[err]] * (_WATCH_RECONNECT_MAX_ATTEMPTS + 1))
    with patch("flyte.remote._action.get_client", return_value=client), patch("flyte.remote._action.ensure_client"):
        with pytest.raises(ConnectError):
            await _collect()
    assert client.run_service.watch_action_details.call_count == _WATCH_RECONNECT_MAX_ATTEMPTS + 1


@pytest.mark.asyncio
async def test_delivered_update_resets_failure_budget(fast_backoff):
    from flyte.remote._action import _WATCH_RECONNECT_MAX_ATTEMPTS

    # More total interruptions than the budget, but every subscription delivers an update
    # before dying — so the consecutive-failure counter keeps resetting and the watch survives.
    streams = [[RUNNING, ConnectError(Code.UNAVAILABLE, "blip")] for _ in range(_WATCH_RECONNECT_MAX_ATTEMPTS + 3)]
    streams.append([SUCCEEDED])
    client = _client_with_streams(streams)
    with patch("flyte.remote._action.get_client", return_value=client), patch("flyte.remote._action.ensure_client"):
        phases = await _collect()
    assert phases[-1] == phase_pb2.ACTION_PHASE_SUCCEEDED
    assert len(phases) == _WATCH_RECONNECT_MAX_ATTEMPTS + 4


def test_transient_classifier():
    import pyqwest

    assert _is_transient_watch_error(ConnectError(Code.UNAVAILABLE, "x"))
    assert _is_transient_watch_error(ConnectError(Code.DEADLINE_EXCEEDED, "x"))
    assert _is_transient_watch_error(ConnectError(Code.CANCELED, "x"))
    assert not _is_transient_watch_error(ConnectError(Code.INTERNAL, "x"))
    assert not _is_transient_watch_error(ConnectError(Code.PERMISSION_DENIED, "x"))
    assert _is_transient_watch_error(ConnectionResetError("reset"))
    assert _is_transient_watch_error(TimeoutError("timed out"))
    assert not _is_transient_watch_error(ValueError("nope"))
    assert _is_transient_watch_error(pyqwest.StreamError("Error reading content", 1))


def test_wrapped_stream_reset_is_transient():
    """The shape connectrpc actually raises: an RST_STREAM mapped onto ConnectError.

    `_client_async._send_request_bidi_stream` does `raise rst_err from e`, and
    `maybe_map_stream_reset` sends NO_ERROR/INTERNAL_ERROR/PROTOCOL_ERROR resets to
    Code.INTERNAL. Only the __cause__ separates an ALB resetting an idle stream from a
    real server-side INTERNAL, which is built from the response body and has no cause.
    """
    import pyqwest

    try:
        raise ConnectError(Code.INTERNAL, "Error reading content") from pyqwest.StreamError(
            "Error reading content", 1
        )
    except ConnectError as e:
        assert _is_transient_watch_error(e)

    # A server-side INTERNAL with no transport cause stays fatal.
    try:
        raise ConnectError(Code.INTERNAL, "boom") from ValueError("server bug")
    except ConnectError as e:
        assert not _is_transient_watch_error(e)
