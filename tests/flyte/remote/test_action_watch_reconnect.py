"""ActionDetails.watch reconnects when the stream ends before a terminal phase.

A quiet watch stream (long image build, no phase change) can be dropped by an idle proxy such as
an ALB. Without a reconnect the caller sees the stream end, treats the action as finished, and
reads a still-running phase as a failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.workflow import run_definition_pb2, run_service_pb2

from flyte.remote._action import ActionDetails


def _action_id() -> identifier_pb2.ActionIdentifier:
    return identifier_pb2.ActionIdentifier(
        run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="r"),
        name="a0",
    )


def _resp(phase: phase_pb2.ActionPhase) -> run_service_pb2.WatchActionDetailsResponse:
    return run_service_pb2.WatchActionDetailsResponse(
        details=run_definition_pb2.ActionDetails(
            id=_action_id(),
            status=run_definition_pb2.ActionStatus(phase=phase),
        )
    )


def _stream(*items):
    """Build an async iterator that yields `items`; a ConnectError item is raised instead."""

    async def _gen():
        for item in items:
            if isinstance(item, ConnectError):
                raise item
            yield item

    return _gen()


def _client_yielding(*streams):
    """A mock client whose watch_action_details returns each stream in turn."""
    client = MagicMock()
    client.run_service.watch_action_details = MagicMock(side_effect=list(streams))
    return client


async def _collect(action_id):
    return [d.pb2.status.phase async for d in ActionDetails.watch.aio(action_id=action_id)]


RUNNING = phase_pb2.ACTION_PHASE_RUNNING
SUCCEEDED = phase_pb2.ACTION_PHASE_SUCCEEDED


@pytest.mark.asyncio
@patch("flyte.remote._action.asyncio.sleep")
@patch("flyte.remote._action.ensure_client", MagicMock())
async def test_reconnects_when_stream_ends_before_terminal(mock_sleep):
    """Stream 1 ends mid-build (the ALB drop); the watch must reconnect and see the terminal phase."""
    client = _client_yielding(
        _stream(_resp(RUNNING)),  # dropped without a terminal phase
        _stream(_resp(RUNNING), _resp(SUCCEEDED)),
    )
    with patch("flyte.remote._action.get_client", return_value=client):
        phases = await _collect(_action_id())

    assert phases == [RUNNING, RUNNING, SUCCEEDED]
    assert client.run_service.watch_action_details.call_count == 2
    mock_sleep.assert_awaited_once()


@pytest.mark.asyncio
@patch("flyte.remote._action.ensure_client", MagicMock())
async def test_no_reconnect_after_terminal_phase():
    client = _client_yielding(_stream(_resp(SUCCEEDED)))
    with patch("flyte.remote._action.get_client", return_value=client):
        phases = await _collect(_action_id())

    assert phases == [SUCCEEDED]
    assert client.run_service.watch_action_details.call_count == 1


@pytest.mark.asyncio
@patch("flyte.remote._action.ensure_client", MagicMock())
async def test_cancelled_stream_stops_without_reconnect():
    client = _client_yielding(_stream(ConnectError(Code.CANCELED, "cancelled")))
    with patch("flyte.remote._action.get_client", return_value=client):
        phases = await _collect(_action_id())

    assert phases == []
    assert client.run_service.watch_action_details.call_count == 1


@pytest.mark.asyncio
@patch("flyte.remote._action.ensure_client", MagicMock())
async def test_other_errors_still_raise():
    client = _client_yielding(_stream(ConnectError(Code.PERMISSION_DENIED, "nope")))
    with patch("flyte.remote._action.get_client", return_value=client):
        with pytest.raises(ConnectError):
            await _collect(_action_id())


@pytest.mark.asyncio
@patch("flyte.remote._action.asyncio.sleep")
@patch("flyte.remote._action.ensure_client", MagicMock())
async def test_backoff_grows_while_streams_end_empty(mock_sleep):
    client = _client_yielding(_stream(), _stream(), _stream(_resp(SUCCEEDED)))
    with patch("flyte.remote._action.get_client", return_value=client):
        await _collect(_action_id())

    delays = [call.args[0] for call in mock_sleep.await_args_list]
    assert delays == [2.0, 4.0]
