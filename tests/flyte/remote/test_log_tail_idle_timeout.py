"""Tests for Logs.tail terminating when a completed action's log stream stays open."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.workflow import run_definition_pb2

from flyte.remote._action import Action, ActionDetails
from flyte.remote._logs import Logs


def _make_action_id() -> identifier_pb2.ActionIdentifier:
    return identifier_pb2.ActionIdentifier(run=identifier_pb2.RunIdentifier(name="run-1"), name="a0")


class _NeverYieldingStream:
    """A server stream that is accepted but never delivers a message or closes.

    This is what the log plane does for an action whose pod has already exited: the
    stream opens, then stays silent forever.
    """

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


def _client_returning(stream) -> MagicMock:
    client = MagicMock()
    client.dataproxy_service.tail_logs = MagicMock(return_value=stream)
    return client


class TestTailIdleTimeout:
    @pytest.mark.asyncio
    async def test_returns_when_stream_is_silent_and_idle_timeout_set(self):
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=_client_returning(_NeverYieldingStream())),
        ):

            async def collect():
                return [line async for line in Logs.tail.aio(action_id=_make_action_id(), idle_timeout=0.2)]

            # Without the idle timeout this never returns and the wait_for below fires.
            lines = await asyncio.wait_for(collect(), timeout=10)

        assert lines == []

    @pytest.mark.asyncio
    async def test_show_logs_arms_idle_timeout_for_a_completed_action(self):
        pb2 = run_definition_pb2.ActionDetails()
        pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        pb2.status.attempts = 1
        details = ActionDetails(pb2=pb2)

        action = Action(pb2=run_definition_pb2.Action(id=_make_action_id()))

        with (
            patch.object(Action, "details", return_value=details) as mock_details,
            patch("flyte.remote._action.Logs.create_viewer") as mock_viewer,
        ):
            mock_details.return_value = details

            async def _details(self):
                return details

            with patch.object(Action, "details", _details):
                await action.show_logs.aio()

        _, kwargs = mock_viewer.call_args
        assert kwargs.get("idle_timeout") is not None, (
            "a completed action's log stream never closes on its own, so the tail must be bounded"
        )
