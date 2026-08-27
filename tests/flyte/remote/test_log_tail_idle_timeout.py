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


class TestTailStopsWhenActionFinishesDuringTail:
    """`run --follow` attaches while the action is still running.

    The action is therefore never terminal when show_logs decides whether to bound the
    tail, so a one-time terminality check leaves the stream unbounded and the CLI hangs
    once the pod exits. Terminality has to be re-checked while streaming.
    """

    @pytest.mark.asyncio
    async def test_stops_once_the_action_becomes_terminal(self):
        terminal = {"value": False}

        async def is_terminal():
            # Terminal only from the second idle tick onward, mimicking a run that
            # finishes a moment after the tail attaches.
            if terminal["value"]:
                return True
            terminal["value"] = True
            return False

        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=_client_returning(_NeverYieldingStream())),
        ):

            async def collect():
                return [
                    line
                    async for line in Logs.tail.aio(
                        action_id=_make_action_id(), idle_timeout=0.1, is_terminal=is_terminal
                    )
                ]

            lines = await asyncio.wait_for(collect(), timeout=10)

        assert lines == []
        assert terminal["value"] is True

    @pytest.mark.asyncio
    async def test_keeps_waiting_while_the_action_is_still_running(self):
        checks = {"count": 0}

        async def never_terminal():
            checks["count"] += 1
            return False

        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=_client_returning(_NeverYieldingStream())),
        ):

            async def collect():
                return [
                    line
                    async for line in Logs.tail.aio(
                        action_id=_make_action_id(), idle_timeout=0.05, is_terminal=never_terminal
                    )
                ]

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(collect(), timeout=1.0)

        # A quiet but still-running action must not be cut off.
        assert checks["count"] > 1

    @pytest.mark.asyncio
    async def test_show_logs_arms_the_tail_for_a_running_action(self):
        pb2 = run_definition_pb2.ActionDetails()
        pb2.status.phase = phase_pb2.ACTION_PHASE_RUNNING
        pb2.status.attempts = 1
        details = ActionDetails(pb2=pb2)

        action = Action(pb2=run_definition_pb2.Action(id=_make_action_id()))

        async def _details(self):
            return details

        with (
            patch.object(Action, "details", _details),
            patch("flyte.remote._action.Logs.create_viewer") as mock_viewer,
        ):
            await action.show_logs.aio()

        _, kwargs = mock_viewer.call_args
        assert kwargs.get("idle_timeout") is not None, "a running action can finish mid-tail, so it must be bounded"
        assert kwargs.get("is_terminal") is not None, "bounding a running action requires a terminality re-check"
