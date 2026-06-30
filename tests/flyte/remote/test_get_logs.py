"""Tests for Action.get_logs and Run.get_logs."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.logs.dataplane import payload_pb2
from flyteidl2.workflow import run_definition_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from flyte.remote._action import Action, ActionDetails
from flyte.remote._logs import _format_line
from flyte.remote._run import Run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action_id(run_name: str = "run-1", action_name: str = "a0") -> identifier_pb2.ActionIdentifier:
    run_id = identifier_pb2.RunIdentifier(name=run_name)
    return identifier_pb2.ActionIdentifier(run=run_id, name=action_name)


def _make_action(run_name: str = "run-1", action_name: str = "a0") -> Action:
    action_id = _make_action_id(run_name=run_name, action_name=action_name)
    pb2 = run_definition_pb2.Action(id=action_id)
    return Action(pb2=pb2)


def _make_run(run_name: str = "run-1", action_name: str = "a0") -> Run:
    action_id = _make_action_id(run_name=run_name, action_name=action_name)
    action_pb2 = run_definition_pb2.Action(id=action_id)
    run_pb2 = run_definition_pb2.Run(action=action_pb2)
    return Run(pb2=run_pb2)


def _make_logline(
    message: str,
    originator: payload_pb2.LogLineOriginator = payload_pb2.LogLineOriginator.USER,
    ts: datetime.datetime | None = None,
) -> payload_pb2.LogLine:
    logline = payload_pb2.LogLine(message=message, originator=originator)
    if ts is not None:
        timestamp = Timestamp()
        timestamp.FromDatetime(ts)
        logline.timestamp.CopyFrom(timestamp)
    return logline


def _make_done_details(attempts: int = 1) -> ActionDetails:
    """Return an ActionDetails whose done() == True and .attempts == attempts."""
    pb2 = run_definition_pb2.ActionDetails()
    pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
    pb2.status.attempts = attempts
    return ActionDetails(pb2=pb2)


async def _async_gen_from(lines: list):
    for line in lines:
        yield line


# ---------------------------------------------------------------------------
# _format_line (pure-function tests — no mocking needed)
# ---------------------------------------------------------------------------


class TestFormatLine:
    def test_user_line_plain_text(self):
        logline = _make_logline("hello world")
        result = _format_line(logline, show_ts=False, filter_system=False)
        assert result is not None
        assert "hello world" in result.plain

    def test_user_line_with_timestamp(self):
        ts = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        logline = _make_logline("hello world", ts=ts)
        result = _format_line(logline, show_ts=True, filter_system=False)
        assert result is not None
        assert "hello world" in result.plain
        assert "2024-06-01" in result.plain

    def test_system_line_kept_when_not_filtering(self):
        logline = _make_logline("system message", originator=payload_pb2.LogLineOriginator.SYSTEM)
        result = _format_line(logline, show_ts=False, filter_system=False)
        assert result is not None

    def test_system_line_filtered_out(self):
        logline = _make_logline("system message", originator=payload_pb2.LogLineOriginator.SYSTEM)
        result = _format_line(logline, show_ts=False, filter_system=True)
        assert result is None

    def test_flyte_bracket_line_filtered_out(self):
        logline = _make_logline("[flyte] internal message")
        result = _format_line(logline, show_ts=False, filter_system=True)
        assert result is None

    def test_flyte_errors_line_not_filtered(self):
        # Lines containing "flyte.errors" should NOT be filtered even if they contain "[flyte]"
        logline = _make_logline("[flyte] flyte.errors something")
        result = _format_line(logline, show_ts=False, filter_system=True)
        assert result is not None

    def test_flyte_bracket_line_kept_when_not_filtering(self):
        logline = _make_logline("[flyte] internal message")
        result = _format_line(logline, show_ts=False, filter_system=False)
        assert result is not None

    def test_returns_none_for_system_line_with_filter_system_enabled(self):
        """SYSTEM originator is always None when filter_system=True."""
        logline = _make_logline("anything", originator=payload_pb2.LogLineOriginator.SYSTEM)
        assert _format_line(logline, show_ts=False, filter_system=True) is None

    def test_plain_property_strips_rich_markup(self):
        logline = _make_logline("clean text")
        result = _format_line(logline, show_ts=False, filter_system=False)
        assert result is not None
        assert isinstance(result.plain, str)
        assert "clean text" in result.plain


# ---------------------------------------------------------------------------
# Action.get_logs
# ---------------------------------------------------------------------------


class TestActionGetLogs:
    """Tests for Action.get_logs — mocks out details() and Logs.tail.aio()."""

    @pytest.mark.asyncio
    async def test_yields_user_messages(self):
        action = _make_action()
        details = _make_done_details(attempts=1)

        lines = [
            _make_logline("line one"),
            _make_logline("line two"),
            _make_logline("line three"),
        ]

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=details)),
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from(lines))
            results = []
            async for line in action.get_logs.aio():
                results.append(line)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert "line one" in results[0]
        assert "line two" in results[1]
        assert "line three" in results[2]

    @pytest.mark.asyncio
    async def test_filters_system_lines(self):
        action = _make_action()
        details = _make_done_details(attempts=1)

        lines = [
            _make_logline("user message", originator=payload_pb2.LogLineOriginator.USER),
            _make_logline("system message", originator=payload_pb2.LogLineOriginator.SYSTEM),
            _make_logline("another user message", originator=payload_pb2.LogLineOriginator.USER),
        ]

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=details)),
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from(lines))
            results = []
            async for line in action.get_logs.aio(filter_system=True):
                results.append(line)

        assert len(results) == 2
        assert "user message" in results[0]
        assert "another user message" in results[1]

    @pytest.mark.asyncio
    async def test_does_not_filter_system_lines_by_default(self):
        action = _make_action()
        details = _make_done_details(attempts=1)

        lines = [
            _make_logline("user message", originator=payload_pb2.LogLineOriginator.USER),
            _make_logline("system message", originator=payload_pb2.LogLineOriginator.SYSTEM),
        ]

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=details)),
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from(lines))
            results = []
            async for line in action.get_logs.aio():
                results.append(line)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_includes_timestamp_when_show_ts(self):
        action = _make_action()
        details = _make_done_details(attempts=1)
        ts = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

        lines = [_make_logline("msg with ts", ts=ts)]

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=details)),
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from(lines))
            results = []
            async for line in action.get_logs.aio(show_ts=True):
                results.append(line)

        assert len(results) == 1
        assert "2024-06-01" in results[0]
        assert "msg with ts" in results[0]

    @pytest.mark.asyncio
    async def test_uses_latest_attempt_from_details(self):
        action = _make_action()
        details = _make_done_details(attempts=3)

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=details)),
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from([]))
            async for _ in action.get_logs.aio():
                pass
            mock_tail.aio.assert_called_once_with(action_id=action.action_id, attempt=3)

    @pytest.mark.asyncio
    async def test_uses_explicit_attempt(self):
        action = _make_action()
        details = _make_done_details(attempts=5)

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=details)),
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from([]))
            async for _ in action.get_logs.aio(attempt=2):
                pass
            mock_tail.aio.assert_called_once_with(action_id=action.action_id, attempt=2)

    @pytest.mark.asyncio
    async def test_waits_for_logs_if_not_running_or_done(self):
        """If details shows action is not yet running/done, wait() should be called."""
        action = _make_action()

        # First call: not running, not done
        pending_pb2 = run_definition_pb2.ActionDetails()
        pending_pb2.status.phase = phase_pb2.ACTION_PHASE_QUEUED
        pending_pb2.status.attempts = 1
        pending_details = ActionDetails(pb2=pending_pb2)

        # Second call (after wait): done
        done_details = _make_done_details(attempts=1)

        details_side_effect = [pending_details, done_details]

        with (
            patch.object(Action, "details", new=AsyncMock(side_effect=details_side_effect)),
            patch.object(Action, "wait", new=AsyncMock()) as mock_wait,
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from([]))
            async for _ in action.get_logs.aio():
                pass
            mock_wait.assert_awaited_once_with(wait_for="logs-ready")

    @pytest.mark.asyncio
    async def test_does_not_wait_if_already_running(self):
        action = _make_action()

        running_pb2 = run_definition_pb2.ActionDetails()
        running_pb2.status.phase = phase_pb2.ACTION_PHASE_RUNNING
        running_pb2.status.attempts = 1
        running_details = ActionDetails(pb2=running_pb2)

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=running_details)),
            patch.object(Action, "wait", new=AsyncMock()) as mock_wait,
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from([]))
            async for _ in action.get_logs.aio():
                pass
            mock_wait.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_filters_flyte_bracket_lines(self):
        action = _make_action()
        details = _make_done_details(attempts=1)

        lines = [
            _make_logline("real output"),
            _make_logline("[flyte] internal"),
            _make_logline("more output"),
        ]

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=details)),
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from(lines))
            results = []
            async for line in action.get_logs.aio(filter_system=True):
                results.append(line)

        assert len(results) == 2
        assert "real output" in results[0]
        assert "more output" in results[1]

    @pytest.mark.asyncio
    async def test_empty_log_stream(self):
        action = _make_action()
        details = _make_done_details(attempts=1)

        with (
            patch.object(Action, "details", new=AsyncMock(return_value=details)),
            patch("flyte.remote._action.Logs.tail") as mock_tail,
        ):
            mock_tail.aio = MagicMock(return_value=_async_gen_from([]))
            results = []
            async for line in action.get_logs.aio():
                results.append(line)

        assert results == []


# ---------------------------------------------------------------------------
# Run.get_logs
# ---------------------------------------------------------------------------


class TestRunGetLogs:
    """Tests for Run.get_logs — delegates to action.get_logs.aio()."""

    @pytest.mark.asyncio
    async def test_delegates_to_action_get_logs(self):
        run = _make_run()

        expected = ["line one", "line two"]

        async def fake_get_logs(*args, **kwargs):
            for line in expected:
                yield line

        with patch.object(run.action.__class__, "get_logs") as mock_get_logs:
            mock_get_logs.aio = MagicMock(return_value=fake_get_logs())
            results = []
            async for line in run.get_logs.aio():
                results.append(line)

        assert results == expected

    @pytest.mark.asyncio
    async def test_passes_attempt_to_action(self):
        run = _make_run()

        async def fake_get_logs(*args, **kwargs):
            return
            yield  # make it a generator

        with patch.object(run.action.__class__, "get_logs") as mock_get_logs:
            mock_get_logs.aio = MagicMock(return_value=fake_get_logs())
            async for _ in run.get_logs.aio(attempt=2):
                pass
            mock_get_logs.aio.assert_called_once_with(2, filter_system=False, show_ts=False)

    @pytest.mark.asyncio
    async def test_passes_filter_system_and_show_ts(self):
        run = _make_run()

        async def fake_get_logs(*args, **kwargs):
            return
            yield

        with patch.object(run.action.__class__, "get_logs") as mock_get_logs:
            mock_get_logs.aio = MagicMock(return_value=fake_get_logs())
            async for _ in run.get_logs.aio(filter_system=True, show_ts=True):
                pass
            mock_get_logs.aio.assert_called_once_with(None, filter_system=True, show_ts=True)
