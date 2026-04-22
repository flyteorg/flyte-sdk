"""
Tests for the event system: _event.py, remote/_event.py, local controller event
methods, and PendingEvent.
"""

from __future__ import annotations

import sys
import threading
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte.errors
from flyte._event import EventWebhook, _Event, new_event
from flyte._internal.controllers._local_controller import _substitute_callback_uri
from flyte.cli._tui._tracker import PendingEvent

# event_service_pb2 doesn't exist yet (proto not compiled), so inject a mock
# before importing the remote Event module.
_mock_event_service_pb2 = MagicMock()
_mock_event_service_pb2.EventPayload = MagicMock
sys.modules.setdefault("flyteidl2.workflow.event_service_pb2", _mock_event_service_pb2)

# Ensure the module-level name is wired up
import flyte.remote._event as _remote_event_mod  # noqa: E402
from flyte.remote._event import Event, _encode_payload  # noqa: E402

_remote_event_mod.event_service_pb2 = _mock_event_service_pb2


# ---------------------------------------------------------------------------
# _Event dataclass
# ---------------------------------------------------------------------------


class TestEventCreation:
    def test_defaults(self):
        e = _Event(name="ev1")
        assert e.name == "ev1"
        assert e.prompt == "Approve?"
        assert e.prompt_type == "text"
        assert e.data_type is bool
        assert e.description == ""
        assert e.timeout is None
        assert e._timeout_seconds is None

    def test_custom_fields(self):
        e = _Event(
            name="ev2",
            prompt="Continue?",
            prompt_type="markdown",
            data_type=str,
            description="some desc",
        )
        assert e.prompt == "Continue?"
        assert e.prompt_type == "markdown"
        assert e.data_type is str
        assert e.description == "some desc"

    @pytest.mark.parametrize("dt", [bool, int, float, str])
    def test_valid_data_types(self, dt):
        e = _Event(name="ev", data_type=dt)
        assert e.data_type is dt

    @pytest.mark.parametrize("dt", [list, dict, bytes, object])
    def test_invalid_data_types(self, dt):
        with pytest.raises(TypeError, match="Invalid data_type"):
            _Event(name="ev", data_type=dt)


class TestEventTimeout:
    def test_timeout_int_seconds(self):
        e = _Event(name="ev", timeout=30)
        assert e._timeout_seconds == 30.0

    def test_timeout_float_seconds(self):
        e = _Event(name="ev", timeout=1.5)
        assert e._timeout_seconds == 1.5

    def test_timeout_timedelta(self):
        e = _Event(name="ev", timeout=timedelta(minutes=2))
        assert e._timeout_seconds == 120.0

    def test_timeout_none(self):
        e = _Event(name="ev", timeout=None)
        assert e._timeout_seconds is None

    def test_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            _Event(name="ev", timeout=0)

    def test_timeout_negative_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            _Event(name="ev", timeout=-5)

    def test_timeout_negative_timedelta_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            _Event(name="ev", timeout=timedelta(seconds=-1))


class TestEventWait:
    @pytest.mark.asyncio
    async def test_wait_outside_task_context_raises(self):
        e = _Event(name="ev")
        with pytest.raises(RuntimeError, match="Events can only be awaited within a task context"):
            await e.wait.aio()


# ---------------------------------------------------------------------------
# new_event factory
# ---------------------------------------------------------------------------


class TestNewEvent:
    @pytest.mark.asyncio
    async def test_new_event_outside_context(self):
        """Outside task context, new_event still returns an _Event (no-op registration)."""
        e = await new_event.aio("my_event", prompt="Go?", data_type=int, timeout=10)
        assert isinstance(e, _Event)
        assert e.name == "my_event"
        assert e.prompt == "Go?"
        assert e.data_type is int
        assert e._timeout_seconds == 10.0

    @pytest.mark.asyncio
    async def test_new_event_registers_in_task_context(self):
        """In a task context, new_event calls controller.register_event."""
        mock_controller = MagicMock()
        mock_controller.register_event = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            e = await new_event.aio("reg_event")

        assert e.name == "reg_event"
        mock_controller.register_event.assert_awaited_once_with(e)

    @pytest.mark.asyncio
    async def test_new_event_with_timeout_timedelta(self):
        e = await new_event.aio("td_event", timeout=timedelta(seconds=45))
        assert e._timeout_seconds == 45.0


# ---------------------------------------------------------------------------
# PendingEvent (TUI tracker)
# ---------------------------------------------------------------------------


class TestPendingEvent:
    def test_set_and_wait(self):
        pe = PendingEvent(
            event_name="ev",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )
        pe.set_result(True)
        result = pe.wait_for_result()
        assert result is True
        assert pe.timed_out is False

    def test_timeout_expires(self):
        pe = PendingEvent(
            event_name="ev",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )
        result = pe.wait_for_result(timeout=0.05)
        assert result is None
        assert pe.timed_out is True

    def test_result_before_timeout(self):
        pe = PendingEvent(
            event_name="ev",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )

        def signal():
            pe.set_result(42)

        t = threading.Timer(0.02, signal)
        t.start()
        result = pe.wait_for_result(timeout=2.0)
        assert result == 42
        assert pe.timed_out is False
        t.join()

    def test_no_timeout(self):
        """Without timeout, wait blocks until signaled."""
        pe = PendingEvent(
            event_name="ev",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=str,
        )

        def signal():
            pe.set_result("hello")

        t = threading.Timer(0.02, signal)
        t.start()
        result = pe.wait_for_result(timeout=None)
        assert result == "hello"
        assert pe.timed_out is False
        t.join()


# ---------------------------------------------------------------------------
# Local controller: register_event / wait_for_event
# ---------------------------------------------------------------------------


class TestLocalControllerEvents:
    @pytest.fixture
    def controller(self):
        """Create a LocalController with mocked dependencies."""
        from flyte._internal.controllers._local_controller import LocalController

        mock_recorder = MagicMock()
        mock_recorder.record_event_waiting.return_value = None  # non-TUI mode
        controller = LocalController.__new__(LocalController)
        controller._registered_events = {}
        controller._recorder = mock_recorder
        return controller

    @pytest.mark.asyncio
    async def test_register_event(self, controller):
        e = _Event(name="ev1")
        await controller.register_event(e)
        assert "ev1" in controller._registered_events
        assert controller._registered_events["ev1"] is e

    @pytest.mark.asyncio
    async def test_register_non_event_raises(self, controller):
        with pytest.raises(TypeError, match="Expected _Event"):
            await controller.register_event("not-an-event")

    @pytest.mark.asyncio
    async def test_wait_non_event_raises(self, controller):
        with pytest.raises(TypeError, match="Expected _Event"):
            await controller.wait_for_event("not-an-event")

    @pytest.mark.asyncio
    async def test_wait_for_event_tui_mode(self, controller):
        """TUI mode: PendingEvent is returned by recorder, result comes from set_result."""
        pe = PendingEvent(
            event_name="ev",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )
        controller._recorder.record_event_waiting.return_value = pe

        mock_ctx = MagicMock()
        mock_tctx = MagicMock()
        mock_tctx.action.name = "act-1"
        mock_ctx.data.task_context = mock_tctx

        # Signal the event from another thread
        def signal():
            pe.set_result(True)

        t = threading.Timer(0.02, signal)
        t.start()

        e = _Event(name="ev")
        with patch("flyte._internal.controllers._local_controller.internal_ctx", return_value=mock_ctx):
            result = await controller.wait_for_event(e)

        assert result is True
        t.join()

    @pytest.mark.asyncio
    async def test_wait_for_event_tui_timeout(self, controller):
        """TUI mode: timeout triggers EventTimedoutError."""
        pe = PendingEvent(
            event_name="ev",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )
        controller._recorder.record_event_waiting.return_value = pe

        mock_ctx = MagicMock()
        mock_tctx = MagicMock()
        mock_tctx.action.name = "act-1"
        mock_ctx.data.task_context = mock_tctx

        e = _Event(name="ev", timeout=0.05)

        with patch("flyte._internal.controllers._local_controller.internal_ctx", return_value=mock_ctx):
            with pytest.raises(flyte.errors.EventTimedoutError, match="not signaled within"):
                await controller.wait_for_event(e)

    @pytest.mark.asyncio
    async def test_wait_for_event_console_timeout(self, controller):
        """Non-TUI mode with timeout: timeout triggers EventTimedoutError."""
        controller._recorder.record_event_waiting.return_value = None  # non-TUI

        mock_ctx = MagicMock()
        mock_tctx = MagicMock()
        mock_tctx.action.name = "act-1"
        mock_ctx.data.task_context = mock_tctx

        stop = threading.Event()

        # Make console prompt block until stop is set (so the thread can be cleaned up)
        def blocking_prompt(event):
            stop.wait()
            return True

        e = _Event(name="ev", timeout=0.05)

        try:
            with (
                patch("flyte._internal.controllers._local_controller.internal_ctx", return_value=mock_ctx),
                patch.object(controller, "_prompt_event_console", side_effect=blocking_prompt),
            ):
                with pytest.raises(flyte.errors.EventTimedoutError, match="not signaled within"):
                    await controller.wait_for_event(e)
        finally:
            stop.set()  # unblock the executor thread so it can exit


# ---------------------------------------------------------------------------
# remote Event — _encode_payload
# ---------------------------------------------------------------------------


class TestEncodePayload:
    def test_encode_bool(self):
        p = _encode_payload(True)
        assert p.bool_value is True

    def test_encode_int(self):
        p = _encode_payload(42)
        assert p.int_value == 42

    def test_encode_float(self):
        p = _encode_payload(3.14)
        assert p.float_value == 3.14

    def test_encode_str(self):
        p = _encode_payload("hello")
        assert p.string_value == "hello"

    def test_encode_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported payload type"):
            _encode_payload([1, 2, 3])


# ---------------------------------------------------------------------------
# remote Event — get / listall / signal
# ---------------------------------------------------------------------------


class TestRemoteEventGet:
    @pytest.mark.asyncio
    async def test_get_returns_event(self):
        mock_event_pb = MagicMock()
        mock_resp = MagicMock()
        mock_resp.event = mock_event_pb

        mock_client = MagicMock()
        mock_client.event_service.GetEvent = AsyncMock(return_value=mock_resp)

        mock_cfg = MagicMock()
        mock_cfg.org = "org"
        mock_cfg.project = "proj"
        mock_cfg.domain = "dev"

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=mock_cfg),
        ):
            result = await Event.get.aio("my_event", run_name="run1", action_name="act1")

        assert result is not None
        assert result.pb2 is mock_event_pb

    @pytest.mark.asyncio
    async def test_get_not_found_returns_none(self):
        import grpc

        # Create a real-ish AioRpcError (it's a subclass of grpc.RpcError which is Exception)
        class FakeAioRpcError(grpc.aio.AioRpcError):
            def __init__(self):
                pass

            def code(self):
                return grpc.StatusCode.NOT_FOUND

        mock_client = MagicMock()
        mock_client.event_service.GetEvent = AsyncMock(side_effect=FakeAioRpcError())

        mock_cfg = MagicMock()
        mock_cfg.org = "org"
        mock_cfg.project = "proj"
        mock_cfg.domain = "dev"

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=mock_cfg),
        ):
            result = await Event.get.aio("missing", run_name="run1", action_name="act1")

        assert result is None


class TestRemoteEventListall:
    @pytest.mark.asyncio
    async def test_listall_with_run_and_action(self):
        ev1 = MagicMock()
        ev2 = MagicMock()
        mock_resp = MagicMock()
        mock_resp.events = [ev1, ev2]
        mock_resp.token = ""

        mock_client = MagicMock()
        mock_client.event_service.ListEvents = AsyncMock(return_value=mock_resp)

        mock_cfg = MagicMock()
        mock_cfg.org = "org"
        mock_cfg.project = "proj"
        mock_cfg.domain = "dev"

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=mock_cfg),
        ):
            events = []
            async for e in Event.listall.aio(run_name="run1", action_name="act1"):
                events.append(e)

        assert len(events) == 2
        assert events[0].pb2 is ev1
        assert events[1].pb2 is ev2

    @pytest.mark.asyncio
    async def test_listall_run_only(self):
        mock_resp = MagicMock()
        mock_resp.events = []
        mock_resp.token = ""

        mock_client = MagicMock()
        mock_client.event_service.ListEvents = AsyncMock(return_value=mock_resp)

        mock_cfg = MagicMock()
        mock_cfg.org = "org"
        mock_cfg.project = "proj"
        mock_cfg.domain = "dev"

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=mock_cfg),
        ):
            events = []
            async for e in Event.listall.aio(run_name="run1"):
                events.append(e)

        assert len(events) == 0
        # Verify ListEvents was called (action_id=None is passed in the proto constructor)
        mock_client.event_service.ListEvents.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_listall_pagination(self):
        ev1 = MagicMock()
        ev2 = MagicMock()
        resp1 = MagicMock()
        resp1.events = [ev1]
        resp1.token = "page2"
        resp2 = MagicMock()
        resp2.events = [ev2]
        resp2.token = ""

        mock_client = MagicMock()
        mock_client.event_service.ListEvents = AsyncMock(side_effect=[resp1, resp2])

        mock_cfg = MagicMock()
        mock_cfg.org = "org"
        mock_cfg.project = "proj"
        mock_cfg.domain = "dev"

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=mock_cfg),
        ):
            events = []
            async for e in Event.listall.aio(run_name="run1"):
                events.append(e)

        assert len(events) == 2
        assert mock_client.event_service.ListEvents.await_count == 2


class TestRemoteEventSignal:
    @pytest.mark.asyncio
    async def test_signal_with_bool(self):
        mock_client = MagicMock()
        mock_client.event_service.SignalEvent = AsyncMock()

        mock_pb = MagicMock()
        mock_pb.event_id = MagicMock()

        event = Event(pb2=mock_pb)

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
        ):
            await event.signal.aio(True)

        mock_client.event_service.SignalEvent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_signal_with_string(self):
        mock_client = MagicMock()
        mock_client.event_service.SignalEvent = AsyncMock()

        mock_pb = MagicMock()
        mock_pb.event_id = MagicMock()

        event = Event(pb2=mock_pb)

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
        ):
            await event.signal.aio("go ahead")

        mock_client.event_service.SignalEvent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_signal_invalid_type_raises(self):
        event = Event(pb2=MagicMock())
        with pytest.raises(TypeError, match="payload must be bool, int, float, or str"):
            await event.signal.aio([1, 2])

    @pytest.mark.asyncio
    async def test_signal_with_int(self):
        mock_client = MagicMock()
        mock_client.event_service.SignalEvent = AsyncMock()

        mock_pb = MagicMock()
        mock_pb.event_id = MagicMock()

        event = Event(pb2=mock_pb)

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
        ):
            await event.signal.aio(42)

        mock_client.event_service.SignalEvent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_signal_with_float(self):
        mock_client = MagicMock()
        mock_client.event_service.SignalEvent = AsyncMock()

        mock_pb = MagicMock()
        mock_pb.event_id = MagicMock()

        event = Event(pb2=mock_pb)

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
        ):
            await event.signal.aio(3.14)

        mock_client.event_service.SignalEvent.assert_awaited_once()


# ---------------------------------------------------------------------------
# EventWebhook dataclass
# ---------------------------------------------------------------------------


class TestEventWebhook:
    def test_basic_creation(self):
        wh = EventWebhook(url="https://example.com/hook")
        assert wh.url == "https://example.com/hook"
        assert wh.payload is None

    def test_with_payload(self):
        wh = EventWebhook(
            url="https://example.com/hook",
            payload={"callback": "{callback_uri}", "event": "approval"},
        )
        assert wh.url == "https://example.com/hook"
        assert wh.payload == {"callback": "{callback_uri}", "event": "approval"}

    def test_event_with_webhook(self):
        wh = EventWebhook(url="https://example.com/hook")
        e = _Event(name="ev", webhook=wh)
        assert e.webhook is wh

    def test_event_webhook_defaults_none(self):
        e = _Event(name="ev")
        assert e.webhook is None


class TestNewEventWebhook:
    @pytest.mark.asyncio
    async def test_new_event_with_webhook(self):
        wh = EventWebhook(url="https://example.com/hook", payload={"cb": "{callback_uri}"})
        e = await new_event.aio("wh_event", webhook=wh)
        assert e.webhook is wh

    @pytest.mark.asyncio
    async def test_new_event_registers_with_webhook_in_task_context(self):
        mock_controller = MagicMock()
        mock_controller.register_event = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        wh = EventWebhook(url="https://example.com/hook")
        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            e = await new_event.aio("wh_reg_event", webhook=wh)

        assert e.webhook is wh
        mock_controller.register_event.assert_awaited_once_with(e)


# ---------------------------------------------------------------------------
# _substitute_callback_uri helper
# ---------------------------------------------------------------------------


class TestSubstituteCallbackUri:
    def test_string_replacement(self):
        assert _substitute_callback_uri("{callback_uri}", "http://x") == "http://x"

    def test_string_partial_replacement(self):
        result = _substitute_callback_uri("url={callback_uri}&done", "http://x")
        assert result == "url=http://x&done"

    def test_string_no_placeholder(self):
        assert _substitute_callback_uri("no placeholder", "http://x") == "no placeholder"

    def test_dict_replacement(self):
        result = _substitute_callback_uri(
            {"callback": "{callback_uri}", "static": "val"},
            "http://x",
        )
        assert result == {"callback": "http://x", "static": "val"}

    def test_nested_dict(self):
        result = _substitute_callback_uri(
            {"outer": {"inner": "{callback_uri}"}},
            "http://x",
        )
        assert result == {"outer": {"inner": "http://x"}}

    def test_list_replacement(self):
        result = _substitute_callback_uri(["{callback_uri}", "other"], "http://x")
        assert result == ["http://x", "other"]

    def test_non_string_passthrough(self):
        assert _substitute_callback_uri(42, "http://x") == 42
        assert _substitute_callback_uri(True, "http://x") is True
        assert _substitute_callback_uri(None, "http://x") is None


# ---------------------------------------------------------------------------
# Local controller: webhook firing
# ---------------------------------------------------------------------------


class TestEventWaitConditionProtocol:
    """Tests for the event wait() condition protocol: timeout, failure, and output handling."""

    @pytest.mark.asyncio
    async def test_wait_raises_event_timedout_error(self):
        """wait() should raise EventTimedoutError when the controller reports timeout."""
        mock_controller = MagicMock()
        mock_controller.wait_for_event = AsyncMock(
            side_effect=flyte.errors.EventTimedoutError("Event 'ev' was not signaled within the timeout period.")
        )

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        e = _Event(name="ev", data_type=bool, timeout=10)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            with pytest.raises(flyte.errors.EventTimedoutError, match="not signaled within"):
                await e.wait.aio()

    @pytest.mark.asyncio
    async def test_wait_raises_event_failed_error(self):
        """wait() should raise EventFailedError when the controller reports failure."""
        mock_controller = MagicMock()
        mock_controller.wait_for_event = AsyncMock(
            side_effect=flyte.errors.EventFailedError("Event 'ev' condition action failed.")
        )

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        e = _Event(name="ev", data_type=str)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            with pytest.raises(flyte.errors.EventFailedError, match="failed"):
                await e.wait.aio()

    @pytest.mark.asyncio
    async def test_wait_returns_bool_true(self):
        """wait() should return True for a bool event signaled with True."""
        mock_controller = MagicMock()
        mock_controller.wait_for_event = AsyncMock(return_value=True)

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        e = _Event(name="approve", data_type=bool)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await e.wait.aio()
            assert result is True

    @pytest.mark.asyncio
    async def test_wait_returns_bool_false(self):
        """wait() should return False for a bool event signaled with False."""
        mock_controller = MagicMock()
        mock_controller.wait_for_event = AsyncMock(return_value=False)

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        e = _Event(name="approve", data_type=bool)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await e.wait.aio()
            assert result is False

    @pytest.mark.asyncio
    async def test_wait_returns_int(self):
        """wait() should return an int for an int event."""
        mock_controller = MagicMock()
        mock_controller.wait_for_event = AsyncMock(return_value=42)

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        e = _Event(name="count", data_type=int)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await e.wait.aio()
            assert result == 42
            assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_wait_returns_float(self):
        """wait() should return a float for a float event."""
        mock_controller = MagicMock()
        mock_controller.wait_for_event = AsyncMock(return_value=3.14)

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        e = _Event(name="threshold", data_type=float)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await e.wait.aio()
            assert abs(result - 3.14) < 1e-6

    @pytest.mark.asyncio
    async def test_wait_returns_str(self):
        """wait() should return a str for a str event."""
        mock_controller = MagicMock()
        mock_controller.wait_for_event = AsyncMock(return_value="go ahead")

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        e = _Event(name="input", data_type=str)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await e.wait.aio()
            assert result == "go ahead"


class TestLocalControllerWebhook:
    @pytest.fixture
    def controller(self):
        from flyte._internal.controllers._local_controller import LocalController

        mock_recorder = MagicMock()
        mock_recorder.record_event_waiting.return_value = None
        controller = LocalController.__new__(LocalController)
        controller._registered_events = {}
        controller._recorder = mock_recorder
        return controller

    @pytest.mark.asyncio
    async def test_register_event_without_webhook_does_not_fire(self, controller):
        e = _Event(name="ev_no_wh")
        with patch.object(controller, "_fire_event_webhook", new_callable=AsyncMock) as mock_fire:
            await controller.register_event(e)
        mock_fire.assert_not_awaited()
        assert "ev_no_wh" in controller._registered_events

    @pytest.mark.asyncio
    async def test_register_event_with_webhook_fires(self, controller):
        wh = EventWebhook(url="https://example.com/hook", payload={"cb": "{callback_uri}"})
        e = _Event(name="ev_wh", webhook=wh)
        with patch.object(controller, "_fire_event_webhook", new_callable=AsyncMock) as mock_fire:
            await controller.register_event(e)
        mock_fire.assert_awaited_once_with(e)
        assert "ev_wh" in controller._registered_events

    @pytest.mark.asyncio
    async def test_fire_event_webhook_posts_with_substituted_payload(self, controller):
        wh = EventWebhook(
            url="https://example.com/hook",
            payload={"callback": "{callback_uri}", "event": "test"},
        )
        e = _Event(name="my_ev", webhook=wh)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await controller._fire_event_webhook(e)

        mock_client.post.assert_awaited_once_with(
            "https://example.com/hook",
            json={"callback": "local://events/my_ev/signal", "event": "test"},
            headers={"Content-Type": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_fire_event_webhook_no_payload(self, controller):
        wh = EventWebhook(url="https://example.com/hook")
        e = _Event(name="ev_nopayload", webhook=wh)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await controller._fire_event_webhook(e)

        mock_client.post.assert_awaited_once_with(
            "https://example.com/hook",
            json=None,
            headers={"Content-Type": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_fire_event_webhook_exception_logged_not_raised(self, controller):
        wh = EventWebhook(url="https://example.com/hook")
        e = _Event(name="ev_fail", webhook=wh)

        with patch("httpx.AsyncClient", side_effect=Exception("connection error")):
            # Should not raise
            await controller._fire_event_webhook(e)
