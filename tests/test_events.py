"""
Tests for the event system: _event.py, remote/_event.py, local controller event
methods, and PendingEvent.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte.errors
from flyte._event import EventWebhook, _Event, new_event
from flyte._internal.controllers._local_controller import _substitute_callback_uri
from flyte.cli._tui._tracker import PendingEvent
from flyte.remote._event import Event, _encode_payload

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


def _mock_condition_action(name: str, parent: str = "", run_name: str = "run1") -> MagicMock:
    """Build a mock condition Action whose Event.name resolves to ``name``.

    ``id`` is a real ActionIdentifier so ``signal`` can build a real SignalEventRequest.
    """
    from flyteidl2.common import identifier_pb2

    action = MagicMock()
    action.id = identifier_pb2.ActionIdentifier(
        run=identifier_pb2.RunIdentifier(name=run_name),
        name=name,
    )
    action.metadata.HasField.return_value = True
    action.metadata.condition.name = name
    action.metadata.parent = parent
    return action


def _mock_list_actions_client(*pages) -> MagicMock:
    """Build a mock client whose run_service.list_actions yields the given pages.

    Each page is a list of actions; tokens chain the pages and the last is empty.
    """
    responses = []
    for i, actions in enumerate(pages):
        resp = MagicMock()
        resp.actions = actions
        resp.token = f"page{i + 1}" if i < len(pages) - 1 else ""
        responses.append(resp)

    mock_client = MagicMock()
    mock_client.run_service.list_actions = AsyncMock(side_effect=responses)
    return mock_client


_CFG = {"org": "org", "project": "proj", "domain": "dev"}


def _mock_cfg() -> MagicMock:
    cfg = MagicMock(**_CFG)
    return cfg


class TestRemoteEventGet:
    @pytest.mark.asyncio
    async def test_get_returns_event(self):
        target = _mock_condition_action("my_event", parent="act1")
        mock_client = _mock_list_actions_client([_mock_condition_action("other"), target])

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=_mock_cfg()),
        ):
            result = await Event.get.aio("my_event", run_name="run1", action_name="act1")

        assert result is not None
        assert result.pb2 is target
        assert result.name == "my_event"

    @pytest.mark.asyncio
    async def test_get_not_found_returns_none(self):
        mock_client = _mock_list_actions_client([_mock_condition_action("something_else")])

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=_mock_cfg()),
        ):
            result = await Event.get.aio("missing", run_name="run1")

        assert result is None


class TestRemoteEventListall:
    @pytest.mark.asyncio
    async def test_listall_filters_by_parent_action(self):
        match = _mock_condition_action("e1", parent="act1")
        other = _mock_condition_action("e2", parent="act2")
        mock_client = _mock_list_actions_client([match, other])

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=_mock_cfg()),
        ):
            events = [e async for e in Event.listall.aio(run_name="run1", action_name="act1")]

        assert [e.pb2 for e in events] == [match]

    @pytest.mark.asyncio
    async def test_listall_filters_condition_actions_server_side(self):
        mock_client = _mock_list_actions_client([])

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=_mock_cfg()),
        ):
            events = [e async for e in Event.listall.aio(run_name="run1")]

        assert events == []
        mock_client.run_service.list_actions.assert_awaited_once()
        # The request must carry a server-side action_type == CONDITION (3) filter.
        req = mock_client.run_service.list_actions.await_args.args[0]
        filt = req.request.filters[0]
        assert filt.field == "action_type"
        assert list(filt.values) == ["3"]

    @pytest.mark.asyncio
    async def test_listall_pagination(self):
        ev1 = _mock_condition_action("e1")
        ev2 = _mock_condition_action("e2")
        mock_client = _mock_list_actions_client([ev1], [ev2])

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
            patch("flyte.remote._event.get_init_config", return_value=_mock_cfg()),
        ):
            events = [e async for e in Event.listall.aio(run_name="run1")]

        assert [e.pb2 for e in events] == [ev1, ev2]
        assert mock_client.run_service.list_actions.await_count == 2


class TestRemoteEventSignal:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [True, "go ahead", 42, 3.14])
    async def test_signal_sends_signal_event(self, payload):
        mock_client = MagicMock()
        mock_client.run_service.signal_event = AsyncMock()

        event = Event(pb2=_mock_condition_action("e1", parent="act1"))

        with (
            patch("flyte.remote._event.ensure_client"),
            patch("flyte.remote._event.get_client", return_value=mock_client),
        ):
            await event.signal.aio(payload)

        mock_client.run_service.signal_event.assert_awaited_once()
        req = mock_client.run_service.signal_event.await_args.args[0]
        assert req.action_id == event.pb2.id
        assert req.parent_action_name == "act1"

    @pytest.mark.asyncio
    async def test_signal_invalid_type_raises(self):
        event = Event(pb2=MagicMock())
        with pytest.raises(TypeError, match="payload must be bool, int, float, or str"):
            await event.signal.aio([1, 2])


# ---------------------------------------------------------------------------
# Event.expected_type (auto-discovered from ActionMetadata.condition.type)
# ---------------------------------------------------------------------------


def _mock_pb2_with_type(simple: int | None, has_condition: bool = True) -> MagicMock:
    """Build a MagicMock Action proto with metadata.condition.type.simple.

    The ConditionActionMetadata.type field is a recent backend addition; we mock
    HasField/type access so tests don't depend on flyteidl2 stubs that include it.
    """
    pb = MagicMock()

    def _has_field(name):
        if name == "condition":
            return has_condition
        if name == "type":
            return simple is not None
        raise ValueError(f"unexpected HasField({name!r}) in test")

    pb.metadata.HasField.side_effect = lambda n: _has_field(n) if n == "condition" else None
    pb.metadata.condition.HasField.side_effect = lambda n: _has_field(n) if n == "type" else None
    if simple is not None:
        pb.metadata.condition.type.simple = simple
    return pb


class TestRemoteEventRichRepr:
    def test_yields_core_fields(self):
        from flyteidl2.common import identifier_pb2
        from flyteidl2.workflow import run_definition_pb2

        action = run_definition_pb2.Action(
            id=identifier_pb2.ActionIdentifier(
                run=identifier_pb2.RunIdentifier(name="run1"),
                name="cond-hash",
            ),
            metadata=run_definition_pb2.ActionMetadata(
                action_type=run_definition_pb2.ACTION_TYPE_CONDITION,
                condition=run_definition_pb2.ConditionActionMetadata(name="approve"),
                parent="a0",
            ),
        )
        pairs = dict(Event(pb2=action).__rich_repr__())
        assert pairs["name"] == "approve"
        assert pairs["action"] == "cond-hash"
        assert pairs["run"] == "run1"
        assert pairs["parent"] == "a0"
        # No `type` yielded when expected_type is unavailable.
        assert "type" not in pairs


class TestRemoteEventExpectedType:
    @pytest.mark.parametrize(
        "simple_value, py_type",
        [
            ("BOOLEAN", bool),
            ("INTEGER", int),
            ("FLOAT", float),
            ("STRING", str),
        ],
    )
    def test_expected_type_for_simple(self, simple_value, py_type):
        from flyteidl2.core import types_pb2

        pb = _mock_pb2_with_type(getattr(types_pb2, simple_value))
        assert Event(pb2=pb).expected_type is py_type

    def test_expected_type_no_condition_metadata(self):
        pb = _mock_pb2_with_type(simple=None, has_condition=False)
        assert Event(pb2=pb).expected_type is None

    def test_expected_type_condition_without_type_field(self):
        pb = _mock_pb2_with_type(simple=None, has_condition=True)
        assert Event(pb2=pb).expected_type is None

    def test_expected_type_proto_stub_missing_type_field(self):
        """When flyteidl2 stubs don't yet know about ConditionActionMetadata.type,
        HasField('type') raises ValueError. The property must gracefully return None.
        """
        pb = MagicMock()
        pb.metadata.HasField.return_value = True  # has 'condition'
        pb.metadata.condition.HasField.side_effect = ValueError("no field 'type'")
        assert Event(pb2=pb).expected_type is None

    def test_expected_type_unsupported_simple_returns_none(self):
        from flyteidl2.core import types_pb2

        pb = _mock_pb2_with_type(types_pb2.DATETIME)
        assert Event(pb2=pb).expected_type is None


# ---------------------------------------------------------------------------
# CLI: flyte signal event ...
# ---------------------------------------------------------------------------


class TestSignalCliCoerce:
    def test_bool_truthy(self):
        from flyte.cli._signal import _coerce

        for v in ("true", "True", "1", "YES", "y", "T"):
            assert _coerce(v, bool) is True

    def test_bool_falsy(self):
        from flyte.cli._signal import _coerce

        for v in ("false", "False", "0", "NO", "n", "F"):
            assert _coerce(v, bool) is False

    def test_bool_invalid(self):
        import rich_click as click

        from flyte.cli._signal import _coerce

        with pytest.raises(click.BadParameter):
            _coerce("maybe", bool)

    def test_int(self):
        from flyte.cli._signal import _coerce

        assert _coerce("42", int) == 42

    def test_int_invalid(self):
        import rich_click as click

        from flyte.cli._signal import _coerce

        with pytest.raises(click.BadParameter):
            _coerce("notanint", int)

    def test_float(self):
        from flyte.cli._signal import _coerce

        assert _coerce("3.14", float) == 3.14

    def test_float_invalid(self):
        import rich_click as click

        from flyte.cli._signal import _coerce

        with pytest.raises(click.BadParameter):
            _coerce("notafloat", float)

    def test_str_passthrough(self):
        from flyte.cli._signal import _coerce

        assert _coerce("hello", str) == "hello"
        # No interpretation of bool-like strings when expected is str.
        assert _coerce("true", str) == "true"


class TestSignalCliDisplayPrompt:
    def test_text_prompt_prints_raw(self):
        from flyteidl2.workflow import run_definition_pb2

        from flyte.cli._signal import _display_prompt

        console = MagicMock()
        _display_prompt(console, "approve?", run_definition_pb2.CONDITION_PROMPT_TYPE_TEXT)
        console.print.assert_called_once_with("approve?")

    def test_markdown_prompt_wraps_in_markdown(self):
        from flyteidl2.workflow import run_definition_pb2
        from rich.markdown import Markdown

        from flyte.cli._signal import _display_prompt

        console = MagicMock()
        _display_prompt(console, "**bold**", run_definition_pb2.CONDITION_PROMPT_TYPE_MARKDOWN)
        console.print.assert_called_once()
        arg = console.print.call_args.args[0]
        assert isinstance(arg, Markdown)

    def test_empty_prompt_is_no_op(self):
        from flyte.cli._signal import _display_prompt

        console = MagicMock()
        _display_prompt(console, "", 0)
        console.print.assert_not_called()


class TestSignalCliPromptForValue:
    def test_bool_uses_confirm(self):
        from flyte.cli import _signal

        with patch.object(_signal.click, "confirm", return_value=True) as conf:
            assert _signal._prompt_for_value(bool) is True
            conf.assert_called_once_with("Approve?", default=True)

    @pytest.mark.parametrize("py_type", [int, float, str])
    def test_scalar_uses_prompt(self, py_type):
        from flyte.cli import _signal

        sentinel = {int: 7, float: 1.5, str: "hi"}[py_type]
        with patch.object(_signal.click, "prompt", return_value=sentinel) as p:
            assert _signal._prompt_for_value(py_type) == sentinel
            p.assert_called_once_with("Value", type=py_type)

    def test_unsupported_raises(self):
        import rich_click as click

        from flyte.cli._signal import _prompt_for_value

        with pytest.raises(click.ClickException):
            _prompt_for_value(list)


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
