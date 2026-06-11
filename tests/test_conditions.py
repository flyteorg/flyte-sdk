"""
Tests for the condition system: _condition.py, remote/_condition.py, local controller
condition methods, and PendingCondition.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte.errors
from flyte._condition import ConditionWebhook, _Condition, new_condition
from flyte._internal.controllers._local_controller import _substitute_callback_uri
from flyte.cli._tui._tracker import PendingCondition
from flyte.remote._condition import Condition, _encode_payload

# ---------------------------------------------------------------------------
# _Condition dataclass
# ---------------------------------------------------------------------------


class TestConditionCreation:
    def test_defaults(self):
        c = _Condition(name="cond1")
        assert c.name == "cond1"
        assert c.prompt == "Approve?"
        assert c.prompt_type == "text"
        assert c.data_type is bool
        assert c.description == ""
        assert c.timeout is None
        assert c._timeout_seconds is None

    def test_custom_fields(self):
        c = _Condition(
            name="cond2",
            prompt="Continue?",
            prompt_type="markdown",
            data_type=str,
            description="some desc",
        )
        assert c.prompt == "Continue?"
        assert c.prompt_type == "markdown"
        assert c.data_type is str
        assert c.description == "some desc"

    @pytest.mark.parametrize("dt", [bool, int, float, str])
    def test_valid_data_types(self, dt):
        c = _Condition(name="cond", data_type=dt)
        assert c.data_type is dt

    @pytest.mark.parametrize("dt", [list, dict, bytes, object])
    def test_invalid_data_types(self, dt):
        with pytest.raises(TypeError, match="Invalid data_type"):
            _Condition(name="cond", data_type=dt)


class TestConditionTimeout:
    def test_timeout_int_seconds(self):
        c = _Condition(name="cond", timeout=30)
        assert c._timeout_seconds == 30.0

    def test_timeout_float_seconds(self):
        c = _Condition(name="cond", timeout=1.5)
        assert c._timeout_seconds == 1.5

    def test_timeout_timedelta(self):
        c = _Condition(name="cond", timeout=timedelta(minutes=2))
        assert c._timeout_seconds == 120.0

    def test_timeout_none(self):
        c = _Condition(name="cond", timeout=None)
        assert c._timeout_seconds is None

    def test_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            _Condition(name="cond", timeout=0)

    def test_timeout_negative_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            _Condition(name="cond", timeout=-5)

    def test_timeout_negative_timedelta_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            _Condition(name="cond", timeout=timedelta(seconds=-1))


class TestConditionWait:
    @pytest.mark.asyncio
    async def test_wait_outside_task_context_raises(self):
        c = _Condition(name="cond")
        with pytest.raises(RuntimeError, match="Conditions can only be awaited within a task context"):
            await c.wait.aio()


# ---------------------------------------------------------------------------
# new_condition factory
# ---------------------------------------------------------------------------


class TestNewCondition:
    @pytest.mark.asyncio
    async def test_new_condition_outside_context(self):
        """Outside task context, new_condition still returns a _Condition (no-op registration)."""
        c = await new_condition.aio("my_condition", prompt="Go?", data_type=int, timeout=10)
        assert isinstance(c, _Condition)
        assert c.name == "my_condition"
        assert c.prompt == "Go?"
        assert c.data_type is int
        assert c._timeout_seconds == 10.0

    @pytest.mark.asyncio
    async def test_new_condition_registers_in_task_context(self):
        """In a task context, new_condition calls controller.register_condition."""
        mock_controller = MagicMock()
        mock_controller.register_condition = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            c = await new_condition.aio("reg_condition")

        assert c.name == "reg_condition"
        mock_controller.register_condition.assert_awaited_once_with(c)

    @pytest.mark.asyncio
    async def test_new_condition_with_timeout_timedelta(self):
        c = await new_condition.aio("td_condition", timeout=timedelta(seconds=45))
        assert c._timeout_seconds == 45.0


# ---------------------------------------------------------------------------
# PendingCondition (TUI tracker)
# ---------------------------------------------------------------------------


class TestPendingCondition:
    def test_set_and_wait(self):
        pc = PendingCondition(
            condition_name="cond",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )
        pc.set_result(True)
        result = pc.wait_for_result()
        assert result is True
        assert pc.timed_out is False

    def test_timeout_expires(self):
        pc = PendingCondition(
            condition_name="cond",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )
        result = pc.wait_for_result(timeout=0.05)
        assert result is None
        assert pc.timed_out is True

    def test_result_before_timeout(self):
        pc = PendingCondition(
            condition_name="cond",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )

        def signal():
            pc.set_result(42)

        t = threading.Timer(0.02, signal)
        t.start()
        result = pc.wait_for_result(timeout=2.0)
        assert result == 42
        assert pc.timed_out is False
        t.join()

    def test_no_timeout(self):
        """Without timeout, wait blocks until signaled."""
        pc = PendingCondition(
            condition_name="cond",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=str,
        )

        def signal():
            pc.set_result("hello")

        t = threading.Timer(0.02, signal)
        t.start()
        result = pc.wait_for_result(timeout=None)
        assert result == "hello"
        assert pc.timed_out is False
        t.join()


# ---------------------------------------------------------------------------
# Local controller: register_condition / wait_for_condition
# ---------------------------------------------------------------------------


class TestLocalControllerConditions:
    @pytest.fixture
    def controller(self):
        """Create a LocalController with mocked dependencies."""
        from flyte._internal.controllers import TaskCallSequencer
        from flyte._internal.controllers._local_controller import LocalController

        mock_recorder = MagicMock()
        mock_recorder.record_condition_waiting.return_value = None  # non-TUI mode
        controller = LocalController.__new__(LocalController)
        controller._registered_conditions = {}
        controller._recorder = mock_recorder
        controller._sequencer = TaskCallSequencer()
        return controller

    @pytest.mark.asyncio
    async def test_register_condition(self, controller):
        c = _Condition(name="cond1")
        await controller.register_condition(c)
        assert "cond1" in controller._registered_conditions
        assert controller._registered_conditions["cond1"] is c

    @pytest.mark.asyncio
    async def test_register_non_condition_raises(self, controller):
        with pytest.raises(TypeError, match="Expected _Condition"):
            await controller.register_condition("not-a-condition")

    @pytest.mark.asyncio
    async def test_wait_non_condition_raises(self, controller):
        with pytest.raises(TypeError, match="Expected _Condition"):
            await controller.wait_for_condition("not-a-condition")

    @pytest.mark.asyncio
    async def test_wait_for_condition_tui_mode(self, controller):
        """TUI mode: PendingCondition is returned by recorder, result comes from set_result."""
        pc = PendingCondition(
            condition_name="cond",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )
        controller._recorder.record_condition_waiting.return_value = pc

        mock_ctx = MagicMock()
        mock_tctx = MagicMock()
        mock_tctx.action.name = "act-1"
        mock_tctx.task_action = None
        mock_ctx.data.task_context = mock_tctx

        # Signal the condition from another thread
        def signal():
            pc.set_result(True)

        t = threading.Timer(0.02, signal)
        t.start()

        c = _Condition(name="cond")
        with patch("flyte._internal.controllers._local_controller.internal_ctx", return_value=mock_ctx):
            result = await controller.wait_for_condition(c)

        assert result is True
        t.join()

    @pytest.mark.asyncio
    async def test_wait_for_condition_tui_timeout(self, controller):
        """TUI mode: timeout triggers ConditionTimedoutError."""
        pc = PendingCondition(
            condition_name="cond",
            action_id="act-1",
            prompt="ok?",
            prompt_type="text",
            data_type=bool,
        )
        controller._recorder.record_condition_waiting.return_value = pc

        mock_ctx = MagicMock()
        mock_tctx = MagicMock()
        mock_tctx.action.name = "act-1"
        mock_tctx.task_action = None
        mock_ctx.data.task_context = mock_tctx

        c = _Condition(name="cond", timeout=0.05)

        with patch("flyte._internal.controllers._local_controller.internal_ctx", return_value=mock_ctx):
            with pytest.raises(flyte.errors.ConditionTimedoutError, match="not signaled within"):
                await controller.wait_for_condition(c)

    @pytest.mark.asyncio
    async def test_wait_for_condition_console_timeout(self, controller):
        """Non-TUI mode with timeout: timeout triggers ConditionTimedoutError."""
        controller._recorder.record_condition_waiting.return_value = None  # non-TUI

        mock_ctx = MagicMock()
        mock_tctx = MagicMock()
        mock_tctx.action.name = "act-1"
        mock_tctx.task_action = None
        mock_ctx.data.task_context = mock_tctx

        stop = threading.Event()

        # Make console prompt block until stop is set (so the thread can be cleaned up)
        def blocking_prompt(condition):
            stop.wait()
            return True

        c = _Condition(name="cond", timeout=0.05)

        try:
            with (
                patch("flyte._internal.controllers._local_controller.internal_ctx", return_value=mock_ctx),
                patch.object(controller, "_prompt_condition_console", side_effect=blocking_prompt),
            ):
                with pytest.raises(flyte.errors.ConditionTimedoutError, match="not signaled within"):
                    await controller.wait_for_condition(c)
        finally:
            stop.set()  # unblock the executor thread so it can exit


# ---------------------------------------------------------------------------
# remote Condition — _encode_payload
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
# remote Condition — get / listall / signal
# ---------------------------------------------------------------------------


def _mock_condition_action(name: str, parent: str = "", run_name: str = "run1") -> MagicMock:
    """Build a mock condition Action whose Condition.name resolves to ``name``.

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


class TestRemoteConditionGet:
    @pytest.mark.asyncio
    async def test_get_returns_condition(self):
        target = _mock_condition_action("my_condition", parent="act1")
        mock_client = _mock_list_actions_client([_mock_condition_action("other"), target])

        with (
            patch("flyte.remote._condition.ensure_client"),
            patch("flyte.remote._condition.get_client", return_value=mock_client),
            patch("flyte.remote._condition.get_init_config", return_value=_mock_cfg()),
        ):
            result = await Condition.get.aio("my_condition", run_name="run1", action_name="act1")

        assert result is not None
        assert result.pb2 is target
        assert result.name == "my_condition"

    @pytest.mark.asyncio
    async def test_get_not_found_returns_none(self):
        mock_client = _mock_list_actions_client([_mock_condition_action("something_else")])

        with (
            patch("flyte.remote._condition.ensure_client"),
            patch("flyte.remote._condition.get_client", return_value=mock_client),
            patch("flyte.remote._condition.get_init_config", return_value=_mock_cfg()),
        ):
            result = await Condition.get.aio("missing", run_name="run1")

        assert result is None


class TestRemoteConditionListall:
    @pytest.mark.asyncio
    async def test_listall_filters_by_parent_action(self):
        match = _mock_condition_action("c1", parent="act1")
        other = _mock_condition_action("c2", parent="act2")
        mock_client = _mock_list_actions_client([match, other])

        with (
            patch("flyte.remote._condition.ensure_client"),
            patch("flyte.remote._condition.get_client", return_value=mock_client),
            patch("flyte.remote._condition.get_init_config", return_value=_mock_cfg()),
        ):
            conditions = [c async for c in Condition.listall.aio(run_name="run1", action_name="act1")]

        assert [c.pb2 for c in conditions] == [match]

    @pytest.mark.asyncio
    async def test_listall_filters_condition_actions_server_side(self):
        mock_client = _mock_list_actions_client([])

        with (
            patch("flyte.remote._condition.ensure_client"),
            patch("flyte.remote._condition.get_client", return_value=mock_client),
            patch("flyte.remote._condition.get_init_config", return_value=_mock_cfg()),
        ):
            conditions = [c async for c in Condition.listall.aio(run_name="run1")]

        assert conditions == []
        mock_client.run_service.list_actions.assert_awaited_once()
        # The request must carry a server-side action_type == CONDITION (3) filter.
        req = mock_client.run_service.list_actions.await_args.args[0]
        filt = req.request.filters[0]
        assert filt.field == "action_type"
        assert list(filt.values) == ["3"]

    @pytest.mark.asyncio
    async def test_listall_pagination(self):
        c1 = _mock_condition_action("c1")
        c2 = _mock_condition_action("c2")
        mock_client = _mock_list_actions_client([c1], [c2])

        with (
            patch("flyte.remote._condition.ensure_client"),
            patch("flyte.remote._condition.get_client", return_value=mock_client),
            patch("flyte.remote._condition.get_init_config", return_value=_mock_cfg()),
        ):
            conditions = [c async for c in Condition.listall.aio(run_name="run1")]

        assert [c.pb2 for c in conditions] == [c1, c2]
        assert mock_client.run_service.list_actions.await_count == 2


class TestRemoteConditionSignal:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [True, "go ahead", 42, 3.14])
    async def test_signal_sends_signal_event(self, payload):
        mock_client = MagicMock()
        mock_client.run_service.signal_event = AsyncMock()

        condition = Condition(pb2=_mock_condition_action("c1", parent="act1"))

        with (
            patch("flyte.remote._condition.ensure_client"),
            patch("flyte.remote._condition.get_client", return_value=mock_client),
        ):
            await condition.signal.aio(payload)

        mock_client.run_service.signal_event.assert_awaited_once()
        req = mock_client.run_service.signal_event.await_args.args[0]
        assert req.action_id == condition.pb2.id
        assert req.parent_action_name == "act1"

    @pytest.mark.asyncio
    async def test_signal_invalid_type_raises(self):
        condition = Condition(pb2=MagicMock())
        with pytest.raises(TypeError, match="payload must be bool, int, float, or str"):
            await condition.signal.aio([1, 2])


# ---------------------------------------------------------------------------
# Condition.expected_type (auto-discovered from ActionMetadata.condition.type)
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


class TestRemoteConditionRichRepr:
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
        pairs = dict(Condition(pb2=action).__rich_repr__())
        assert pairs["name"] == "approve"
        assert pairs["action"] == "cond-hash"
        assert pairs["run"] == "run1"
        assert pairs["parent"] == "a0"
        # No `type` yielded when expected_type is unavailable.
        assert "type" not in pairs


class TestRemoteConditionExpectedType:
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
        assert Condition(pb2=pb).expected_type is py_type

    def test_expected_type_no_condition_metadata(self):
        pb = _mock_pb2_with_type(simple=None, has_condition=False)
        assert Condition(pb2=pb).expected_type is None

    def test_expected_type_condition_without_type_field(self):
        pb = _mock_pb2_with_type(simple=None, has_condition=True)
        assert Condition(pb2=pb).expected_type is None

    def test_expected_type_proto_stub_missing_type_field(self):
        """When flyteidl2 stubs don't yet know about ConditionActionMetadata.type,
        HasField('type') raises ValueError. The property must gracefully return None.
        """
        pb = MagicMock()
        pb.metadata.HasField.return_value = True  # has 'condition'
        pb.metadata.condition.HasField.side_effect = ValueError("no field 'type'")
        assert Condition(pb2=pb).expected_type is None

    def test_expected_type_unsupported_simple_returns_none(self):
        from flyteidl2.core import types_pb2

        pb = _mock_pb2_with_type(types_pb2.DATETIME)
        assert Condition(pb2=pb).expected_type is None


# ---------------------------------------------------------------------------
# CLI: flyte signal condition ...
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
# ConditionWebhook dataclass
# ---------------------------------------------------------------------------


class TestConditionWebhook:
    def test_basic_creation(self):
        wh = ConditionWebhook(url="https://example.com/hook")
        assert wh.url == "https://example.com/hook"
        assert wh.payload is None

    def test_with_payload(self):
        wh = ConditionWebhook(
            url="https://example.com/hook",
            payload={"callback": "{callback_uri}", "condition": "approval"},
        )
        assert wh.url == "https://example.com/hook"
        assert wh.payload == {"callback": "{callback_uri}", "condition": "approval"}

    def test_condition_with_webhook(self):
        wh = ConditionWebhook(url="https://example.com/hook")
        c = _Condition(name="cond", webhook=wh)
        assert c.webhook is wh

    def test_condition_webhook_defaults_none(self):
        c = _Condition(name="cond")
        assert c.webhook is None


class TestNewConditionWebhook:
    @pytest.mark.asyncio
    async def test_new_condition_with_webhook(self):
        wh = ConditionWebhook(url="https://example.com/hook", payload={"cb": "{callback_uri}"})
        c = await new_condition.aio("wh_condition", webhook=wh)
        assert c.webhook is wh

    @pytest.mark.asyncio
    async def test_new_condition_registers_with_webhook_in_task_context(self):
        mock_controller = MagicMock()
        mock_controller.register_condition = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        wh = ConditionWebhook(url="https://example.com/hook")
        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            c = await new_condition.aio("wh_reg_condition", webhook=wh)

        assert c.webhook is wh
        mock_controller.register_condition.assert_awaited_once_with(c)


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
# Condition wait() protocol
# ---------------------------------------------------------------------------


class TestConditionWaitProtocol:
    """Tests for the condition wait() protocol: timeout, failure, and output handling."""

    @pytest.mark.asyncio
    async def test_wait_raises_condition_timedout_error(self):
        """wait() should raise ConditionTimedoutError when the controller reports timeout."""
        mock_controller = MagicMock()
        mock_controller.wait_for_condition = AsyncMock(
            side_effect=flyte.errors.ConditionTimedoutError(
                "Condition 'cond' was not signaled within the timeout period."
            )
        )

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        c = _Condition(name="cond", data_type=bool, timeout=10)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            with pytest.raises(flyte.errors.ConditionTimedoutError, match="not signaled within"):
                await c.wait.aio()

    @pytest.mark.asyncio
    async def test_wait_raises_condition_failed_error(self):
        """wait() should raise ConditionFailedError when the controller reports failure."""
        mock_controller = MagicMock()
        mock_controller.wait_for_condition = AsyncMock(
            side_effect=flyte.errors.ConditionFailedError("Condition 'cond' condition action failed.")
        )

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        c = _Condition(name="cond", data_type=str)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            with pytest.raises(flyte.errors.ConditionFailedError, match="failed"):
                await c.wait.aio()

    @pytest.mark.asyncio
    async def test_wait_returns_bool_true(self):
        """wait() should return True for a bool condition signaled with True."""
        mock_controller = MagicMock()
        mock_controller.wait_for_condition = AsyncMock(return_value=True)

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        c = _Condition(name="approve", data_type=bool)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await c.wait.aio()
            assert result is True

    @pytest.mark.asyncio
    async def test_wait_returns_bool_false(self):
        """wait() should return False for a bool condition signaled with False."""
        mock_controller = MagicMock()
        mock_controller.wait_for_condition = AsyncMock(return_value=False)

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        c = _Condition(name="approve", data_type=bool)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await c.wait.aio()
            assert result is False

    @pytest.mark.asyncio
    async def test_wait_returns_int(self):
        """wait() should return an int for an int condition."""
        mock_controller = MagicMock()
        mock_controller.wait_for_condition = AsyncMock(return_value=42)

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        c = _Condition(name="count", data_type=int)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await c.wait.aio()
            assert result == 42
            assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_wait_returns_float(self):
        """wait() should return a float for a float condition."""
        mock_controller = MagicMock()
        mock_controller.wait_for_condition = AsyncMock(return_value=3.14)

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        c = _Condition(name="threshold", data_type=float)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await c.wait.aio()
            assert abs(result - 3.14) < 1e-6

    @pytest.mark.asyncio
    async def test_wait_returns_str(self):
        """wait() should return a str for a str condition."""
        mock_controller = MagicMock()
        mock_controller.wait_for_condition = AsyncMock(return_value="go ahead")

        mock_ctx = MagicMock()
        mock_ctx.is_task_context.return_value = True

        c = _Condition(name="input", data_type=str)

        with (
            patch("flyte._context.internal_ctx", return_value=mock_ctx),
            patch("flyte._internal.controllers.get_controller", return_value=mock_controller),
        ):
            result = await c.wait.aio()
            assert result == "go ahead"


# ---------------------------------------------------------------------------
# Local controller: webhook firing
# ---------------------------------------------------------------------------


class TestLocalControllerWebhook:
    @pytest.fixture
    def controller(self):
        from flyte._internal.controllers._local_controller import LocalController

        mock_recorder = MagicMock()
        mock_recorder.record_condition_waiting.return_value = None
        controller = LocalController.__new__(LocalController)
        controller._registered_conditions = {}
        controller._recorder = mock_recorder
        return controller

    @pytest.mark.asyncio
    async def test_register_condition_without_webhook_does_not_fire(self, controller):
        c = _Condition(name="cond_no_wh")
        with patch.object(controller, "_fire_condition_webhook", new_callable=AsyncMock) as mock_fire:
            await controller.register_condition(c)
        mock_fire.assert_not_awaited()
        assert "cond_no_wh" in controller._registered_conditions

    @pytest.mark.asyncio
    async def test_register_condition_with_webhook_fires(self, controller):
        wh = ConditionWebhook(url="https://example.com/hook", payload={"cb": "{callback_uri}"})
        c = _Condition(name="cond_wh", webhook=wh)
        with patch.object(controller, "_fire_condition_webhook", new_callable=AsyncMock) as mock_fire:
            await controller.register_condition(c)
        mock_fire.assert_awaited_once_with(c)
        assert "cond_wh" in controller._registered_conditions

    @pytest.mark.asyncio
    async def test_fire_condition_webhook_posts_with_substituted_payload(self, controller):
        wh = ConditionWebhook(
            url="https://example.com/hook",
            payload={"callback": "{callback_uri}", "condition": "test"},
        )
        c = _Condition(name="my_cond", webhook=wh)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await controller._fire_condition_webhook(c)

        mock_client.post.assert_awaited_once_with(
            "https://example.com/hook",
            json={"callback": "local://conditions/my_cond/signal", "condition": "test"},
            headers={"Content-Type": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_fire_condition_webhook_no_payload(self, controller):
        wh = ConditionWebhook(url="https://example.com/hook")
        c = _Condition(name="cond_nopayload", webhook=wh)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await controller._fire_condition_webhook(c)

        mock_client.post.assert_awaited_once_with(
            "https://example.com/hook",
            json=None,
            headers={"Content-Type": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_fire_condition_webhook_exception_logged_not_raised(self, controller):
        wh = ConditionWebhook(url="https://example.com/hook")
        c = _Condition(name="cond_fail", webhook=wh)

        with patch("httpx.AsyncClient", side_effect=Exception("connection error")):
            # Should not raise
            await controller._fire_condition_webhook(c)
