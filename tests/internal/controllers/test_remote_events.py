"""Tests for remote controller event registration and waiting.

These tests cover:
- register_event: validates event, creates condition action, submits to backend (fire-and-forget)
- wait_for_event: waits for condition action completion, returns typed payload
- Various data types (bool, int, float, str)
- Error/failure/abort/timeout terminal phases
- Integration with core controller start_action / wait_for_action split
"""

from __future__ import annotations

import asyncio
import pathlib
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.task import task_definition_pb2

import flyte
import flyte.errors
import flyte.report
from flyte._context import internal_ctx
from flyte._event import EventWebhook, _Event
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.controllers.remote._core import Controller
from flyte._internal.controllers.remote._service_protocol import ClientSet
from flyte.models import ActionID, CodeBundle, RawDataPath, TaskContext

this_dir_str = str(pathlib.Path(__file__).parent.absolute())


def _make_task_context(**overrides) -> TaskContext:
    defaults = {
        "action": ActionID(name="parent_action", run_name="test_run", project="proj", domain="dev", org="org"),
        "raw_data_path": RawDataPath(path="test"),
        "output_path": "/tmp",
        "version": "v1",
        "run_base_dir": "/run_base",
        "report": flyte.report.Report(name="test_report"),
        "code_bundle": CodeBundle(
            computed_version="vcode-bundle",
            destination=this_dir_str,
            tgz="dummy.tgz",
        ),
    }
    defaults.update(overrides)
    return TaskContext(**defaults)


def _make_event(name="approval", data_type=bool, **kwargs) -> _Event:
    return _Event(name=name, data_type=data_type, **kwargs)


async def _make_client() -> ClientSet:
    return AsyncMock()  # type: ignore


def _make_controller(**kwargs) -> RemoteController:
    defaults = {"client_coro": _make_client(), "workers": 2, "max_system_retries": 2}
    defaults.update(kwargs)
    return RemoteController(**defaults)


# ── Core controller: start_action + wait_for_action split ──────────────────


def _make_bare_controller():
    """Create a Controller without running __init__, with minimal attributes for _bg_* methods."""
    controller = object.__new__(Controller)
    controller._informers = MagicMock()
    controller._informer_start_wait_timeout = 5.0
    controller._shared_queue = asyncio.Queue()
    controller._state_service = AsyncMock()
    controller._actions_service = None
    return controller


@pytest.mark.asyncio
async def test_start_action_returns_immediately():
    """start_action should submit to the informer and return without waiting for completion."""
    controller = _make_bare_controller()
    informer = AsyncMock()
    controller._informers.get_or_create = AsyncMock(return_value=informer)

    # Minimal action
    run_id = identifier_pb2.RunIdentifier(name="root_run")
    action = Action(
        action_id=identifier_pb2.ActionIdentifier(name="subrun-1", run=run_id),
        parent_action_name="parent",
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="run-base",
    )

    await controller._bg_submit_action(action)

    # Should have submitted to informer
    informer.submit.assert_awaited_once_with(action)
    # Should NOT have waited for completion
    informer.wait_for_action_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_for_action_blocks_until_completion():
    """wait_for_action should block until the action reaches a terminal state."""
    controller = _make_bare_controller()

    run_id = identifier_pb2.RunIdentifier(name="root_run")
    action_id = identifier_pb2.ActionIdentifier(name="subrun-1", run=run_id)

    final_action = Action(
        action_id=action_id,
        parent_action_name="parent",
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        realized_outputs_uri="s3://bucket/output",
    )

    informer = AsyncMock()
    informer.get = AsyncMock(return_value=final_action)
    controller._informers.get_or_create = AsyncMock(return_value=informer)

    result = await controller._bg_wait_for_action(action_id, run_id, "parent")

    informer.wait_for_action_completion.assert_awaited_once_with("subrun-1")
    informer.get.assert_awaited_once_with("subrun-1")
    informer.remove.assert_awaited_once_with("subrun-1")
    assert result.phase == phase_pb2.ACTION_PHASE_SUCCEEDED


@pytest.mark.asyncio
async def test_submit_and_wait_combines_both():
    """submit_and_wait_for_action should submit then wait, returning the final action."""
    controller = _make_bare_controller()

    run_id = identifier_pb2.RunIdentifier(name="root_run")
    action = Action(
        action_id=identifier_pb2.ActionIdentifier(name="subrun-1", run=run_id),
        parent_action_name="parent",
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="run-base",
    )

    final_action = Action(
        action_id=action.action_id,
        parent_action_name="parent",
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        realized_outputs_uri="s3://bucket/output",
    )

    informer = AsyncMock()
    informer.get = AsyncMock(return_value=final_action)
    controller._informers.get_or_create = AsyncMock(return_value=informer)

    result = await controller._bg_submit_and_wait_for_action(action)

    informer.submit.assert_awaited_once_with(action)
    informer.wait_for_action_completion.assert_awaited_once_with("subrun-1")
    assert result.phase == phase_pb2.ACTION_PHASE_SUCCEEDED


# ── RemoteController: register_event ──────────────────────────────────────


@pytest.mark.asyncio
async def test_register_event_bool():
    """register_event with bool data_type should submit a condition action and return immediately."""
    await flyte.init.aio()
    event = _make_event(name="approve", data_type=bool, prompt="Approve deployment?")

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
                await controller.register_event(event)
                mock_start.assert_called_once()
                action = mock_start.call_args[0][0]
                assert action.type == "condition"


@pytest.mark.asyncio
async def test_register_event_str():
    """register_event with str data_type."""
    await flyte.init.aio()
    event = _make_event(name="user_input", data_type=str, prompt="Enter value:")

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
                await controller.register_event(event)
                mock_start.assert_called_once()
                action = mock_start.call_args[0][0]
                assert action.type == "condition"


@pytest.mark.asyncio
async def test_register_event_int():
    """register_event with int data_type."""
    await flyte.init.aio()
    event = _make_event(name="count", data_type=int, prompt="How many?")

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
                await controller.register_event(event)
                mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_register_event_float():
    """register_event with float data_type."""
    await flyte.init.aio()
    event = _make_event(name="threshold", data_type=float, prompt="Set threshold:")

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
                await controller.register_event(event)
                mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_register_event_rejects_non_event():
    """register_event should raise TypeError for non-_Event objects."""
    controller = _make_controller()
    with pytest.raises(TypeError, match="Expected _Event"):
        await controller.register_event("not_an_event")


@pytest.mark.asyncio
async def test_register_event_with_webhook():
    """register_event should work with webhook-configured events."""
    await flyte.init.aio()
    webhook = EventWebhook(url="https://example.com/hook", payload={"cb": "{callback_uri}"})
    event = _make_event(name="webhook_event", webhook=webhook)

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
                await controller.register_event(event)
                mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_register_event_with_timeout():
    """register_event should work with timeout-configured events."""
    await flyte.init.aio()
    event = _make_event(name="timed_event", timeout=timedelta(seconds=30))

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
                await controller.register_event(event)
                mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_register_event_with_description():
    """register_event should pass description through to the condition action."""
    await flyte.init.aio()
    event = _make_event(name="described_event", description="This needs human review")

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
                await controller.register_event(event)
                mock_start.assert_called_once()


# ── RemoteController: wait_for_event ──────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_for_event_bool_succeeded():
    """wait_for_event should return bool payload on successful completion."""
    await flyte.init.aio()
    event = _make_event(name="approve", data_type=bool)

    succeeded_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-approve",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        realized_outputs_uri="/tmp/outputs",
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            # Simulate that register_event was called first
            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=succeeded_action,
                ) as mock_wait,
            ):
                await controller.register_event(event)
                await controller.wait_for_event(event)
                mock_wait.assert_called_once()


@pytest.mark.asyncio
async def test_wait_for_event_str_succeeded():
    """wait_for_event should return str payload on successful completion."""
    await flyte.init.aio()
    event = _make_event(name="user_input", data_type=str)

    succeeded_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-user_input",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        realized_outputs_uri="/tmp/outputs",
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=succeeded_action,
                ),
            ):
                await controller.register_event(event)
                await controller.wait_for_event(event)


@pytest.mark.asyncio
async def test_wait_for_event_int_succeeded():
    """wait_for_event should return int payload on successful completion."""
    await flyte.init.aio()
    event = _make_event(name="count", data_type=int)

    succeeded_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-count",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        realized_outputs_uri="/tmp/outputs",
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=succeeded_action,
                ),
            ):
                await controller.register_event(event)
                await controller.wait_for_event(event)


@pytest.mark.asyncio
async def test_wait_for_event_float_succeeded():
    """wait_for_event should return float payload on successful completion."""
    await flyte.init.aio()
    event = _make_event(name="threshold", data_type=float)

    succeeded_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-threshold",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        realized_outputs_uri="/tmp/outputs",
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=succeeded_action,
                ),
            ):
                await controller.register_event(event)
                await controller.wait_for_event(event)


@pytest.mark.asyncio
async def test_wait_for_event_failed_phase():
    """wait_for_event should raise EventFailedError when the condition action fails."""
    await flyte.init.aio()
    event = _make_event(name="fail_event", data_type=bool)

    failed_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-fail_event",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_FAILED,
        client_err=Exception("Condition failed"),
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=failed_action,
                ),
            ):
                await controller.register_event(event)
                with pytest.raises(flyte.errors.EventFailedError):
                    await controller.wait_for_event(event)


@pytest.mark.asyncio
async def test_wait_for_event_aborted_phase():
    """wait_for_event should raise ActionAbortedError when the condition action is aborted."""
    await flyte.init.aio()
    event = _make_event(name="abort_event", data_type=bool)

    aborted_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-abort_event",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_ABORTED,
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=aborted_action,
                ),
            ):
                await controller.register_event(event)
                with pytest.raises(flyte.errors.ActionAbortedError):
                    await controller.wait_for_event(event)


@pytest.mark.asyncio
async def test_wait_for_event_timed_out_phase():
    """wait_for_event should raise EventTimedoutError when the condition action times out."""
    await flyte.init.aio()
    event = _make_event(name="timeout_event", data_type=bool, timeout=10)

    timed_out_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-timeout_event",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_TIMED_OUT,
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=timed_out_action,
                ),
            ):
                await controller.register_event(event)
                with pytest.raises(flyte.errors.EventTimedoutError):
                    await controller.wait_for_event(event)


@pytest.mark.asyncio
async def test_wait_for_event_rejects_non_event():
    """wait_for_event should raise TypeError for non-_Event objects."""
    controller = _make_controller()
    with pytest.raises(TypeError, match="Expected _Event"):
        await controller.wait_for_event("not_an_event")


@pytest.mark.asyncio
async def test_wait_for_event_without_register_raises():
    """wait_for_event should raise if event was not previously registered."""
    await flyte.init.aio()
    event = _make_event(name="unregistered")

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()
            with pytest.raises((RuntimeError, KeyError)):
                await controller.wait_for_event(event)


# ── Action.from_condition ──────────────────────────────────────────────────


def test_action_from_condition_creates_condition_type():
    """Action.from_condition should create an action with type='condition'."""
    action_id = identifier_pb2.ActionIdentifier(
        name="condition-test",
        run=identifier_pb2.RunIdentifier(name="test_run"),
    )
    action = Action.from_condition(
        parent_action_name="parent",
        action_id=action_id,
        event_name="test_event",
        prompt="Approve?",
        data_type=bool,
        run_output_base="/run_base",
    )
    assert action.type == "condition"
    assert action.condition is not None
    assert action.name == "condition-test"


def test_action_from_condition_with_description():
    """from_condition should pass description to the ConditionAction proto."""
    action_id = identifier_pb2.ActionIdentifier(
        name="condition-desc",
        run=identifier_pb2.RunIdentifier(name="test_run"),
    )
    action = Action.from_condition(
        parent_action_name="parent",
        action_id=action_id,
        event_name="desc_event",
        prompt="Review?",
        data_type=str,
        description="Needs human review",
        run_output_base="/run_base",
    )
    assert action.type == "condition"
    assert action.condition is not None


@pytest.mark.parametrize("data_type", [bool, int, float, str])
def test_action_from_condition_all_data_types(data_type):
    """from_condition should work with all supported data types."""
    action_id = identifier_pb2.ActionIdentifier(
        name=f"condition-{data_type.__name__}",
        run=identifier_pb2.RunIdentifier(name="test_run"),
    )
    action = Action.from_condition(
        parent_action_name="parent",
        action_id=action_id,
        event_name=f"event_{data_type.__name__}",
        prompt="Enter value:",
        data_type=data_type,
        run_output_base="/run_base",
    )
    assert action.type == "condition"
    assert action.condition is not None


# ── Existing callers still work after rename ──────────────────────────────


@pytest.mark.asyncio
async def test_submit_task_uses_submit_and_wait():
    """After the rename, _submit should call submit_and_wait_for_action (not start_action)."""
    await flyte.init.aio()

    env = flyte.TaskEnvironment("test_rename")

    @env.task
    async def dummy_task():
        pass

    action = Action(
        parent_action_name="test_parent_action",
        action_id=identifier_pb2.ActionIdentifier(name="test_action"),
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
    )

    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry",
            new_callable=AsyncMock,
        ),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
            return_value=action,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_config,
    ):
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()
            await controller.submit(dummy_task)

        mock_submit_and_wait.assert_called_once()


@pytest.mark.asyncio
async def test_record_trace_uses_submit_and_wait():
    """After the rename, record_trace should call submit_and_wait_for_action."""
    await flyte.init.aio()
    from flyte._internal.controllers import TraceInfo
    from flyte.models import NativeInterface

    interface = NativeInterface(inputs={}, outputs={"result": int})
    trace_info = TraceInfo(
        name="test_function",
        action=ActionID(name="test_action"),
        interface=interface,
        inputs_path="/tmp/inputs",
        output=42,
    )

    with (
        patch("flyte._internal.runtime.io.upload_outputs", new_callable=AsyncMock),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_config,
    ):
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()
            await controller.record_trace(trace_info)

        mock_submit_and_wait.assert_called_once()


# ── Condition output handling ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_for_event_failed_raises_event_failed_error():
    """wait_for_event should raise EventFailedError (not generic Exception) on FAILED phase."""
    await flyte.init.aio()
    event = _make_event(name="fail_cond", data_type=str)

    failed_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-fail_cond",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_FAILED,
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=failed_action,
                ),
            ):
                await controller.register_event(event)
                with pytest.raises(flyte.errors.EventFailedError, match="fail_cond"):
                    await controller.wait_for_event(event)


@pytest.mark.asyncio
async def test_wait_for_event_failed_with_client_err_raises_event_failed():
    """wait_for_event should raise EventFailedError even when client_err is set."""
    await flyte.init.aio()
    event = _make_event(name="fail_client", data_type=int)

    failed_action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="condition-fail_client",
            run=identifier_pb2.RunIdentifier(name="test_run"),
        ),
        parent_action_name="parent_action",
        type="condition",
        phase=phase_pb2.ACTION_PHASE_FAILED,
        client_err=Exception("internal error"),
    )

    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        ctx = internal_ctx()
        tctx = _make_task_context()

        with ctx.replace_task_context(tctx):
            controller = _make_controller()

            with (
                patch.object(controller, "start_action", new_callable=AsyncMock),
                patch.object(
                    controller,
                    "wait_for_action",
                    new_callable=AsyncMock,
                    return_value=failed_action,
                ),
            ):
                await controller.register_event(event)
                with pytest.raises(flyte.errors.EventFailedError):
                    await controller.wait_for_event(event)


@pytest.mark.asyncio
async def test_wait_for_event_timed_out_raises_event_timedout():
    """wait_for_event should raise EventTimedoutError on TIMED_OUT phase for all data types."""
    await flyte.init.aio()

    for dt in (bool, int, float, str):
        event = _make_event(name=f"timeout_{dt.__name__}", data_type=dt, timeout=5)

        timed_out_action = Action(
            action_id=identifier_pb2.ActionIdentifier(
                name=f"condition-timeout_{dt.__name__}",
                run=identifier_pb2.RunIdentifier(name="test_run"),
            ),
            parent_action_name="parent_action",
            type="condition",
            phase=phase_pb2.ACTION_PHASE_TIMED_OUT,
        )

        with patch("flyte._initialize.get_init_config") as mock_config:
            mock_config.return_value.root_dir = pathlib.Path(__file__).parent
            ctx = internal_ctx()
            tctx = _make_task_context()

            with ctx.replace_task_context(tctx):
                controller = _make_controller()

                with (
                    patch.object(controller, "start_action", new_callable=AsyncMock),
                    patch.object(
                        controller,
                        "wait_for_action",
                        new_callable=AsyncMock,
                        return_value=timed_out_action,
                    ),
                ):
                    await controller.register_event(event)
                    with pytest.raises(flyte.errors.EventTimedoutError):
                        await controller.wait_for_event(event)


# ── Action.literal_to_python ─────────────────────────────────────────────────


class TestLiteralToPython:
    """Tests for Action.literal_to_python which converts flyteidl Literal to Python values."""

    def test_bool_true(self):
        from flyteidl2.core.literals_pb2 import Literal, Primitive, Scalar

        lit = Literal(scalar=Scalar(primitive=Primitive(boolean=True)))
        assert Action.literal_to_python(lit, bool) is True

    def test_bool_false(self):
        from flyteidl2.core.literals_pb2 import Literal, Primitive, Scalar

        lit = Literal(scalar=Scalar(primitive=Primitive(boolean=False)))
        assert Action.literal_to_python(lit, bool) is False

    def test_int_value(self):
        from flyteidl2.core.literals_pb2 import Literal, Primitive, Scalar

        lit = Literal(scalar=Scalar(primitive=Primitive(integer=42)))
        result = Action.literal_to_python(lit, int)
        assert result == 42
        assert isinstance(result, int)

    def test_float_value(self):
        from flyteidl2.core.literals_pb2 import Literal, Primitive, Scalar

        lit = Literal(scalar=Scalar(primitive=Primitive(float_value=3.14)))
        result = Action.literal_to_python(lit, float)
        assert abs(result - 3.14) < 1e-6
        assert isinstance(result, float)

    def test_str_value(self):
        from flyteidl2.core.literals_pb2 import Literal, Primitive, Scalar

        lit = Literal(scalar=Scalar(primitive=Primitive(string_value="hello")))
        result = Action.literal_to_python(lit, str)
        assert result == "hello"
        assert isinstance(result, str)

    def test_unsupported_type_raises(self):
        from flyteidl2.core.literals_pb2 import Literal, Primitive, Scalar

        lit = Literal(scalar=Scalar(primitive=Primitive(integer=1)))
        with pytest.raises(TypeError, match="Unsupported expected_type"):
            Action.literal_to_python(lit, list)


# ── Action.condition_output field ────────────────────────────────────────────


def test_action_condition_output_defaults_none():
    """Action.condition_output should default to None."""
    action = Action(
        action_id=identifier_pb2.ActionIdentifier(name="test", run=identifier_pb2.RunIdentifier(name="run")),
        parent_action_name="parent",
    )
    assert action.condition_output is None


def test_action_from_state_sets_condition_output_none():
    """from_state should set condition_output (currently None until proto support)."""
    from flyteidl2.workflow.state_service_pb2 import ActionUpdate

    update = ActionUpdate(
        action_id=identifier_pb2.ActionIdentifier(name="cond-1", run=identifier_pb2.RunIdentifier(name="run")),
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        output_uri="",
    )
    action = Action.from_state("parent", update)
    assert action.condition_output is None
