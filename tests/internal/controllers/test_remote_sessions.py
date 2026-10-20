"""Tests for sessions: child actions that publish an address on ActionUpdate while they run.

These tests cover:
- Action: ActionUpdate.address is kept as the latest value and cleared with it
- ActionCache: ready events fire on an address, on completion, and on removal
- Core controller: the worker fires ready on the first address, and terminal still fires completion
- RemoteController: register_action / wait_for_registered_action / abort_registered_action for sessions
"""

from __future__ import annotations

import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from flyteidl2.common import address_pb2, identifier_pb2, phase_pb2
from flyteidl2.core import execution_pb2, literals_pb2
from flyteidl2.task import task_definition_pb2
from flyteidl2.workflow import state_service_pb2

import flyte
import flyte.errors
import flyte.report
from flyte._condition import _Condition
from flyte._context import internal_ctx
from flyte._internal.controllers._local_controller import LocalController
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.controllers.remote._core import Controller
from flyte._internal.controllers.remote._informer import ActionCache
from flyte._session import _Session, new_session
from flyte.models import ActionID, CodeBundle, RawDataPath, TaskContext

this_dir_str = str(pathlib.Path(__file__).parent.absolute())

RUN = identifier_pb2.RunIdentifier(name="test_run")


def _address(ip: str = "10.0.0.7", path: str = "/s/abc") -> address_pb2.ActionAddress:
    return address_pb2.ActionAddress(
        pod=address_pb2.PodAddress(namespace="ns", name="agent-0", ip=ip, port=8080),
        path=path,
        cluster="c1",
    )


def _update(name: str, phase: int, address: address_pb2.ActionAddress | None = None) -> state_service_pb2.ActionUpdate:
    u = state_service_pb2.ActionUpdate(action_id=identifier_pb2.ActionIdentifier(name=name, run=RUN), phase=phase)
    if address is not None:
        u.address.CopyFrom(address)
    return u


def _spec(retries: int = 0) -> task_definition_pb2.TaskSpec:
    spec = task_definition_pb2.TaskSpec()
    spec.task_template.id.name = "sandbox"
    spec.task_template.type = "gvisor-session"
    if retries:
        spec.task_template.metadata.retries.retries = retries
    return spec


def _session_action(name: str = "s-1") -> Action:
    return Action.from_session(
        parent_action_name="parent_action",
        action_id=identifier_pb2.ActionIdentifier(name=name, run=RUN),
        task_spec=_spec(),
        run_output_base="/run_base",
        inputs_uri="/run_base/s-1/inputs.pb",
    )


def _make_task_context() -> TaskContext:
    return TaskContext(
        action=ActionID(name="parent_action", run_name="test_run", project="proj", domain="dev", org="org"),
        raw_data_path=RawDataPath(path="test"),
        output_path="/tmp",
        version="v1",
        run_base_dir="/run_base",
        report=flyte.report.Report(name="test_report"),
        code_bundle=CodeBundle(computed_version="vcode-bundle", destination=this_dir_str, tgz="dummy.tgz"),
    )


async def _make_client():
    return AsyncMock()


def _make_controller() -> RemoteController:
    controller = RemoteController(client_coro=_make_client(), workers=2, max_system_retries=2)
    controller._actions_service = AsyncMock()
    return controller


# ── Action ─────────────────────────────────────────────────────────────────────


def test_merge_state_keeps_the_latest_address_and_clears_it():
    action = _session_action()
    action.merge_state(_update("s-1", phase_pb2.ACTION_PHASE_RUNNING, _address(ip="10.0.0.7")))
    assert action.address.pod.ip == "10.0.0.7"

    action.merge_state(_update("s-1", phase_pb2.ACTION_PHASE_RUNNING, _address(ip="10.0.0.8")))
    assert action.address.pod.ip == "10.0.0.8"

    # The server clears the address when the action goes terminal.
    action.merge_state(_update("s-1", phase_pb2.ACTION_PHASE_ABORTED))
    assert action.address is None
    assert action.is_terminal()


def test_from_state_carries_the_address():
    # A running session in the watch snapshot, before this process submitted it.
    action = Action.from_state("parent_action", _update("s-1", phase_pb2.ACTION_PHASE_RUNNING, _address()))
    assert action.address.path == "/s/abc"
    assert action.is_started()
    assert not action.is_terminal()


def test_from_state_without_an_address():
    action = Action.from_state("parent_action", _update("t-1", phase_pb2.ACTION_PHASE_SUCCEEDED))
    assert action.address is None


def test_from_session_launches_as_a_task_and_never_retries():
    spec = _spec(retries=3)
    action = Action.from_session(
        parent_action_name="parent_action",
        action_id=identifier_pb2.ActionIdentifier(name="s-1", run=RUN),
        task_spec=spec,
        run_output_base="/run_base",
        inputs_uri="/in",
        queue="q",
    )
    assert action.type == "task"
    assert action.queue == "q"
    assert action.friendly_name == "sandbox"
    assert not action.task.task_template.metadata.HasField("retries")
    # The caller's spec is left alone.
    assert spec.task_template.metadata.retries.retries == 3


# ── ActionCache ready events ───────────────────────────────────────────────────


async def _wait_briefly(coro) -> bool:
    return await asyncio.wait_for(coro, timeout=1)


@pytest.mark.asyncio
async def test_ready_event_fires_on_fire_ready_event():
    cache = ActionCache("parent_action")
    await cache.submit(_session_action())
    waiter = asyncio.create_task(cache.wait_for_ready("s-1"))
    await asyncio.sleep(0)
    assert not waiter.done()
    await cache.fire_ready_event("s-1")
    assert await _wait_briefly(waiter)


@pytest.mark.asyncio
async def test_completion_also_fires_ready():
    cache = ActionCache("parent_action")
    await cache.submit(_session_action())
    await cache.fire_completion_event("s-1")
    assert await _wait_briefly(cache.wait_for_ready("s-1"))


@pytest.mark.asyncio
async def test_remove_wakes_a_ready_waiter():
    cache = ActionCache("parent_action")
    await cache.submit(_session_action())
    waiter = asyncio.create_task(cache.wait_for_ready("s-1"))
    await asyncio.sleep(0)
    await cache.remove("s-1")
    assert await _wait_briefly(waiter)


@pytest.mark.asyncio
async def test_fire_all_completion_events_fires_ready():
    cache = ActionCache("parent_action")
    await cache.submit(_session_action())
    waiter = asyncio.create_task(cache.wait_for_ready("s-1"))
    await asyncio.sleep(0)
    await cache.fire_all_completion_events()
    assert await _wait_briefly(waiter)


@pytest.mark.asyncio
async def test_wait_for_ready_of_an_unknown_action_returns_false():
    cache = ActionCache("parent_action")
    assert await cache.wait_for_ready("nope") is False


@pytest.mark.asyncio
async def test_update_before_submit_keeps_the_address():
    # The watch can deliver the address before submit registers the action.
    cache = ActionCache("parent_action")
    await cache.observe_state(_update("s-1", phase_pb2.ACTION_PHASE_RUNNING, _address()))
    merged = await cache.submit(_session_action())
    assert merged.address is not None
    assert merged.is_started()


# ── Core controller worker ─────────────────────────────────────────────────────


def _bare_controller_with_informer():
    controller = object.__new__(Controller)
    informer = AsyncMock()
    controller._informers = MagicMock()
    controller._informers.get = AsyncMock(return_value=informer)
    controller._informers.get_or_create = AsyncMock(return_value=informer)
    controller._informer_start_wait_timeout = 5.0
    controller._shared_queue = asyncio.Queue()
    controller._state_service = AsyncMock()
    controller._actions_service = AsyncMock()
    return controller, informer


@pytest.mark.asyncio
async def test_process_fires_ready_on_an_address():
    controller, informer = _bare_controller_with_informer()
    action = _session_action()
    action.merge_state(_update("s-1", phase_pb2.ACTION_PHASE_RUNNING, _address()))
    await controller._bg_process(action)
    informer.fire_ready_event.assert_awaited_once_with("s-1")
    informer.fire_completion_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_running_without_address_fires_nothing():
    controller, informer = _bare_controller_with_informer()
    action = _session_action()
    action.merge_state(_update("s-1", phase_pb2.ACTION_PHASE_RUNNING))
    await controller._bg_process(action)
    informer.fire_ready_event.assert_not_awaited()
    informer.fire_completion_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_terminal_fires_completion():
    controller, informer = _bare_controller_with_informer()
    action = _session_action()
    action.merge_state(_update("s-1", phase_pb2.ACTION_PHASE_FAILED))
    await controller._bg_process(action)
    informer.fire_completion_event.assert_awaited_once_with("s-1")


@pytest.mark.asyncio
async def test_wait_for_action_ready_keeps_the_action_cached():
    controller, informer = _bare_controller_with_informer()
    action = _session_action()
    ready = _session_action()
    ready.merge_state(_update("s-1", phase_pb2.ACTION_PHASE_RUNNING, _address()))
    informer.get = AsyncMock(return_value=ready)

    result = await controller._bg_wait_for_action_ready(action)

    informer.wait_for_action_ready.assert_awaited_once_with("s-1")
    informer.remove.assert_not_awaited()
    assert result.address is not None


# ── RemoteController ───────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def task_ctx():
    await flyte.init.aio()
    with patch("flyte._initialize.get_init_config") as mock_config:
        mock_config.return_value.root_dir = pathlib.Path(__file__).parent
        with internal_ctx().replace_task_context(_make_task_context()):
            yield


def _ready(action: Action, address: address_pb2.ActionAddress | None = None, phase=phase_pb2.ACTION_PHASE_RUNNING):
    out = Action(action_id=action.action_id, parent_action_name=action.parent_action_name, phase=phase, started=True)
    out.address = address
    return out


@pytest.mark.asyncio
async def test_register_action_starts_a_session(task_ctx):
    controller = _make_controller()
    session = _Session(name="sandbox", task_spec=_spec(retries=2), queue="q")
    with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
        await controller.register_action(session)
    action = mock_start.call_args[0][0]
    assert action.type == "task"
    assert action.parent_action_name == "parent_action"
    assert action.queue == "q"
    assert action.inputs_uri
    assert not action.task.task_template.metadata.HasField("retries")
    assert session.action_name == action.name


@pytest.mark.asyncio
async def test_sessions_with_the_same_name_get_distinct_actions(task_ctx):
    controller = _make_controller()
    a, b = _Session(name="sandbox", task_spec=_spec()), _Session(name="sandbox", task_spec=_spec())
    with patch.object(controller, "start_action", new_callable=AsyncMock):
        await controller.register_action(a)
        await controller.register_action(b)
    assert a.action_name != b.action_name


@pytest.mark.asyncio
async def test_register_action_needs_the_actions_service(task_ctx):
    controller = _make_controller()
    controller._actions_service = None
    with pytest.raises(flyte.errors.RuntimeSystemError, match="actions service"):
        await controller.register_action(_Session(name="sandbox", task_spec=_spec()))


@pytest.mark.asyncio
async def test_register_action_delegates_conditions(task_ctx):
    controller = _make_controller()
    condition = _Condition(name="approve", data_type=bool)
    with patch.object(controller, "register_condition", new_callable=AsyncMock) as mock_reg:
        await controller.register_action(condition)
    mock_reg.assert_awaited_once_with(condition)


@pytest.mark.asyncio
async def test_register_action_rejects_other_types(task_ctx):
    controller = _make_controller()
    with pytest.raises(TypeError):
        await controller.register_action(object())


@pytest.mark.asyncio
async def test_wait_returns_the_address(task_ctx):
    controller = _make_controller()
    session = _Session(name="sandbox", task_spec=_spec())
    with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
        await controller.register_action(session)
    action = mock_start.call_args[0][0]
    with patch.object(
        controller, "wait_for_action_ready", new_callable=AsyncMock, return_value=_ready(action, _address())
    ):
        address = await controller.wait_for_registered_action(session)
    assert address.pod.name == "agent-0"
    # Still running: still tracked, so it can be aborted.
    assert session.action_name in controller._sessions


@pytest.mark.asyncio
async def test_wait_raises_when_aborted_before_ready(task_ctx):
    controller = _make_controller()
    session = _Session(name="sandbox", task_spec=_spec())
    with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
        await controller.register_action(session)
    action = mock_start.call_args[0][0]
    with patch.object(
        controller,
        "wait_for_action_ready",
        new_callable=AsyncMock,
        return_value=_ready(action, phase=phase_pb2.ACTION_PHASE_ABORTED),
    ):
        with pytest.raises(flyte.errors.ActionAbortedError):
            await controller.wait_for_registered_action(session)
    assert session.action_name not in controller._sessions


@pytest.mark.asyncio
async def test_wait_raises_the_error_when_failed_before_ready(task_ctx):
    controller = _make_controller()
    session = _Session(name="sandbox", task_spec=_spec())
    with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
        await controller.register_action(session)
    action = mock_start.call_args[0][0]
    failed = _ready(action, phase=phase_pb2.ACTION_PHASE_FAILED)
    failed.err = execution_pb2.ExecutionError(
        code="SandboxFailed", message="image pull failed", kind=execution_pb2.ExecutionError.SYSTEM
    )
    with patch.object(controller, "wait_for_action_ready", new_callable=AsyncMock, return_value=failed):
        with pytest.raises(Exception, match="image pull failed"):
            await controller.wait_for_registered_action(session)


@pytest.mark.asyncio
async def test_wait_raises_when_succeeded_without_an_address(task_ctx):
    controller = _make_controller()
    session = _Session(name="sandbox", task_spec=_spec())
    with patch.object(controller, "start_action", new_callable=AsyncMock) as mock_start:
        await controller.register_action(session)
    action = mock_start.call_args[0][0]
    with patch.object(
        controller,
        "wait_for_action_ready",
        new_callable=AsyncMock,
        return_value=_ready(action, phase=phase_pb2.ACTION_PHASE_SUCCEEDED),
    ):
        with pytest.raises(flyte.errors.RuntimeSystemError, match="ended before it was ready"):
            await controller.wait_for_registered_action(session)


@pytest.mark.asyncio
async def test_wait_times_out_and_aborts(task_ctx):
    controller = _make_controller()
    session = _Session(name="sandbox", task_spec=_spec(), ready_timeout=0.05)
    with patch.object(controller, "start_action", new_callable=AsyncMock):
        await controller.register_action(session)

    async def never(_action):
        await asyncio.sleep(10)

    with (
        patch.object(controller, "wait_for_action_ready", side_effect=never),
        patch.object(controller, "get_action", new_callable=AsyncMock, return_value=None),
        patch.object(controller, "cancel_action", new_callable=AsyncMock) as mock_cancel,
    ):
        with pytest.raises(flyte.errors.RuntimeSystemError, match="not ready within"):
            await controller.wait_for_registered_action(session)
    mock_cancel.assert_awaited_once()
    assert session.action_name not in controller._sessions


@pytest.mark.asyncio
async def test_wait_without_register_raises(task_ctx):
    controller = _make_controller()
    with pytest.raises(RuntimeError, match="not started"):
        await controller.wait_for_registered_action(_Session(name="sandbox", task_spec=_spec()))


@pytest.mark.asyncio
async def test_abort_cancels_once(task_ctx):
    controller = _make_controller()
    session = _Session(name="sandbox", task_spec=_spec())
    with patch.object(controller, "start_action", new_callable=AsyncMock):
        await controller.register_action(session)
    with (
        patch.object(controller, "get_action", new_callable=AsyncMock, return_value=None),
        patch.object(controller, "cancel_action", new_callable=AsyncMock) as mock_cancel,
    ):
        await controller.abort_registered_action(session, "done")
        await controller.abort_registered_action(session, "done again")
    mock_cancel.assert_awaited_once()


# ── flyte._session ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_new_session_outside_a_task_raises():
    with pytest.raises(RuntimeError, match="within a task"):
        await new_session.aio("sandbox", task_spec=_spec())


def test_session_rejects_a_non_positive_ready_timeout():
    with pytest.raises(ValueError):
        _Session(name="sandbox", task_spec=_spec(), ready_timeout=0)


@pytest.mark.asyncio
async def test_local_controller_does_not_run_sessions():
    controller = object.__new__(LocalController)
    with pytest.raises(NotImplementedError):
        await controller.register_action(_Session(name="sandbox", task_spec=_spec()))


def test_condition_output_is_untouched_by_addresses():
    # A condition's value and a session's address are separate fields on the update.
    cond = Action(
        action_id=identifier_pb2.ActionIdentifier(name="c-1", run=RUN), parent_action_name="p", type="condition"
    )
    u = _update("c-1", phase_pb2.ACTION_PHASE_SUCCEEDED)
    u.value.CopyFrom(literals_pb2.Literal())
    cond.merge_state(u)
    assert cond.condition_output is not None
    assert cond.address is None
