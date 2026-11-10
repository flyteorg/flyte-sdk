"""An enqueue refused by an org limit (FAILED_PRECONDITION with a cloudidl.common.OrgLimitExceeded detail)
fails the child at once with a user error the parent can catch, instead of retrying and surfacing as a
system error that makes the platform retry the whole parent."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.common import identifier_pb2, phase_pb2
from google.protobuf.any_pb2 import Any

import flyte.errors
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._core import Controller

_MSG = '"acme:p:d:r1" would exceed org "acme"\'s max_actions_per_run quota (3 current + 1 requested > 3)'


def _org_limit_detail() -> Any:
    # The open-source SDK has no cloudidl protos; only the type URL identifies the detail.
    return Any(type_url="type.googleapis.com/cloudidl.common.OrgLimitExceeded", value=b"")


def _controller() -> Controller:
    from aiolimiter import AsyncLimiter

    controller = object.__new__(Controller)
    controller._running = True
    controller._shared_queue = asyncio.Queue()
    controller._max_retries = 5
    controller._min_backoff_on_err = 0.1
    controller._max_backoff_on_err = 0.1
    controller._rate_limiter = AsyncLimiter(100, 1.0)
    controller._enqueue_timeout = 0.01
    controller._consecutive_launch_timeouts = 0
    controller._actions_service = AsyncMock()
    informer = AsyncMock()
    controller._informers = AsyncMock()
    controller._informers.get = AsyncMock(return_value=informer)
    return controller


def _action() -> Action:
    return Action(
        parent_action_name="parent",
        action_id=identifier_pb2.ActionIdentifier(name="child", run=identifier_pb2.RunIdentifier(name="r1")),
        type="trace",
    )


@pytest.mark.asyncio
async def test_org_limit_refusal_fails_child_once_as_user_error():
    controller = _controller()
    action = _action()
    await controller._shared_queue.put(action)
    attempts = 0

    async def enqueue(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        controller._running = False
        raise ConnectError(Code.FAILED_PRECONDITION, _MSG, details=[_org_limit_detail()])

    controller._actions_service.enqueue.side_effect = enqueue

    with patch("flyte._internal.controllers.remote._core.asyncio.sleep", new_callable=AsyncMock) as sleep:
        await controller._bg_run(worker_id="w1")

    assert attempts == 1, "an org limit refusal is not retried"
    sleep.assert_not_awaited()
    assert isinstance(action.client_err, flyte.errors.OrgLimitExceededError)
    assert isinstance(action.client_err, flyte.errors.RuntimeUserError)
    assert not isinstance(action.client_err, flyte.errors.RuntimeSystemError)
    assert "max_actions_per_run" in str(action.client_err)


@pytest.mark.asyncio
async def test_failed_precondition_without_org_limit_detail_still_retries():
    controller = _controller()
    controller._actions_service.enqueue.side_effect = ConnectError(Code.FAILED_PRECONDITION, "wrong shard")

    with pytest.raises(flyte.errors.SlowDownError):
        await controller._bg_launch(_action())


class _RustAction:
    """The fields handle_action_failure reads from a Rust controller action."""

    def __init__(self, client_err: str):
        self.err_bytes = None
        self.client_err = client_err
        self.phase_value = phase_pb2.ACTION_PHASE_FAILED
        self.name = "child"


def test_rust_client_error_with_org_limit_prefix_is_user_error():
    from flyte._internal.controllers.remote._org_limits import rust_client_error

    exc = rust_client_error(f"Org limit exceeded: {_MSG}")

    assert isinstance(exc, flyte.errors.OrgLimitExceededError)
    assert str(exc).endswith(_MSG)


def test_rust_client_error_otherwise_system_error():
    from flyte._internal.controllers.remote._org_limits import rust_client_error

    exc = rust_client_error("Runtime error: Precondition failed: wrong shard")

    assert isinstance(exc, flyte.errors.RuntimeSystemError)
    assert "wrong shard" in str(exc)


@pytest.mark.asyncio
async def test_rust_controller_org_limit_error_is_user_error():
    pytest.importorskip("flyte_controller_base")
    from flyte._internal.controllers.remote._r_controller import handle_action_failure

    exc = await handle_action_failure(_RustAction(f"Org limit exceeded: {_MSG}"), "square")

    assert isinstance(exc, flyte.errors.OrgLimitExceededError)
    assert _MSG in str(exc)


@pytest.mark.asyncio
async def test_rust_controller_other_errors_stay_system_errors():
    pytest.importorskip("flyte_controller_base")
    from flyte._internal.controllers.remote._r_controller import handle_action_failure

    exc = await handle_action_failure(_RustAction("Runtime error: Precondition failed: wrong shard"), "square")

    assert isinstance(exc, flyte.errors.RuntimeSystemError)
