"""
Tests for `Run.raise_for_status()`: a finished run raises the same exception a parent task would
have raised for the equivalent sub-action, so runs launched with `flyte.run` compose like actions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.core import execution_pb2
from flyteidl2.workflow import run_definition_pb2

import flyte.errors
from flyte._internal.runtime import convert
from flyte.remote._action import ActionDetails
from flyte.remote._run import Run

RUN_NAME = "run-abc"
_ACTION_ID = identifier_pb2.ActionIdentifier(
    run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name=RUN_NAME),
    name="a0",
)


_root_details: dict[str, ActionDetails] = {}


@pytest.fixture(autouse=True)
def root_action_details():
    """raise_for_status reads the root action's details from the API; serve the one _run() built."""
    fake = MagicMock()
    fake.get_details.aio = AsyncMock(side_effect=lambda action_id: _root_details[action_id.run.name])
    with patch("flyte.remote._run.ActionDetails", fake):
        yield fake.get_details.aio
    _root_details.clear()


def _run(
    phase: phase_pb2.ActionPhase,
    error_info: run_definition_pb2.ErrorInfo | None = None,
    abort_info: run_definition_pb2.AbortInfo | None = None,
) -> Run:
    details = run_definition_pb2.ActionDetails(id=_ACTION_ID, status=run_definition_pb2.ActionStatus(phase=phase))
    if error_info is not None:
        details.error_info.CopyFrom(error_info)
    if abort_info is not None:
        details.abort_info.CopyFrom(abort_info)
    _root_details[RUN_NAME] = ActionDetails(details)
    return Run(
        run_definition_pb2.Run(
            action=run_definition_pb2.Action(id=_ACTION_ID, status=run_definition_pb2.ActionStatus(phase=phase))
        )
    )


def _error(code: str, message: str = "boom", kind=run_definition_pb2.ErrorInfo.KIND_USER):
    return run_definition_pb2.ErrorInfo(code=code, message=message, kind=kind)


def test_succeeded_does_not_raise(root_action_details):
    assert _run(phase_pb2.ACTION_PHASE_SUCCEEDED).raise_for_status() is None
    root_action_details.assert_not_called()


def test_not_done_raises_run_not_done():
    with pytest.raises(flyte.errors.RuntimeUserError) as exc_info:
        _run(phase_pb2.ACTION_PHASE_RUNNING).raise_for_status()
    assert exc_info.value.code == "RunNotDoneError"


@pytest.mark.parametrize(
    ("code", "expected_type"),
    [
        ("QUEUED_TIMEOUT_EXCEEDED", flyte.errors.MaxQueuedTimeExceededError),
        ("MAX_RUNTIME_EXCEEDED", flyte.errors.MaxRuntimeExceededError),
        ("DEADLINE_EXCEEDED", flyte.errors.DeadlineExceededError),
        ("TIMED_OUT", flyte.errors.TaskTimeoutError),
    ],
)
def test_timed_out_raises_cause_specific_error(code, expected_type):
    msg = "timed out waiting for resources: queued_timeout of 1m0s exceeded (attempt=0)"
    with pytest.raises(flyte.errors.TaskTimeoutError) as exc_info:
        _run(phase_pb2.ACTION_PHASE_TIMED_OUT, _error(code, msg)).raise_for_status()
    assert type(exc_info.value) is expected_type
    assert RUN_NAME in str(exc_info.value)
    assert msg in str(exc_info.value)


def test_timed_out_without_error_info_raises_base_timeout():
    with pytest.raises(flyte.errors.TaskTimeoutError) as exc_info:
        _run(phase_pb2.ACTION_PHASE_TIMED_OUT).raise_for_status()
    assert type(exc_info.value) is flyte.errors.TaskTimeoutError


def test_reads_error_from_root_action_details(root_action_details):
    # GetRunDetails omits the root action's error_info, so the error must come from the root
    # action's own details.
    with pytest.raises(flyte.errors.MaxQueuedTimeExceededError):
        _run(phase_pb2.ACTION_PHASE_TIMED_OUT, _error("QUEUED_TIMEOUT_EXCEEDED")).raise_for_status()
    root_action_details.assert_awaited_once_with(_ACTION_ID)


def test_failed_by_reraised_queued_timeout_keeps_its_type():
    # The root re-raised a sub-action's MaxQueuedTimeExceededError, so the run FAILED with the class
    # name as its code.
    run = _run(phase_pb2.ACTION_PHASE_FAILED, _error("MaxQueuedTimeExceededError", "no B300 capacity"))
    with pytest.raises(flyte.errors.MaxQueuedTimeExceededError, match="no B300 capacity"):
        run.raise_for_status()


def test_failed_user_error_keeps_code():
    run = _run(phase_pb2.ACTION_PHASE_FAILED, _error("ValueError", "bad input"))
    with pytest.raises(flyte.errors.RuntimeUserError, match="bad input") as exc_info:
        run.raise_for_status()
    assert exc_info.value.code == "ValueError"


def test_failed_system_error():
    run = _run(
        phase_pb2.ACTION_PHASE_FAILED, _error("OOMKilled", "pod died", kind=run_definition_pb2.ErrorInfo.KIND_SYSTEM)
    )
    with pytest.raises(flyte.errors.RuntimeSystemError, match="pod died"):
        run.raise_for_status()


def test_failed_without_error_info_raises_system_error():
    with pytest.raises(flyte.errors.RuntimeSystemError) as exc_info:
        _run(phase_pb2.ACTION_PHASE_FAILED).raise_for_status()
    assert exc_info.value.code == "UnknownRunFailure"


def test_aborted_raises_action_aborted_with_reason():
    run = _run(phase_pb2.ACTION_PHASE_ABORTED, abort_info=run_definition_pb2.AbortInfo(reason="user cancelled"))
    with pytest.raises(flyte.errors.ActionAbortedError, match="user cancelled"):
        run.raise_for_status()


@pytest.mark.parametrize(
    "code",
    ["MaxQueuedTimeExceededError", "RetriesExhausted|MaxQueuedTimeExceededError"],
)
def test_sub_action_failed_with_reraised_timeout_keeps_its_type(code):
    # Same mapping on the controller path: a grandchild's queued timeout, re-raised by the child,
    # reaches the parent as MaxQueuedTimeExceededError rather than a generic RuntimeUserError.
    err = execution_pb2.ExecutionError(kind=execution_pb2.ExecutionError.USER, code=code, message="no capacity")
    exc = convert.convert_error_to_native(err)
    assert type(exc) is flyte.errors.MaxQueuedTimeExceededError
    assert str(exc) == "no capacity"
