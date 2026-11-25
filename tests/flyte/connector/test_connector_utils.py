import pytest
from flyteidl2.core.execution_pb2 import TaskExecution

from flyte.connectors.utils import convert_to_flyte_phase, is_terminal_phase


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        (TaskExecution.UNDEFINED, False),
        (TaskExecution.QUEUED, False),
        (TaskExecution.INITIALIZING, False),
        (TaskExecution.RUNNING, False),
        (TaskExecution.SUCCEEDED, True),
        (TaskExecution.ABORTED, True),
        (TaskExecution.FAILED, True),
        (TaskExecution.RETRYABLE_FAILED, True),
    ],
)
def test_is_terminal_phase(phase: TaskExecution.Phase, expected: bool) -> None:
    assert is_terminal_phase(phase) is expected


def test_internal_error_is_a_terminal_retryable_failure() -> None:
    phase = convert_to_flyte_phase("internal_error")

    assert phase == TaskExecution.RETRYABLE_FAILED
    assert is_terminal_phase(phase)


def test_retryable_failure_breaks_connector_polling() -> None:
    phase = TaskExecution.RUNNING
    get_calls = 0

    while not is_terminal_phase(phase):
        get_calls += 1
        phase = TaskExecution.RETRYABLE_FAILED

    assert get_calls == 1
