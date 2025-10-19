from flyteidl2.core.execution_pb2 import TaskExecution


def is_terminal_phase(phase: TaskExecution.Phase) -> bool:
    """
    Return true if the phase is terminal.
    """
    return phase in [TaskExecution.SUCCEEDED, TaskExecution.ABORTED, TaskExecution.FAILED]


def convert_to_flyte_phase(state: str) -> TaskExecution.Phase:
    """
    Convert the state from the connector to the phase in flyte.
    """
    state = state.lower()
    if state in ["failed", "timeout", "timedout", "canceled", "cancelled", "skipped"]:
        return TaskExecution.FAILED
    if state in ["internal_error"]:
        return TaskExecution.RETRYABLE_FAILED
    elif state in ["done", "succeeded", "success", "completed"]:
        return TaskExecution.SUCCEEDED
    elif state in ["running", "terminating"]:
        return TaskExecution.RUNNING
    elif state in ["pending"]:
        return TaskExecution.INITIALIZING
    raise ValueError(f"Unrecognized state: {state}")
