from dataclasses import dataclass
from datetime import timedelta


@dataclass
class Timeout:
    """
    Timeout class to define a timeout for a task.
    The task timeout can be set to a maximum runtime and a maximum queued time.
    Maximum runtime is the maximum time the task can run for (in one attempt).
    Maximum queued time is the maximum time the task can stay in the queue before it starts executing.

    Example usage:
    ```python
    timeout = Timeout(max_runtime=timedelta(minutes=5), max_queued_time=timedelta(minutes=10))
    @env.task(timeout=timeout)
    async def my_task():
        pass
    ```
    :param max_runtime: timedelta or int - Maximum runtime for the task. If specified int, it will be converted to
    timedelta as seconds.
    :param max_queued_time: optional, timedelta or int - Maximum queued time for the task. If specified int,
    it will be converted to timedelta as seconds. Defaults to None.

    """

    max_runtime: timedelta | int
    max_queued_time: timedelta | int | None = None


TimeoutType = Timeout | int | timedelta


def timeout_from_request(timeout: TimeoutType) -> Timeout:
    """
    Converts a timeout request into a Timeout object.
    """
    if isinstance(timeout, Timeout):
        return timeout
    else:
        if isinstance(timeout, int):
            timeout = timedelta(seconds=timeout)
        elif isinstance(timeout, timedelta):
            pass
        else:
            raise ValueError("Timeout must be an instance of Timeout, int, or timedelta.")
        return Timeout(max_runtime=timeout)
