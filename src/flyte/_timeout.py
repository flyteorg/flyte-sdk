"""
Wall-clock bounds on a Flyte action.

Three orthogonal fields, each independently optional. A field is treated as
**unlimited** when it is unset (``None``) or zero (``0`` / ``timedelta(0)``) —
the default state of an action is "no time bound."

- ``max_runtime``     — wall-clock spent in the RUNNING phase. Per attempt;
                        resets on each retry. Enforced by the lease worker.
- ``max_queued_time`` — wall-clock spent in QUEUED + WAITING_FOR_RESOURCES
                        before the action enters INITIALIZING. Per attempt;
                        resets on each retry. Maps to proto
                        ``TimeoutStrategy.queued_timeout``.
- ``deadline``        — max elapsed wall-clock from the first QUEUED timestamp
                        to a terminal phase, across **all** attempts (user *or*
                        system). One-time, absolute. The strongest of the
                        three: when it fires mid-attempt, the action terminates
                        as TIMED_OUT regardless of remaining retry budget or
                        per-attempt timer state.

Bare ``int`` (seconds) and bare ``timedelta`` are accepted on the task
``timeout=`` parameter and interpreted as ``max_runtime``. ``timeout=0`` is
equivalent to leaving the timeout unset (unlimited).
"""

from dataclasses import dataclass
from datetime import timedelta


@dataclass
class Timeout:
    """
    Timeout bounds for a task. See module docstring for semantics.

    Example::

        flyte.Timeout(
            max_runtime=timedelta(minutes=30),
            max_queued_time=timedelta(minutes=15),
            deadline=timedelta(hours=2),
        )

    :param max_runtime: Per-attempt RUNNING-phase bound. ``int`` is interpreted
                        as seconds. ``None`` or ``0`` means unlimited.
    :param max_queued_time: Per-attempt queue-wait bound. ``int`` is
                            interpreted as seconds. ``None`` or ``0`` means
                            unlimited.
    :param deadline: Absolute wall-clock budget across all attempts. ``int``
                     is interpreted as seconds. ``None`` or ``0`` means
                     unlimited.
    """

    max_runtime: timedelta | int | None = None
    max_queued_time: timedelta | int | None = None
    deadline: timedelta | int | None = None


TimeoutType = Timeout | int | timedelta


def timeout_from_request(timeout: TimeoutType) -> Timeout:
    """
    Normalize a user-supplied timeout into a :class:`Timeout`.

    A bare ``int`` (seconds) or ``timedelta`` is interpreted as
    ``max_runtime`` for backward compatibility with the original single-bound
    API.
    """
    if isinstance(timeout, Timeout):
        return timeout
    if isinstance(timeout, int):
        return Timeout(max_runtime=timedelta(seconds=timeout))
    if isinstance(timeout, timedelta):
        return Timeout(max_runtime=timeout)
    raise ValueError("Timeout must be an instance of Timeout, int, or timedelta.")
