"""Observation hooks around task and trace execution.

Flyte core never imports an observability SDK. Instead it exposes the two moments that
matter to one — a task starting and a `flyte.trace` step running — and lets an
out-of-tree package subscribe. `flyteplugins-otel` is the first subscriber; it turns
these callbacks into OpenTelemetry spans.

The reason this is a context manager rather than a plain callback is that an observer
usually needs its span to be active while the user's code runs, so that any spans an
instrumentation library creates underneath nest inside it rather than floating up as trace roots.

The other reason core has to be involved at all is durability. A resumed run serves
already-completed steps out of its durable log without re-executing them, so nothing in
the user's code path fires for those steps and an observer that only saw live execution
would report a trace with holes in it. `StepInfo.replayed` marks those steps so an
observer can record them anyway.

Observers are best-effort by construction: a failure inside one is logged and swallowed,
never surfaced to the task being observed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from flyte._logging import logger

if TYPE_CHECKING:
    from flyte.models import ActionID

__all__ = [
    "Observer",
    "Recorder",
    "StepInfo",
    "TaskInfo",
    "has_observers",
    "observe_step",
    "observe_task",
    "register_observer",
    "unregister_observer",
]


@dataclass(frozen=True)
class TaskInfo:
    """A task starting to execute, in the container that will run it.

    `custom_context` is the run's propagated key-value context. It is the carrier an
    observer reads a W3C `traceparent` out of, so a Flyte run can nest under a trace that
    started outside it, and the one it writes back into so sub-actions inherit the parent.
    Flyte copies it into every sub-action's inputs, which makes it durable and cross-pod.
    """

    name: str
    action: ActionID
    custom_context: Mapping[str, str] = field(default_factory=dict)
    version: str = ""


@dataclass(frozen=True)
class StepInfo:
    """One `flyte.trace` step.

    `action` is the step's own deterministic sub-action id, and `task_action` is the
    task that owns it — the pairing an observer needs to nest a step span under its task
    span. `replayed` is True when a resumed run served this step from its durable log
    rather than executing it, in which case the enclosing block does no work and the
    recorded span has no meaningful duration.
    """

    name: str
    action: ActionID
    task_action: ActionID | None
    replayed: bool = False


@dataclass
class Recorder:
    """Handed to the caller inside an observation block, to report an outcome.

    `flyte.trace` catches the user's exception so it can be persisted to the durable
    log, which means the exception never propagates out through the observation block and
    an observer cannot pick it up from `__exit__`. Callers pass it here instead.
    """

    error: BaseException | None = None

    def record_error(self, error: BaseException) -> None:
        self.error = error


class Observer(Protocol):
    """Receives task and step execution boundaries.

    Both methods must return a context manager. Implementations should assume they are
    called on hot paths and keep the non-observing case cheap.
    """

    def task_span(self, info: TaskInfo, recorder: Recorder) -> contextlib.AbstractContextManager[object]: ...

    def step_span(self, info: StepInfo, recorder: Recorder) -> contextlib.AbstractContextManager[object]: ...


_observers: list[Observer] = []


def register_observer(observer: Observer) -> None:
    """Subscribe an observer. Registering the same object twice is a no-op."""
    if observer not in _observers:
        _observers.append(observer)


def unregister_observer(observer: Observer) -> None:
    """Remove a previously registered observer, if present."""
    with contextlib.suppress(ValueError):
        _observers.remove(observer)


def has_observers() -> bool:
    """Whether any observer is registered. Callers use this to skip setup work."""
    return bool(_observers)


@contextlib.contextmanager
def _guarded(observer: Observer, method: str, info: object, recorder: Recorder) -> Generator[None, None, None]:
    """Run one observer's context manager with both ends guarded.

    Entry and exit are caught separately because a failure at either end is the observer's
    problem and must not become the task's. An observer that fails to start is simply
    dropped for this block and one that fails to finish is logged.

    Exit is always reported as clean even when the body raised. Observers learn about
    failures from the recorder rather than from `__exit__`, so there is nothing to hand
    over, and passing a live exception into span-ending teardown only gives it a second
    chance to raise.
    """
    manager = None
    try:
        manager = getattr(observer, method)(info, recorder)
        manager.__enter__()
    except Exception:
        logger.debug(f"Observer {observer!r} failed to start a span for {info!r}", exc_info=True)
        manager = None

    try:
        yield
    finally:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                logger.debug(f"Observer {observer!r} failed to end a span for {info!r}", exc_info=True)


@contextlib.contextmanager
def _fan_out(method: str, info: object) -> Generator[Recorder, None, None]:
    """Enter every observer's context manager around the body, and never let one break it.

    `ExitStack` unwinds whatever entered, in reverse, so one observer's teardown cannot
    strand another's.
    """
    recorder = Recorder()
    if not _observers:
        yield recorder
        return

    with contextlib.ExitStack() as stack:
        # A snapshot, so an observer registering or unregistering from inside the block
        # cannot mutate the list being iterated.
        for observer in list(_observers):
            stack.enter_context(_guarded(observer, method, info, recorder))
        yield recorder


def observe_task(info: TaskInfo) -> contextlib.AbstractContextManager[Recorder]:
    """Observe a task's execution. Yields a `Recorder`."""
    return _fan_out("task_span", info)


def observe_step(info: StepInfo) -> contextlib.AbstractContextManager[Recorder]:
    """Observe one `flyte.trace` step, executed or replayed. Yields a `Recorder`."""
    return _fan_out("step_span", info)
