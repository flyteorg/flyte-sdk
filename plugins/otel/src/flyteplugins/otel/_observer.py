"""Turning Flyte's task and trace callbacks into OpenTelemetry spans."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Generator, Mapping
from typing import TYPE_CHECKING, Any, Optional

import flyte
from flyte._logging import logger
from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from ._ids import trace_id_for_run

if TYPE_CHECKING:
    from flyte._observe import Recorder, StepInfo, TaskInfo
    from flyte.models import ActionID

__all__ = ["OtelObserver", "RunScopedIdGenerator"]

# Set for the instant a root span is being created, so the id generator below can hand back
# the run's derived trace id instead of a random one. A ContextVar rather than a plain
# attribute because tasks may start spans from several coroutines on the same thread.
_pinned_trace_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("flyte_pinned_trace_id", default=None)


class RunScopedIdGenerator(IdGenerator):
    """Hands back the run's derived trace id when one has been pinned.

    OpenTelemetry gives no way to ask for a specific trace id when starting a span; the
    tracer always asks its provider's id generator. Pinning the value around the one call
    that needs it is the supported way in.

    Everything else is delegated, so wrapping a provider that was configured elsewhere keeps
    whatever id generation it already had.
    """

    def __init__(self, delegate: IdGenerator | None = None):
        self._delegate = delegate or RandomIdGenerator()

    def generate_span_id(self) -> int:
        return self._delegate.generate_span_id()

    def generate_trace_id(self) -> int:
        pinned = _pinned_trace_id.get()
        return pinned if pinned is not None else self._delegate.generate_trace_id()


def _action_attributes(action: "ActionID") -> dict[str, Any]:
    """The Flyte identifiers that let a span be traced back to the run that produced it.

    These are also what a Grafana data link queries on to jump from a span back into the
    Flyte UI, so the names here are effectively public API.
    """
    attributes: dict[str, Any] = {"flyte.action_name": action.name}
    if action.run_name:
        attributes["flyte.run_name"] = action.run_name
    if action.project:
        attributes["flyte.project"] = action.project
    if action.domain:
        attributes["flyte.domain"] = action.domain
    if action.org:
        attributes["flyte.org"] = action.org
    return attributes


def _inbound_context(carrier: Mapping[str, str] | None) -> Context | None:
    """Read a parent span context out of a Flyte custom_context carrier.

    This is the same W3C carrier the documented `custom_context` propagation pattern uses,
    so a run submitted inside a caller's span joins that caller's trace rather than starting
    its own. Returns None when the carrier holds nothing usable, which is the ordinary case
    for a run kicked off without any surrounding trace.
    """
    if not carrier:
        return None
    try:
        context = extract(dict(carrier))
    except Exception:
        logger.debug("Could not extract a trace context from custom_context", exc_info=True)
        return None
    span_context = trace_api.get_current_span(context).get_span_context()
    return context if span_context.is_valid else None


@contextlib.contextmanager
def _propagate_to_sub_actions() -> Generator[None, None, None]:
    """Publish the active span into custom_context for the duration of the block.

    Best effort: propagation failing is not a reason to fail the task, and outside a task
    context `flyte.custom_context` is itself a no-op.
    """
    carrier: dict[str, str] = {}
    try:
        inject(carrier)
    except Exception:
        logger.debug("Could not inject a trace context for sub-actions", exc_info=True)

    if not carrier:
        yield
        return

    # Driven by hand rather than with `with`, so that only entering and leaving the context
    # are guarded. Wrapping the body too would swallow the task's own exceptions.
    manager = None
    try:
        manager = flyte.custom_context(**carrier)
        manager.__enter__()
    except Exception:
        logger.debug("Could not set custom_context for sub-actions", exc_info=True)
        manager = None

    try:
        yield
    finally:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                logger.debug("Could not restore custom_context after the task", exc_info=True)


def _finish(span: Span, recorder: "Recorder") -> None:
    """Stamp the outcome on a span before it ends.

    `flyte.trace` swallows the user's exception so it can be written to the durable log,
    so the error arrives on the recorder rather than propagating through the span's own
    exception handling.
    """
    error = recorder.error
    if error is None:
        span.set_status(Status(StatusCode.OK))
        return
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))


class OtelObserver:
    """A `flyte._observe.Observer` that records spans.

    Task spans are roots pinned to the run's derived trace id. Every attempt of a task
    therefore starts its own subtree and because the trace id is shared those subtrees all
    collect into the one trace: a resumed run reads as the attempt that crashed followed by
    the attempt that finished.
    """

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    @contextlib.contextmanager
    def task_span(self, info: "TaskInfo", recorder: "Recorder") -> Generator[None, None, None]:
        span = self._start_task_span(info)

        with trace_api.use_span(span, end_on_exit=True, record_exception=False, set_status_on_exception=False):
            # Hand this span down as the parent for anything spawned inside the task. Flyte
            # copies custom_context into every sub-action's inputs, so a child task running in
            # another pod picks the parent up with no wiring of our own, and because inputs are
            # durable it survives a resume too.
            with _propagate_to_sub_actions():
                try:
                    yield
                finally:
                    _finish(span, recorder)

    def _start_task_span(self, info: "TaskInfo") -> Span:
        """Start the task span under an inbound trace when there is one.

        A `traceparent` in the run's context means something outside Flyte already started
        this trace: the caller who submitted the run or the parent task that spawned this
        one. Nesting under it is what joins the two halves, and it keeps working across a
        resume because the carrier travels with the action's persisted inputs.

        With nothing inbound this is a root span, and the trace id is derived from the run so
        that separate attempts still converge on one trace.
        """
        attributes = {**_action_attributes(info.action), "flyte.task_name": info.name}
        parent = _inbound_context(info.custom_context)

        if parent is not None:
            return self._tracer.start_span(
                info.name,
                context=parent,
                kind=SpanKind.INTERNAL,
                attributes=attributes,
                record_exception=False,
                set_status_on_exception=False,
            )

        token = _pinned_trace_id.set(trace_id_for_run(info.action))
        try:
            # An empty Context makes this a root span, which is what forces the tracer to
            # mint a trace id at all, and therefore to consult the pinned value.
            return self._tracer.start_span(
                info.name,
                context=Context(),
                kind=SpanKind.INTERNAL,
                attributes=attributes,
                record_exception=False,
                set_status_on_exception=False,
            )
        finally:
            _pinned_trace_id.reset(token)

    @contextlib.contextmanager
    def step_span(self, info: "StepInfo", recorder: "Recorder") -> Generator[None, None, None]:
        attributes = {
            **_action_attributes(info.action),
            "flyte.step_name": info.name,
            "flyte.replayed": info.replayed,
        }
        if info.task_action is not None:
            attributes["flyte.task_action_name"] = info.task_action.name

        # A replayed step opens and closes this block without doing any work, so it lands as
        # a near-instant span. That is the honest shape: the step did not run here, it was
        # read back from the durable log, and flyte.replayed says so.
        with self._tracer.start_as_current_span(
            info.name,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield
            finally:
                _finish(span, recorder)


def make_id_generator(existing: Optional[IdGenerator] = None) -> IdGenerator:
    """Return an id generator that honours pinned trace ids, wrapping one if given."""
    if isinstance(existing, RunScopedIdGenerator):
        return existing
    return RunScopedIdGenerator(existing)


def adopt_provider(tracer_provider: object) -> bool:
    """Make an already configured TracerProvider able to honour run derived trace ids.

    Someone who set up their own provider should not have to give it up to get durable traces.
    Swapping in a wrapping id generator leaves their sampler, resource and exporters exactly as they were.

    Returns whether the swap happened. A provider without an `id_generator` (the API's
    no-op provider, or a stub) is left alone and simply gets random trace ids.
    """
    existing = getattr(tracer_provider, "id_generator", None)
    if existing is None:
        return False
    try:
        tracer_provider.id_generator = make_id_generator(existing)  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Could not install a run scoped id generator on the supplied provider", exc_info=True)
        return False
    return True
