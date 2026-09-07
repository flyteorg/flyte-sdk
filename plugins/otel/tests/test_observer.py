import pytest
from flyte._observe import StepInfo, TaskInfo, observe_step, observe_task
from flyte.models import ActionID
from opentelemetry.trace import StatusCode

from flyteplugins.otel import trace_id_for_run

ACTION = ActionID(name="a0", run_name="run-abc", project="proj", domain="dev", org="acme")
STEP_ACTION = ActionID(name="a0-step-1", run_name="run-abc", project="proj", domain="dev", org="acme")


def by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def test_task_span_uses_the_runs_derived_trace_id(spans):
    with observe_task(TaskInfo(name="my_task", action=ACTION)):
        pass

    span = spans.get_finished_spans()[0]
    assert span.context.trace_id == trace_id_for_run(ACTION)
    assert span.attributes["flyte.run_name"] == "run-abc"
    assert span.attributes["flyte.project"] == "proj"
    assert span.attributes["flyte.domain"] == "dev"
    assert span.attributes["flyte.org"] == "acme"
    assert span.attributes["flyte.task_name"] == "my_task"


def test_task_span_nests_under_an_inbound_traceparent(spans):
    """The documented custom_context pattern hands us a W3C carrier; we must join that trace.

    This is what makes a run submitted inside a caller's span show up as part of the caller's
    trace rather than as a disconnected one of our own.
    """
    from opentelemetry.propagate import inject
    from opentelemetry.sdk.trace import TracerProvider

    # An entirely separate provider, standing in for the caller's own OTel setup.
    caller = TracerProvider().get_tracer("caller")
    with caller.start_as_current_span("workflow_run") as outer:
        carrier = {}
        inject(carrier)
    assert "traceparent" in carrier

    with observe_task(TaskInfo(name="my_task", action=ACTION, custom_context=carrier)):
        pass

    span = spans.get_finished_spans()[0]
    outer_context = outer.get_span_context()
    assert span.context.trace_id == outer_context.trace_id
    assert span.parent.span_id == outer_context.span_id
    # The inbound trace wins over the derived one, otherwise the two halves stay split.
    assert span.context.trace_id != trace_id_for_run(ACTION)


def test_an_unusable_carrier_falls_back_to_the_derived_trace_id(spans):
    """Junk or unrelated keys in custom_context must not cost us a trace."""
    with observe_task(TaskInfo(name="my_task", action=ACTION, custom_context={"project": "my-project"})):
        pass
    assert spans.get_finished_spans()[0].context.trace_id == trace_id_for_run(ACTION)


def test_step_spans_nest_under_the_task_span(spans):
    with observe_task(TaskInfo(name="my_task", action=ACTION)):
        with observe_step(StepInfo(name="call_model", action=STEP_ACTION, task_action=ACTION)):
            pass

    recorded = by_name(spans)
    task, step = recorded["my_task"], recorded["call_model"]
    assert step.parent.span_id == task.context.span_id
    assert step.context.trace_id == task.context.trace_id
    assert step.attributes["flyte.task_action_name"] == "a0"
    assert step.attributes["flyte.replayed"] is False


def test_replayed_step_is_marked(spans):
    with observe_task(TaskInfo(name="my_task", action=ACTION)):
        with observe_step(StepInfo(name="call_model", action=STEP_ACTION, task_action=ACTION, replayed=True)):
            pass

    assert by_name(spans)["call_model"].attributes["flyte.replayed"] is True


def test_error_recorded_on_the_recorder_sets_span_status(spans):
    """flyte.trace swallows the exception, so the observer only learns of it this way."""
    with observe_step(StepInfo(name="call_model", action=STEP_ACTION, task_action=ACTION)) as recorder:
        recorder.record_error(ValueError("model refused"))

    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert "model refused" in span.status.description
    assert span.events, "the exception should be recorded as a span event"


def test_success_sets_ok_status(spans):
    with observe_step(StepInfo(name="call_model", action=STEP_ACTION, task_action=ACTION)):
        pass
    assert spans.get_finished_spans()[0].status.status_code is StatusCode.OK


def test_exception_escaping_the_block_still_ends_the_span(spans):
    with pytest.raises(RuntimeError):
        with observe_task(TaskInfo(name="my_task", action=ACTION)):
            raise RuntimeError("boom")

    assert len(spans.get_finished_spans()) == 1


def test_a_broken_observer_does_not_break_the_task():
    """Observation is best effort; a bad observer must never fail the work it watches."""
    import flyte._observe as observe

    class Broken:
        def task_span(self, info, recorder):
            raise RuntimeError("observer is broken")

        def step_span(self, info, recorder):
            raise RuntimeError("observer is broken")

    observe.register_observer(Broken())
    try:
        with observe_task(TaskInfo(name="my_task", action=ACTION)):
            pass
    finally:
        for leftover in list(observe._observers):
            observe.unregister_observer(leftover)


def test_an_observer_that_fails_on_teardown_does_not_break_the_task():
    """A span exporter blowing up at the end of the block is still the observer's problem."""
    import contextlib

    import flyte._observe as observe

    @contextlib.contextmanager
    def explodes_on_exit(*_args):
        yield
        raise RuntimeError("export failed")

    class BreaksAtTheEnd:
        task_span = explodes_on_exit
        step_span = explodes_on_exit

    body_completed = False
    observe.register_observer(BreaksAtTheEnd())
    try:
        with observe_task(TaskInfo(name="my_task", action=ACTION)):
            body_completed = True
    finally:
        for leftover in list(observe._observers):
            observe.unregister_observer(leftover)

    assert body_completed


def test_one_broken_observer_does_not_stop_a_working_one(spans):
    """Observers are independent; a bad one must not cost you the spans from a good one."""
    import flyte._observe as observe

    class Broken:
        def task_span(self, info, recorder):
            raise RuntimeError("observer is broken")

        def step_span(self, info, recorder):
            raise RuntimeError("observer is broken")

    observe.register_observer(Broken())
    with observe_task(TaskInfo(name="my_task", action=ACTION)):
        pass

    assert [span.name for span in spans.get_finished_spans()] == ["my_task"]


def test_no_observers_registered_is_a_cheap_no_op():
    with observe_task(TaskInfo(name="my_task", action=ACTION)) as recorder:
        assert recorder.error is None
