"""The behaviour this plugin exists for: one durable run reads as one trace.

Each test drives two independent "processes" against the same run identity, the way a crash
and its resume actually happen, and checks what a backend would end up holding.
"""

import flyte._observe as observe
from flyte._observe import StepInfo, TaskInfo, observe_step, observe_task
from flyte.models import ActionID
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flyteplugins.otel import init, shutdown

RUN = ActionID(name="a0", run_name="run-abc", project="proj", domain="dev", org="acme")


def step(n):
    return ActionID(name=f"a0-step-{n}", run_name="run-abc", project="proj", domain="dev", org="acme")


class Process:
    """One container's worth of SDK lifetime: init, record, export, exit."""

    def __init__(self):
        self.exporter = InMemorySpanExporter()

    def __enter__(self):
        init(exporter=self.exporter, disable_batch=True, set_global=False, service_name="test")
        return self

    def __exit__(self, *exc):
        shutdown()
        for leftover in list(observe._observers):
            observe.unregister_observer(leftover)
        return False

    @property
    def spans(self):
        return self.exporter.get_finished_spans()


def test_a_crash_and_its_resume_land_in_one_trace():
    with Process() as first:
        with observe_task(TaskInfo(name="agent", action=RUN)) as task:
            for n in (1, 2):
                with observe_step(StepInfo(name=f"step_{n}", action=step(n), task_action=RUN)):
                    pass
            with observe_step(StepInfo(name="step_3", action=step(3), task_action=RUN)) as recorder:
                recorder.record_error(RuntimeError("pod evicted"))
            task.record_error(RuntimeError("pod evicted"))

    with Process() as second:
        with observe_task(TaskInfo(name="agent", action=RUN)):
            # Steps 1 and 2 are served from the durable log and never re-execute.
            for n in (1, 2):
                with observe_step(StepInfo(name=f"step_{n}", action=step(n), task_action=RUN, replayed=True)):
                    pass
            for n in (3, 4):
                with observe_step(StepInfo(name=f"step_{n}", action=step(n), task_action=RUN)):
                    pass

    trace_ids = {span.context.trace_id for span in first.spans + second.spans}
    assert len(trace_ids) == 1, "the crash and the resume must record into a single trace"


def test_the_resumed_trace_has_no_holes_in_it():
    """Without replay spans, every step completed before the crash would be missing."""
    with Process() as resumed:
        with observe_task(TaskInfo(name="agent", action=RUN)):
            for n in (1, 2):
                with observe_step(StepInfo(name=f"step_{n}", action=step(n), task_action=RUN, replayed=True)):
                    pass
            for n in (3, 4):
                with observe_step(StepInfo(name=f"step_{n}", action=step(n), task_action=RUN)):
                    pass

    steps = {span.name: span for span in resumed.spans if span.name.startswith("step_")}
    assert set(steps) == {"step_1", "step_2", "step_3", "step_4"}
    assert [steps[f"step_{n}"].attributes["flyte.replayed"] for n in (1, 2, 3, 4)] == [True, True, False, False]


def test_each_attempt_is_its_own_subtree():
    """Attempts share a trace id but not span ids, so a retry never collides with its original."""
    with Process() as first:
        with observe_task(TaskInfo(name="agent", action=RUN)):
            pass
    with Process() as second:
        with observe_task(TaskInfo(name="agent", action=RUN)):
            pass

    first_span, second_span = first.spans[0], second.spans[0]
    assert first_span.context.trace_id == second_span.context.trace_id
    assert first_span.context.span_id != second_span.context.span_id
    assert first_span.parent is None and second_span.parent is None


def test_a_different_run_gets_a_different_trace():
    other = ActionID(name="a0", run_name="run-xyz", project="proj", domain="dev", org="acme")
    with Process() as proc:
        with observe_task(TaskInfo(name="agent", action=RUN)):
            pass
        with observe_task(TaskInfo(name="agent", action=other)):
            pass

    assert len({span.context.trace_id for span in proc.spans}) == 2
