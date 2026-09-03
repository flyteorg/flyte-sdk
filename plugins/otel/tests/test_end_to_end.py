"""Proves the core hooks actually fire during a real run, not just in isolation."""

import logging

import flyte
import flyte._observe as observe
import pytest
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

env = flyte.TaskEnvironment(name="otel_e2e")


@flyte.trace
async def call_model(x: int) -> int:
    return x * 2


@flyte.trace
async def call_tool(x: int) -> int:
    return x + 1


@env.task
async def agent(n: int = 2) -> int:
    total = 0
    for i in range(n):
        total += await call_model(i)
        total += await call_tool(i)
    return total


@pytest.mark.asyncio
async def test_a_run_produces_a_task_span_with_its_trace_steps_nested_inside(spans):
    await flyte.init.aio()
    flyte.run(agent, n=2)

    recorded = spans.get_finished_spans()
    names = [span.name for span in recorded]
    # The task span carries the environment qualified task name.
    assert "otel_e2e.agent" in names, f"expected a task span, got {names}"
    assert names.count("call_model") == 2
    assert names.count("call_tool") == 2

    task = next(span for span in recorded if span.name == "otel_e2e.agent")
    steps = [span for span in recorded if span.name in {"call_model", "call_tool"}]

    assert {span.context.trace_id for span in steps} == {task.context.trace_id}
    assert {span.parent.span_id for span in steps} == {task.context.span_id}
    assert all(span.attributes["flyte.replayed"] is False for span in steps)
    assert task.attributes["flyte.run_name"]


@env.task
async def late_init(n: int = 1) -> int:
    from flyteplugins.otel import init

    init(service_name="too-late", exporter=ConsoleSpanExporter(), disable_batch=True)
    return n


@pytest.mark.asyncio
async def test_initializing_inside_a_task_warns_that_the_task_span_was_missed():
    """The task span opens before the body runs, so in-task init silently loses it.

    The failure mode is a trace with steps but no task span to hang them from, which is hard
    to diagnose from the output, so it has to say something.

    Captured off the flyte logger directly rather than through caplog, because that logger
    sets propagate = False and so never reaches the root handler caplog installs. The handler
    goes on after flyte.init, which clears the logger's handlers.
    """
    from flyteplugins.otel import shutdown

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    await flyte.init.aio()

    handler = Capture(level=logging.WARNING)
    flyte_logger = logging.getLogger("flyte")
    flyte_logger.addHandler(handler)

    try:
        flyte.run(late_init, n=1)
    finally:
        flyte_logger.removeHandler(handler)
        shutdown()
        for leftover in list(observe._observers):
            observe.unregister_observer(leftover)

    assert any("initialized from inside a running task" in record.getMessage() for record in records), (
        f"expected a late-init warning, saw {[r.getMessage() for r in records]}"
    )


@env.task
async def child(x: int) -> int:
    return x * 10


@env.task
async def parent(n: int = 2) -> int:
    return sum([await child(i) for i in range(n)])


@pytest.mark.asyncio
async def test_a_child_task_nests_under_its_parent(spans):
    """Cross-task parenting, via the traceparent we publish into custom_context.

    Flyte copies custom_context into each sub-action's inputs, so this holds across pods and
    across a resume without the plugin propagating anything itself.
    """
    await flyte.init.aio()
    flyte.run(parent, n=2)

    recorded = spans.get_finished_spans()
    parent_span = next(span for span in recorded if span.name == "otel_e2e.parent")
    children = [span for span in recorded if span.name == "otel_e2e.child"]

    assert len(children) == 2, f"expected two child spans, got {[s.name for s in recorded]}"
    assert {span.context.trace_id for span in children} == {parent_span.context.trace_id}
    assert {span.parent.span_id for span in children if span.parent} == {parent_span.context.span_id}


@flyte.trace
async def no_outputs(x: int):
    print(f"got {x}", flush=True)


@env.task
async def agent_without_outputs(n: int = 2) -> int:
    for i in range(n):
        await no_outputs(i)
    return n


@pytest.mark.asyncio
async def test_a_step_that_executes_is_never_marked_replayed(spans):
    """The durable lookup reports a hit for steps that go on to execute anyway.

    A traced function with no outputs is the clearest case: its action is on record, so the
    lookup succeeds, but there is nothing to return and it runs regardless. Treating that
    hit as a replay would double count every such step and claim work was skipped when it
    was not.
    """
    await flyte.init.aio()
    flyte.run(agent_without_outputs, n=2)

    steps = [span for span in spans.get_finished_spans() if span.name == "no_outputs"]
    assert len(steps) == 2, f"expected one span per call, got {len(steps)}"
    assert all(span.attributes["flyte.replayed"] is False for span in steps)


# --- sync tasks and sync traces ---


@flyte.trace
def sync_step(x: int) -> int:
    return x * 2


@flyte.trace
def sync_stream(n: int):
    yield from range(n)


@env.task
def sync_task(n: int = 2) -> int:
    total = 0
    for i in range(n):
        total += sync_step(i)
    total += sum(sync_stream(n))
    return total


@pytest.mark.asyncio
async def test_sync_tasks_and_sync_traces_are_recorded(spans):
    """flyte.trace has four wrappers; the sync and sync-generator ones need covering too."""
    await flyte.init.aio()
    flyte.run(sync_task, n=2)

    recorded = spans.get_finished_spans()
    task = next(s for s in recorded if s.name == "otel_e2e.sync_task")
    steps = [s for s in recorded if s.name in {"sync_step", "sync_stream"}]

    assert [s.name for s in steps].count("sync_step") == 2
    assert [s.name for s in steps].count("sync_stream") == 1
    assert {s.parent.span_id for s in steps} == {task.context.span_id}
    assert len({s.context.trace_id for s in recorded}) == 1


@pytest.mark.asyncio
async def test_an_unobserved_run_never_describes_its_steps(monkeypatch):
    """With nothing registered, a traced step must not be described for nobody to read.

    _step used to be evaluated as an argument to observe_step, so every traced call built a
    StepInfo even when no observer existed to receive it. Guarding that is only worth
    anything for as long as it stays guarded, and the cost is invisible in output — no span
    is produced either way — so nothing else would catch the regression.

    Deliberately takes no spans fixture: that fixture is what registers an observer.
    """
    import flyte._trace as trace

    assert not observe.has_observers(), "another test left an observer registered"

    calls = 0
    real_step = trace._step

    def counting_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_step(*args, **kwargs)

    monkeypatch.setattr(trace, "_step", counting_step)

    await flyte.init.aio()
    flyte.run(agent, n=2)

    assert calls == 0, f"described {calls} steps with no observer registered"
