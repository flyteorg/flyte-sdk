"""What a real Flyte run produces: nested spans, and generations carrying Flyte identity."""

import flyte
import pytest
from agento11y import GenerationStart, ModelRef, assistant_text_message
from agento11y.context import agent_name_from_context, agent_version_from_context, conversation_id_from_context

from flyteplugins.agento11y import get_client, init

env = flyte.TaskEnvironment(name="ao11y_e2e")

seen: dict[str, object] = {}


@flyte.trace
async def think(prompt: str) -> str:
    client = get_client()
    with client.start_generation(GenerationStart(model=ModelRef(provider="openai", name="gpt-4o"))) as rec:
        rec.set_result(output=[assistant_text_message("hi")])
    return "ok"


@env.task
async def agent(n: int = 2) -> str:
    seen["run_name"] = flyte.ctx().action.run_name
    seen["conversation_id"] = conversation_id_from_context()
    seen["agent_name"] = agent_name_from_context()
    seen["agent_version"] = agent_version_from_context()
    for i in range(n):
        await think(f"q{i}")
    return "done"


@pytest.mark.asyncio
async def test_a_run_produces_one_nested_trace_with_bound_identity(clean, spans, generations):
    init(
        service_name="my-agent",
        exporter=spans,
        disable_batch=True,
        set_global=False,
        client_options={"generation_exporter": generations},
    )

    await flyte.init.aio()
    flyte.run(agent, n=2)
    get_client().shutdown()

    recorded = spans.get_finished_spans()
    names = [s.name for s in recorded]
    task = next(s for s in recorded if s.name == "ao11y_e2e.agent")
    steps = [s for s in recorded if s.name == "think"]
    gens = [s for s in recorded if s.name.startswith("generateText")]

    # Task span, trace steps under it, generation spans under those.
    assert len(steps) == 2, f"expected two trace steps, got {names}"
    assert len(gens) == 2, f"expected two generation spans, got {names}"
    assert len({s.context.trace_id for s in recorded}) == 1, "the whole run must be one trace"
    assert {s.parent.span_id for s in steps} == {task.context.span_id}
    assert {s.parent.span_id for s in gens} == {s.context.span_id for s in steps}


@pytest.mark.asyncio
async def test_flyte_identity_is_bound_onto_the_agento11y_context(clean, spans, generations):
    """Grafana groups by conversation and compares by agent version; Flyte knows both."""
    init(
        service_name="my-agent",
        exporter=spans,
        disable_batch=True,
        set_global=False,
        client_options={"generation_exporter": generations},
    )

    await flyte.init.aio()
    flyte.run(agent, n=1)
    get_client().shutdown()

    assert seen["conversation_id"] == seen["run_name"]
    assert seen["agent_name"] == "ao11y_e2e.agent"
    # Local runs may carry an empty version; when set it must be the task's.
    assert seen["agent_version"] in (None, "", flyte.ctx().version if flyte.ctx() else "")


@pytest.mark.asyncio
async def test_generations_sent_to_grafana_carry_the_runs_trace_id(clean, spans, generations):
    """This is the link between a generation in Grafana and the Flyte run that made it."""
    init(
        service_name="my-agent",
        exporter=spans,
        disable_batch=True,
        set_global=False,
        client_options={"generation_exporter": generations},
    )

    await flyte.init.aio()
    flyte.run(agent, n=2)
    get_client().shutdown()

    task = next(s for s in spans.get_finished_spans() if s.name == "ao11y_e2e.agent")
    assert generations.generations, "no generations were exported"
    for generation in generations.generations:
        assert generation.trace_id == format(task.context.trace_id, "032x")
        assert generation.conversation_id == seen["run_name"]


@pytest.mark.asyncio
async def test_conversation_binding_can_be_turned_off(clean, spans, generations):
    """An agent whose conversations outlive a single run supplies its own id."""
    init(
        service_name="my-agent",
        exporter=spans,
        disable_batch=True,
        set_global=False,
        bind_conversation=False,
        client_options={"generation_exporter": generations},
    )

    await flyte.init.aio()
    flyte.run(agent, n=1)
    get_client().shutdown()

    assert seen["conversation_id"] is None
    assert seen["agent_name"] == "ao11y_e2e.agent"


# --- sync tasks ---

sync_seen: dict[str, object] = {}


@flyte.trace
def sync_think(prompt: str) -> str:
    client = get_client()
    with client.start_generation(GenerationStart(model=ModelRef(provider="openai", name="gpt-4o"))) as rec:
        rec.set_result(output=[assistant_text_message("hi")])
    return "ok"


@env.task
def sync_agent(n: int = 2) -> str:
    sync_seen["run_name"] = flyte.ctx().action.run_name
    sync_seen["conversation_id"] = conversation_id_from_context()
    sync_seen["agent_name"] = agent_name_from_context()
    for i in range(n):
        sync_think(f"q{i}")
    return "done"


@pytest.mark.asyncio
async def test_a_sync_task_gets_the_same_nesting_and_binding(clean, spans, generations):
    """Sync tasks run the same observer path, so they must not lose either behaviour."""
    init(
        service_name="my-agent",
        exporter=spans,
        disable_batch=True,
        set_global=False,
        client_options={"generation_exporter": generations},
    )

    await flyte.init.aio()
    flyte.run(sync_agent, n=2)
    get_client().shutdown()

    recorded = spans.get_finished_spans()
    gens = [s for s in recorded if s.name.startswith("generateText")]
    steps = [s for s in recorded if s.name == "sync_think"]

    assert len(gens) == 2
    assert len({s.context.trace_id for s in recorded}) == 1
    assert {s.parent.span_id for s in gens} == {s.context.span_id for s in steps}
    assert sync_seen["conversation_id"] == sync_seen["run_name"]
    assert sync_seen["agent_name"] == "ao11y_e2e.sync_agent"
