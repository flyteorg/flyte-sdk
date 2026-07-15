"""``run_agent`` — run a Hermes agent on Flyte.

Hermes owns the agent loop (it drives the model + tools). ``run_agent`` runs that
loop inside your ``@env.task``: it builds an agent with Flyte-task tools, drives
the agent, and returns the final answer. Each tool call runs as a durable Flyte
child action (its own container/resources, with retries and caching).

Observability: the run timeline — tool calls and AI message turns — is rendered
into the Flyte task report.

The adapter minimizes delta between native Hermes code and Flyte integration by
exposing tools that are drop-in replacements for Hermes tool instances.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import ReportTimeline, flush_report

from ._durable import durable_run
from ._memory import load_transcript, resolve_memory, save_transcript
from ._tools import _coerce_tool

if typing.TYPE_CHECKING:
    from hermes import Agent as _Agent
else:
    _Agent = None


async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: str | None = None,
    instructions: str | None = None,
    agent: typing.Any = None,
    name: str = "hermes-agent",
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    **run_kwargs: typing.Any,
) -> str:
    """Run a Hermes agent with the given tools and prompt; return the final text.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a pre-built ``agent`` or ``tools`` + ``model`` to have one
    built for you.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates.
        agent: A pre-built Hermes agent. Mutually exclusive with ``tools``.
        model: Model name for the built agent.
        name: Agent name (for debugging/observability).
        durable: Record/replay the agent run via ``flyte.trace`` so a retried task
            replays the recorded answer instead of re-driving the model. Hermes is
            a template SDK with no per-turn model hook, so durability is recorded at
            run granularity here (see ``_durable``); the real adapters record per
            model turn.
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory.
            When set, conversation history is persisted to a keyed ``MemoryStore``
            and resumed on a later run with the same key.
        **run_kwargs: Additional kwargs forwarded to the agent's run method.

    Returns:
        The agent's final output as a string.
    """
    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("Hermes agent")

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    # Cross-run memory: load the prior transcript (if any) and resume the thread.
    store = await resolve_memory(memory_key)
    history = await load_transcript(store)

    # Build the agent if not provided
    if agent is None:
        if _Agent is None:
            from hermes import Agent as _HermesAgent
        else:
            _HermesAgent = _Agent

        if model is None:
            model = "gpt-4o"

        # Attach the tools natively to the agent (the framework convention), so a
        # built agent and a bring-your-own agent are driven the same way below.
        agent = _HermesAgent(
            name=name,
            model=model,
            system_prompt=instructions or f"You are a helpful assistant named {name}.",
            tools=[_coerce_tool(t) for t in tools],
        )

    # Run the agent. A pre-built agent already owns its tools, so nothing is
    # injected here. Prior history (when present) resumes the conversation.
    call_kwargs = dict(run_kwargs)
    if history:
        call_kwargs["message_history"] = history

    async def _drive() -> typing.Any:
        return await agent.run(input, **call_kwargs)

    # Durability: record the run so a retried task replays the answer instead of
    # re-driving the model (see ``_durable`` for why this is run-grained here).
    if durable:
        final = await durable_run(
            _drive, input=input, history=history, tools=tools, model=model, instructions=instructions
        )
    else:
        result = await _drive()
        final = str(result) if result is not None else ""

    # Persist the updated transcript for the next run with this memory_key.
    if store is not None:
        await save_transcript(
            store,
            [*history, {"role": "user", "content": input}, {"role": "assistant", "content": final}],
        )

    if observability:
        await flush_report()

    return final or ""
