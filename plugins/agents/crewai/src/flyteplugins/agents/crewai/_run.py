"""``run_agent`` — run a CrewAI agent on Flyte.

CrewAI owns the agent loop (it drives the model + tools). ``run_agent`` runs that
loop inside your ``@env.task``: it builds an agent with Flyte-task tools, drives
the agent via ``Agent.kickoff_async``, and returns the final answer. Each tool
call runs as a durable Flyte child action (its own container/resources, with
retries and caching).

Observability: the run timeline — tool calls and AI message turns — is rendered
into the Flyte task report.

The adapter minimizes delta between native CrewAI code and Flyte integration by
exposing tools that are drop-in ``crewai.tools.BaseTool`` instances.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import ReportTimeline, flush_report

from . import _memory
from ._durable import make_durable_llm
from ._tools import _coerce_tool

if typing.TYPE_CHECKING:
    from crewai import Agent as _CrewAgent
else:
    _CrewAgent = None


def _extract_text(result: typing.Any) -> str:
    """Pull the final answer out of a CrewAI ``LiteAgentOutput`` (or fallback)."""
    if result is None:
        return ""
    raw = getattr(result, "raw", None)
    if isinstance(raw, str):
        return raw
    return str(result)


async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: str | None = None,
    instructions: str | None = None,
    agent: typing.Any = None,
    name: str = "crewai-agent",
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    **run_kwargs: typing.Any,
) -> str:
    """Run a CrewAI agent with the given tools and prompt; return the final text.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a pre-built ``agent`` (with its own tools already attached) or
    ``tools`` + ``model`` to have one built for you — not both.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates. Attached
            natively to the built ``Agent(tools=...)``. Ignored when ``agent`` is
            given (a pre-built agent carries its own tools).
        model: Model name (e.g. ``"gpt-4o"``) for the built agent.
        instructions: Extra guidance folded into the built agent's backstory.
        agent: A pre-built CrewAI ``Agent``. Mutually exclusive with ``tools``.
        name: Agent name (for debugging/observability).
        durable: Record/replay each model turn via ``flyte.trace``. Applied only
            when ``run_agent`` builds the agent (the builder sets a durable
            ``llm``); a pre-built ``agent`` keeps its own ``llm`` and is not
            rewrapped, so its turns are not durable.
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory.
            When set, conversation history is persisted to a keyed ``MemoryStore``
            and resumed on a later run with the same key.
        **run_kwargs: Additional kwargs forwarded to ``Agent.kickoff_async``.

    Returns:
        The agent's final output as a string.
    """
    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("CrewAI agent")

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    # Build the agent if not provided, attaching the Flyte-task tools natively.
    if agent is None:
        if _CrewAgent is None:
            from crewai import Agent as _LocalAgent
        else:
            _LocalAgent = _CrewAgent

        backstory = f"You are a helpful assistant named {name}."
        if instructions:
            backstory = f"{backstory}\n\n{instructions}"

        # When durable, drive the agent with a durable ``LLM`` so every model
        # turn is recorded via ``flyte.trace`` and replayed on retry. For a
        # PREBUILT agent we cannot safely rewrap its ``llm``, so durability is
        # only applied on this builder path (see the module docstring / limitation).
        if durable:
            llm: typing.Any = make_durable_llm(model)
        else:
            llm = model or "gpt-4o"

        agent = _LocalAgent(
            role="Assistant",
            goal="Answer the user's question accurately and concisely.",
            backstory=backstory,
            tools=[_coerce_tool(t) for t in tools],
            llm=llm,
        )

    # Cross-run memory: load the prior transcript (if any) and prepend it to the
    # new user turn, so ``kickoff_async`` continues the conversation. Best-effort.
    store = await _memory.resolve_memory(memory_key)
    if store is not None:
        history = await _memory.load_history(store)
        kickoff_input: typing.Any = _memory.build_input(history, input)
    else:
        kickoff_input = input

    # Drive the agent loop. ``kickoff_async`` takes the prompt (a str, or a list
    # of ``{"role", "content"}`` message dicts when resuming memory) as its input;
    # a pre-built agent already carries its own tools, so none are injected here.
    result = await agent.kickoff_async(kickoff_input, **run_kwargs)

    final = _extract_text(result)

    # Persist this turn to the transcript for the next run with the same key.
    if store is not None:
        await _memory.save_turn(store, input, final)

    if observability:
        await flush_report()

    return final or ""
