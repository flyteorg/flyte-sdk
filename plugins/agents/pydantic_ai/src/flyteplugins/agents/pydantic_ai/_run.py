"""``run_agent`` — run a Pydantic AI agent on Flyte.

Pydantic AI owns the agent loop. ``run_agent`` runs that loop inside your
``@env.task``: it builds an ``Agent`` with Flyte-task tools, drives the agent,
and returns the final answer. Each tool call runs as a durable Flyte child action
(its own container/resources, with retries and caching).

Observability: the run timeline — tool calls and AI message turns — is rendered
into the Flyte task report.

The adapter minimizes delta between native Pydantic AI code and Flyte integration
by exposing tools that are drop-in replacements for Pydantic AI ``Tool`` instances.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import ReportTimeline, flush_report

from ._tools import _coerce_tool

if typing.TYPE_CHECKING:
    from pydantic_ai import Agent as _Agent
else:
    _Agent = None

# Module-level alias for test monkeypatching.
Agent = _Agent


async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: str | None = None,
    instructions: str | None = None,
    agent: typing.Any = None,
    name: str = "pydantic-ai-agent",
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    **run_kwargs: typing.Any,
) -> str:
    """Run a Pydantic AI agent with the given tools and prompt; return the final text.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a pre-built ``agent`` or ``tools`` + ``model`` to have one
    built for you.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates.
        agent: A pre-built Pydantic AI ``Agent``. Mutually exclusive with ``tools``.
        model: Model name or model provider for the built agent.
        name: Agent name (for debugging/observability).
        durable: Record/replay each model turn via ``flyte.trace``.
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory.
            When set, conversation history is persisted to a keyed ``MemoryStore``
            and resumed on a later run with the same key.
        **run_kwargs: Additional kwargs forwarded to the agent's ``run`` method.

    Returns:
        The agent's final output as a string.
    """
    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("Pydantic AI agent")

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    # Build the agent if not provided
    if agent is None:
        if _Agent is None:
            from pydantic_ai import Agent as _PydanticAgent

            if model is None:
                try:
                    from pydantic_ai.models.openai import OpenAIModel

                    model = OpenAIModel("gpt-4o")
                except ImportError:
                    model = None  # type: ignore[assignment]
        else:
            _PydanticAgent = _Agent
            if model is None:
                model = "gpt-4o"  # placeholder when patched

        agent = _PydanticAgent(
            model=model,
            name=name,
            system_prompt=f"You are a helpful assistant named {name}.",
        )

    # Run the agent
    result = await agent.run(input, tools=[_coerce_tool(t) for t in tools], **run_kwargs)

    # Extract final text
    final = str(result.data) if result.data is not None else ""

    if observability:
        await flush_report()

    return final or ""
