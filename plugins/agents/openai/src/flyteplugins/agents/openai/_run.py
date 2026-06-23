"""``run_agent`` — the one-call path to run an OpenAI agent durably on Flyte."""

from __future__ import annotations

import typing

from agents import Agent, RunConfig, Runner
from flyte._task import TaskTemplate
from flyteplugins.agents.core import flush_report, resolve_memory

from ._durable import FlyteModelProvider
from ._memory import FlyteSession
from ._observability import install_flyte_tracing
from ._tools import function_tool


def _normalize_tools(tools: typing.Sequence[typing.Any]) -> list[typing.Any]:
    """Accept bare ``@env.task`` templates as tools by wrapping them on the fly."""
    normalized: list[typing.Any] = []
    for t in tools or ():
        normalized.append(function_tool(t) if isinstance(t, TaskTemplate) else t)
    return normalized


async def run_agent(
    input: str | list[typing.Any],
    *,
    agent: Agent | None = None,
    tools: typing.Sequence[typing.Any] = (),
    model: str = "gpt-4.1",
    instructions: str | None = None,
    name: str = "flyte-agent",
    max_turns: int = 10,
    durable: bool = True,
    observability: bool = True,
    run_config: RunConfig | None = None,
    memory_key: str | None = None,
) -> str:
    """Run an OpenAI Agents SDK agent with Flyte providing the runtime.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each model turn is recorded via ``flyte.trace`` (replayed on
    retry) and each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a fully-built ``agent`` (keeping its handoffs/guardrails), or
    ``tools`` + ``instructions`` + ``model`` to have one built for you. ``tools``
    may be :func:`function_tool`-wrapped tools or bare ``@env.task`` templates
    (wrapped automatically).

    Args:
        input: The user prompt (or a list of input items).
        agent: A pre-built ``agents.Agent``. Mutually exclusive with ``tools``.
        tools: Tools to expose (when ``agent`` is not given).
        model: Model name (when ``agent`` is not given).
        instructions: System instructions (when ``agent`` is not given).
        name: Agent name (when ``agent`` is not given).
        max_turns: Maximum model to tool turns and vice versa before the SDK raises.
        durable: Record/replay each model turn via ``flyte.trace``.
        observability: Render the run timeline into the Flyte task report.
        run_config: A custom ``RunConfig``; ``model_provider`` is wrapped for
            durability unless ``durable=False``.
        memory_key: Stable id (e.g. a user/thread id) for **cross-run memory**.
            When set, conversation history is loaded from and saved to a durable,
            keyed ``MemoryStore`` (via the SDK's ``Session``), so a later run with
            the same key continues the conversation. ``None`` disables memory.

    Returns:
        The agent's final output as a string.
    """
    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    if agent is None:
        agent = Agent(
            name=name,
            instructions=instructions or "You are a helpful assistant.",
            model=model,
            tools=_normalize_tools(tools),
        )

    config = run_config or RunConfig()
    if durable:
        config.model_provider = FlyteModelProvider(config.model_provider)
    if observability:
        install_flyte_tracing()

    store = await resolve_memory(memory_key)
    session = FlyteSession(store) if store is not None else None

    try:
        result = await Runner.run(agent, input, max_turns=max_turns, run_config=config, session=session)
    finally:
        if observability:
            await flush_report()

    final = result.final_output
    return final if isinstance(final, str) else str(final)
