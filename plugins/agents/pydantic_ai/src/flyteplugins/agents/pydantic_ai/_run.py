"""``run_agent`` — run a Pydantic AI agent on Flyte.

Pydantic AI owns the agent loop. ``run_agent`` runs that loop inside your
``@env.task``: it drives an ``Agent`` whose tools are Flyte tasks, and returns
the final answer. Each tool call runs as a durable Flyte child action (its own
container/resources, with retries and caching).

Provide either a pre-built ``agent`` (with its tools already attached via
``Agent(tools=[...])``) or ``tools`` + ``model`` to have one built for you.

Observability: the run timeline is rendered into the Flyte task report.
"""

from __future__ import annotations

import contextlib
import typing

from flyte._logging import logger
from flyteplugins.agents.core import ReportTimeline, flush_report

from ._memory import load_history, resolve_memory, save_history
from ._tools import _coerce_tool

if typing.TYPE_CHECKING:
    from pydantic_ai import Agent as _AgentType
else:
    _AgentType = None

# Module-level hook for test monkeypatching. When patched to a callable it is used
# to build the agent; otherwise the real ``pydantic_ai.Agent`` is imported lazily.
# Both names are kept so tests may patch either ``_Agent`` or ``Agent``.
_Agent = None
Agent = _AgentType


def _result_text(result: typing.Any) -> str:
    """Extract the final text from a Pydantic AI run result.

    Pydantic AI 2.x exposes the final answer on ``result.output``; older/stub
    results may only carry ``.data``. Prefer ``.output`` and fall back to ``.data``.
    """
    value = getattr(result, "output", None)
    if value is None:
        value = getattr(result, "data", None)
    return "" if value is None else str(value)


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

    Provide either a pre-built ``agent`` (with its tools already attached) or
    ``tools`` + ``model`` to have one built for you — not both.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates. Used only
            when no ``agent`` is passed; the built agent attaches them natively.
        agent: A pre-built Pydantic AI ``Agent`` (tools already attached).
            Mutually exclusive with ``tools``.
        model: Model name or provider for the built agent (default ``openai:gpt-4o``).
        instructions: System prompt / instructions for the built agent.
        name: Agent name (for debugging/observability).
        durable: Record/replay each model turn via ``flyte.trace``. On the builder
            path the inferred model is wrapped in ``FlyteModel``; on the prebuilt-
            agent path durability is applied via ``agent.override(model=...)`` when
            the agent's model can be obtained (best-effort otherwise).
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory. When
            set, prior conversation history is loaded from a durable, keyed
            ``MemoryStore`` and passed as ``message_history=``; after the run the
            full history is saved back, so a later run with the same key continues
            the conversation. Best-effort — a memory failure never breaks a run.
        **run_kwargs: Additional kwargs forwarded to ``agent.run`` (e.g. an explicit
            ``message_history=``, which takes precedence over loaded memory).

    Returns:
        The agent's final output as a string.
    """
    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("Pydantic AI agent")

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    # ``override_cm`` swaps the agent's model for a durable wrapper for the run
    # (prebuilt-agent path). For the builder path we wrap the model before
    # constructing the Agent, so no override is needed.
    override_cm: typing.Any = contextlib.nullcontext()

    if agent is None:
        agent = _build_agent(
            tools=tools,
            model=model,
            instructions=instructions,
            name=name,
            durable=durable,
        )
    elif durable:
        override_cm = _durable_override(agent)

    # Cross-run memory: load prior history and seed ``message_history=`` (unless the
    # caller passed their own). Best-effort — a memory failure never breaks a run.
    store = await resolve_memory(memory_key)
    if store is not None and "message_history" not in run_kwargs:
        history = await load_history(store)
        if history:
            run_kwargs["message_history"] = history

    # Pydantic AI's ``Agent.run`` takes NO ``tools=`` kwarg — tools are attached at
    # construction. Drive the loop and pull the final text off ``result.output``.
    with override_cm:
        result = await agent.run(input, **run_kwargs)
    final = _result_text(result)

    # Persist the full history (prior + this run's new turns) back to memory.
    if store is not None:
        await save_history(store, result)

    if observability:
        await flush_report()

    return final


def _durable_override(agent: typing.Any) -> typing.Any:
    """Return a context manager that swaps the agent's model for a durable :class:`FlyteModel`.

    Uses ``Agent.override(model=...)`` so the swap is scoped to the run. Best-effort:
    if the agent's model can't be obtained/wrapped (e.g. a stub agent in tests, or a
    prebuilt agent without an accessible ``Model``), durability is skipped for the
    prebuilt-agent path rather than breaking the run.
    """
    try:
        from ._durable import FlyteModel

        inner = getattr(agent, "model", None)
        override = getattr(agent, "override", None)
        if inner is None or not callable(override):
            logger.warning("Could not apply model-turn durability to the prebuilt agent; running without it.")
            return contextlib.nullcontext()
        return override(model=FlyteModel(inner))
    except Exception:  # pragma: no cover - durability must never break a run
        logger.warning("Could not apply model-turn durability to the prebuilt agent; running without it.")
        return contextlib.nullcontext()


def _build_agent(
    *,
    tools: typing.Sequence[typing.Any],
    model: str | None,
    instructions: str | None,
    name: str,
    durable: bool = True,
) -> typing.Any:
    """Build a Pydantic AI ``Agent`` with the given Flyte-task tools attached natively.

    When ``durable`` (and using the real ``pydantic_ai.Agent``) the model string is
    resolved via ``infer_model`` and wrapped in :class:`FlyteModel` so each model
    turn records/replays via ``flyte.trace``. Best-effort: if the model can't be
    inferred/wrapped, falls back to the plain model string.
    """
    # ``_Agent`` / ``Agent`` may be monkeypatched (tests) to a builder callable;
    # otherwise import the real ``pydantic_ai.Agent`` lazily.
    agent_factory = _Agent or Agent
    is_real_agent = agent_factory is None
    if agent_factory is None:
        from pydantic_ai import Agent as agent_factory  # type: ignore[no-redef]

    coerced = [_coerce_tool(t) for t in tools]
    system_prompt = instructions or f"You are a helpful assistant named {name}."

    model_arg: typing.Any = model or "openai:gpt-4o"
    if durable and is_real_agent:
        model_arg = _durable_model(model_arg)

    return agent_factory(
        model_arg,
        name=name,
        system_prompt=system_prompt,
        tools=coerced,
    )


def _durable_model(model: str) -> typing.Any:
    """Wrap an inferred model in :class:`FlyteModel` for durable turns; fall back to the string.

    Best-effort: ``infer_model`` can raise (e.g. a missing provider API key at
    build time); if it does we return the original string so the run proceeds
    without model-turn durability rather than failing.
    """
    try:
        from pydantic_ai.models import infer_model

        from ._durable import FlyteModel

        return FlyteModel(infer_model(model))
    except Exception:  # pragma: no cover - durability must never break a run
        logger.warning("Could not apply model-turn durability to the built agent; running without it.")
        return model
