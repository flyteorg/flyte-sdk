"""``run_agent`` — run a LangChain agent on Flyte.

LangChain's agent framework owns the loop. ``run_agent`` runs that loop inside your
``@env.task``: it builds an agent (a compiled LangGraph graph) with Flyte-task tools,
drives it, and returns the final answer. Each tool call runs as a durable Flyte child
action (its own container/resources, with retries and caching).

In langchain 1.x the agent is built with ``langchain.agents.create_agent(model, tools,
system_prompt=...)``, which returns a compiled graph that is driven with a messages
state: ``await graph.ainvoke({"messages": [{"role": "user", "content": input}]})``, and
the final text is ``result["messages"][-1].content``.

Observability: the run timeline is rendered into the Flyte task report.

The adapter minimizes delta between native LangChain code and Flyte integration by
exposing tools that are drop-in ``BaseTool`` instances.
"""

from __future__ import annotations

import typing

from flyte.syncify import syncify
from flyteplugins.agents.core import ReportTimeline, flush_report

from ._durable import DurableChatModel
from ._memory import load_history, resolve_memory, save_history
from ._tools import _coerce_tool

# Module-level alias for test monkeypatching: ``_create_agent`` lets tests
# substitute the graph builder without constructing a real model or graph.
_create_agent = None


def _final_text(result: typing.Any) -> str:
    """Extract the agent's final text from a compiled-graph result.

    ``create_agent`` graphs return a messages state ``{"messages": [...]}``; the
    final answer is the content of the last message. Falls back gracefully for
    other shapes (e.g. a legacy ``{"output": ...}`` dict or a bare string).
    """
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            content = getattr(messages[-1], "content", messages[-1])
            return content if isinstance(content, str) else str(content)
        if "output" in result:
            return str(result["output"])
        return ""
    return str(result)


@syncify
async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: typing.Any = None,
    instructions: str | None = None,
    agent: typing.Any = None,
    name: str = "langchain-agent",
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    **agent_kwargs: typing.Any,
) -> str:
    """Run a LangChain agent with the given tools and prompt; return the final text.

    ``run_agent`` is syncified: call it synchronously as ``run_agent(...)`` or
    asynchronously as ``await run_agent.aio(...)``.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a pre-built ``agent`` (a compiled graph from ``create_agent``)
    or ``tools`` + ``model`` to have one built for you.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates.
        model: A LangChain-compatible chat model (e.g.
            ``ChatOpenAI(model="gpt-4o")``). Required when ``agent`` is not
            given — one is built using this model.
        instructions: System prompt for the built agent.
        agent: A pre-built LangChain agent (a compiled ``create_agent`` graph).
            Mutually exclusive with ``tools``.
        name: Agent name (for debugging/observability).
        durable: Record/replay each model turn via ``flyte.trace``. Applies when
            a model is being built (``tools`` + ``model``, or a caller-passed
            ``BaseChatModel``) — the model is wrapped in :class:`DurableChatModel`.
            A fully pre-built compiled ``agent`` cannot be rewrapped, so its model
            turns are not made durable (its tool calls remain durable regardless).
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory.
            When set, conversation history is persisted to a keyed ``MemoryStore``
            and resumed on a later run with the same key.
        **agent_kwargs: Additional kwargs forwarded to ``create_agent``.

    Returns:
        The agent's final output as a string.
    """
    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("LangChain agent")

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    # Build the agent (a compiled graph) if not provided.
    if agent is None:
        if _create_agent is None:
            from langchain.agents import create_agent as create_agent_fn
        else:
            create_agent_fn = _create_agent

        if model is None:
            raise ValueError(
                "Provide `model=` when building the agent (or pass a pre-built `agent=`). "
                'For example: `model=ChatOpenAI(model="gpt-4o")`.'
            )

        # Wrap the inner chat model so each model turn is durable/replayable. We
        # can only do this on the builder path (a fully pre-built compiled ``agent``
        # owns its own model, which we cannot reach to rewrap).
        if durable:
            model = _wrap_durable(model)

        tool_objs = [_coerce_tool(t) for t in tools]
        system_prompt = instructions or f"You are a helpful assistant named {name}."
        agent = create_agent_fn(model, tool_objs, system_prompt=system_prompt, **agent_kwargs)

    # Cross-run memory: load prior history and prepend it to the messages passed
    # to the graph, then persist the full transcript back after the run.
    store = await resolve_memory(memory_key)
    prior = await load_history(store)
    messages = [*prior, {"role": "user", "content": input}]

    result = await agent.ainvoke({"messages": messages})

    await save_history(store, _result_messages(result))

    final = _final_text(result)

    if observability:
        await flush_report()

    return final or ""


def _wrap_durable(model: typing.Any) -> typing.Any:
    """Wrap a chat model in :class:`DurableChatModel` when possible.

    Best-effort: only ``BaseChatModel`` instances are wrappable; anything else
    (or any failure) is returned unchanged so durability never breaks a run.
    """
    try:
        from langchain_core.language_models.chat_models import BaseChatModel

        if isinstance(model, BaseChatModel) and not isinstance(model, DurableChatModel):
            return DurableChatModel(inner=model)
    except Exception:  # pragma: no cover - durability is best-effort, never fatal
        pass
    return model


def _result_messages(result: typing.Any) -> list[typing.Any]:
    """Extract the message list from a compiled-graph result (empty on other shapes)."""
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            return list(messages)
    return []
