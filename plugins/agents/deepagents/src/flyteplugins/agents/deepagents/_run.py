"""``run_agent`` — run a LangChain Deep Agent on Flyte.

Deep Agents (LangChain's agent harness) owns the loop: ``create_deep_agent``
returns a compiled LangGraph graph with built-in planning (todos), a virtual
filesystem, and subagents. ``run_agent`` runs that loop inside your
``@env.task``: it builds a deep agent with Flyte-task tools, drives it, and
returns the final answer. Each tool call runs as a durable Flyte child action
(its own container/resources, with retries and caching).

The graph is driven with a messages state: ``await graph.ainvoke({"messages":
[{"role": "user", "content": input}]})``, and the final text is
``result["messages"][-1].content``. The result state also carries ``files`` —
the agent's virtual filesystem — which ``memory_key`` persists across runs
alongside the conversation.

Observability: the run timeline is rendered into the Flyte task report.

The adapter minimizes delta between native Deep Agents code and Flyte
integration by exposing tools that are drop-in ``BaseTool`` instances.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import ReportTimeline, flush_report, sync_variant

from ._durable import DurableChatModel
from ._memory import load_files, load_history, resolve_memory, save_state
from ._tools import _coerce_tool

# Module-level alias for test monkeypatching: tests substitute the graph builder
# without importing a real provider model.
_create_deep_agent = None


def _resolve_chat_model(model: typing.Any) -> typing.Any:
    """Return a LangChain chat model. A model instance passes through; a
    ``provider:model`` string resolves via ``init_chat_model``. ``None`` is an
    error — the caller must choose a model."""
    if model is None:
        raise ValueError(
            "Provide `model=` when building the agent (or pass a pre-built `agent=`). "
            'For example: `model="anthropic:claude-sonnet-4-6"` or a chat-model instance.'
        )
    from langchain_core.language_models.chat_models import BaseChatModel

    if isinstance(model, BaseChatModel):
        return model
    from langchain.chat_models import init_chat_model

    return init_chat_model(model)


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


def _final_text(result: typing.Any) -> str:
    """Extract the agent's final text from a compiled-graph result.

    Deep agent graphs return a messages state ``{"messages": [...], "files":
    {...}}``; the final answer is the content of the last message. Falls back
    gracefully for other shapes.
    """
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            content = getattr(messages[-1], "content", messages[-1])
            return content if isinstance(content, str) else str(content)
        return ""
    return str(result)


def _result_messages(result: typing.Any) -> list[typing.Any]:
    """Extract the message list from a compiled-graph result (empty on other shapes)."""
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            return list(messages)
    return []


async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: typing.Any = None,
    instructions: str | None = None,
    agent: typing.Any = None,
    name: str = "deep-agent",
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    **agent_kwargs: typing.Any,
) -> str:
    """Run a Deep Agent with the given tools and prompt; return the final text.

    Await this from an async task as ``await run_agent(...)``; from a sync task
    use :func:`run_agent_sync` instead.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a pre-built ``agent`` (a compiled graph from
    ``create_deep_agent``) or ``tools`` + ``model`` to have one built for you.
    Deep-Agents-specific options — ``subagents=``, ``skills=``, ``backend=``,
    ``interrupt_on=``, … — pass through ``**agent_kwargs`` on the builder path.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates.
        model: A LangChain chat model instance or a ``provider:model`` string
            (e.g. ``"anthropic:claude-sonnet-4-6"``). Required when ``agent``
            is not given.
        instructions: System prompt for the built agent.
        agent: A pre-built deep agent (a compiled ``create_deep_agent`` graph).
            Mutually exclusive with ``tools``. To get durable model turns on this
            path, build it with ``create_deep_agent(model=DurableChatModel(inner=...))``.
        name: Agent name (for debugging/observability).
        durable: Record/replay each model turn via ``flyte.trace``. Applies when
            the agent is being built — the resolved model is wrapped in
            :class:`DurableChatModel`. A fully pre-built compiled ``agent`` cannot
            be rewrapped (wrap its model yourself, see above); its tool calls
            remain durable regardless.
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory.
            When set, the conversation *and* the agent's virtual filesystem are
            persisted to a keyed ``MemoryStore`` and resumed on a later run with
            the same key.
        **agent_kwargs: Additional kwargs forwarded to ``create_deep_agent``
            (``subagents=``, ``skills=``, ``backend=``, ...).

    Returns:
        The agent's final output as a string.
    """
    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("Deep agent")

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    # Build the agent (a compiled graph) if not provided.
    if agent is None:
        if _create_deep_agent is None:
            from deepagents import create_deep_agent as create_deep_agent_fn
        else:
            create_deep_agent_fn = _create_deep_agent

        chat_model = _resolve_chat_model(model)
        # Wrap the chat model so each model turn is durable/replayable. We can
        # only do this on the builder path (a fully pre-built compiled ``agent``
        # owns its own model, which we cannot reach to rewrap).
        if durable:
            chat_model = _wrap_durable(chat_model)

        tool_objs = [_coerce_tool(t) for t in tools]
        system_prompt = instructions or f"You are a helpful assistant named {name}."
        agent = create_deep_agent_fn(
            model=chat_model,
            tools=tool_objs,
            system_prompt=system_prompt,
            **agent_kwargs,
        )

    # Cross-run memory: load the prior conversation and virtual filesystem and
    # seed the run with both, then persist the updated state back after.
    store = await resolve_memory(memory_key)
    prior = await load_history(store)
    files = await load_files(store)

    state: dict[str, typing.Any] = {"messages": [*prior, {"role": "user", "content": input}]}
    if files:
        state["files"] = files

    result = await agent.ainvoke(state)

    await save_state(
        store,
        _result_messages(result),
        files=result.get("files") if isinstance(result, dict) else None,
    )

    final = _final_text(result)

    if observability:
        await flush_report()

    return final or ""


run_agent_sync = sync_variant(run_agent)
