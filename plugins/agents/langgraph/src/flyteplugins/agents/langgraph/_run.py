"""``run_agent`` — drive a LangGraph graph on Flyte.

The intended devex is that *you* build the ``StateGraph`` (with
:func:`~flyteplugins.agents.langgraph.ai_node` /
:func:`~flyteplugins.agents.langgraph.tool_node`), compile it, and hand the
compiled graph to ``run_agent(agent=...)``. ``run_agent`` runs that graph inside
your ``@env.task``: each model turn is durable (replayed on retry) and each tool
call runs as a durable Flyte child action.

As a convenience, passing ``tools`` (instead of ``agent``) builds a default
tool-calling graph for you from the same ``ai_node`` / ``tool_node`` building
blocks.

Observability: the run timeline — model turns and tool calls — is rendered into
the Flyte task report.
"""

from __future__ import annotations

import typing

from flyte.syncify import syncify
from flyteplugins.agents.core import ReportTimeline, flush_report

from ._nodes import ai_node, tool_node
from ._tools import _coerce_tool

try:  # langgraph is a hard dependency; guarded only so imports never hard-crash.
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import tools_condition
except Exception:  # pragma: no cover - langgraph missing
    END = START = MessagesState = StateGraph = tools_condition = None  # type: ignore[assignment]

# Module-level aliases so the builder path can be redirected in tests.
_StateGraph = StateGraph
_MessagesState = MessagesState
_START = START
_tools_condition = tools_condition


def _resolve_chat_model(model: typing.Any) -> typing.Any:
    """Return a LangChain chat model for the default-graph builder.

    A chat-model instance passes through unchanged. A ``provider:model`` string
    resolves via ``langchain.chat_models.init_chat_model`` (requires the
    ``langchain`` package). ``None`` is an error — the caller must choose a model.
    """
    if model is None:
        raise ValueError(
            "Provide `model=` when building the agent (or pass a pre-built `agent=`). "
            'For example: `model=ChatOpenAI(model="gpt-4o")`.'
        )
    if not isinstance(model, str):
        return model
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as e:
        raise ImportError(
            f"Resolving the model string {model!r} requires the `langchain` package. "
            "Pass a chat-model instance instead, or install `langchain` to use "
            "`provider:model` strings."
        ) from e

    return init_chat_model(model)


def _build_default_graph(
    *,
    model: typing.Any,
    tools: typing.Sequence[typing.Any],
    durable: bool,
    observability: bool,
) -> typing.Any:
    """Build the standard tool-calling graph from ``ai_node`` + ``tool_node``."""
    chat_model = _resolve_chat_model(model)
    builder = _StateGraph(_MessagesState)
    builder.add_node("ai", ai_node(chat_model, tools, name="ai", durable=durable, observability=observability))
    builder.add_node("tools", tool_node(tools, name="tools", observability=observability))
    builder.add_edge(_START, "ai")
    builder.add_conditional_edges("ai", _tools_condition)
    builder.add_edge("tools", "ai")
    return builder.compile()


def _final_text(result: typing.Any) -> str:
    """Extract the final assistant text from a graph's output state."""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
            if content is not None:
                return content if isinstance(content, str) else str(content)
        return ""
    return str(result) if result is not None else ""


@syncify
async def run_agent(
    input: str | typing.Any,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: typing.Any = None,
    instructions: str | None = None,
    agent: typing.Any = None,
    name: str = "langgraph-agent",
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    **run_kwargs: typing.Any,
) -> str:
    """Run a LangGraph graph and return the final text.

    ``run_agent`` is syncified: call it synchronously as ``run_agent(...)`` or
    asynchronously as ``await run_agent.aio(...)``.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each model turn is recorded via ``flyte.trace`` (replayed on
    retry) and each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a pre-built ``agent`` (a compiled ``StateGraph`` you built
    with :func:`ai_node` / :func:`tool_node`) or ``tools`` to have a default
    tool-calling graph built for you. The two are mutually exclusive.

    Args:
        input: The user prompt (a ``str``) or a full graph input state (a dict).
        tools: ``@tool``-wrapped tools or bare ``@env.task`` templates (used only
            when ``agent`` is not given).
        model: A LangChain chat-model instance (e.g. ``ChatOpenAI(model="gpt-4o")``)
            or a ``provider:model`` string (resolved via ``init_chat_model``, which
            requires the ``langchain`` package). Required when building the graph
            (i.e. when ``agent`` is not given).
        instructions: System prompt prepended to a built graph's messages.
        agent: A pre-built compiled LangGraph graph. Mutually exclusive with ``tools``.
        name: Graph name (for debugging/observability).
        durable: Record each model turn via ``flyte.trace`` (built graphs only).
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory. When
            set, the conversation transcript is persisted to a keyed ``MemoryStore``
            and resumed on a later run with the same key.
        **run_kwargs: Additional kwargs forwarded to the graph's ``ainvoke``.

    Returns:
        The graph's final assistant message as a string.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from ._memory import load_messages, resolve_memory, save_messages

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("LangGraph agent")

    # Cross-run memory: the prior transcript (if any) is prepended to the run's
    # messages, and the full transcript is persisted back afterwards.
    store = await resolve_memory(memory_key)
    prior = await load_messages(store)

    if agent is None:
        agent = _build_default_graph(
            model=model,
            tools=[_coerce_tool(t) for t in tools],
            durable=durable,
            observability=observability,
        )
        seed: list[typing.Any] = []
        # The system prompt is only needed once; on resumed runs it already lives
        # in the prior transcript.
        if instructions and not prior:
            seed.append(SystemMessage(content=instructions))
        seed.extend(prior)
        seed.append(HumanMessage(content=input))
        input_state: typing.Any = {"messages": seed}
    elif isinstance(input, str):
        input_state = {"messages": [*prior, HumanMessage(content=input)]}
    else:
        input_state = input or {}
        if prior:
            input_state = {**input_state, "messages": [*prior, *input_state.get("messages", [])]}

    try:
        result = await agent.ainvoke(input_state, **run_kwargs)
    finally:
        if observability:
            await flush_report()

    if store is not None and isinstance(result, dict) and result.get("messages"):
        await save_messages(store, result["messages"])

    return _final_text(result)
