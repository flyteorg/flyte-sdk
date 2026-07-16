"""LangGraph node factories that make Flyte the durable runtime under a graph.

The intended devex: you build the ``StateGraph`` yourself, and these two factories
provide the nodes that Flyte makes durable and observable.

- :func:`ai_node` — the model-calling node. It binds your ``@tool``-wrapped tasks
  to the chat model and runs one model turn. Each turn is recorded as a durable
  ``flyte.trace`` leaf (via :func:`~flyteplugins.agents.core.durable_step`), so a
  crash/retry replays the recorded response instead of re-calling (and re-billing)
  the model.
- :func:`tool_node` — the tool-executing node. It runs the tool calls the model
  emitted; each ``@tool``-wrapped task runs as a durable Flyte child action (its
  own container/resources, retries, caching).

Both render their turns into the Flyte task report. Wire them into a standard
tool-calling loop::

    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import tools_condition

    builder = StateGraph(MessagesState)
    builder.add_node("ai", ai_node(model, tools))
    builder.add_node("tools", tool_node(tools))
    builder.add_edge(START, "ai")
    builder.add_conditional_edges("ai", tools_condition)
    builder.add_edge("tools", "ai")
    graph = builder.compile()
"""

from __future__ import annotations

import json
import typing

from flyteplugins.agents.core import ReportTimeline, abbrev, durable_step, fingerprint

if typing.TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# A node is an (optionally async) callable ``state -> partial_state``.
Node = typing.Callable[[dict], typing.Awaitable[dict]]


def _message_text(message: typing.Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def ai_node(
    model: "BaseChatModel",
    tools: typing.Sequence[typing.Any],
    *,
    name: str = "ai",
    durable: bool = True,
    observability: bool = True,
) -> Node:
    """Build the model-calling node for a tool-calling graph.

    The returned node binds ``tools`` to ``model`` and runs a single model turn
    over ``state["messages"]``, appending the model's response. Pass
    ``@tool``-wrapped tasks (or any LangChain ``BaseTool``) as ``tools``.

    Args:
        model: A LangChain chat model (e.g. ``ChatOpenAI(model="gpt-4o")``).
        tools: The tools to expose to the model.
        name: Node label (used for the graph node and the trace/report entry).
        durable: Record each model turn via ``flyte.trace`` so retries replay it.
        observability: Render each model turn into the Flyte task report.

    Returns:
        An async node ``state -> {"messages": [ai_message]}``.
    """
    bound = model.bind_tools(list(tools))
    timeline = ReportTimeline() if observability else None

    async def _ai(state: dict) -> dict:
        from langchain_core.messages import message_to_dict, messages_from_dict, messages_to_dict

        messages = state["messages"]

        async def _call() -> typing.Any:
            return await bound.ainvoke(messages)

        if durable:
            key = fingerprint({"node": name, "messages": messages_to_dict(list(messages))})
            response = await durable_step(
                key,
                _call,
                name=f"{name}:model",
                dumps=lambda m: json.dumps(message_to_dict(m)),
                loads=lambda s: messages_from_dict([json.loads(s)])[0],
            )
        else:
            response = await _call()

        if timeline is not None:
            tool_calls = getattr(response, "tool_calls", None) or []
            if tool_calls:
                detail = "→ " + ", ".join(tc["name"] for tc in tool_calls)
            else:
                detail = abbrev(_message_text(response), 200)
            timeline.row(icon="🤖", label=name, meta="assistant", detail=detail)

        return {"messages": [response]}

    _ai.__name__ = name
    return _ai


def tool_node(
    tools: typing.Sequence[typing.Any],
    *,
    name: str = "tools",
    observability: bool = True,
) -> Node:
    """Build the tool-executing node for a tool-calling graph.

    The returned node reads the tool calls from the last message and runs each
    one, appending a ``ToolMessage`` per call. ``@tool``-wrapped tasks run as
    durable Flyte child actions; anything else runs as the tool defines.

    Args:
        tools: The tools available to execute (``@tool``-wrapped tasks or any
            LangChain ``BaseTool``).
        name: Node label (used for the report entry).
        observability: Render each tool call/result into the Flyte task report.

    Returns:
        An async node ``state -> {"messages": [tool_message, ...]}``.
    """
    registry = {getattr(t, "name", getattr(t, "__name__", "")): t for t in tools}
    timeline = ReportTimeline() if observability else None

    async def _tools(state: dict) -> dict:
        from langchain_core.messages import ToolMessage

        messages = state["messages"]
        last = messages[-1] if messages else None
        calls = getattr(last, "tool_calls", None) or []

        results: list[typing.Any] = []
        for call in calls:
            tool_name = call["name"]
            args = call.get("args", {}) or {}
            call_id = call.get("id", "")
            if timeline is not None:
                timeline.row(icon="🛠️", label=tool_name, meta="tool", detail=abbrev(str(args), 160))

            selected = registry.get(tool_name)
            if selected is None:
                output = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    output = await selected.ainvoke(args)
                except Exception as exc:  # surface tool errors back to the model
                    output = f"Error: {exc}"

            if timeline is not None:
                timeline.row(icon="🔧", label=tool_name, meta="tool result", detail=abbrev(str(output), 160))
            results.append(ToolMessage(content=str(output), tool_call_id=call_id, name=tool_name))

        return {"messages": results}

    _tools.__name__ = name
    return _tools
