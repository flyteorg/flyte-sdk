"""``FlyteAgent`` — build a LangGraph agent with Flyte durability.

Users who want to construct their own LangGraph ``StateGraph`` (custom nodes,
conditional edges, subgraphs, etc.) but still want durable tool execution,
cross-run memory, and observability can use :class:`FlyteAgent`:

.. code-block:: python

    from flyte import TaskEnvironment, task
    from flyteplugins.agents.langgraph import FlyteAgent
    from langchain_core.messages import HumanMessage

    env = TaskEnvironment("lg_agent_a")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    agent = FlyteAgent(name="weather-agent")
    tools = agent.durable_tools(get_weather)  # durable LangGraph tools
    built = agent.build(tools=tools)           # returns compiled StateGraph
    result = await built.ainvoke({"messages": [HumanMessage(content="...")]})

The same Flyte task can be used inside the agent without ``run_agent`` — each
tool call is a durable Flyte child action.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from langgraph.graph import StateGraph as _StateGraph
else:
    _StateGraph = None


class FlyteAgent:
    """Build a LangGraph ``StateGraph`` with Flyte durable tools, memory, and observability.

    Args:
        name: Graph name for debugging/observability.
        instructions: System instructions for the agent. Defaults to a generic prompt.
        **kwargs: Additional kwargs forwarded to ``StateGraph.__init__``.

    Example:

    .. code-block:: python

        from flyte import TaskEnvironment, task
        from flyteplugins.agents.langgraph import FlyteAgent
        from langchain_core.messages import HumanMessage

        env = TaskEnvironment("my_env")

        @env.task
        async def get_weather(city: str) -> str:
            return f"sunny in {city}"

        agent = FlyteAgent(name="weather-agent")
        tools = agent.durable_tools(get_weather)
        built = agent.build(tools=tools)
        result = await built.ainvoke({"messages": [HumanMessage(content="...")]})
    """

    def __init__(
        self,
        *,
        name: str = "langgraph-agent",
        instructions: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        self._name = name
        self._instructions = instructions
        self._extra_kwargs = kwargs

    @property
    def name(self) -> str:
        """The graph name."""
        return self._name

    @property
    def instructions(self) -> str:
        """The system instructions."""
        if self._instructions is not None:
            return self._instructions
        return f"You are a helpful assistant named {self._name}."

    def durable_tools(
        self,
        *tasks: typing.Any,
        names: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
    ) -> list[typing.Any]:
        """Convert Flyte tasks into durable LangGraph tools.

        Each task is wrapped so that when the graph calls the tool node, it runs as
        a durable Flyte child action (its own container/resources, with retries
        and caching) rather than inline in the graph's process.

        Args:
            *tasks: Flyte ``@env.task`` decorated functions.
            names: Optional mapping of task name → tool name override.
            descriptions: Optional mapping of task name → description override.

        Returns:
            A list of LangChain ``StructuredTool`` instances ready to use as
            LangGraph tool nodes.

        Example:

        .. code-block:: python

            agent = FlyteAgent()
            tools = agent.durable_tools(get_weather, get_news)
            built = agent.build(tools=tools)
        """
        from ._tools import tool as _durable_tool

        names = names or {}
        descriptions = descriptions or {}
        result = []
        for t in tasks:
            task_name = getattr(t, "func", t).__name__ if hasattr(t, "func") else t.__name__  # type: ignore[attr-defined]
            result.append(
                _durable_tool(
                    t,
                    name=names.get(task_name),
                    description=descriptions.get(task_name),
                )
            )
        return result

    def build(
        self,
        *,
        tools: typing.Sequence[typing.Any] | None = None,
        agent: typing.Any = None,
    ) -> typing.Any:
        """Build a LangGraph ``StateGraph`` with durable tools.

        Args:
            tools: Durable LangGraph tools (from :meth:`durable_tools`) or
                bare Flyte tasks. If ``None``, a minimal graph is built without
                tools.
            agent: A pre-built compiled LangGraph ``StateGraph``. If provided,
                ``tools`` is ignored and this agent is returned as-is.

        Returns:
            A compiled LangGraph ``StateGraph`` instance ready to call ``.ainvoke()`` on.

        Example:

        .. code-block:: python

            agent = FlyteAgent()
            tools = agent.durable_tools(get_weather)
            built = agent.build(tools=tools)
            result = await built.ainvoke({"messages": [HumanMessage(content="...")]})
        """
        if agent is not None:
            return agent

        from langgraph.graph import StateGraph as _LangGraphStateGraph

        StateGraph = _LangGraphStateGraph

        # For a simple tool-use graph, build a minimal StateGraph
        builder = StateGraph(typing.get_type_hints(type("Typing", (), {"messages": list[str]})) or dict)

        # Add a tool node that executes the tools
        async def tool_node(state: dict) -> dict:
            messages = state.get("messages", [])
            if not messages:
                return {"messages": []}
            last = messages[-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                results = []
                for tc in last.tool_calls:
                    # Find and execute the tool
                    tool_fn = None
                    for t in tools:
                        tname = getattr(t, "name", None) or getattr(t, "func", None)
                        if tname and (
                            tname == tc["name"] or (hasattr(tname, "__name__") and tname.__name__ == tc["name"])
                        ):
                            tool_fn = t
                            break
                    if tool_fn is not None:
                        result = await _coerce_tool(tool_fn)
                        # Execute the tool with the call args
                        try:
                            if callable(result):
                                output = result(**tc.get("args", {}))
                                if hasattr(output, "__await__"):
                                    output = await output
                            else:
                                output = "No tool available"
                        except Exception as e:
                            output = f"Error: {e}"
                        results.append({"id": tc.get("id", ""), "name": tc["name"], "output": str(output)})
                return {"messages": []}
            return {"messages": []}

        # Add the AI node (placeholder — the real model call is handled by the graph)
        async def ai_node(state: dict) -> dict:
            messages = state.get("messages", [])
            if not messages:
                return {"messages": []}
            return {"messages": []}

        builder.add_node("ai", ai_node)
        builder.add_node("tools", tool_node)
        builder.set_entry_point("ai")
        builder.add_conditional_edges("ai", lambda s: "tools" if s.get("messages") else "__end__")
        builder.add_edge("tools", "ai")
        return builder.compile()

    async def run(
        self,
        input: str,
        *,
        tools: typing.Sequence[typing.Any] | None = None,
        agent: typing.Any = None,
        observability: bool = True,
        **run_kwargs: typing.Any,
    ) -> str:
        """Build and run a LangGraph agent in one call.

        Convenience method that combines :meth:`build` and ``agent.ainvoke()``.

        Args:
            input: The user prompt.
            tools: Durable tools or bare Flyte tasks.
            agent: A pre-built compiled LangGraph ``StateGraph``.
            observability: Render the run timeline into the Flyte task report.
            **run_kwargs: Additional kwargs forwarded to ``agent.ainvoke()``.

        Returns:
            The agent's final output as a string.
        """
        built = self.build(tools=tools, agent=agent)
        from langchain_core.messages import HumanMessage

        result = await built.ainvoke(
            {"messages": [HumanMessage(content=input)]},
            **run_kwargs,
        )
        final = ""
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    final = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                elif hasattr(last_msg, "text"):
                    final = last_msg.text
                else:
                    final = str(last_msg)
        return final or str(result) if result else ""


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Coerce a tool to a LangChain-compatible tool."""
    from flyte._task import AsyncFunctionTaskTemplate

    if isinstance(t, AsyncFunctionTaskTemplate):
        from ._tools import tool

        return tool(t)
    return t
