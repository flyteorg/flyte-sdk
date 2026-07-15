"""``FlyteAgent`` — build a LangChain agent with Flyte durability.

Users who want to construct their own LangChain agent (custom prompts,
multiple models, chains, etc.) but still want durable tool execution,
cross-run memory, and observability can use :class:`FlyteAgent`:

.. code-block:: python

    from flyte import TaskEnvironment, task
    from flyteplugins.agents.langchain import FlyteAgent
    from langchain_openai import ChatOpenAI

    env = TaskEnvironment("lc_agent_a")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    agent = FlyteAgent(name="weather-agent")
    tools = agent.durable_tools(get_weather)  # durable LangChain tools
    built = agent.build(tools=tools, model=ChatOpenAI())  # returns AgentExecutor
    result = await built.ainvoke({"input": "What's the weather?"})

The same Flyte task can be used inside the agent without ``run_agent`` — each
tool call is a durable Flyte child action.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from langchain_openai import ChatOpenAI as _ChatOpenAI

    from langchain.agents import AgentExecutor as _AgentExecutor
else:
    _AgentExecutor = None
    _ChatOpenAI = None


class FlyteAgent:
    """Build a LangChain ``AgentExecutor`` with Flyte durable tools, memory, and observability.

    Args:
        model: A LangChain-compatible chat model (e.g. ``ChatOpenAI()``).
            When ``None``, defaults to ``ChatOpenAI()``.
        name: Agent name for debugging/observability.
        instructions: System instructions for the agent. Defaults to a generic prompt.
        **kwargs: Additional kwargs forwarded to ``AgentExecutor.__init__``.

    Example:

    .. code-block:: python

        from flyte import TaskEnvironment, task
        from flyteplugins.agents.langchain import FlyteAgent
        from langchain_openai import ChatOpenAI

        env = TaskEnvironment("my_env")

        @env.task
        async def get_weather(city: str) -> str:
            return f"sunny in {city}"

        agent = FlyteAgent(name="weather-agent")
        tools = agent.durable_tools(get_weather)
        built = agent.build(tools=tools, model=ChatOpenAI())
        result = await built.ainvoke({"input": "What's the weather?"})
    """

    def __init__(
        self,
        model: typing.Any | None = None,
        *,
        name: str = "langchain-agent",
        instructions: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        self._model = model
        self._name = name
        self._instructions = instructions
        self._extra_kwargs = kwargs

    @property
    def model(self) -> typing.Any:
        """The model configuration."""
        if self._model is None:
            if _ChatOpenAI is None:
                from langchain_openai import ChatOpenAI as _LangChainChatOpenAI

                return _LangChainChatOpenAI()
            return _ChatOpenAI()
        return self._model

    @property
    def name(self) -> str:
        """The agent name."""
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
        """Convert Flyte tasks into durable LangChain tools.

        Each task is wrapped so that when the agent calls the tool, it runs as
        a durable Flyte child action (its own container/resources, with retries
        and caching) rather than inline in the agent's process.

        Args:
            *tasks: Flyte ``@env.task`` decorated functions.
            names: Optional mapping of task name → tool name override.
            descriptions: Optional mapping of task name → description override.

        Returns:
            A list of LangChain ``StructuredTool`` instances ready to pass to
            ``AgentExecutor`` or ``create_tool_calling_agent``.

        Example:

        .. code-block:: python

            agent = FlyteAgent()
            tools = agent.durable_tools(get_weather, get_news)
            built = agent.build(tools=tools, model=ChatOpenAI())
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
        model: typing.Any | None = None,
        agent: typing.Any = None,
    ) -> typing.Any:
        """Build a LangChain ``AgentExecutor`` with durable tools.

        Args:
            tools: Durable LangChain tools (from :meth:`durable_tools`) or
                bare Flyte tasks. If ``None``, the agent is built without tools.
            model: A LangChain-compatible chat model. Defaults to ``ChatOpenAI()``.
            agent: A pre-built ``AgentExecutor``. If provided, ``tools`` and
                ``model`` are ignored and this agent is returned as-is.

        Returns:
            A ``AgentExecutor`` instance ready to call ``.ainvoke()`` on.

        Example:

        .. code-block:: python

            agent = FlyteAgent()
            tools = agent.durable_tools(get_weather)
            built = agent.build(tools=tools, model=ChatOpenAI())
            result = await built.ainvoke({"input": "What's the weather?"})
        """
        if agent is not None:
            return agent

        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        try:
            from langchain.agents import AgentExecutor as _AgentExecutor
        except ImportError:
            _AgentExecutor = None

        try:
            from langchain.agents import create_tool_calling_agent as _create_tool_calling_agent
        except ImportError:
            _create_tool_calling_agent = None

        if model is None:
            if _ChatOpenAI is None:
                from langchain_openai import ChatOpenAI as _LangChainChatOpenAI

                model = _LangChainChatOpenAI()
            else:
                model = _ChatOpenAI()

        if tools:
            from ._tools import _coerce_tool

            tool_objs = [_coerce_tool(t) for t in tools]
        else:
            tool_objs = []

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.instructions),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        if _AgentExecutor is None:
            raise ImportError(
                "langchain 1.x removed AgentExecutor. Install langchain<1 or use langgraph as the agent backend."
            )

        if _create_tool_calling_agent is None:
            raise ImportError(
                "langchain 1.x removed create_tool_calling_agent. "
                "Install langchain<1 or use langgraph as the agent backend."
            )

        agent_creator = _create_tool_calling_agent(model, tool_objs, prompt)
        return _AgentExecutor(
            agent=agent_creator,
            tools=tool_objs,
            name=self._name,
            handle_parsing_errors=True,
            max_iterations=self._extra_kwargs.pop("max_iterations", 10),
            **self._extra_kwargs,
        )

    async def run(
        self,
        input: str,
        *,
        tools: typing.Sequence[typing.Any] | None = None,
        model: typing.Any | None = None,
        agent: typing.Any = None,
        observability: bool = True,
        **run_kwargs: typing.Any,
    ) -> str:
        """Build and run a LangChain agent in one call.

        Convenience method that combines :meth:`build` and ``agent.ainvoke()``.

        Args:
            input: The user prompt.
            tools: Durable tools or bare Flyte tasks.
            model: A LangChain-compatible chat model.
            agent: A pre-built ``AgentExecutor``.
            observability: Render the run timeline into the Flyte task report.
            **run_kwargs: Additional kwargs forwarded to ``agent.ainvoke()``.

        Returns:
            The agent's final output as a string.
        """
        built = self.build(tools=tools, model=model, agent=agent)

        result = await built.ainvoke(
            {"input": input, "chat_history": []},
            **run_kwargs,
        )
        final = result.get("output", "") if isinstance(result, dict) else str(result)
        return final or ""
