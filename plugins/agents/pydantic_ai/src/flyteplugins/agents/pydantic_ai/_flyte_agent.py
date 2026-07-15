"""``FlyteAgent`` — build a Pydantic AI agent with Flyte durability.

Users who want to construct their own ``pydantic_ai.Agent`` (custom prompts,
multiple models, hand-offs, etc.) but still want durable tool execution,
cross-run memory, and observability can use :class:`FlyteAgent`:

.. code-block:: python

    from flyte import TaskEnvironment, task
    from flyteplugins.agents.pydantic_ai import FlyteAgent

    env = TaskEnvironment("my_env")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    agent = FlyteAgent("gpt-4o", name="weather-agent")
    tools = agent.durable_tools(get_weather)  # durable Pydantic AI tools
    built = agent.build(tools=tools)           # returns pydantic_ai.Agent
    result = await built.run("What's the weather in Seattle?")

The same Flyte task can be used inside the agent without ``run_agent`` — each
tool call is a durable Flyte child action.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pydantic_ai import Agent as _PydanticAgent
else:
    _PydanticAgent = None


class FlyteAgent:
    """Build a Pydantic AI ``Agent`` with Flyte durable tools, memory, and observability.

    Args:
        model: Model name (e.g. ``"gpt-4o"``) or a model provider instance.
            When ``None``, defaults to ``"gpt-4o"``.
        name: Agent name for debugging/observability.
        system_prompt: System prompt for the agent. Defaults to a generic prompt.
        **kwargs: Additional kwargs forwarded to ``pydantic_ai.Agent.__init__``.

    Example:

    .. code-block:: python

        from flyte import TaskEnvironment, task
        from flyteplugins.agents.pydantic_ai import FlyteAgent

        env = TaskEnvironment("my_env")

        @env.task
        async def get_weather(city: str) -> str:
            return f"sunny in {city}"

        agent = FlyteAgent("gpt-4o", name="weather-agent")
        tools = agent.durable_tools(get_weather)
        built = agent.build(tools=tools)
        result = await built.run("What's the weather?")
    """

    def __init__(
        self,
        model: str | typing.Any | None = None,
        *,
        name: str = "pydantic-ai-agent",
        system_prompt: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        self._model = model
        self._name = name
        self._system_prompt = system_prompt
        self._extra_kwargs = kwargs

    @property
    def model(self) -> str | typing.Any:
        """The model configuration."""
        if self._model is None:
            return "openai:gpt-4o"
        return self._model

    @property
    def name(self) -> str:
        """The agent name."""
        return self._name

    @property
    def system_prompt(self) -> str:
        """The system prompt."""
        if self._system_prompt is not None:
            return self._system_prompt
        return f"You are a helpful assistant named {self._name}."

    def durable_tools(
        self,
        *tasks: typing.Any,
        names: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
    ) -> list[typing.Any]:
        """Convert Flyte tasks into durable Pydantic AI tools.

        Each task is wrapped so that when the agent calls the tool, it runs as
        a durable Flyte child action (its own container/resources, with retries
        and caching) rather than inline in the agent's process.

        Args:
            *tasks: Flyte ``@env.task`` decorated functions.
            names: Optional mapping of task name → tool name override.
            descriptions: Optional mapping of task name → description override.

        Returns:
            A list of Pydantic AI ``Tool`` instances (or callables that pass
            ``inspect.isfunction()`` checks) ready to pass to ``Agent.run()``.

        Example:

        .. code-block:: python

            agent = FlyteAgent("gpt-4o")
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
        """Build a Pydantic AI ``Agent`` with durable tools.

        Args:
            tools: Durable Pydantic AI tools (from :meth:`durable_tools`) or
                bare Flyte tasks. If ``None``, the agent is built without tools.
            agent: A pre-built ``pydantic_ai.Agent``. If provided, ``tools``
                is ignored and this agent is returned as-is.

        Returns:
            A ``pydantic_ai.Agent`` instance ready to call ``.run()`` on.

        Example:

        .. code-block:: python

            agent = FlyteAgent("gpt-4o")
            tools = agent.durable_tools(get_weather)
            built = agent.build(tools=tools)
            result = await built.run("What's the weather?")
        """
        if agent is not None:
            return agent

        if _PydanticAgent is None:
            from pydantic_ai import Agent as _LocalAgent
        else:
            _LocalAgent = _PydanticAgent

        return _LocalAgent(
            model=self.model,
            name=self._name,
            system_prompt=self.system_prompt,
            **self._extra_kwargs,
        )

    async def run(
        self,
        input: str,
        *,
        tools: typing.Sequence[typing.Any] | None = None,
        agent: typing.Any = None,
        observability: bool = True,
        **run_kwargs: typing.Any,
    ) -> str:
        """Build and run a Pydantic AI agent in one call.

        Convenience method that combines :meth:`build` and ``agent.run()``.

        Args:
            input: The user prompt.
            tools: Durable tools or bare Flyte tasks.
            agent: A pre-built ``pydantic_ai.Agent``.
            observability: Render the run timeline into the Flyte task report.
            **run_kwargs: Additional kwargs forwarded to ``agent.run()``.

        Returns:
            The agent's final output as a string.
        """
        built = self.build(tools=tools, agent=agent)
        from ._tools import _coerce_tool

        if tools:
            coerced_tools = [_coerce_tool(t) for t in tools]
        else:
            coerced_tools = []

        result = await built.run(input, tools=coerced_tools, **run_kwargs)
        return str(result.data) if result.data is not None else ""
