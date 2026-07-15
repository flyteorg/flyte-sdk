"""``run_agent`` — run a LangChain agent on Flyte.

LangChain's agent framework owns the loop. ``run_agent`` runs that loop inside your
``@env.task``: it builds an agent with Flyte-task tools, drives the agent executor,
and returns the final answer. Each tool call runs as a durable Flyte child action
(its own container/resources, with retries and caching).

Observability: the run timeline — tool calls and AI message turns — is rendered into
the Flyte task report.

The adapter minimizes delta between native LangChain code and Flyte integration by
exposing tools that are drop-in replacements for LangChain ``BaseTool`` instances.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import ReportTimeline, flush_report

from ._tools import _coerce_tool

if typing.TYPE_CHECKING:
    from langchain_openai import ChatOpenAI as _ChatOpenAI

    from langchain.agents import (
        AgentExecutor as _AgentExecutor,
    )
    from langchain.agents import (
        create_tool_calling_agent as _create_tool_calling_agent,
    )
else:
    _AgentExecutor = None
    _create_tool_calling_agent = None
    _ChatOpenAI = None

# Module-level aliases for test monkeypatching.
ChatOpenAI = _ChatOpenAI


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

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a pre-built ``agent`` or ``tools`` + ``model`` to have one
    built for you.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates.
        model: A LangChain-compatible chat model (e.g. ``ChatOpenAI()``). When
            ``agent`` is not given, one is built using this model.
        agent: A pre-built LangChain agent executor. Mutually exclusive with ``tools``.
        name: Agent name (for debugging/observability).
        durable: Record/replay each model turn via ``flyte.trace``.
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory.
            When set, conversation history is persisted to a keyed ``MemoryStore``
            and resumed on a later run with the same key.
        **agent_kwargs: Additional kwargs forwarded to the agent builder.

    Returns:
        The agent's final output as a string.
    """
    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("LangChain agent")

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    # Build the agent if not provided
    if agent is None:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        if _AgentExecutor is None:
            from langchain.agents import (
                AgentExecutor as _LangChainAgentExecutor,
            )
            from langchain.agents import (
                create_tool_calling_agent as _create_tool_calling_agent,
            )
        else:
            _LangChainAgentExecutor = _AgentExecutor
            _create_tool_calling_agent = _AgentExecutor

        if model is None:
            if _ChatOpenAI is None:
                from langchain_openai import ChatOpenAI as _LangChainChatOpenAI
            else:
                _LangChainChatOpenAI = _ChatOpenAI
            model = _LangChainChatOpenAI()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are a helpful assistant named {name}."),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        tool_objs = [_coerce_tool(t) for t in tools]
        agent_creator = _create_tool_calling_agent(model, tool_objs, prompt)
        agent = _LangChainAgentExecutor(
            agent=agent_creator,
            tools=tool_objs,
            name=name,
            handle_parsing_errors=True,
            max_iterations=agent_kwargs.pop("max_iterations", 10),
            **agent_kwargs,
        )

    # Run the agent
    from langchain_core.messages import HumanMessage

    _ = [HumanMessage(content=input)]  # used for observability tracking
    result = await agent.ainvoke(
        {"input": input, "chat_history": []},
    )

    # Extract final text
    final = result.get("output", "") if isinstance(result, dict) else str(result)

    if observability:
        await flush_report()

    return final or ""
