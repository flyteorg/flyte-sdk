"""``run_agent`` — run a Hermes (``hermes-agent``) agent on Flyte.

Hermes owns the agent loop: its ``AIAgent`` (the ``run_agent`` top-level module
of the ``hermes-agent`` package) drives the model and dispatches tools from a
process-global registry. ``run_agent`` runs that loop inside your ``@env.task``:
it builds an ``AIAgent`` scoped to exactly the Flyte-task tools you pass (via a
custom Hermes toolset), drives ``AIAgent.run_conversation`` (a blocking call,
bridged off the event loop with ``asyncio.to_thread``), and returns the final
answer. Each tool call runs as a durable Flyte child action (its own
container/resources, with retries and caching).

Observability: the run timeline is rendered into the Flyte task report.

The adapter minimizes delta between native Hermes code and Flyte integration:
bring your own pre-configured ``AIAgent`` (with ``enabled_toolsets`` including
:data:`~flyteplugins.agents.hermes.FLYTE_TOOLSET`) or let ``run_agent`` build
one from ``tools`` + ``model``.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import typing

from flyteplugins.agents.core import ReportTimeline, flush_report, sync_variant

from ._memory import load_transcript, resolve_memory, save_transcript
from ._tools import tool

if typing.TYPE_CHECKING:
    from run_agent import AIAgent as _AIAgent  # hermes-agent's top-level module
else:
    _AIAgent = None

_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Route anything not already Hermes-registered through :func:`tool`."""
    if getattr(t, "__hermes_registered__", False):
        return t
    return tool(t)


def _scoped_toolset(agent_name: str, registered: typing.Sequence[typing.Any]) -> str:
    """Create (or refresh) a Hermes toolset holding exactly this agent's tools.

    :func:`tool` registers every tool under the shared ``FLYTE_TOOLSET``; scoping
    each built agent to a named subset keeps two agents in one process from
    seeing each other's tools.
    """
    from toolsets import create_custom_toolset  # hermes-agent

    toolset = f"flyte-{re.sub(r'[^A-Za-z0-9_-]+', '-', agent_name).strip('-') or 'agent'}"
    names = [getattr(t, "__name__", str(t)) for t in registered]
    create_custom_toolset(toolset, f"Flyte tools for agent {agent_name!r}", tools=names)
    return toolset


async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: str | None = None,
    instructions: str | None = None,
    agent: typing.Any = None,
    name: str = "hermes-agent",
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    **agent_kwargs: typing.Any,
) -> str:
    """Run a Hermes agent with the given tools and prompt; return the final text.

    Await this from an async task as ``await run_agent(...)``; from a sync task
    use :func:`run_agent_sync` instead.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    Within it, each tool call runs as a durable Flyte child action. Give the
    enclosing task ``retries=...`` for self-healing and ``report=True`` to see
    the agent timeline.

    Provide either a pre-built ``agent`` (an ``AIAgent`` with its own
    ``enabled_toolsets``) or ``tools`` + ``model`` to have one built for you.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates.
        agent: A pre-built Hermes ``AIAgent``. Mutually exclusive with ``tools``.
        model: Model name for the built agent. Required when ``agent`` is not
            given (there is no default model).
        instructions: System prompt. On the builder path this becomes the
            agent's ``ephemeral_system_prompt``; with a pre-built agent it is
            passed as this run's ``system_message``.
        name: Agent name (used for the scoped toolset and observability).
        durable: Accepted for the shared adapter contract, but currently a
            no-op for Hermes: ``hermes-agent`` exposes no per-model-turn hook
            (the model client is buried inside ``AIAgent``), so completed model
            turns cannot be recorded/replayed via ``flyte.trace`` the way the
            openai/langchain adapters do. Tool calls are durable regardless —
            each runs as a Flyte child action with retries and caching — so a
            retried task still self-heals at tool granularity.
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory.
            When set, conversation history is persisted to a keyed ``MemoryStore``
            and resumed on a later run with the same key (passed to Hermes as
            ``conversation_history``).
        **agent_kwargs: Extra keyword arguments for the built ``AIAgent``
            (e.g. ``api_key=``, ``base_url=``, ``provider=``,
            ``max_iterations=``). Only valid on the builder path. When none of
            ``api_key``/``base_url``/``provider`` are given and
            ``OPENAI_API_KEY`` is set, the built agent is pointed at OpenAI
            with that key (Hermes otherwise only reads credentials from its own
            ``hermes setup`` config, which a fresh container doesn't have).

    Returns:
        The agent's final output as a string.
    """
    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("Hermes agent")

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")
    if agent is not None and agent_kwargs:
        raise ValueError("`**agent_kwargs` configure the built agent; don't pass them with a pre-built `agent=`.")

    # Cross-run memory: load the prior transcript (if any) and resume the thread.
    store = await resolve_memory(memory_key)
    history = await load_transcript(store)

    system_message: str | None = None
    if agent is None:
        if model is None:
            raise ValueError("Provide `model=` when building the agent (or pass a pre-built `agent=`).")

        if _AIAgent is None:
            from run_agent import AIAgent as _HermesAgent  # hermes-agent's top-level module
        else:
            _HermesAgent = _AIAgent

        enabled_toolsets: list[str] = []
        if tools:
            registered = [_coerce_tool(t) for t in tools]
            enabled_toolsets = [_scoped_toolset(name, registered)]

        has_credentials = any(k in agent_kwargs for k in ("api_key", "base_url", "provider"))
        if not has_credentials and os.environ.get("OPENAI_API_KEY"):
            agent_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
            agent_kwargs["base_url"] = _OPENAI_BASE_URL

        agent = _HermesAgent(
            model=model,
            ephemeral_system_prompt=instructions or f"You are a helpful assistant named {name}.",
            enabled_toolsets=enabled_toolsets,
            quiet_mode=True,
            **agent_kwargs,
        )
    else:
        # A pre-built agent keeps its own prompt; explicit instructions ride
        # along as this run's system message.
        system_message = instructions

    # Drive the agent. ``run_conversation`` is synchronous (Hermes bridges its
    # async tool handlers internally), so run it off the event loop;
    # ``to_thread`` propagates the Flyte task context into the worker thread.
    call_kwargs: dict[str, typing.Any] = {}
    if system_message:
        call_kwargs["system_message"] = system_message
    if history:
        call_kwargs["conversation_history"] = list(history)

    result = await asyncio.to_thread(lambda: agent.run_conversation(input, **call_kwargs))
    if inspect.isawaitable(result):  # tolerate async fakes/wrappers
        result = await result

    final = result.get("final_response") if isinstance(result, dict) else result
    final = str(final) if final is not None else ""

    # Persist the updated transcript for the next run with this memory_key.
    if store is not None:
        await save_transcript(
            store,
            [*history, {"role": "user", "content": input}, {"role": "assistant", "content": final}],
        )

    if observability:
        await flush_report()

    return final


run_agent_sync = sync_variant(run_agent)
