"""``run_agent`` — run a Google ADK agent on Flyte using the SDK's own loop.

ADK's ``Runner`` owns the agent loop (it drives the model + tools and yields
``Event``s). ``run_agent`` runs that loop inside your ``@env.task``: it builds an
``LlmAgent`` with Flyte-task tools, drives ``Runner.run_async``, renders the events
into the Flyte report, and returns the final answer.

Durability via the seam below the loop: with ``durable=True`` the agent's model is
wrapped (:class:`FlyteLlm`) so each turn is recorded for replay. Cross-run memory
via ``memory_key``: the session transcript is persisted to a keyed ``MemoryStore``
and restored on the next run.

API keys are read from the environment (e.g. ``GOOGLE_API_KEY``) — wire them as
Flyte secrets.
"""

from __future__ import annotations

import typing
import uuid

from flyte._task import AsyncFunctionTaskTemplate
from flyteplugins.agents.core import ReportTimeline, abbrev, flush_report, tool

from ._durable import durable_model
from ._memory import load_memory, save_memory


def _coerce_tool(t: typing.Any) -> typing.Any:
    return tool(t) if isinstance(t, AsyncFunctionTaskTemplate) else t


def _content_text(content: typing.Any) -> str:
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


def _render(timeline: ReportTimeline, event: typing.Any) -> None:
    content = getattr(event, "content", None)
    for part in getattr(content, "parts", None) or []:
        call = getattr(part, "function_call", None)
        resp = getattr(part, "function_response", None)
        text = getattr(part, "text", None)
        if call is not None:
            timeline.row(
                icon="🛠️",
                label=getattr(call, "name", ""),
                meta="tool",
                detail=abbrev(getattr(call, "args", ""), 160),
            )
        elif resp is not None:
            timeline.row(
                icon="🔧",
                label=getattr(resp, "name", ""),
                meta="tool result",
                detail=abbrev(getattr(resp, "response", ""), 160),
            )
        elif text and text.strip():
            timeline.row(icon="💬", label="assistant", detail=abbrev(text, 200))


def _run_config(max_llm_calls: int | None) -> typing.Any:
    """Build an ADK ``RunConfig`` that caps model calls; ``None`` → ADK's default (500)."""
    if max_llm_calls is None:
        return None
    from google.adk.agents.run_config import RunConfig

    return RunConfig(max_llm_calls=max_llm_calls)


class _UsageSink:
    """Tally model-turn count + token usage across ADK's event stream.

    Each model-response ``Event`` carries genai ``usage_metadata`` (tool-result events
    don't), so events with usage = model turns and we sum their token counts. Gemini's
    ``cached_content_token_count`` is surfaced as ``cached`` (its context cache, like
    Claude's cache-read tokens), ``thoughts_token_count`` as ``thinking`` for those models.
    """

    def __init__(self) -> None:
        self.turns = self.prompt = self.completion = self.total = self.cached = self.thinking = 0

    def add(self, event: typing.Any) -> None:
        um = getattr(event, "usage_metadata", None)
        if um is None:
            return
        self.turns += 1
        self.prompt += getattr(um, "prompt_token_count", 0) or 0
        self.completion += getattr(um, "candidates_token_count", 0) or 0
        self.total += getattr(um, "total_token_count", 0) or 0
        self.cached += getattr(um, "cached_content_token_count", 0) or 0
        self.thinking += getattr(um, "thoughts_token_count", 0) or 0

    def detail(self) -> str:
        out = (
            f"{self.turns} model turns · {self.prompt} prompt · "
            f"{self.completion} completion · {self.total} total tokens"
        )
        if self.thinking:
            out += f" · {self.thinking} thinking"
        if self.cached:
            out += f" · {self.cached} cached"
        return out


async def run_agent(
    input: str,
    *,
    agent: typing.Any = None,
    tools: typing.Sequence[typing.Any] = (),
    model: str = "gemini-2.0-flash",
    instructions: str | None = None,
    name: str = "assistant",
    max_llm_calls: int | None = None,
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    app_name: str = "flyte-agent",
    user_id: str = "flyte-user",
) -> str:
    """Run a Google ADK agent with the given tools and prompt; return the final text.

    Call this from inside an ``@env.task`` — that task is the durable parent, and each
    tool the agent calls runs as a durable Flyte child action. Provide either a
    pre-built ``agent`` (an ADK ``LlmAgent``/``BaseAgent``) or ``tools`` + ``model`` +
    ``instructions`` to have one built.

    Args:
        input: The user prompt.
        agent: A pre-built ADK agent. Mutually exclusive with ``tools``.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates.
        model: Model name for the built agent (e.g. ``gemini-2.0-flash``).
        instructions: System instruction for the built agent.
        name: Agent name (a valid Python identifier). ADK injects this into the system
            prompt as the model's "internal name", so it can surface in replies — keep it
            natural (defaults to ``"assistant"``; avoid a brand-y/internal label).
        max_llm_calls: Cap on model (LLM) calls before ADK raises
            ``LlmCallsLimitExceededError`` (its runaway-loop guard, via
            ``RunConfig.max_llm_calls``); ``None`` uses ADK's default of 500. Counts LLM
            calls, not conversational turns (a tool round is ~2 calls). For a wall-clock
            bound on the whole run, set ``timeout=`` on the enclosing ``@env.task``.
        durable: Wrap the model so each turn is recorded/replayed via ``flyte.trace``.
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (user/thread) for cross-run memory. When set, the session
            transcript is persisted and restored so a later run continues the conversation.
        app_name: ADK app name (namespacing).
        user_id: ADK user id.
    """
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.genai import types

    if agent is not None and tools:
        raise ValueError("Pass either `agent` (with its own tools) or `tools`, not both.")

    if agent is None:
        agent = LlmAgent(
            name=name,
            model=durable_model(model) if durable else model,
            instruction=instructions or "You are a helpful assistant.",
            tools=[_coerce_tool(t) for t in tools],
        )

    session_service = InMemorySessionService()
    store, prior_events = await load_memory(memory_key)
    session_id = memory_key or uuid.uuid4().hex
    session = await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    for event in prior_events:
        await session_service.append_event(session, event)

    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    timeline = ReportTimeline() if observability else None
    usage = _UsageSink() if observability else None
    if timeline is not None:
        timeline.heading("Google ADK agent")

    final = ""
    message = types.Content(role="user", parts=[types.Part.from_text(text=input)])
    run_config = _run_config(max_llm_calls)
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message, run_config=run_config
    ):
        if timeline is not None:
            _render(timeline, event)
        if usage is not None:
            usage.add(event)
        if event.is_final_response() and event.content is not None:
            final = _content_text(event.content) or final

    if memory_key:
        latest = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        await save_memory(store, getattr(latest, "events", session.events))

    if timeline is not None and usage is not None and usage.turns:
        timeline.row(icon="📊", label="usage", meta="model", detail=usage.detail())
    if observability:
        await flush_report()
    return final
