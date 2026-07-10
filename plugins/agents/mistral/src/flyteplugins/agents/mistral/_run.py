"""``run_agent`` — run a Mistral agent.

Durability via the seam below the loop: ``run_async`` makes each model turn by
calling ``self.start_async``/``self.append_async`` on the conversations object.
When ``durable=True`` we wrap those two methods so each turn is recorded via
``durable_step`` (a ``flyte.trace`` leaf) — so a crash/retry replays completed
turns from their recorded ``ConversationResponse`` and completed tool calls are
cache hits.

The same seam also drives observability: the SDK's ``RunResult`` exposes no token
usage, but each turn's ``ConversationResponse`` does — so the wrapper tallies the
model-turn count + token usage and we render it as a summary row in the report. On a
retry, turns served from their durable record are counted as cached tokens so the row
shows they weren't re-billed (rather than passing a free replay off as fresh spend).

The API key is read from the environment — wire it as a Flyte secret.
"""

from __future__ import annotations

import os
import typing

from flyte._logging import logger
from flyte._task import AsyncFunctionTaskTemplate
from flyteplugins.agents.core import (
    ReportTimeline,
    abbrev,
    durable_step,
    fingerprint,
    flush_report,
    jsonable,
    resolve_memory,
    tool,
)

# Semantic fields of a start/append call used to key the durable turn.
_TURN_KEY_FIELDS = ("inputs", "model", "agent_id", "instructions", "tools", "conversation_id")

# Path-addressed memory slot holding a thread's server-side conversation id.
_MEMORY_CONV_PATH = "mistral/conversation_id"


def _coerce_tool(t: typing.Any) -> typing.Callable:
    return tool(t) if isinstance(t, AsyncFunctionTaskTemplate) else t


def _entry_type(entry: typing.Any) -> str | None:
    return getattr(entry, "type", None)


def _entry_text(entry: typing.Any) -> str:
    content = getattr(entry, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c if isinstance(c, str) else (getattr(c, "text", "") or "") for c in content)
    return getattr(content, "text", "") or ""


def _final_text(output_entries: list[typing.Any]) -> str:
    return "".join(_entry_text(e) for e in output_entries if _entry_type(e) == "message.output")


def _render(timeline: ReportTimeline, output_entries: list[typing.Any]) -> None:
    for entry in output_entries:
        kind = _entry_type(entry)
        if kind == "function.call":
            timeline.row(
                icon="🛠️",
                label=getattr(entry, "name", ""),
                meta="tool",
                detail=abbrev(getattr(entry, "arguments", ""), 160),
            )
        elif kind == "message.output":
            timeline.row(icon="💬", label="assistant", detail=abbrev(_entry_text(entry), 200))


class _UsageSink:
    """Tally model-turn count + token usage across the SDK's loop.

    ``RunResult`` surfaces no usage, but each model turn's ``ConversationResponse``
    carries ``.usage`` — and every turn flows through our start/append wrapper, so we
    tally it there. Tokens for a turn served from a durable record on a retry (no real
    model call) are counted as ``cached``, so the row shows they weren't re-billed.
    """

    def __init__(self) -> None:
        self.turns = self.prompt = self.completion = self.total = self.connector = self.cached = 0

    def add(self, response: typing.Any, *, cached: bool = False) -> None:
        self.turns += 1
        usage = getattr(response, "usage", None)
        turn_total = (getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
        if usage is not None:
            self.prompt += getattr(usage, "prompt_tokens", 0) or 0
            self.completion += getattr(usage, "completion_tokens", 0) or 0
            self.total += turn_total
            self.connector += getattr(usage, "connector_tokens", 0) or 0
        if cached:
            self.cached += turn_total

    def detail(self) -> str:
        out = (
            f"{self.turns} model turns · {self.prompt} prompt · "
            f"{self.completion} completion · {self.total} total tokens"
        )
        if self.connector:
            out += f" · {self.connector} connector"
        if self.cached:
            out += f" · {self.cached} cached"
        return out


def _install_turn_hooks(conversations: typing.Any, *, durable: bool, usage: _UsageSink | None) -> None:
    """Wrap ``run_async``'s internal ``start``/``append`` for durability + usage.

    ``run_async`` calls ``self.start_async``/``self.append_async`` for each model
    turn, so shadowing those instance methods lets us (a) record/replay each turn via
    ``durable_step`` (the seam below the SDK's loop) when ``durable``, and (b) tally
    tokens from each turn's ``ConversationResponse`` when ``usage`` is given — neither
    of which the SDK's ``RunResult`` exposes. The response round-trips through pydantic
    JSON (verified faithful, incl. the polymorphic outputs).
    """

    from mistralai.client.models import ConversationResponse

    original = {"start": conversations.start_async, "append": conversations.append_async}

    def _wrap(phase: str):
        orig = original[phase]

        async def _turn(**kwargs: typing.Any) -> typing.Any:
            cached = False
            if durable:
                # ``durable_step`` invokes ``run`` only on a fresh execution; on a retry
                # it returns the recorded result without calling it. Flip a flag inside
                # the closure so we know per-turn whether the model was actually called
                # (fresh) or the turn came from its record (counted as cached, not re-billed).
                fired = False

                async def _do() -> typing.Any:
                    nonlocal fired
                    fired = True
                    return await orig(**kwargs)

                key = fingerprint(
                    {
                        "phase": phase,
                        **{k: jsonable(kwargs[k]) for k in _TURN_KEY_FIELDS if kwargs.get(k) is not None},
                    }
                )
                response = await durable_step(
                    key,
                    _do,
                    name=f"conversation_{phase}",
                    dumps=lambda r: r.model_dump_json(),
                    loads=ConversationResponse.model_validate_json,
                )
                cached = not fired
            else:
                response = await orig(**kwargs)
            if usage is not None:
                usage.add(response, cached=cached)
            return response

        return _turn

    conversations.start_async = _wrap("start")
    conversations.append_async = _wrap("append")


async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: str | None = "mistral-large-latest",
    instructions: str | None = None,
    timeout_ms: int | None = None,
    durable: bool = True,
    observability: bool = True,
    agent_id: str | None = None,
    api_key_env_var: str = "MISTRAL_API_KEY",
    memory_key: str | None = None,
) -> str:
    """Run a Mistral agent with the given tools and prompt; return the final text.

    Call this from inside an ``@env.task`` — that task is the durable parent.
    The Mistral SDK runs the agent loop; each tool the agent calls runs as a
    durable Flyte child action, and (with ``durable=True``) each model turn is
    recorded for replay. Pass ``agent_id`` to drive a pre-created server-side
    agent instead of an inline ``model``.

    Args:
        input: The user prompt.
        tools: ``tool``-wrapped tools or bare ``@env.task`` templates.
        model: Model for an inline run (when ``agent_id`` is not given).
        instructions: System instructions.
        timeout_ms: Per-turn request timeout (ms), applied by the SDK to each model
            call inside its loop; ``None`` uses the SDK default. This bounds a single
            hung turn — it is not a whole-run cap (Mistral exposes no turn-count
            limit). To bound the entire agent run, set ``timeout=`` on the enclosing
            ``@env.task`` (the durable parent), which caps all turns + tool calls.
        durable: Record/replay each conversation turn via ``flyte.trace``.
        observability: Render the run timeline into the Flyte task report.
        agent_id: Reuse an existing server-side agent (instead of ``model``).
        api_key_env_var: Env var holding the Mistral API key (wire as a secret).
        memory_key: Stable id (e.g. a user/thread id) for cross-run memory.
            When set, the thread's server-side ``conversation_id`` is persisted in a
            keyed ``MemoryStore`` and reused, so a later run with the same key
            continues the conversation. ``None`` disables memory.
    """

    from mistralai.client import Mistral
    from mistralai.extra.run.context import RunContext

    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"Mistral API key not found in '{api_key_env_var}'. Provide it as a Flyte secret, e.g. "
            f'flyte.Secret(key="mistral_api_key", as_env_var="{api_key_env_var}").'
        )

    client = Mistral(api_key=api_key)
    conversations = client.beta.conversations

    timeline = ReportTimeline() if observability else None
    usage = _UsageSink() if observability else None
    if timeline is not None:
        timeline.heading("Mistral agent")

    # Wrap the per-turn seam for durable replay (``durable``) and/or token tallying
    # (``usage``) — the SDK's ``RunResult`` surfaces neither, but every turn flows here.
    if durable or usage is not None:
        try:
            _install_turn_hooks(conversations, durable=durable, usage=usage)
        except Exception:  # pragma: no cover - never break the run over hook wiring
            logger.warning("Could not install conversation-turn hooks; continuing without replay/usage.")

    # Cross-run memory: Mistral keeps the transcript server-side, so we persist just
    # the conversation id and continue that conversation when the key recurs.
    store = await resolve_memory(memory_key)
    conversation_id = await store.read_json.aio(_MEMORY_CONV_PATH) if store is not None else None

    run_ctx_kwargs = {"agent_id": agent_id} if agent_id is not None else {"model": model}
    if conversation_id:
        run_ctx_kwargs["conversation_id"] = conversation_id

    async with RunContext(**run_ctx_kwargs) as run_ctx:
        for fn in (_coerce_tool(t) for t in tools):
            run_ctx.register_func(fn)

        result = await conversations.run_async(run_ctx, inputs=input, instructions=instructions, timeout_ms=timeout_ms)

    if store is not None and result.conversation_id:
        await store.write_json.aio(_MEMORY_CONV_PATH, result.conversation_id, actor="mistral-agent")
        await store.save.aio()

    output_entries = list(result.output_entries or [])
    if timeline is not None:
        _render(timeline, output_entries)
        if usage is not None and usage.turns:
            timeline.row(icon="📊", label="usage", meta="model", detail=usage.detail())
        await flush_report()
    return _final_text(output_entries)
