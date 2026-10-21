"""Render a Hermes run into the Flyte task report.

Hermes has hooks (`hermes_cli.plugins.VALID_HOOKS`, registered with
`register_hook`) and middleware (`hermes_cli.middleware`), but the adapter
already sits on the two seams that carry everything a timeline needs, so no
Hermes-specific plumbing is added here beyond one more middleware callback.

Model turns come from the same `llm_execution` middleware that `_durable` uses:
every provider call in `agent.conversation_loop` goes through
`hermes_cli.middleware.run_llm_execution_middleware(request, next_call, **context)`,
which hands over the request kwargs (model, messages, tools), the provider
response, and the surrounding context, and lets the callback time `next_call`
itself. Unlike the `pre_llm_call` / `post_llm_call` hooks it is one callback per
turn with both halves in scope, so a turn's duration needs no pairing of two
events, and unlike `_durable` this callback is a pure observer: it never
replaces the response.

With durability on, `_durable` renders the model-turn rows instead and this
callback steps aside. A replayed turn is answered from its trace record, so
`next_call` is never invoked and an observer below the durable middleware
would never see the turn at all, which is what left the report with no model
turns on a retried attempt. Handing rendering to the durable callback whenever
its gate is on keeps every turn on the timeline, exactly once, whether it ran
live or was replayed. A replayed row is marked replayed and carries neither a
duration nor a token usage summary, because on this attempt the turn cost
neither.

Tool calls come from the adapter's own tool wrapper (`_tools`), which already
wraps every Flyte tool Hermes can dispatch. Reading them there rather than from
the `tool_execution` middleware or the `pre_tool_call` / `post_tool_call` hooks
keeps the timeline correct for a bring-your-own `AIAgent` and needs no second
registration.

Both seams are process-global, exactly like the durable middleware, so what
scopes them to one run is a `contextvars.ContextVar` holding the recorder for
the run in flight. `run_agent` drives the synchronous Hermes loop with
`asyncio.to_thread`, which copies the current context into the worker thread,
so the recorder reaches the callbacks on that thread. With the contextvar unset
every seam is a no-op, which is what `observability=False` and any other agent
in the process get.

Rendering never breaks a run: every recorder entry point swallows its own
failures and logs them at debug.
"""

from __future__ import annotations

import contextvars
import time
import typing

from flyte._logging import logger
from flyteplugins.agents.core import ReportTimeline, abbrev

from ._durable import _to_plain, is_durable

# Holds the recorder for the `run_agent(observability=True)` call in flight.
# Unset everywhere else, which is what makes the process-global seams no-ops.
_RECORDER: contextvars.ContextVar["_RunRecorder | None"] = contextvars.ContextVar("flyte_hermes_recorder", default=None)

_MIDDLEWARE_KIND = "llm_execution"

_MODEL_ICON = "🧠"
_REPLAY_ICON = "♻️"
_TOOL_ICON = "🛠️"

_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_input_tokens",
)


def _ms(elapsed: float) -> str:
    """Format an elapsed-seconds measurement the way `duration_ms` formats an ISO gap."""
    return f"{elapsed * 1000:.0f} ms"


def _last_message(request: typing.Mapping[str, typing.Any]) -> str:
    """Preview of the newest message in a provider request, for the row detail."""
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(getattr(part, "text", "")) for part in content
        ).strip()
    return str(content) if content else ""


def _usage(response: typing.Any) -> str:
    """Token usage as a compact `key=value` string, empty when the provider sent none."""
    usage = _to_plain(getattr(response, "usage", None))
    if not isinstance(usage, dict):
        return ""
    parts = [f"{k}={usage[k]}" for k in _USAGE_FIELDS if usage.get(k) is not None]
    return ", ".join(parts)


def _response_text_and_calls(response: typing.Any) -> tuple[str, list[str]]:
    """Assistant text and requested tool names from a provider response.

    Best-effort across the response shapes Hermes reads: chat completions and
    bedrock (`choices[0].message`), anthropic messages (`content` blocks) and
    codex (`output` items).
    """
    plain = _to_plain(response)
    if not isinstance(plain, dict):
        return (str(plain) if plain is not None else ""), []

    choices = plain.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        text = message.get("content") or message.get("reasoning_content") or ""
        calls = [
            str((call.get("function") or {}).get("name") or call.get("name") or "")
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict)
        ]
        return str(text or ""), [c for c in calls if c]

    blocks = plain.get("content") if isinstance(plain.get("content"), list) else plain.get("output")
    if isinstance(blocks, list):
        texts = [str(b.get("text") or "") for b in blocks if isinstance(b, dict) and b.get("text")]
        calls = [
            str(b.get("name") or "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") in ("tool_use", "function_call") and b.get("name")
        ]
        return " ".join(t for t in texts if t), calls

    content = plain.get("content")
    return (str(content) if content else ""), []


def _detail(prompt: str, text: str, calls: list[str], usage: str) -> str:
    """Assemble the HTML detail cell of a model-turn row."""
    parts: list[str] = []
    if prompt or text:
        parts.append(f"<code>{abbrev(prompt, 160)}</code> → <code>{abbrev(text, 240)}</code>")
    if calls:
        parts.append(f"calls <code>{abbrev(', '.join(calls), 160)}</code>")
    if usage:
        parts.append(f'<span style="opacity:.7">{abbrev(usage)}</span>')
    return " · ".join(parts)


class _RunRecorder:
    """Append one `ReportTimeline` row per model turn and per tool call.

    One instance is created per `run_agent(observability=True)` call and lives
    in `_RECORDER` for the duration of that run.
    """

    def __init__(self, timeline: ReportTimeline):
        self._timeline = timeline
        self.turns = 0
        self.tool_calls = 0

    def model_turn(
        self,
        request: typing.Mapping[str, typing.Any],
        response: typing.Any,
        elapsed: float,
        error: typing.Any = None,
        replayed: bool = False,
    ) -> None:
        self.turns += 1
        try:
            messages = request.get("messages")
            count = len(messages) if isinstance(messages, list) else 0
            text, calls = ("", []) if error is not None else _response_text_and_calls(response)
            # A replayed turn was answered from the trace record of an earlier
            # attempt, so it has no duration and no token cost to report.
            meta = " · ".join(
                p
                for p in (
                    "model turn",
                    "replayed" if replayed else "",
                    f"{count} msgs" if count else "",
                    "" if replayed else _ms(elapsed),
                )
                if p
            )
            self._timeline.row(
                icon=_REPLAY_ICON if replayed else _MODEL_ICON,
                label=str(request.get("model") or "model"),
                meta=meta,
                detail=_detail(_last_message(request), text, calls, "" if replayed else _usage(response)),
                error=error,
            )
        except Exception:  # pragma: no cover - rendering must never break the run
            logger.debug("Hermes timeline failed to render a model turn", exc_info=True)

    def tool_call(
        self,
        name: str,
        args: typing.Any,
        result: typing.Any,
        elapsed: float,
        error: typing.Any = None,
    ) -> None:
        self.tool_calls += 1
        try:
            detail = f"<code>{abbrev(args, 160)}</code> → <code>{abbrev(result, 240)}</code>"
            self._timeline.row(
                icon=_TOOL_ICON,
                label=name,
                meta=" · ".join(p for p in ("tool", _ms(elapsed)) if p),
                detail=detail,
                error=error,
            )
        except Exception:  # pragma: no cover - rendering must never break the run
            logger.debug("Hermes timeline failed to render a tool call", exc_info=True)


def start_recording(timeline: ReportTimeline) -> contextvars.Token:
    """Open the recorder for the current context; returns a reset token."""
    return _RECORDER.set(_RunRecorder(timeline))


def stop_recording(token: contextvars.Token) -> None:
    """Restore the recorder to its prior state."""
    _RECORDER.reset(token)


def current_recorder() -> _RunRecorder | None:
    """The recorder for the run in flight, or None when observability is off."""
    return _RECORDER.get()


def record_tool_call(
    name: str,
    args: typing.Any,
    result: typing.Any,
    elapsed: float,
    error: typing.Any = None,
) -> None:
    """Add a tool-call row, or do nothing when no run is being observed.

    Called from the adapter's tool wrapper in `_tools`, which is the one place
    every Flyte tool call passes through.
    """
    recorder = _RECORDER.get()
    if recorder is not None:
        recorder.tool_call(name, args, result, elapsed, error=error)


def record_model_turn(
    request: typing.Mapping[str, typing.Any],
    response: typing.Any,
    elapsed: float,
    error: typing.Any = None,
    replayed: bool = False,
) -> None:
    """Add a model-turn row, or do nothing when no run is being observed.

    Called from the durable middleware in `_durable`, which owns rendering
    while its gate is on so replayed turns reach the timeline too.
    """
    recorder = _RECORDER.get()
    if recorder is not None:
        recorder.model_turn(request, response, elapsed, error=error, replayed=replayed)


def observe_llm_execution(
    request: typing.Any,
    next_call: typing.Callable[[typing.Any], typing.Any],
    **context: typing.Any,
) -> typing.Any:
    """Hermes `llm_execution` middleware that times a turn and renders it.

    A pure observer: the provider response is returned unchanged, and
    `next_call` is invoked exactly once. With no recorder in context this is a
    plain passthrough, which is the case for every agent in the process not
    started by `run_agent(observability=True)`.

    It is also a passthrough while the durable gate is on. The durable
    middleware renders every turn in that case, including the replayed ones
    this callback can never see, and stepping aside here is what keeps a live
    turn from being rendered twice. This holds whichever order the two
    callbacks were appended to the process-global chain in.
    """
    recorder = _RECORDER.get()
    if recorder is None or is_durable() or not isinstance(request, dict):
        return next_call(request)

    started = time.perf_counter()
    try:
        response = next_call(request)
    except Exception as exc:
        recorder.model_turn(request, None, time.perf_counter() - started, error=exc)
        raise
    recorder.model_turn(request, response, time.perf_counter() - started)
    return response


def ensure_registered() -> None:
    """Register the observer middleware on the Hermes plugin manager, once.

    The chain reads the manager's middleware list directly, so registering is
    just appending; identity is the idempotency check, which keeps repeated
    `run_agent` calls in one process from stacking duplicate callbacks.
    """
    from hermes_cli.plugins import get_plugin_manager

    callbacks = get_plugin_manager()._middleware.setdefault(_MIDDLEWARE_KIND, [])
    if not any(cb is observe_llm_execution for cb in callbacks):
        callbacks.append(observe_llm_execution)
