"""Durable, replayable model turns for Hermes.

Hermes owns its agent loop: `AIAgent.run_conversation` drives the provider
itself, once per turn, from inside `agent.conversation_loop`. The seam below
that loop is the `llm_execution` middleware that hermes-agent exposes from
version 0.17 on: every provider call goes through
`hermes_cli.middleware.run_llm_execution_middleware(request, next_call, **context)`,
and a registered callback wraps `next_call`, which performs the real call and
returns the provider response object.

That is exactly the shape `flyteplugins.agents.core.durable_step` wants, so the
callback here records each model turn as a `flyte.trace` leaf keyed by a
fingerprint of the request. Inside a Flyte task a crashed or retried run
replays completed turns from their recorded responses instead of re-calling
(and re-billing) the model. Outside a task context `flyte.trace` is a
pass-through, so the same code runs locally unchanged.

Three facts shape the implementation.

The middleware list lives on the process-global Hermes plugin manager, so a
registered callback would otherwise apply to every agent in the process,
durable or not. Durability is therefore gated on a `contextvars.ContextVar`
that `run_agent` sets only while a `durable=True` call is in flight. The Hermes
loop is synchronous and the adapter drives it with `asyncio.to_thread`, which
copies the current context into the worker thread, so the gate reaches the
callback on that thread. With the gate unset the callback is a pure
passthrough.

`durable_step` is async while the middleware callback is synchronous, so the
callback bridges with `flyte._utils.asyn.run_sync`, the same way the CrewAI
adapter does.

Downstream Hermes consumes the response purely by attribute access
(`response.choices[0].message.content` / `.tool_calls` / `.reasoning_content`,
`.finish_reason`, `.usage`, `.model` for the chat-completions and bedrock
transports, `response.content` blocks for `anthropic_messages`,
`response.output` for codex). No `isinstance` check and no method call such as
`model_dump()` is made on the response, which was verified against the
installed hermes-agent. A recursively rebuilt `types.SimpleNamespace` therefore
substitutes for the provider object on replay, which is also what Hermes' own
streaming path returns. `durable_step` round-trips the result through
`dumps`/`loads` even on the recording pass, so the live path sees the same
rebuilt shape as a replay and the two cannot drift.

Streaming turns are already fully drained by the time `next_call` returns, so
they are recorded as a single completed turn. A replayed turn does not re-emit
deltas to display or TTS consumers.

Because a replayed turn never calls `next_call`, the observer middleware in
`_observability` (which sits below this one in the chain) never sees it. So
while the durable gate is on this callback owns report rendering for model
turns: the observer steps aside, and every turn is rendered from here, live
ones with their duration and token usage and replayed ones marked as replayed
with neither. Rendering is best-effort and never breaks a run.
"""

from __future__ import annotations

import contextvars
import json
import time
import types
import typing

from flyte._logging import logger
from flyteplugins.agents.core import durable_step, fingerprint, jsonable

# Set (to True) only while a `run_agent(durable=True)` call is in flight. The
# Hermes middleware list is process-global, so this per-run gate is what keeps
# a non-durable agent in the same process from being recorded.
_DURABLE: contextvars.ContextVar[bool] = contextvars.ContextVar("flyte_hermes_durable", default=False)

_MIDDLEWARE_KIND = "llm_execution"


def enable_durable() -> contextvars.Token:
    """Turn durable model turns on for the current context; returns a reset token."""
    return _DURABLE.set(True)


def disable_durable(token: contextvars.Token) -> None:
    """Restore the durable gate to its prior state."""
    _DURABLE.reset(token)


def is_durable() -> bool:
    """True while a `run_agent(durable=True)` call is in flight in this context."""
    return _DURABLE.get()


def _record(
    request: typing.Mapping[str, typing.Any],
    response: typing.Any,
    elapsed: float,
    error: typing.Any = None,
    replayed: bool = False,
) -> None:
    """Render one model turn into the run report, if a run is being observed.

    The durable callback renders both the live and the replayed case because a
    replayed turn never reaches the observer callback: `next_call` is not
    invoked at all. The observer steps aside whenever the gate is on, so this
    stays exactly one row per turn.
    """
    try:
        from ._observability import record_model_turn

        record_model_turn(request, response, elapsed, error=error, replayed=replayed)
    except Exception:  # pragma: no cover - rendering must never break the run
        logger.debug("Hermes failed to render a model turn", exc_info=True)


def _to_plain(value: typing.Any) -> typing.Any:
    """Recursively convert a provider response into JSON-dumpable plain data.

    Pydantic models (the OpenAI and Anthropic SDK response types) go through
    `model_dump(mode="json")`; `SimpleNamespace` and other attribute bags go
    through their `__dict__`; dicts and lists recurse. Anything left over is
    handed to `jsonable`, which falls back to `str`.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            try:
                return dump()
            except Exception:
                pass
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return jsonable(value)


def _rebuild(value: typing.Any) -> typing.Any:
    """Rebuild recorded plain data into the attribute shape Hermes reads.

    Dicts become `SimpleNamespace` so nested access such as
    `response.choices[0].message.tool_calls[0].function.arguments` works;
    lists stay lists; scalars pass through.
    """
    if isinstance(value, dict):
        return types.SimpleNamespace(**{k: _rebuild(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_rebuild(v) for v in value]
    return value


def _dumps(response: typing.Any) -> str:
    """Serialize a provider response for the trace record."""
    return json.dumps(_to_plain(response), default=str)


def _loads(recorded: str) -> typing.Any:
    """Rebuild a provider response from its recorded form."""
    return _rebuild(json.loads(recorded))


def _tool_names(request: typing.Mapping[str, typing.Any]) -> list[str]:
    """Stable, sorted tool names from a provider request payload.

    Covers both the OpenAI shape (`{"type": "function", "function": {"name": ...}}`)
    and the Anthropic shape (`{"name": ...}`).
    """
    names: list[str] = []
    for entry in request.get("tools") or []:
        if isinstance(entry, dict):
            function = entry.get("function")
            if isinstance(function, dict) and function.get("name"):
                names.append(str(function["name"]))
            elif entry.get("name"):
                names.append(str(entry["name"]))
        else:
            names.append(str(getattr(entry, "name", entry)))
    return sorted(names)


def _request_key(request: typing.Mapping[str, typing.Any], context: typing.Mapping[str, typing.Any]) -> str:
    """Deterministic memo key for one model turn.

    Keys on the semantic request: model, conversation, tool names and api mode.
    `api_call_count` is folded in so two textually identical requests within one
    run stay distinct steps rather than collapsing onto one record.
    """
    payload: dict[str, typing.Any] = {
        "model": jsonable(request.get("model") or context.get("model")),
        "messages": _to_plain(request.get("messages")),
        "tools": _tool_names(request),
        "api_mode": jsonable(context.get("api_mode")),
        "api_call_count": jsonable(context.get("api_call_count")),
    }
    system = request.get("system")
    if system is not None:
        payload["system"] = _to_plain(system)
    return fingerprint(payload)


def durable_llm_execution(
    request: typing.Any,
    next_call: typing.Callable[[typing.Any], typing.Any],
    **context: typing.Any,
) -> typing.Any:
    """Hermes `llm_execution` middleware recording each model turn for replay.

    With the durable gate unset this is a plain passthrough, which is the case
    for every agent in the process that was not started by
    `run_agent(durable=True)`.

    `next_call` is single-use: the Hermes chain raises if a callback calls it
    twice. So the guarded fallback only runs when the durability layer failed
    before `next_call` was reached; a failure after that point re-raises, and
    an exception raised by the real provider call propagates unchanged.

    With the gate on this is also what renders the model-turn row of the run
    report, for live and replayed turns alike. A turn where `next_call` was
    never reached came back from the trace record, so it is rendered as
    replayed.
    """
    if not _DURABLE.get() or not isinstance(request, dict):
        return next_call(request)

    called = False

    def _invoke() -> typing.Any:
        nonlocal called
        called = True
        return next_call(request)

    started = time.perf_counter()
    try:
        from flyte._utils.asyn import run_sync

        response = run_sync(
            durable_step,
            _request_key(request, context),
            lambda: _as_awaitable(_invoke()),
            name="model_turn",
            dumps=_dumps,
            loads=_loads,
        )
    except Exception as exc:
        if called:
            # The provider call already ran; re-raise rather than run it twice.
            _record(request, None, time.perf_counter() - started, error=exc)
            raise
        # Durability never breaks a run: fall back to the plain provider call.
        started = time.perf_counter()
        try:
            response = next_call(request)
        except Exception as fallback_exc:
            _record(request, None, time.perf_counter() - started, error=fallback_exc)
            raise
        _record(request, response, time.perf_counter() - started)
        return response

    # If `called` stayed False, `durable_step` answered from its trace record
    # without running the step, so this turn is a replay of an earlier attempt.
    _record(request, response, time.perf_counter() - started, replayed=not called)
    return response


async def _as_awaitable(value: typing.Any) -> typing.Any:
    """Normalize a maybe-awaitable to an awaited value (`next_call` returns eagerly)."""
    import inspect

    if inspect.isawaitable(value):
        return await value
    return value


def ensure_registered() -> None:
    """Register the durable middleware on the Hermes plugin manager, once.

    The chain reads the manager's middleware list directly, with no enablement
    filter, so registering is just appending. Identity is the idempotency
    check, which keeps repeated `run_agent` calls in one process from stacking
    duplicate callbacks.
    """
    from hermes_cli.plugins import get_plugin_manager

    callbacks = get_plugin_manager()._middleware.setdefault(_MIDDLEWARE_KIND, [])
    if not any(cb is durable_llm_execution for cb in callbacks):
        callbacks.append(durable_llm_execution)
