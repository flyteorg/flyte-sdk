"""Durable, replayable model turns for CrewAI.

CrewAI owns the agent loop and drives the model itself: during
``Agent.kickoff_async`` it calls its ``crewai.LLM`` once per turn. To make that
loop durable we swap in a durable ``LLM``: it records each turn through the
shared :func:`~flyteplugins.agents.core.durable_step` (a ``flyte.trace`` leaf),
so inside a Flyte task a crashed/retried run replays completed turns from their
recorded completions instead of re-calling (and re-billing) the model. Tool
calls run as durable child actions (see :func:`flyteplugins.agents.crewai.tool`),
so the whole run becomes crash-resilient when the enclosing task carries
``retries=...``.

Two CrewAI facts shape the implementation:

- ``crewai.LLM(model=...)`` is a *factory*: ``LLM.__new__`` dispatches on the
  model name and returns a provider-specific subclass (``OpenAICompletion``,
  ``AnthropicCompletion``, ...). A plain ``class D(LLM)`` therefore never sees the
  provider's real ``call``. We instead subclass the *concrete* provider class of a
  probe instance (``type(LLM(model=...))``) and instantiate through it, so the
  durable overrides sit directly on top of the real completion methods.
- ``kickoff_async`` invokes the **synchronous** ``call`` (verified on 1.15.2),
  not ``acall``; we override both so durability applies whichever path runs.

The turn result is the completion — a string, or an arbitrary object for
structured outputs. Strings round-trip as-is; anything else is coerced to a
string for the trace record (structured turns are recorded but rebuilt as text,
which is a lossy but safe fallback and never the case for the default text loop).
The concrete provider class is a heavy import, so the durable subclass is built
lazily on first use.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import durable_step, fingerprint, jsonable

# Sentinel prefix marking a completion we serialized as JSON (non-string result).
# Plain string completions are stored verbatim; on load we only JSON-decode a
# value carrying this prefix, so ordinary strings never get mis-parsed.
_JSON_PREFIX = "\x00json\x00"


def _messages_fingerprint(
    messages: typing.Any,
    tools: typing.Any,
    extra: typing.Mapping[str, typing.Any] | None = None,
) -> str:
    """Deterministic memo key for a model turn — message content + tool names.

    Fingerprints on serializable request identity only: the messages (a str or a
    list of ``{"role", "content"}`` dicts) and the tool *names* — never callables
    or live tool/agent objects.
    """
    payload: dict[str, typing.Any] = {
        "messages": jsonable(messages),
        "tools": sorted(_tool_names(tools)),
    }
    if extra:
        payload.update(extra)
    return fingerprint(payload)


def _tool_names(tools: typing.Any) -> list[str]:
    """Extract stable tool names from CrewAI's ``[{name: BaseTool}, ...]`` (or list)."""
    names: list[str] = []
    for entry in tools or []:
        if isinstance(entry, dict):
            names.extend(str(k) for k in entry.keys())
        else:
            names.append(str(getattr(entry, "name", entry)))
    return names


def _dumps(result: typing.Any) -> str:
    """Serialize a turn result: strings verbatim, everything else JSON-tagged.

    JSON-native structures (dict/list/scalars) serialize directly; SDK objects
    are coerced via ``jsonable`` first. ``default=str`` catches any residue so
    serialization never fails a turn.
    """
    if isinstance(result, str):
        return result
    import json

    if isinstance(result, (dict, list, int, float, bool)) or result is None:
        payload = result
    else:
        payload = jsonable(result)
    return _JSON_PREFIX + json.dumps(payload, default=str)


def _loads(recorded: str) -> typing.Any:
    """Rebuild a turn result from its recorded string form."""
    if recorded.startswith(_JSON_PREFIX):
        import json

        return json.loads(recorded[len(_JSON_PREFIX) :])
    return recorded


def _make_durable_llm_class(model: str | None) -> type:
    """Build a durable ``LLM`` subclass over the concrete provider for ``model``.

    Imported/derived lazily: ``crewai`` and its provider classes are heavy, and
    the concrete class depends on the model name (provider dispatch).
    """
    from crewai import LLM

    probe = LLM(model=model or "gpt-4o")
    concrete_cls = type(probe)

    class DurableLLM(concrete_cls):  # type: ignore[valid-type, misc]
        """A ``crewai.LLM`` whose every model turn is recorded via ``durable_step``.

        ``call`` (used by ``kickoff_async``) and ``acall`` both route the real
        completion through the shared durable step so retries replay recorded
        turns. Durability is guarded: if the trace layer misbehaves the real
        call still runs, so it never breaks a run.
        """

        def call(self, messages: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            from flyte._utils.asyn import run_sync

            tools = args[0] if args else kwargs.get("tools")
            try:
                return run_sync(
                    durable_step,
                    _messages_fingerprint(messages, tools),
                    lambda: _as_awaitable(super(DurableLLM, self).call(messages, *args, **kwargs)),
                    name="model_turn",
                    dumps=_dumps,
                    loads=_loads,
                )
            except Exception:  # pragma: no cover - durability never breaks a run
                return super().call(messages, *args, **kwargs)

        async def acall(self, messages: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            tools = args[0] if args else kwargs.get("tools")
            try:
                return await durable_step(
                    _messages_fingerprint(messages, tools),
                    lambda: _as_awaitable(super(DurableLLM, self).acall(messages, *args, **kwargs)),
                    name="model_turn",
                    dumps=_dumps,
                    loads=_loads,
                )
            except Exception:  # pragma: no cover - durability never breaks a run
                return await _as_awaitable(super().acall(messages, *args, **kwargs))

    return DurableLLM


async def _as_awaitable(value: typing.Any) -> typing.Any:
    """Normalize a maybe-coroutine to an awaited value (``call`` returns eagerly)."""
    import inspect

    if inspect.isawaitable(value):
        return await value
    return value


def make_durable_llm(model: str | None) -> typing.Any:
    """Construct a durable ``crewai.LLM`` instance for ``model`` (or ``gpt-4o``)."""
    cls = _make_durable_llm_class(model)
    return cls(model=model or "gpt-4o")
