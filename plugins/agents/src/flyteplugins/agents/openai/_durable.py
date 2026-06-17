"""Durable, replayable model turns for the OpenAI Agents SDK.

The OpenAI Agents ``Runner`` owns the agent loop. To make that loop durable we
swap in a :class:`FlyteModelProvider`: it wraps the real model so every
``get_response`` (one model turn) is recorded through ``flyte.trace``. Inside a
Flyte task this means a crashed or retried run **replays** completed turns from
their recorded outputs instead of re-calling the LLM — no token re-billing, and
deterministic fast-forward through work already done. Tool calls, meanwhile, run
as durable child actions (see :func:`flyteplugins.agents.openai.function_tool`),
so the whole agent run becomes crash-resilient and self-healing when the
enclosing task carries ``retries=...``.

Why a ContextVar: ``flyte.trace`` keys its memo on the decorated function's
*serializable* arguments. A model turn's real arguments (``Tool`` objects with
callables, model settings, etc.) are not cleanly serializable, so we pass them
out-of-band via a ContextVar and feed ``flyte.trace`` only a deterministic
fingerprint string. The recorded output is the SDK's ``ModelResponse`` serialized
to JSON via pydantic — readable in the Flyte UI and replayed faithfully on retry.
"""

from __future__ import annotations

import hashlib
import json
import typing
from collections.abc import AsyncIterator
from contextvars import ContextVar

import flyte
from pydantic import TypeAdapter

from agents.items import ModelResponse
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider

# Faithful, human-readable serialization of a model turn for the trace record.
# ``ModelResponse`` is a dataclass of OpenAI pydantic types; pydantic round-trips
# it exactly, and storing JSON (a ``str``) makes the recorded turn visible in the
# Flyte UI rather than an opaque pickle blob.
_RESPONSE_ADAPTER: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)

# Carries the real (non-serializable) model call for the trace body to execute.
_PENDING_CALL: ContextVar[typing.Callable[[], typing.Awaitable[typing.Any]]] = ContextVar("flyte_pending_model_call")

# Order of the positional parameters of ``Model.get_response`` we care about for
# fingerprinting (``tracing`` and the keyword-only ids are intentionally ignored).
_GET_RESPONSE_PARAMS = (
    "system_instructions",
    "input",
    "model_settings",
    "tools",
    "output_schema",
    "handoffs",
)


@flyte.trace
async def durable_model_call(request_key: str) -> str:
    """A single, durable model turn, recorded as JSON.

    ``request_key`` is a deterministic fingerprint of the turn's request; it is
    the only thing ``flyte.trace`` sees, so the recorded output is keyed and
    replayed by it. The actual model call is supplied via :data:`_PENDING_CALL`.

    Returns the model response serialized to a JSON ``str`` via pydantic. A
    ``str`` is stored inline and is **human-readable in the Flyte UI** (unlike an
    opaque pickle), and pydantic round-trips the SDK's ``ModelResponse`` —
    including its nested OpenAI types — faithfully. Outside a task context
    ``flyte.trace`` is a transparent pass-through, so this also works unchanged
    for local/unit runs.
    """
    call = _PENDING_CALL.get()
    return _RESPONSE_ADAPTER.dump_json(await call()).decode("utf-8")


def _jsonable(obj: typing.Any) -> typing.Any:
    """Best-effort conversion of an SDK object to something JSON-dumpable."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    for attr in ("to_json_dict", "model_dump"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # pragma: no cover - defensive
                pass
    return str(obj)


def _fingerprint(bound: dict[str, typing.Any]) -> str:
    """Deterministic fingerprint of a model turn's request.

    Must be a pure function of the semantic request so replays line up. We use
    tool/handoff *names* (not the callables) and a JSON dump of the inputs.
    """
    payload = {
        "system": bound.get("system_instructions"),
        "input": _jsonable(bound.get("input")),
        "model_settings": _jsonable(bound.get("model_settings")),
        "tools": sorted(getattr(t, "name", str(t)) for t in (bound.get("tools") or [])),
        "handoffs": sorted(
            getattr(h, "agent_name", getattr(h, "tool_name", str(h))) for h in (bound.get("handoffs") or [])
        ),
        "output_schema": getattr(bound.get("output_schema"), "name", None),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _bind(args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    bound = dict(kwargs)
    for name, value in zip(_GET_RESPONSE_PARAMS, args):
        bound.setdefault(name, value)
    return bound


class FlyteModel(Model):
    """Wrap a :class:`~agents.models.interface.Model` so each turn is durable.

    ``get_response`` is recorded/replayed via ``flyte.trace``. ``stream_response``
    is delegated unchanged: streamed turns are not memoized in this version
    (tool calls remain durable regardless).
    """

    def __init__(self, inner: Model):
        self._inner = inner

    async def get_response(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        request_key = _fingerprint(_bind(args, kwargs))
        token = _PENDING_CALL.set(lambda: self._inner.get_response(*args, **kwargs))
        try:
            return _RESPONSE_ADAPTER.validate_json(await durable_model_call(request_key))
        finally:
            _PENDING_CALL.reset(token)

    def stream_response(self, *args: typing.Any, **kwargs: typing.Any) -> AsyncIterator[typing.Any]:
        return self._inner.stream_response(*args, **kwargs)

    async def close(self) -> None:
        await self._inner.close()


class FlyteModelProvider(ModelProvider):
    """Wrap a ``ModelProvider`` so every model it returns produces durable turns.

    Pass an explicit ``inner`` provider to keep custom routing (Azure, a gateway,
    a local OpenAI-compatible server); defaults to the SDK's ``MultiProvider``.
    Set it on ``RunConfig.model_provider`` (``run_agent`` does this for you).
    """

    def __init__(self, inner: ModelProvider | None = None):
        self._inner = inner or MultiProvider()

    def get_model(self, model_name: str | None) -> Model:
        return FlyteModel(self._inner.get_model(model_name))

    async def aclose(self) -> None:
        await self._inner.aclose()
