"""Durable, replayable model turns for Pydantic AI.

Pydantic AI owns the agent loop and drives a :class:`~pydantic_ai.models.Model`
one turn at a time via ``await model.request(messages, settings, params)``. To make
that loop durable we wrap the real model in a :class:`FlyteModel`: every
``request`` (one model turn) is recorded through the shared
:func:`~flyteplugins.agents.core.durable_step` (a ``flyte.trace`` leaf). Inside a
Flyte task this means a crashed/retried run replays completed turns from their
recorded outputs instead of re-calling (and re-billing) the model. Tool calls run
as durable child actions (see :func:`flyteplugins.agents.pydantic_ai.tool`), so the
whole agent run becomes crash-resilient when the enclosing task carries
``retries=...``.

The turn is recorded as JSON (pydantic round-trips Pydantic AI's ``ModelResponse``
faithfully and stays readable in the Flyte UI). The non-serializable real call is
captured in a closure passed to ``durable_step``; the trace is keyed on a
deterministic fingerprint of the *serializable* request identity — the serialized
messages, model settings, and tool names — never the live objects.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import durable_step, fingerprint, jsonable
from pydantic import TypeAdapter

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import Model

if typing.TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.models import ModelRequestParameters, StreamedResponse
    from pydantic_ai.settings import ModelSettings

# pydantic round-trips ModelResponse faithfully; storing JSON keeps the recorded
# turn human-readable in the Flyte UI.
_RESPONSE_ADAPTER: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)


def _request_fingerprint(
    messages: typing.Any,
    model_settings: typing.Any,
    model_request_parameters: typing.Any,
) -> str:
    """Deterministic memo key for a model turn — serializable identity, not live objects.

    Fingerprints on the serialized messages (via ``ModelMessagesTypeAdapter``), the
    model settings, and the (sorted) tool names off ``model_request_parameters`` —
    never callables or live SDK objects.
    """
    try:
        serialized_messages = ModelMessagesTypeAdapter.dump_python(list(messages))
    except Exception:  # pragma: no cover - defensive; fall back to best-effort coercion
        serialized_messages = jsonable(messages)

    tool_names = sorted(
        getattr(t, "name", str(t))
        for t in list(getattr(model_request_parameters, "function_tools", None) or [])
        + list(getattr(model_request_parameters, "output_tools", None) or [])
    )

    return fingerprint(
        {
            "messages": serialized_messages,
            "model_settings": jsonable(model_settings),
            "tools": tool_names,
            "output_mode": getattr(model_request_parameters, "output_mode", None),
        }
    )


class FlyteModel(Model):
    """Wrap a :class:`~pydantic_ai.models.Model` so each model turn is durable.

    ``request`` is recorded/replayed via ``durable_step``. ``request_stream`` is
    delegated unchanged: streamed turns are not memoized in this version (tool
    calls remain durable regardless). ``model_name`` / ``system`` and any other
    members are forwarded to the inner model.
    """

    def __init__(self, inner: Model):
        self._inner = inner

    async def request(
        self,
        messages: typing.Any,
        model_settings: "ModelSettings | None",
        model_request_parameters: "ModelRequestParameters",
    ) -> ModelResponse:
        return await durable_step(
            _request_fingerprint(messages, model_settings, model_request_parameters),
            lambda: self._inner.request(messages, model_settings, model_request_parameters),
            name="model_turn",
            dumps=lambda response: _RESPONSE_ADAPTER.dump_json(response).decode("utf-8"),
            loads=_RESPONSE_ADAPTER.validate_json,
        )

    def request_stream(
        self,
        messages: typing.Any,
        model_settings: "ModelSettings | None",
        model_request_parameters: "ModelRequestParameters",
        run_context: typing.Any = None,
    ) -> "AsyncIterator[StreamedResponse]":
        # Streamed turns are delegated unchanged (not memoized).
        return self._inner.request_stream(messages, model_settings, model_request_parameters, run_context)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def system(self) -> str:
        return self._inner.system

    def __getattr__(self, name: str) -> typing.Any:
        # Forward any other abstract/concrete members to the inner model.
        # (``__getattr__`` only fires for attributes not found normally, so the
        # explicit members above take precedence and there's no recursion.)
        return getattr(self._inner, name)
