"""Durable, replayable model turns for the OpenAI Agents SDK.

The OpenAI Agents ``Runner`` owns the agent loop. To make that loop durable we
swap in a :class:`FlyteModelProvider`: it wraps the real model so every
``get_response`` (one model turn) is recorded through the shared
:func:`~flyteplugins.agents.core.durable_step` (a ``flyte.trace`` leaf). Inside a
Flyte task this means a crashed/retried run replays completed turns from
their recorded outputs instead of re-calling (and re-billing) the model. Tool
calls run as durable child actions (see :func:`flyteplugins.agents.openai.tool`),
so the whole agent run becomes crash-resilient and self-healing when the enclosing task
carries ``retries=...``.

The turn is recorded as JSON (pydantic round-trips the SDK's ``ModelResponse``
faithfully and stays readable in the Flyte UI). The non-serializable real call is
captured in a closure passed to ``durable_step``.
"""

from __future__ import annotations

import typing
from collections.abc import AsyncIterator

from agents.items import ModelResponse
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider
from flyteplugins.agents.core import durable_step, fingerprint, jsonable
from pydantic import TypeAdapter

# pydantic round-trips ModelResponse (and its nested OpenAI types) faithfully;
# storing JSON keeps the recorded turn human-readable in the Flyte UI.
_RESPONSE_ADAPTER: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)

# Positional parameters of ``Model.get_response`` used for fingerprinting
# (``tracing`` and the keyword-only ids are intentionally ignored).
_GET_RESPONSE_PARAMS = (
    "system_instructions",
    "input",
    "model_settings",
    "tools",
    "output_schema",
    "handoffs",
)


def _bind(args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    bound = dict(kwargs)
    for name, value in zip(_GET_RESPONSE_PARAMS, args):
        bound.setdefault(name, value)
    return bound


def _request_fingerprint(args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]) -> str:
    """Deterministic memo key for a model turn — tool/handoff names, not callables."""
    bound = _bind(args, kwargs)
    return fingerprint(
        {
            "system": bound.get("system_instructions"),
            "input": jsonable(bound.get("input")),
            "model_settings": jsonable(bound.get("model_settings")),
            "tools": sorted(getattr(t, "name", str(t)) for t in (bound.get("tools") or [])),
            "handoffs": sorted(
                getattr(h, "agent_name", getattr(h, "tool_name", str(h))) for h in (bound.get("handoffs") or [])
            ),
            "output_schema": getattr(bound.get("output_schema"), "name", None),
        }
    )


class FlyteModel(Model):
    """Wrap a :class:`~agents.models.interface.Model` so each turn is durable.

    ``get_response`` is recorded/replayed via ``durable_step``. ``stream_response``
    is delegated unchanged: streamed turns are not memoized in this version (tool
    calls remain durable regardless).
    """

    def __init__(self, inner: Model):
        self._inner = inner

    async def get_response(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return await durable_step(
            _request_fingerprint(args, kwargs),
            lambda: self._inner.get_response(*args, **kwargs),
            name="model_turn",
            dumps=lambda response: _RESPONSE_ADAPTER.dump_json(response).decode("utf-8"),
            loads=_RESPONSE_ADAPTER.validate_json,
        )

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
