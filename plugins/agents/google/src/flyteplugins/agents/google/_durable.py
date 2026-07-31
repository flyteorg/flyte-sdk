"""Durable model turns for Google ADK — trace the seam below the loop.

ADK's ``Runner`` owns the loop, but every model turn flows through the agent's
``BaseLlm.generate_content_async``. :class:`FlyteLlm` wraps that method so each
(non-streaming) turn is recorded as a ``durable_step`` (a ``flyte.trace`` leaf):
on a crash/retry the completed turns replay from their recorded ``LlmResponse``
instead of re-calling (and re-billing) the model — while ADK still drives the loop.

Streaming turns are passed through unmemoized; tool calls remain durable
Flyte actions regardless.
"""

from __future__ import annotations

import json
import typing

from flyteplugins.agents.core import durable_step, fingerprint, jsonable

from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse


class FlyteLlm(BaseLlm):
    """A ``BaseLlm`` that records each model turn via ``durable_step`` for replay.

    Wraps an inner ``BaseLlm`` (resolved from the agent's ``model``); ``model`` is set
    to the inner model name so ADK behaves identically. Construct via
    :func:`durable_model`.
    """

    inner: typing.Any = None

    async def generate_content_async(
        self, llm_request: typing.Any, stream: bool = False
    ) -> typing.AsyncGenerator[typing.Any, None]:
        if stream:
            # Streamed turns are not memoized per-turn in this version.
            async for response in self.inner.generate_content_async(llm_request, stream=True):
                yield response
            return

        key = fingerprint({"model": self.model, "request": _request_key(llm_request)})

        async def _run() -> list[dict]:
            collected: list[dict] = []
            async for response in self.inner.generate_content_async(llm_request, stream=False):
                collected.append(response.model_dump(mode="json", exclude_none=True))
            return collected

        recorded = await durable_step(key, _run, name="adk_model_turn", dumps=json.dumps, loads=json.loads)
        for payload in recorded:
            yield LlmResponse.model_validate(payload)


def _request_key(llm_request: typing.Any) -> typing.Any:
    """A deterministic, JSON-able view of the request to key the durable turn."""
    try:
        return jsonable(llm_request.model_dump(mode="json", exclude_none=True))
    except Exception:  # pragma: no cover - fall back to a coarse key
        return jsonable(getattr(llm_request, "contents", None))


def durable_model(model: typing.Any) -> typing.Any:
    """Wrap ``model`` (a name string or ``BaseLlm``) so its turns are durable.

    Returns a :class:`FlyteLlm` over the resolved inner model, or ``model`` unchanged
    when it can't be wrapped (durability is best-effort, never fatal).
    """
    try:
        from google.adk.models import BaseLlm as _BaseLlm
        from google.adk.models import LLMRegistry

        if isinstance(model, str):
            inner = LLMRegistry.new_llm(model)
        elif isinstance(model, _BaseLlm):
            inner = model
        else:
            return model
        return FlyteLlm(model=inner.model, inner=inner)
    except Exception:  # pragma: no cover - never break a run over durability wiring
        return model
