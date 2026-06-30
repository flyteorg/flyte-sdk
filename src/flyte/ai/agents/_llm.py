"""LLM callback abstraction used by :class:`flyte.ai.agents.Agent`.

This module is internal: import :class:`LLMMessage` from
:mod:`flyte.ai.agents` instead. The agent module re-exports the default
litellm-backed callback for back-compat.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class LLMMessage:
    """Provider-agnostic shape returned by :data:`LLMCallable`.

    ``tool_calls`` follows the OpenAI tool-calling convention; provider-specific
    callers should normalize to this shape.
    """

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None


LLMCallable = Callable[
    [str, str, list[dict[str, Any]], list[dict[str, Any]] | None],
    Awaitable[LLMMessage],
]


async def _default_call_llm(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> LLMMessage:
    """Default LLM callback that uses ``litellm.acompletion`` with tool calling.

    Compatible with any provider that litellm supports (OpenAI, Anthropic,
    Gemini, Bedrock, local OpenAI-compatible servers, …).
    """
    try:
        from litellm import acompletion
    except ImportError as exc:  # pragma: no cover - exercised by integration tests only
        raise ImportError(
            "litellm is not installed. Install with `pip install litellm` "
            "or pass `call_llm=...` with a custom callback."
        ) from exc

    full_messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    kwargs: dict[str, Any] = {"model": model, "messages": full_messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = await acompletion(**kwargs)
    choice = response.choices[0]  # type: ignore[index]
    msg = choice.message
    tool_calls: list[dict[str, Any]] = []
    for call in getattr(msg, "tool_calls", None) or []:
        try:
            args_str = call.function.arguments
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except json.JSONDecodeError:
            args = {"_raw": call.function.arguments}
        tool_calls.append(
            {
                "id": getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                "name": call.function.name,
                "arguments": args,
            }
        )
    return LLMMessage(content=getattr(msg, "content", None) or "", tool_calls=tool_calls, raw=response)
