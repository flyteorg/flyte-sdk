"""Durable, replayable model turns for LangChain chat models.

LangChain's ``create_agent`` graph owns the agent loop; each iteration calls the
chat model once (a "model turn"). :class:`DurableChatModel` wraps any
``BaseChatModel`` so every turn is recorded through the shared
:func:`~flyteplugins.agents.core.durable_step` (a ``flyte.trace`` leaf). Inside a
Flyte task this means a crashed/retried run replays completed turns from their
recorded outputs instead of re-calling (and re-billing) the model. Tool calls run
as durable child actions (see :func:`flyteplugins.agents.langchain.tool`), so the
whole agent run becomes crash-resilient when the enclosing task carries
``retries=...``.

The turn is recorded as JSON: the generated messages of the model's
``ChatResult`` are serialized with ``message_to_dict`` and rebuilt with
``messages_from_dict`` (the same round-trip the langgraph adapter uses), which
keeps the recorded turn human-readable in the Flyte UI.

Tool-calling still works because :meth:`DurableChatModel.bind_tools` delegates to
the inner model to format the tools, then re-binds the resulting kwargs to *this*
wrapper — so ``create_agent``'s bound runnable still routes generation through the
durable override.
"""

from __future__ import annotations

import json
import typing

from flyteplugins.agents.core import durable_step, fingerprint
from langchain_core.language_models.chat_models import BaseChatModel

if typing.TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatResult
    from langchain_core.runnables import Runnable


def _dumps_result(result: "ChatResult") -> str:
    """Serialize a ``ChatResult``'s generated messages to JSON (readable in the UI)."""
    from langchain_core.messages import message_to_dict

    return json.dumps([message_to_dict(gen.message) for gen in result.generations])


def _loads_result(payload: str) -> "ChatResult":
    """Rebuild a ``ChatResult`` from the JSON written by :func:`_dumps_result`."""
    from langchain_core.messages import messages_from_dict
    from langchain_core.outputs import ChatGeneration, ChatResult

    messages = messages_from_dict(json.loads(payload))
    return ChatResult(generations=[ChatGeneration(message=m) for m in messages])


class DurableChatModel(BaseChatModel):
    """Wrap a ``BaseChatModel`` so each model turn is durable and replayable.

    ``_agenerate`` (async) and ``_generate`` (sync) delegate to the inner model
    and record the turn via ``durable_step``. Pass an instance to
    ``create_agent(DurableChatModel(inner=model), tools, ...)``; ``bind_tools``
    and other capabilities are delegated to the inner model so tool-calling
    behaves exactly as the inner model does.

    Durability is best-effort: if anything in the durable path raises, the turn
    falls back to a direct inner call so a run is never broken by it.
    """

    inner: BaseChatModel
    """The wrapped chat model that actually generates responses."""

    @property
    def _llm_type(self) -> str:
        return f"durable-{self.inner._llm_type}"

    def _turn_key(self, messages: typing.Sequence["BaseMessage"], **kwargs: typing.Any) -> str:
        """Deterministic memo key for a model turn — serialized messages + bound tool names."""
        from langchain_core.messages import messages_to_dict

        tools = kwargs.get("tools") or []
        tool_names = sorted(
            (t.get("function", {}).get("name") or t.get("name") or str(t)) if isinstance(t, dict) else str(t)
            for t in tools
        )
        return fingerprint(
            {
                "type": self._llm_type,
                "messages": messages_to_dict(list(messages)),
                "tools": tool_names,
                "stop": kwargs.get("stop"),
            }
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> "ChatResult":  # type: ignore[override]
        async def _call() -> "ChatResult":
            return await self.inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        try:
            key = self._turn_key(messages, stop=stop, **kwargs)
            return await durable_step(
                key,
                _call,
                name="model_turn",
                dumps=_dumps_result,
                loads=_loads_result,
            )
        except Exception:  # pragma: no cover - durability must never break a run
            return await _call()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> "ChatResult":  # type: ignore[override]
        # ``durable_step`` is async; the sync path delegates straight through so
        # ``create_agent``'s sync callers keep working (durability applies to the
        # async agent loop, which is the one Flyte tasks drive).
        return self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools: typing.Sequence[typing.Any], **kwargs: typing.Any) -> "Runnable":
        """Format tools via the inner model, but bind them to *this* wrapper.

        The inner model knows how to convert tools into its provider format; we
        reuse that, then re-bind the resulting kwargs to ``self`` so the runnable
        ``create_agent`` invokes still routes generation through the durable
        override (rather than the inner model directly).
        """
        bound = self.inner.bind_tools(tools, **kwargs)
        bound_kwargs = dict(getattr(bound, "kwargs", {}) or {})
        return self.bind(**bound_kwargs)

    def get_num_tokens(self, text: str) -> int:
        return self.inner.get_num_tokens(text)

    def get_num_tokens_from_messages(self, messages, tools=None) -> int:  # type: ignore[override]
        if tools is not None:
            return self.inner.get_num_tokens_from_messages(messages, tools=tools)
        return self.inner.get_num_tokens_from_messages(messages)
