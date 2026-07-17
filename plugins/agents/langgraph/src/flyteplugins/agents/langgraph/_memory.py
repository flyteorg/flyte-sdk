"""Cross-run LangGraph memory — a thin bridge over Flyte's keyed ``MemoryStore``.

LangGraph keeps conversation state in-memory. This module persists the message
transcript to a durable, keyed :class:`~flyte.ai.agents.memory.MemoryStore` (an
object-store slot addressed by ``memory_key``) so a later run with the same key
continues the conversation — across workers and restarts.

The transcript is stored (via ``read_json`` / ``write_json``) as the serialized
LangChain message list, so it round-trips faithfully through
``messages_from_dict`` / ``messages_to_dict``.
"""

from __future__ import annotations

import typing

from flyte._logging import logger
from flyteplugins.agents.core import resolve_memory as _resolve_memory

# Path-addressed slot holding the serialized message transcript inside the MemoryStore.
_MEMORY_HISTORY_PATH = "langgraph/history.json"


async def resolve_memory(memory_key: str | None) -> typing.Any | None:
    """Resolve a keyed MemoryStore for LangGraph cross-run memory, or ``None``.

    Best-effort: returns ``None`` when ``memory_key`` is falsy or no durable
    store can be resolved, so memory never breaks a run.
    """
    if not memory_key:
        return None
    return await _resolve_memory(memory_key)


async def load_messages(store: typing.Any) -> list[typing.Any]:
    """Load the prior conversation as LangChain messages (empty list if none)."""
    if store is None:
        return []
    try:
        from langchain_core.messages import messages_from_dict

        raw = await store.read_json.aio(_MEMORY_HISTORY_PATH, [])
        return messages_from_dict(raw) if raw else []
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not load LangGraph memory; continuing without prior history.")
        return []


async def save_messages(store: typing.Any, messages: typing.Sequence[typing.Any]) -> None:
    """Persist the full conversation transcript back to the keyed store."""
    if store is None or not messages:
        return
    try:
        from langchain_core.messages import messages_to_dict

        await store.write_json.aio(_MEMORY_HISTORY_PATH, messages_to_dict(list(messages)))
        await store.save.aio()
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not persist LangGraph memory; continuing.")
