"""Cross-run LangChain memory — a thin bridge over Flyte's keyed ``MemoryStore``.

LangChain's ``create_agent`` graph is driven with a messages state
(``graph.ainvoke({"messages": [...]})``) and, by default, keeps no state across
runs. This module bridges that: it resolves a keyed :class:`MemoryStore`, loads a
prior conversation from a path-addressed JSON slot, and writes the full
transcript back after the run — so a later run with the same ``memory_key``
continues the conversation.

The transcript is stored as ``messages_to_dict(...)`` output (the same
serialization the langgraph adapter uses) in a single JSON slot, and rebuilt with
``messages_from_dict``. All operations are best-effort: any failure leaves the run
untouched (memory never breaks a run).
"""

from __future__ import annotations

import typing

from flyte._logging import logger
from flyteplugins.agents.core import resolve_memory as _resolve_memory

if typing.TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

# Path-addressed memory slot holding the serialized conversation transcript
# (``messages_to_dict`` output) inside the MemoryStore.
_HISTORY_PATH = "langchain/history.json"


async def resolve_memory(memory_key: str | None) -> typing.Any | None:
    """Resolve a keyed MemoryStore for LangChain cross-run memory, or ``None``.

    Best-effort: returns ``None`` when ``memory_key`` is falsy or no durable
    store can be resolved, so memory never breaks a run.
    """
    if not memory_key:
        return None
    return await _resolve_memory(memory_key)


async def load_history(store: typing.Any) -> list["BaseMessage"]:
    """Load and deserialize the prior conversation from ``store``.

    Returns an empty list when there is no prior history or on any error.
    """
    if store is None:
        return []
    try:
        from langchain_core.messages import messages_from_dict

        raw = await store.read_json.aio(_HISTORY_PATH, [])
        return messages_from_dict(raw) if raw else []
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not load LangChain memory; continuing without prior history.")
        return []


async def save_history(store: typing.Any, messages: typing.Sequence["BaseMessage"]) -> None:
    """Serialize the full conversation ``messages`` and persist them to ``store``.

    Best-effort: logs and returns on any error so memory never breaks a run.
    """
    if store is None or not messages:
        return
    try:
        from langchain_core.messages import messages_to_dict

        await store.write_json.aio(_HISTORY_PATH, messages_to_dict(list(messages)))
        await store.save.aio()
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not save LangChain memory; conversation will not be resumed.")
