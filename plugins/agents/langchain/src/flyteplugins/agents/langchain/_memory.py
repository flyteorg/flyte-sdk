"""Cross-run LangChain memory — a thin handle over Flyte's keyed ``MemoryStore``.

LangChain keeps conversation state in ``InMemoryChatMessageHistory`` by default.
This module provides the bridge: it resolves a keyed ``MemoryStore`` and
persists the thread's ``conversation_id`` (or equivalent) so a later run
with the same ``memory_key`` continues the conversation.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import resolve_memory as _resolve_memory

# Path-addressed memory slot holding a thread's conversation state inside the MemoryStore.
_MEMORY_CONV_PATH = "langchain/conversation_id"


async def resolve_memory(memory_key: str | None) -> typing.Any | None:
    """Resolve a keyed MemoryStore for LangChain cross-run memory, or ``None``.

    Best-effort: returns ``None`` when ``memory_key`` is falsy or no durable
    store can be resolved, so memory never breaks a run.
    """
    if not memory_key:
        return None
    return await _resolve_memory(memory_key)
