"""Cross-run Pydantic AI memory — a thin handle over Flyte's keyed ``MemoryStore``.

Pydantic AI keeps conversation state in-memory: each ``Agent.run`` returns a
result whose ``all_messages()`` is the full message history, and a follow-up run
continues the conversation by passing that history as ``message_history=``. This
module bridges that onto a durable, keyed :class:`~flyte.ai.agents.memory.MemoryStore`
so a later run with the same ``memory_key`` resumes where the last one left off —
even on a different worker.

The message history is serialized with Pydantic AI's ``ModelMessagesTypeAdapter``
(the SDK's own round-trip format) and stored in a single path-addressed JSON slot,
so it stays testable via the store's ``read_json`` / ``write_json`` API. All of it
is best-effort: a memory failure never breaks a run.
"""

from __future__ import annotations

import typing

from flyte._logging import logger
from flyteplugins.agents.core import resolve_memory as _resolve_memory

# Path-addressed memory slot holding a thread's serialized message history.
_MEMORY_HISTORY_PATH = "pydantic_ai/history.json"


async def resolve_memory(memory_key: str | None) -> typing.Any | None:
    """Resolve a keyed MemoryStore for Pydantic AI cross-run memory, or ``None``.

    Best-effort: returns ``None`` when ``memory_key`` is falsy or no durable
    store can be resolved, so memory never breaks a run.
    """
    if not memory_key:
        return None
    return await _resolve_memory(memory_key)


async def load_history(store: typing.Any) -> list[typing.Any]:
    """Load and deserialize the prior message history from ``store``.

    Reads the JSON slot and rebuilds Pydantic AI messages via
    ``ModelMessagesTypeAdapter``. Best-effort: returns ``[]`` on any failure
    (empty/absent slot, deserialization error) so a bad memory read never
    breaks a run. Pass the result as ``message_history=`` to ``agent.run(...)``.
    """
    if store is None:
        return []
    try:
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        raw = await store.read_json.aio(_MEMORY_HISTORY_PATH, default=None)
        if not raw:
            return []
        return list(ModelMessagesTypeAdapter.validate_python(raw))
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not load Pydantic AI history from memory; continuing without prior history.")
        return []


async def save_history(store: typing.Any, result: typing.Any) -> None:
    """Serialize ``result.all_messages()`` and persist it back to ``store``.

    Writes the full history (prior + this run's new turns) to the JSON slot and
    durably saves the store. Best-effort: swallows any failure so a memory write
    never breaks a run.
    """
    if store is None:
        return
    try:
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        messages = result.all_messages()
        serialized = ModelMessagesTypeAdapter.dump_python(messages)
        await store.write_json.aio(_MEMORY_HISTORY_PATH, serialized)
        await store.save.aio()
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not save Pydantic AI history to memory; continuing.")
