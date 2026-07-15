"""Cross-run Hermes memory — a thin bridge over Flyte's keyed ``MemoryStore``.

Hermes keeps conversation state in-memory. This module persists the conversation
transcript to a durable, keyed :class:`~flyte.ai.agents.memory.MemoryStore` (an
object-store slot addressed by ``memory_key``) so a later run with the same key
continues the conversation — across workers and restarts.

The transcript is a plain list of ``{"role": ..., "content": ...}`` turns, stored
via ``read_json`` / ``write_json``.
"""

from __future__ import annotations

import typing

from flyte._logging import logger
from flyteplugins.agents.core import resolve_memory as _resolve_memory

# Path-addressed slot holding the conversation transcript inside the MemoryStore.
_MEMORY_HISTORY_PATH = "hermes/history.json"


async def resolve_memory(memory_key: str | None) -> typing.Any | None:
    """Resolve a keyed MemoryStore for Hermes cross-run memory, or ``None``.

    Best-effort: returns ``None`` when ``memory_key`` is falsy or no durable
    store can be resolved, so memory never breaks a run.
    """
    if not memory_key:
        return None
    return await _resolve_memory(memory_key)


async def load_transcript(store: typing.Any) -> list[dict[str, typing.Any]]:
    """Load the prior conversation transcript (empty list if none)."""
    if store is None:
        return []
    try:
        return list(await store.read_json.aio(_MEMORY_HISTORY_PATH, []))
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not load Hermes memory; continuing without prior history.")
        return []


async def save_transcript(store: typing.Any, transcript: typing.Sequence[dict[str, typing.Any]]) -> None:
    """Persist the conversation transcript back to the keyed store."""
    if store is None:
        return
    try:
        await store.write_json.aio(_MEMORY_HISTORY_PATH, list(transcript))
        await store.save.aio()
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not persist Hermes memory; continuing.")
