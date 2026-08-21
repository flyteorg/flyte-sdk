"""Cross-run harness memory — a session archive in a keyed `MemoryStore`.

The per-run crash-resume session (see `._durable`) is keyed by the action and
backed by a `flyte.Checkpoint` — ephemeral, scoped to one run. This module keys
the same session archive by a stable `memory_key` (a user/thread id) and backs it
with the durable, cross-run `flyte.ai.agents.memory.MemoryStore`, so a later run
with the same key resumes the prior conversation: the harness sees its own JSONL
transcript for a session id it has already used, and continues it.

Because the memory store survives retries too, it subsumes crash-resume: when a
`memory_key` is given, the adapter uses this session instead of the checkpoint one.
"""

from __future__ import annotations

import pathlib
import typing
import uuid

from flyte._logging import logger
from flyteplugins.agents.core import resolve_memory

from ._durable import restore, snapshot

# Fixed namespace so a memory key maps to a stable harness session id across runs.
_SESSION_NS = uuid.UUID("2c7f4b81-9e05-5a62-b7d3-8f1a6c9e2b40")

# Path-addressed slot holding the thread's session archive inside the MemoryStore.
_SESSION_PATH = "deepseek/session.json"


def memory_session_id(memory_key: str) -> str:
    """A stable harness session id for a memory key (same across runs)."""
    return f"flyte-mem-{uuid.uuid5(_SESSION_NS, memory_key).hex}"


class MemorySession:
    """A harness session whose `session_root` is mirrored to a keyed `MemoryStore`.

    Mirrors `flyteplugins.agents.deepseek._durable.CheckpointSession`, so
    `run_agent` drives either one through the same `seed` / `persist` pair.
    """

    def __init__(self, store: typing.Any, session_id: str) -> None:
        self._store = store
        self.session_id = session_id
        self.resumed = False

    async def seed(self, session_root: pathlib.Path) -> bool:
        """Restore the thread's prior session files into `session_root`."""
        try:
            archive = await self._store.read_json.aio(_SESSION_PATH, {})
        except Exception:  # pragma: no cover - memory is best-effort, never fatal
            logger.warning("Could not load DeepSeek cross-run memory; continuing without prior history.")
            return False
        self.resumed = restore(session_root, archive or {})
        return self.resumed

    async def persist(self, session_root: pathlib.Path) -> None:
        """Persist `session_root` back to the keyed store. Never raises."""
        try:
            archive = snapshot(session_root)
            if not archive:
                return
            await self._store.write_json.aio(_SESSION_PATH, archive, actor="deepseek-harness")
            await self._store.save.aio()
        except Exception:  # pragma: no cover - memory is best-effort, never fatal
            logger.warning("Could not persist DeepSeek cross-run memory; continuing.")


async def wire_memory_session(session_root: pathlib.Path, *, memory_key: str | None) -> MemorySession | None:
    """Build a memory-backed session for `memory_key`, seeded from prior runs.

    Returns `None` when memory is off or unavailable (no key, no durable store),
    in which case the caller falls back to the per-run durable session. Never raises.
    """
    if not memory_key:
        return None
    try:
        store = await resolve_memory(memory_key)
        if store is None:
            return None
        session = MemorySession(store, memory_session_id(memory_key))
        await session.seed(session_root)
        return session
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not wire DeepSeek cross-run memory for key %r; continuing without it.", memory_key)
        return None
