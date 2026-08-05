"""Cross-run agent memory — a thin handle over Flyte's keyed ``MemoryStore``.

"Agent memory" = durable state addressable by a stable ``memory_key`` (a user or
thread id), shared across runs and workers. We don't reinvent it: Flyte already
ships a keyed, blob-store-backed store (``flyte.ai.agents.memory.MemoryStore``)
that resolves a deterministic remote path from the key and the active run context
— stripping the per-run scratch so two runs with the same key share one store,
and encapsulating all the storage details.

It is decoupled from the agent loop - importing it does not pull in any agent
harness — so adapters use it purely as a durable store. Each adapter then maps its
SDK's own conversation state onto the returned store.

The store carries two complementary surfaces, so one ``memory_key`` covers both
kinds of memory:

- ``messages`` (``append``/``extend``) — the conversation transcript;
- path-addressed ``read_json``/``write_json``/``list_paths`` — durable named facts
  (the ``remember`` / ``recall`` substrate), with audit + version history.
"""

from __future__ import annotations

import typing

from flyte._logging import logger

if typing.TYPE_CHECKING:
    from flyte.ai.agents.memory import MemoryStore


async def resolve_memory(memory_key: str | None, *, audit: bool = True) -> "MemoryStore | None":
    """Open (or create) the keyed agent-memory store, or ``None`` if unavailable.

    Best-effort: returns ``None`` (and logs) when ``memory_key`` is falsy or no
    durable store can be resolved — e.g. no Flyte context/org, or an invalid key —
    so memory never breaks a run. Call from inside an ``@env.task``; the store's
    remote path is derived from the run context and is stable across runs for the
    same key.

    Args:
        memory_key: A stable single-segment id (a user/thread id). ``None`` or
            empty disables memory.
        audit: Keep the store's append-only audit log (the ``MemoryStore`` default).
    """
    if not memory_key:
        return None
    try:
        from flyte.ai.agents.memory import MemoryStore

        return await MemoryStore.get_or_create.aio(key=memory_key, audit=audit)
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not open agent memory for key %r; continuing without memory.", memory_key)
        return None
