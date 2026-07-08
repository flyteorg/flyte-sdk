"""Cross-run Claude memory — a ``SessionStore`` backed by a keyed ``MemoryStore``.

Claude's session resume materializes a ``SessionStore``'s transcript into the
CLI. The per-run crash-resume store (see :mod:`._durable`) is keyed by the action
and backed by a ``flyte.Checkpoint`` (ephemeral, per-run). This store is keyed by a
stable ``memory_key`` (a user/thread id) and backed by a durable, cross-run
``MemoryStore`` — so a later run with the same key resumes the prior
conversation.

Because the memory store survives retries too, it subsumes crash-resume: when a
``memory_key`` is given, the adapter uses this store instead of the checkpoint one.
"""

from __future__ import annotations

import asyncio
import typing
import uuid

from flyte._logging import logger
from flyteplugins.agents.core import resolve_memory

from ._durable import _skey

# Fixed namespace so a memory key maps to a stable Claude session id across runs.
_SESSION_NS = uuid.UUID("9a7c1e30-2b44-5d96-8a1f-3c6b2e0d7f44")

# Path-addressed slot holding the thread's Claude transcript inside the MemoryStore.
_TRANSCRIPT_PATH = "claude/transcript.json"


def memory_session_id(memory_key: str) -> str:
    """A stable, valid-UUID Claude session id for a memory key (same across runs)."""
    return str(uuid.uuid5(_SESSION_NS, memory_key))


class MemorySessionStore:
    """Duck-typed Claude ``SessionStore`` persisted in a keyed ``MemoryStore``.

    Only ``append``/``load`` are required by the SDK. The whole transcript (keyed by
    the SDK's session/subagent key) is stored as one path-addressed JSON entry; the
    SDK materializes it back into the CLI on resume.
    """

    def __init__(self, store: typing.Any) -> None:
        self._store = store
        self._state: dict[str, list[dict]] = {}
        self._seen: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def seed(self) -> bool:
        """Load the thread's prior transcript into memory; returns whether any existed."""
        data = await self._store.read_json.aio(_TRANSCRIPT_PATH)
        if not data:
            return False
        self._state = {k: list(v) for k, v in data.items()}
        self._seen = {k: {e["uuid"] for e in v if "uuid" in e} for k, v in self._state.items()}
        return bool(self._state)

    async def _persist(self) -> None:
        await self._store.write_json.aio(_TRANSCRIPT_PATH, self._state, actor="claude-agent")
        await self._store.save.aio()

    async def append(self, key: typing.Mapping[str, typing.Any], entries: list[dict]) -> None:
        skey = _skey(key)
        async with self._lock:
            buf = self._state.setdefault(skey, [])
            seen = self._seen.setdefault(skey, set())
            changed = False
            for entry in entries:
                uid = entry.get("uuid")
                if uid is not None and uid in seen:
                    continue  # idempotent: the mirror may re-send entries
                buf.append(entry)
                changed = True
                if uid is not None:
                    seen.add(uid)
            if changed:
                await self._persist()

    async def load(self, key: typing.Mapping[str, typing.Any]) -> list[dict] | None:
        async with self._lock:
            buf = self._state.get(_skey(key))
            return list(buf) if buf else None

    async def list_subkeys(self, key: typing.Mapping[str, typing.Any]) -> list[str]:
        """List the subagent subpaths under a session so resume restores them too.

        Without this the SDK only materializes the main transcript on resume; with it,
        subagent transcripts (mirrored under ``subpath`` keys) come back as well.
        Scoped to ``session_id``; the main transcript (empty subpath) is excluded.
        """
        prefix = f"{key['session_id']}::"
        async with self._lock:
            return [skey[len(prefix) :] for skey in self._state if skey.startswith(prefix) and skey != prefix]


async def wire_memory_session(options: typing.Any, *, memory_key: str | None) -> MemorySessionStore | None:
    """Attach a cross-run, memory-backed resume to ``options`` for ``memory_key``.

    First run for the key pins a deterministic ``session_id``; a later run (whose
    transcript already exists) sets ``resume`` and seeds from the store — so the
    conversation continues. Returns the store, or ``None`` when memory is off /
    unavailable. Never raises.
    """
    if not memory_key:
        return None
    try:
        store = await resolve_memory(memory_key)
        if store is None:
            return None
        session = MemorySessionStore(store)
        had_prior = await session.seed()
        session_id = memory_session_id(memory_key)
        if had_prior:
            options.resume = session_id
        else:
            options.session_id = session_id
        options.session_store = session
        return session
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not wire Claude cross-run memory for key %r; continuing without it.", memory_key)
        return None
