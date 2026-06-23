"""Durable Claude sessions — make ``durable=True`` real via the SDK's session store.

The Claude Agent SDK runs the model loop inside the Claude Code CLI subprocess, so
there is no in-process model-call seam to wrap in ``flyte.trace`` for per-turn replay.
Instead we use the SDK's own **session mirror + resume**: the CLI mirrors the running
transcript to a ``SessionStore`` we provide, and on a retry it resumes from that store
rather than starting over.

We back that store with :class:`flyte.Checkpoint` — the native, retry-surviving
durable prefix the runtime hands each task (``save`` writes this attempt's blob,
``load`` restores the previous attempt's). So a crashed attempt's conversation is
restored on the next attempt instead of replayed from scratch, without us owning
the loop.

Mapping to Flyte primitives:
- **session id** is derived deterministically from the task's ``ActionID``, so every
  retry of the same action targets the same session;
- **persistence** is ``flyte.Checkpoint`` — durable across container restarts.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import typing
import uuid

from flyte._logging import logger

# Fixed namespace so derived session ids are stable across processes and retries.
_SESSION_NS = uuid.UUID("4e1b8a2c-6c2e-5f7a-9b3d-2a1f0c7d4e55")
_PAYLOAD = "payload"


def deterministic_session_id(task_context: typing.Any) -> str:
    """A stable, valid-UUID session id for the current action (same across retries).

    Uses ``task_action`` when present (it stays pinned to the real running task even
    inside a ``@trace`` pseudo-action), falling back to ``action``.
    """
    action = getattr(task_context, "task_action", None) or task_context.action
    seed = f"{action.run_name}/{action.name}"
    return str(uuid.uuid5(_SESSION_NS, seed))


def _skey(key: typing.Mapping[str, typing.Any]) -> str:
    """Flatten a Claude ``SessionKey`` to a single storage key.

    Keyed by ``session_id`` (+ optional subagent ``subpath``); ``project_key`` is
    intentionally ignored — our session ids are already globally unique per action.
    """
    return f"{key['session_id']}::{key.get('subpath') or ''}"


def _read_payload(local: typing.Any) -> dict | None:
    """Parse the checkpoint blob written by a previous attempt, or ``None``.

    Kept sync (local, already-downloaded file IO) so the async store stays free of
    blocking ``pathlib`` calls.
    """
    payload = pathlib.Path(local)
    if payload.is_dir():
        payload = payload / _PAYLOAD
    if not payload.is_file():
        return None
    try:
        return json.loads(payload.read_text())
    except (ValueError, OSError):
        return None


class CheckpointSessionStore:
    """A duck-typed Claude ``SessionStore`` persisted via :class:`flyte.Checkpoint`.

    The SDK requires only ``append`` and ``load`` and probes for the optional methods
    (``list_sessions``/``delete``/...), so we deliberately omit them. The whole store —
    every session/subagent transcript seen this run — is serialized to a single
    checkpoint blob; the SDK materializes it back into the CLI on resume.
    """

    def __init__(self, checkpoint: typing.Any) -> None:
        self._ckpt = checkpoint
        self._state: dict[str, list[dict]] = {}
        self._seen: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def seed_from_prev(self) -> bool:
        """Restore the previous attempt's checkpoint into memory.

        Returns ``True`` if a prior attempt's session existed (i.e. this is a retry
        with state to resume), ``False`` otherwise.
        """
        local = await self._ckpt.load()
        if local is None:
            return False
        data = _read_payload(local)
        if not data:
            return False
        self._state = {k: list(v) for k, v in data.items()}
        self._seen = {k: {e["uuid"] for e in v if "uuid" in e} for k, v in self._state.items()}
        return bool(self._state)

    async def _persist(self) -> None:
        await self._ckpt.save(json.dumps(self._state).encode("utf-8"))

    async def append(self, key: typing.Mapping[str, typing.Any], entries: list[dict]) -> None:
        """Mirror a batch of transcript entries, then persist to the checkpoint."""
        skey = _skey(key)
        async with self._lock:
            buf = self._state.setdefault(skey, [])
            seen = self._seen.setdefault(skey, set())
            changed = False
            for entry in entries:
                uid = entry.get("uuid")
                if uid is not None and uid in seen:
                    continue  # idempotent: the mirror may re-send entries on retry
                buf.append(entry)
                changed = True
                if uid is not None:
                    seen.add(uid)
            if changed:
                await self._persist()

    async def load(self, key: typing.Mapping[str, typing.Any]) -> list[dict] | None:
        """Return the full transcript for ``key`` (for resume), or ``None`` if unseen."""
        async with self._lock:
            buf = self._state.get(_skey(key))
            return list(buf) if buf else None


async def wire_durable_session(options: typing.Any, *, durable: bool) -> CheckpointSessionStore | None:
    """Attach a resume-backed session store to ``options`` when durable and able.

    First attempt pins a deterministic ``session_id``; a retry (whose previous
    checkpoint exists) sets ``resume`` to that same id and seeds the store from the
    prior attempt — so completed turns and tool results are restored from the
    checkpoint instead of recomputed. Returns the store (or ``None`` when durability
    is off / unavailable). Never raises: durability is best-effort and must not break
    a run.
    """
    if not durable:
        return None
    try:
        import flyte

        task_context = flyte.ctx()
        if task_context is None:
            return None
        checkpoint = task_context.checkpoint
        if checkpoint is None:
            return None
        store = CheckpointSessionStore(checkpoint)
        had_prior = await store.seed_from_prev()
        session_id = deterministic_session_id(task_context)
        if had_prior:
            options.resume = session_id
        else:
            options.session_id = session_id
        options.session_store = store
        return store
    except Exception:  # pragma: no cover - durability must never break the run
        logger.warning("Could not wire a durable Claude session; continuing without resume.")
        return None
