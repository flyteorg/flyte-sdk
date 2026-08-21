"""Durable harness sessions — make `durable=True` real via the SDK's session store.

DeepSeek Harness runs the model loop inside its runtime subprocess, so there is
no in-process model-call seam to wrap in `flyte.trace` for per-turn replay (the
same constraint the Claude adapter has). What the harness does own is session
persistence: every session is written as JSONL under `session_root`
(`DSH_SESSION_ROOT`), and prompting an existing `session_id` continues that
session rather than starting a new one.

So durability here is session resume, backed by `flyte.Checkpoint` — the native,
retry-surviving durable prefix the runtime hands each task (`save` writes this
attempt's blob, `load` restores the previous attempt's):

- the session id is derived deterministically from the task's `ActionID`, so
  every retry of the same action targets the same session;
- before the harness starts, a previous attempt's session files are restored
  into `session_root`; after each run they are snapshotted back.

A crashed attempt therefore resumes its conversation instead of restarting it —
without us owning the loop. Tool durability, retries and caching apply either way.
"""

from __future__ import annotations

import json
import pathlib
import typing
import uuid

from flyte._logging import logger

# Fixed namespace so derived session ids are stable across processes and retries.
_SESSION_NS = uuid.UUID("6f2d9c14-8b7e-5a3f-9d21-4c8e5b7a0f13")
_PAYLOAD = "payload"

# A session transcript is JSONL; anything larger than this is left out of the
# snapshot rather than pushed through the checkpoint blob.
_MAX_FILE_BYTES = 32 * 1024 * 1024


def deterministic_session_id(task_context: typing.Any) -> str:
    """A stable harness session id for the current action (same across retries).

    Uses `task_action` when present (it stays pinned to the real running task even
    inside a `@trace` pseudo-action), falling back to `action`.
    """
    action = getattr(task_context, "task_action", None) or task_context.action
    seed = f"{action.run_name}/{action.name}"
    return f"flyte-{uuid.uuid5(_SESSION_NS, seed).hex}"


def snapshot(session_root: pathlib.Path) -> dict[str, str]:
    """Read a `session_root` tree into a `{relative path: text}` mapping.

    Best-effort: unreadable, binary or oversized entries are skipped, so a
    snapshot never fails the run that produced it.
    """
    archive: dict[str, str] = {}
    if not session_root.is_dir():
        return archive
    for path in sorted(session_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                logger.warning("DeepSeek session file %s is too large to persist; skipping.", path.name)
                continue
            archive[str(path.relative_to(session_root))] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.debug("DeepSeek session file %s could not be read for persistence; skipping.", path, exc_info=True)
    return archive


def restore(session_root: pathlib.Path, archive: typing.Mapping[str, str]) -> bool:
    """Materialize an archive back into `session_root`; returns whether anything was written."""
    if not archive:
        return False
    written = False
    for relative, text in archive.items():
        path = session_root / relative
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            written = True
        except OSError:  # pragma: no cover - unwritable session root
            logger.debug("Could not restore DeepSeek session file %s.", relative, exc_info=True)
    return written


def _read_payload(local: typing.Any) -> dict | None:
    """Parse the checkpoint blob written by a previous attempt, or `None`.

    Kept sync (local, already-downloaded file IO) so the async wiring stays free of
    blocking `pathlib` calls.
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


class CheckpointSession:
    """A harness session whose `session_root` is mirrored to a `flyte.Checkpoint`.

    `seed` restores the previous attempt's transcript before the harness starts;
    `persist` snapshots it back after each run. `resumed` reports whether this
    attempt actually picked up a prior conversation.
    """

    def __init__(self, checkpoint: typing.Any, session_id: str) -> None:
        self._ckpt = checkpoint
        self.session_id = session_id
        self.resumed = False

    async def seed(self, session_root: pathlib.Path) -> bool:
        """Restore a prior attempt's session files into `session_root`."""
        local = await self._ckpt.load()
        if local is None:
            return False
        archive = _read_payload(local)
        if not archive:
            return False
        self.resumed = restore(session_root, archive)
        return self.resumed

    async def persist(self, session_root: pathlib.Path) -> None:
        """Snapshot `session_root` into the checkpoint. Never raises."""
        try:
            archive = snapshot(session_root)
            if archive:
                await self._ckpt.save(json.dumps(archive).encode("utf-8"))
        except Exception:  # pragma: no cover - durability must never break the run
            logger.warning("Could not persist the DeepSeek session checkpoint; continuing.")


async def wire_durable_session(session_root: pathlib.Path, *, durable: bool) -> CheckpointSession | None:
    """Build a checkpoint-backed session for this action, seeded from any prior attempt.

    Returns `None` when durability is off or unavailable (no task context, no
    checkpoint — e.g. a local run), in which case the caller runs with a fresh
    session. Never raises: durability is best-effort and must not break a run.
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
        session = CheckpointSession(checkpoint, deterministic_session_id(task_context))
        await session.seed(session_root)
        return session
    except Exception:  # pragma: no cover - durability must never break the run
        logger.warning("Could not wire a durable DeepSeek session; continuing without resume.")
        return None
