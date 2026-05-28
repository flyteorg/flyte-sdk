"""Dir-backed memory for :class:`flyte.ai.agents.Agent`.

This module implements :class:`MemoryStore`, a directory-backed store that
combines a managed conversation transcript with path-addressed artifact files,
plus optional auditing, read-only prefixes, version snapshots, and optimistic
concurrency. It is intentionally decoupled from the agent loop so non-agent
code (sleep cycles, promotion tasks, dashboards) can use the same store with
the same enforcement.

Design influences include the Claude-style "many small files addressed by
path, with audit + version history" pattern.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence, cast

from flyte.io import Dir

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------------

#: Hard-coded path of the conversation transcript inside the memory root.
MESSAGES_PATH = "messages.json"

#: Path of the append-only audit log inside the memory root.
_AUDIT_LOG_PATH = "audit/log.jsonl"

#: Prefixes managed internally by :class:`MemoryStore`. Direct writes through
#: :meth:`MemoryStore.write_text` / :meth:`MemoryStore.write_json` to these
#: prefixes are rejected; they are also excluded from :meth:`list_paths`.
_INTERNAL_PREFIXES: tuple[str, ...] = ("audit/", "meta/", "versions/")


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class MemoryStoreError(RuntimeError):
    """Base class for :class:`MemoryStore` errors."""


class AccessDenied(MemoryStoreError):
    """Raised when a write targets a read-only or reserved prefix."""


class ConcurrencyError(MemoryStoreError):
    """Raised when an ``expected_sha`` precondition does not match the current state."""

    def __init__(self, path: str, expected_sha: str, actual_sha: str):
        super().__init__(f"ConcurrencyError for {path!r}: expected_sha={expected_sha} actual_sha={actual_sha}")
        self.path = path
        self.expected_sha = expected_sha
        self.actual_sha = actual_sha


# ----------------------------------------------------------------------------
# Metadata sidecar
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryMeta:
    """Per-file metadata sidecar (sha256, actor, timestamp, …) for a memory entry."""

    path: str
    sha256: str
    updated_at: str
    updated_by: str
    reason: str
    bytes: int


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ensure_relative_posix(path: str) -> str:
    """Normalize and validate a memory-relative path.

    Rejects absolute paths, empty paths, and any traversal segment (``..``).
    Returns a forward-slash-joined string regardless of host OS.
    """
    p = pathlib.Path(path)
    if p.is_absolute() or str(path).startswith("/"):
        raise MemoryStoreError(f"Path must be relative, got {path!r}")
    parts: list[str] = []
    for part in p.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise MemoryStoreError(f"Path traversal is not allowed, got {path!r}")
        parts.append(part)
    if not parts:
        raise MemoryStoreError("Empty path is not allowed")
    return "/".join(parts)


# ----------------------------------------------------------------------------
# MemoryStore
# ----------------------------------------------------------------------------


@dataclass
class MemoryStore:
    """Conversation transcript + path-addressed artifact memory backed by :class:`flyte.io.Dir`.

    The construct combines two complementary stores:

    - ``messages``: the live LLM conversation transcript (managed by
      :class:`~flyte.ai.agents.Agent`; mutate via :meth:`append` /
      :meth:`extend` only).
    - **Path-addressed files** under a working directory ``root``. Use
      :meth:`write_text` / :meth:`read_text` / :meth:`write_json` /
      :meth:`read_json` / :meth:`list_paths` for arbitrary named blobs that
      should round-trip through Flyte object storage.

    Persistence is :class:`flyte.io.Dir`-only: call :meth:`save_to_dir` at the
    end of a run and :meth:`load_from_dir` at the start of the next.

    The on-disk layout under ``root`` looks like::

        <root>/messages.json                            # transcript
        <root>/<your/path>.{txt,json,…}                 # path-addressed entries
        <root>/meta/<sanitized_path>.json               # per-entry metadata
        <root>/audit/log.jsonl                          # opt-in audit trail
        <root>/versions/<sanitized_path>/<ts>_<sha>.txt # opt-in version history

    Optional capabilities (off-by-default unless noted):

    - ``read_only_prefixes``: block direct writes into one or more prefixes
      (e.g. ``("memory/",)``). Useful when the agent must stage proposals
      under ``user/`` and a separate trusted pipeline (sleep cycle, reviewer)
      promotes them.
    - ``audit`` *(default: True)*: append every successful write to
      ``audit/log.jsonl``. Cheap and easy to disable.
    - ``keep_versions``: snapshot every successful write under
      ``versions/<sanitized_path>/<ts>_<sha>.txt`` for full history (≈ 2x
      storage on every mutation).

    Optimistic concurrency is supported via the ``expected_sha=`` argument on
    :meth:`write_text` / :meth:`write_json`; mismatches raise
    :class:`ConcurrencyError`.

    Parameters
    ----------
    messages:
        Pre-existing conversation transcript. Defaults to empty.
    root:
        Local working directory backing the store. When omitted, a fresh
        temporary directory is created. When pointing at an existing directory
        that contains ``messages.json``, the transcript is auto-loaded.
    read_only_prefixes:
        Prefixes that direct writes are not permitted to target.
    audit:
        Enable the append-only audit log.
    keep_versions:
        Snapshot every successful write under ``versions/``.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    root: pathlib.Path | str | None = None
    read_only_prefixes: tuple[str, ...] = ()
    audit: bool = True
    keep_versions: bool = False

    def __post_init__(self) -> None:
        explicit_root = self.root is not None
        if explicit_root:
            resolved = pathlib.Path(cast("pathlib.Path | str", self.root))
        else:
            resolved = pathlib.Path(tempfile.mkdtemp(prefix="flyte_agent_mem_"))
        resolved.mkdir(parents=True, exist_ok=True)
        self.root = resolved

        # Auto-load messages.json when pointing at an existing directory and
        # the user did not pre-seed messages explicitly.
        if explicit_root and not self.messages:
            mp = resolved / MESSAGES_PATH
            if mp.exists():
                try:
                    self.messages = json.loads(mp.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    logger.warning("messages.json under %s was not valid JSON; ignoring.", resolved)

    @property
    def _root(self) -> pathlib.Path:
        """Return :attr:`root` narrowed to :class:`pathlib.Path` (always set after ``__post_init__``)."""
        return cast(pathlib.Path, self.root)

    # ------------------------------------------------------------------
    # Transcript helpers
    # ------------------------------------------------------------------

    def append(self, message: dict[str, Any]) -> None:
        """Append a single message to the conversation transcript."""
        self.messages.append(message)

    def extend(self, messages: Sequence[dict[str, Any]]) -> None:
        """Append a sequence of messages to the conversation transcript."""
        self.messages.extend(messages)

    # ------------------------------------------------------------------
    # Path-addressed I/O
    # ------------------------------------------------------------------

    def _abs(self, rel_path: str) -> pathlib.Path:
        rel = _ensure_relative_posix(rel_path)
        return self._root / rel

    def _meta_path(self, rel_path: str) -> pathlib.Path:
        rel = _ensure_relative_posix(rel_path)
        return self._root / "meta" / (rel.replace("/", "__") + ".json")

    def _versions_dir(self, rel_path: str) -> pathlib.Path:
        rel = _ensure_relative_posix(rel_path)
        return self._root / "versions" / rel.replace("/", "__")

    def _audit_path(self) -> pathlib.Path:
        return self._root / _AUDIT_LOG_PATH

    def _assert_can_write(self, rel: str) -> None:
        if rel == MESSAGES_PATH:
            raise AccessDenied("messages.json is managed by MemoryStore; mutate via append() / extend().")
        if any(rel.startswith(p) for p in _INTERNAL_PREFIXES):
            raise AccessDenied(f"Writes to {rel!r} are not allowed (reserved prefix)")
        if any(rel.startswith(p) for p in self.read_only_prefixes):
            raise AccessDenied(f"Writes to {rel!r} are not allowed (read-only prefix)")

    def exists(self, rel_path: str) -> bool:
        """Return ``True`` if a memory file exists at ``rel_path``."""
        return self._abs(rel_path).exists()

    def read_text(self, rel_path: str, default: str = "") -> str:
        """Return the UTF-8 contents of ``rel_path`` (or ``default`` if missing)."""
        try:
            return self._abs(rel_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return default

    def read_json(self, rel_path: str, default: Any = None) -> Any:
        """Return the JSON-decoded contents of ``rel_path`` (or ``default`` if empty/missing)."""
        text = self.read_text(rel_path, default="")
        if not text.strip():
            return default
        return json.loads(text)

    def list_paths(self, prefix: str = "") -> list[str]:
        """List memory file paths under ``prefix`` (POSIX-relative).

        Internal bookkeeping (``audit/``, ``meta/``, ``versions/``) and the
        conversation transcript (``messages.json``) are excluded.
        """
        prefix_norm = "" if not prefix else _ensure_relative_posix(prefix)
        base = self._root / prefix_norm if prefix_norm else self._root
        if not base.exists():
            return []
        out: list[str] = []
        for p in base.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(self._root).as_posix()
            if rel == MESSAGES_PATH:
                continue
            if any(rel.startswith(internal) for internal in _INTERNAL_PREFIXES):
                continue
            out.append(rel)
        out.sort()
        return out

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_meta(self, rel_path: str) -> MemoryMeta | None:
        """Return the :class:`MemoryMeta` sidecar for ``rel_path`` if present."""
        mp = self._meta_path(rel_path)
        if not mp.exists():
            return None
        try:
            raw = json.loads(mp.read_text(encoding="utf-8"))
            return MemoryMeta(**raw)
        except Exception as exc:
            raise MemoryStoreError(f"Failed to read meta for {rel_path!r}: {exc}") from exc

    def current_sha(self, rel_path: str) -> str:
        """Return the sha256 of ``rel_path`` (empty string if it does not exist)."""
        meta = self.get_meta(rel_path)
        if meta is not None:
            return meta.sha256
        if not self.exists(rel_path):
            return ""
        return _sha256_bytes(self._abs(rel_path).read_bytes())

    # ------------------------------------------------------------------
    # Writes (audited + optionally versioned)
    # ------------------------------------------------------------------

    def write_text(
        self,
        rel_path: str,
        content: str,
        *,
        actor: str = "agent",
        reason: str = "",
        expected_sha: str | None = None,
    ) -> MemoryMeta:
        """Write ``content`` to ``rel_path`` with optional concurrency + audit + versioning.

        Args:
            rel_path: Destination path, relative to the memory root. Must not
                escape the root and must not target a reserved or read-only
                prefix.
            content: UTF-8 string to write.
            actor: Free-form identifier of the writer (typically the tool or
                agent name). Recorded in the audit log + metadata sidecar.
            reason: Optional human-readable explanation.
            expected_sha: When provided, the write succeeds only if the
                current sha256 of ``rel_path`` matches. Mismatches raise
                :class:`ConcurrencyError`.

        Returns:
            The :class:`MemoryMeta` describing the new content.
        """
        rel = _ensure_relative_posix(rel_path)
        self._assert_can_write(rel)

        p = self._abs(rel)
        old_sha = self.current_sha(rel)
        if expected_sha is not None and expected_sha != old_sha:
            raise ConcurrencyError(rel, expected_sha=expected_sha, actual_sha=old_sha)

        p.parent.mkdir(parents=True, exist_ok=True)
        new_sha = _sha256_text(content)
        p.write_text(content, encoding="utf-8")

        version_file_rel = ""
        if self.keep_versions:
            vdir = self._versions_dir(rel)
            vdir.mkdir(parents=True, exist_ok=True)
            ts = _utc_ts().replace(":", "-")
            vpath = vdir / f"{ts}_{new_sha}.txt"
            if vpath.exists():
                # Defend against same-second collisions.
                salt = _sha256_bytes(os.urandom(8))[:8]
                vpath = vdir / f"{ts}_{new_sha}_{salt}.txt"
            vpath.write_text(content, encoding="utf-8")
            version_file_rel = vpath.relative_to(self._root).as_posix()

        meta = MemoryMeta(
            path=rel,
            sha256=new_sha,
            updated_at=_utc_ts(),
            updated_by=actor,
            reason=reason,
            bytes=len(content.encode("utf-8")),
        )
        meta_p = self._meta_path(rel)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")

        if self.audit:
            self._append_audit(
                {
                    "ts": meta.updated_at,
                    "op": "create" if not old_sha else "update",
                    "path": rel,
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                    "actor": actor,
                    "reason": reason,
                    "version_file": version_file_rel,
                }
            )
        return meta

    def write_json(
        self,
        rel_path: str,
        obj: Any,
        *,
        actor: str = "agent",
        reason: str = "",
        expected_sha: str | None = None,
    ) -> MemoryMeta:
        """JSON-encode ``obj`` and write it via :meth:`write_text`."""
        content = json.dumps(obj, indent=2, sort_keys=True, default=str)
        return self.write_text(rel_path, content, actor=actor, reason=reason, expected_sha=expected_sha)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _append_audit(self, event: dict[str, Any]) -> None:
        ap = self._audit_path()
        ap.parent.mkdir(parents=True, exist_ok=True)
        with ap.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    def audit_tail(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the last ``n`` audit events (most recent last).

        Returns an empty list when auditing is disabled or the log does not
        exist yet.
        """
        ap = self._audit_path()
        if not ap.exists():
            return []
        lines = ap.read_text(encoding="utf-8").splitlines()
        tail = lines[-n:] if n > 0 else lines
        out: list[dict[str, Any]] = []
        for raw_line in tail:
            line = raw_line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------
    # Sync to / from flyte.io.Dir
    # ------------------------------------------------------------------

    def _flush_messages(self) -> None:
        """Persist the live transcript to ``messages.json`` under the working root."""
        p = self._root / MESSAGES_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.messages, default=str, indent=2), encoding="utf-8")

    async def save_to_dir(self, remote_destination: str | None = None) -> Dir:
        """Serialize this memory to a remote directory.

        Flushes the conversation transcript to ``messages.json`` under the
        working root, then uploads the entire root via
        :meth:`flyte.io.Dir.from_local`. Audit log, metadata sidecars, and any
        version snapshots are uploaded alongside the live memory files.
        """
        self._flush_messages()
        return await Dir.from_local(str(self._root), remote_destination=remote_destination)

    @classmethod
    async def load_from_dir(
        cls,
        dir: Dir,
        *,
        read_only_prefixes: tuple[str, ...] = (),
        audit: bool = True,
        keep_versions: bool = False,
    ) -> "MemoryStore":
        """Restore memory previously written by :meth:`save_to_dir`.

        Downloads ``dir`` into a fresh local working directory and
        re-hydrates the conversation transcript. Subsequent writes are
        appended in the local copy and re-uploaded by the next
        :meth:`save_to_dir` call.
        """
        local_root = pathlib.Path(tempfile.mkdtemp(prefix="flyte_agent_mem_"))
        await dir.download(local_path=str(local_root))
        return cls(
            root=local_root,
            read_only_prefixes=read_only_prefixes,
            audit=audit,
            keep_versions=keep_versions,
        )
