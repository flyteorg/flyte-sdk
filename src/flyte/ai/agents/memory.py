"""Dir-backed memory for :class:`flyte.ai.agents.Agent`.

This module implements :class:`MemoryStore`, a directory-backed store that
combines a managed conversation transcript with path-addressed artifact files,
plus optional auditing, read-only prefixes, version snapshots, and optimistic
concurrency. It is intentionally decoupled from the agent loop so non-agent
code (sleep cycles, promotion tasks, dashboards) can use the same store with
the same enforcement.

Design influences include the Claude-style "many small files addressed by
path, with audit + version history" pattern.

The path-addressed I/O methods (``read_text``, ``read_json``, ``get_meta``,
``current_sha``, ``write_text``, ``write_json``) — like the keyed-store methods
``create`` / ``get_or_create`` / ``save`` — are :func:`~flyte.syncify.syncify`-wrapped:
call them synchronously (``memory.read_text(...)``) or await the ``.aio(...)``
companion in async code. The remaining helpers (``flush_messages``, ``audit_tail``)
stay async-by-default with explicit ``*_sync`` companions, and ``list_paths`` is
synchronous.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import shutil
import tempfile
import weakref
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence, cast
from urllib.parse import quote, urlparse

from mashumaro.types import SerializableType

import flyte.storage as storage
from flyte._context import internal_ctx
from flyte.io import Dir
from flyte.models import PathRewrite
from flyte.syncify import syncify

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------------

#: Hard-coded path of the conversation transcript inside the memory root.
MESSAGES_PATH = "messages.json"

#: Path of the append-only audit log inside the memory root.
_AUDIT_LOG_PATH = "audit/log.jsonl"

#: Top-level directory names managed internally by :class:`MemoryStore`.
#: Direct writes through :meth:`MemoryStore.write_text` /
#: :meth:`MemoryStore.write_json` to these prefixes are rejected; they are
#: also excluded from :meth:`list_paths`.
_INTERNAL_DIRS: frozenset[str] = frozenset({"audit", "meta", "versions"})

#: ``"<dir>/"`` prefixes derived from :data:`_INTERNAL_DIRS`. Used for
#: relative-path comparisons (``rel.startswith(prefix)``).
_INTERNAL_PREFIXES: tuple[str, ...] = tuple(f"{d}/" for d in sorted(_INTERNAL_DIRS))

#: Chunk size for streaming sha256 of files on disk.
_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB

#: Namespace fragments + schema version that locate keyed MemoryStores under the
#: active storage root: ``{storage_root}/agents/memory-store/v0/...``. Bump
#: :data:`_MEMORY_SCHEMA_VERSION` when the persisted layout changes in a way that
#: older stores cannot be read by newer code (or vice versa).
_MEMORY_NAMESPACE: tuple[str, ...] = ("agents", "memory-store")
_MEMORY_SCHEMA_VERSION = "v0"

#: Per-run scratch segment that ``flyte.run`` appends to raw-data paths as
#: ``{run_base_dir}/rd/{run_id}``. Stripped from *local* storage roots so that
#: repeated runs sharing a memory key resolve to the same store.
_RAW_DATA_SCRATCH_SEGMENT = "rd"

#: Directory name of the default *local* raw-data root laid down by ``flyte.run``
#: as ``/tmp/flyte/raw_data/{action_name}`` (see ``flyte._run``). Unlike the remote
#: layout there is no ``/rd/{run_id}`` tail to strip, so the per-run ``{action_name}``
#: segment is stripped by anchoring at this directory instead.
_LOCAL_RAW_DATA_DIR = "raw_data"


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


def _sha256_file(path: pathlib.Path) -> str:
    """Stream-hash a file in 1 MiB chunks to avoid loading large blobs into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _encode_filename(rel: str) -> str:
    """Reversibly encode a relative POSIX path into a single filename.

    URL-encodes ``/`` to ``%2F`` (and any other special chars to their
    ``%XX`` form) so that ``"a/b"`` and ``"a__b"`` no longer collide on
    metadata sidecars / version directories.
    """
    return quote(rel, safe="")


def _ensure_namespace_segment(value: str, *, name: str) -> str:
    """Validate a blob-store namespace segment.

    Memory-store keys become durable object-store prefixes, so they must be
    stable single path segments rather than relative paths. This keeps
    ``{memory_key}`` from escaping or reshaping the managed memory namespace
    (see :data:`_MEMORY_NAMESPACE`).
    """
    if not value:
        raise MemoryStoreError(f"{name} must not be empty")
    if value in (".", "..") or "/" in value or "\\" in value:
        raise MemoryStoreError(f"{name} must be a single path segment, got {value!r}")
    return value


def _join_remote_path(*parts: str) -> str:
    """Join URI / path fragments with POSIX separators.

    The leading fragment keeps its scheme and root intact (so ``s3://bucket``
    survives), while every subsequent fragment is stripped of surrounding
    slashes before joining. Empty fragments are ignored.
    """
    segments = [p for p in parts if p]
    if not segments:
        return ""
    head, *rest = segments
    return "/".join([head.rstrip("/"), *(p.strip("/") for p in rest)])


def _normalize_raw_data_path(raw_data_path: str, path_rewrite: PathRewrite | None = None) -> str:
    """Return a canonical raw-data path before storage-root normalization.

    Hosted runtimes may mount object storage at ``path_rewrite.new_prefix`` while
    the context still carries the logical ``old_prefix`` URI. Rewriting here keeps
    keyed memory paths anchored to the same bucket across mount- and URI-based
    raw-data prefixes.
    """
    trimmed = raw_data_path.rstrip("/")
    if path_rewrite is None:
        return trimmed
    new_prefix = path_rewrite.new_prefix.rstrip("/")
    old_prefix = path_rewrite.old_prefix.rstrip("/")
    if trimmed == new_prefix or trimmed.startswith(f"{new_prefix}/"):
        return old_prefix + trimmed[len(new_prefix) :]
    return trimmed


def _memory_storage_root(raw_data_path: str, *, path_rewrite: PathRewrite | None = None) -> str:
    """Return the stable storage root used for keyed memory stores.

    Raw-data paths can carry bucket-internal sharding and per-run prefixes
    (e.g. ``s3://bucket/w6/org/project/domain/.../rd/<run_id>``). Keyed memories
    must be independent of those so two runs with the same key resolve to one
    store, so we normalize:

    - **Remote URIs** are anchored at the provider root (``scheme://netloc``),
      dropping every bucket-internal prefix.
    - **Local paths** strip the per-run tail: either a trailing ``/rd/<run_id>``
      scratch suffix (see :data:`_RAW_DATA_SCRATCH_SEGMENT`) or, for the default
      ``.../raw_data/<action_name>`` local layout, the per-run ``<action_name>``
      segment (anchoring at :data:`_LOCAL_RAW_DATA_DIR`).
    """
    trimmed = _normalize_raw_data_path(raw_data_path, path_rewrite)

    parsed = urlparse(trimmed)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"

    # Local roots differ from ``trimmed`` when they end in a per-run tail.
    parent, scratch, run_id = trimmed.rsplit("/", 2) if trimmed.count("/") >= 2 else (trimmed, "", "")
    if run_id:
        # ``<parent>/rd/<run_id>``: drop both the scratch marker and the run id.
        if scratch == _RAW_DATA_SCRATCH_SEGMENT:
            return parent
        # ``<parent>/raw_data/<action_name>``: keep the raw-data root, drop the
        # per-run action name so two runs sharing a key resolve to one store.
        if scratch == _LOCAL_RAW_DATA_DIR:
            return f"{parent}/{scratch}"
    return trimmed


def _memory_storage_root_from_context() -> str:
    """Return the storage root for keyed memory stores from the Flyte context."""
    raw = internal_ctx().raw_data
    return _memory_storage_root(raw.path, path_rewrite=raw.path_rewrite)


def _current_org() -> str:
    """Return the current Flyte organization from task context or init config."""
    tctx = internal_ctx().data.task_context
    if tctx is not None and tctx.action.org:
        return tctx.action.org

    from flyte._initialize import _get_init_config

    cfg = _get_init_config()
    if cfg is None or cfg.org is None:
        raise MemoryStoreError(
            "Organization has not been initialized. Pass org=... to MemoryStore.create/get_or_create/exists."
        )
    return cfg.org


def _cleanup_temp_root(path: pathlib.Path) -> None:
    """Best-effort cleanup of an auto-created temporary memory root."""
    shutil.rmtree(path, ignore_errors=True)


class _MemoryStoreExists:
    """Descriptor that preserves ``store.exists(path)`` and adds ``MemoryStore.exists(key=...)``."""

    def __get__(self, obj: "MemoryStore | None", owner: type["MemoryStore"]):
        if obj is not None:
            return obj._path_exists

        async def exists_for_key(
            *,
            key: str,
            org: str | None = None,
            project: str | None = None,
            domain: str | None = None,
        ) -> bool:
            remote_path = owner.remote_path_for_key(key=key, org=org, project=project, domain=domain)
            return await owner._remote_store_exists(remote_path)

        return exists_for_key


# ----------------------------------------------------------------------------
# MemoryStore
# ----------------------------------------------------------------------------


@dataclass
class MemoryStore(SerializableType):
    """Conversation transcript + path-addressed artifact memory backed by :class:`flyte.io.Dir`.

    The construct combines two complementary stores:

    - ``messages``: the live LLM conversation transcript (managed by
      :class:`~flyte.ai.agents.Agent`; mutate via :meth:`append` /
      :meth:`extend` only).
    - **Path-addressed files** under a working directory ``root``. Use
      :meth:`write_text` / :meth:`read_text` / :meth:`write_json` /
      :meth:`read_json` / :meth:`list_paths` for arbitrary named blobs that
      should round-trip through Flyte object storage.

    Persistence is :class:`flyte.io.Dir`-backed. Obtain a store via
    :meth:`create` or :meth:`get_or_create`; it saves to a deterministic
    blob-store namespace under the active Flyte raw-data bucket, derived from
    its ``key``. :meth:`save` always targets that deterministic
    :attr:`remote_path`. :meth:`create`, :meth:`get_or_create`, and
    :meth:`save` are sync-by-default (``MemoryStore.create(...)``) with an
    ``.aio(...)`` companion for async call sites, mirroring the rest of the
    Flyte SDK.

    The on-disk layout under ``root`` looks like::

        <root>/messages.json                           # transcript
        <root>/<your/path>.{txt,json,…}                # path-addressed entries
        <root>/meta/<encoded_path>.json                # per-entry metadata
        <root>/audit/log.jsonl                         # opt-in audit trail
        <root>/versions/<encoded_path>/<ts>_<sha>.txt  # opt-in version history

    Optional capabilities (off-by-default unless noted):

    - ``read_only_prefixes``: block direct writes into one or more prefixes
      (e.g. ``("memory/",)``). Useful when the agent must stage proposals
      under ``user/`` and a separate trusted pipeline (sleep cycle, reviewer)
      promotes them.
    - ``audit`` *(default: True)*: append every successful write to
      ``audit/log.jsonl``. Cheap and easy to disable.
    - ``keep_versions``: snapshot every successful write under
      ``versions/<encoded_path>/<ts>_<sha>.txt`` for full history (≈ 2x
      storage on every mutation).

    Optimistic concurrency is supported via the ``expected_sha=`` argument on
    :meth:`write_text` / :meth:`write_json`; mismatches raise
    :class:`ConcurrencyError`.

    Public I/O methods are async by default. Each one has a ``*_sync``
    companion that runs the same logic on the calling thread; the async
    version simply dispatches the sync version to a background thread via
    :func:`asyncio.to_thread`.

    Every :class:`MemoryStore` is **keyed**: it is bound to a deterministic
    blob-store namespace derived from its ``key``. Obtain one via
    :meth:`create` or :meth:`get_or_create` (the recommended entry points);
    direct construction is supported for serialization / advanced use but still
    requires a ``key``. There is no such thing as an unkeyed / ephemeral store.

    Parameters
    ----------
    key:
        Deterministic memory key (a single path segment). Determines the
        durable :attr:`remote_path` under the active raw-data root.
    messages:
        Pre-existing conversation transcript. Defaults to empty.
    root:
        Local working directory backing the store. When omitted, a fresh
        temporary directory is created (and automatically cleaned up when
        the :class:`MemoryStore` is garbage-collected). When pointing at an
        existing directory that contains ``messages.json``, the transcript
        is auto-loaded. This is an internal staging directory; callers
        normally never set it.
    remote_path:
        Durable destination for :meth:`save`. Usually resolved from ``key``
        (and the Flyte context) by :meth:`create` / :meth:`get_or_create`;
        when omitted it is resolved lazily on first :meth:`save` / hydration.
    read_only_prefixes:
        Prefixes that direct writes are not permitted to target.
    audit:
        Enable the append-only audit log.
    keep_versions:
        Snapshot every successful write under ``versions/``.
    """

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    root: pathlib.Path | str | None = None
    remote_path: str | None = None
    read_only_prefixes: tuple[str, ...] = ()
    audit: bool = True
    keep_versions: bool = False

    # ``_root_real`` caches ``self.root.resolve()`` for the symlink-escape
    # check in ``_abs`` (computed once, after we know ``root`` is a Path).
    _root_real: pathlib.Path = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Every store is keyed; an empty/invalid key is rejected up front.
        self.key = _ensure_namespace_segment(self.key, name="key")

        explicit_root = self.root is not None
        if explicit_root:
            resolved = pathlib.Path(cast("pathlib.Path | str", self.root))
        else:
            resolved = pathlib.Path(tempfile.mkdtemp(prefix="flyte_agent_mem_"))
            # Best-effort cleanup of the auto-created tempdir on GC.
            weakref.finalize(self, _cleanup_temp_root, resolved)
        resolved.mkdir(parents=True, exist_ok=True)
        self.root = resolved
        self._root_real = resolved.resolve()

        # Whether the local working root still needs to be populated from the
        # keyed ``remote_path``. Stores produced locally (fresh or via an
        # explicit root) are already authoritative; only stores rebuilt from a
        # serialized payload (see :meth:`_deserialize`) defer the download.
        self._needs_remote_hydration = False

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

    def _require_remote_path(self) -> str:
        """Return :attr:`remote_path`, resolving it from :attr:`key` on first use.

        Resolution needs the active Flyte context (org / project / domain); it is
        deferred until a durable operation (:meth:`save` / hydration) actually
        needs it so a store can be constructed outside a task context.
        """
        if self.remote_path is None:
            self.remote_path = self.remote_path_for_key(key=self.key)
        return self.remote_path

    # ------------------------------------------------------------------
    # Flyte I/O serialization (mashumaro SerializableType)
    # ------------------------------------------------------------------

    def _serialize(self) -> dict[str, Any]:
        """Serialize to a Flyte-portable payload (msgpack-native types only).

        The local working directory is intentionally omitted: keyed stores are
        re-hydrated from :attr:`remote_path` on first local access (see
        :meth:`_ensure_hydrated`), and the transcript travels inline so common
        message-only consumers need no download.
        """
        return {
            "messages": self.messages,
            "key": self.key,
            "remote_path": self.remote_path,
            "read_only_prefixes": list(self.read_only_prefixes),
            "audit": self.audit,
            "keep_versions": self.keep_versions,
        }

    @classmethod
    def _deserialize(cls, value: dict[str, Any]) -> "MemoryStore":
        """Rebuild a store from a :meth:`_serialize` payload.

        The store is created against a fresh local working directory; when it is
        keyed, the working directory is downloaded from :attr:`remote_path` lazily
        on first artifact access / save.
        """
        store = cls(
            messages=list(value.get("messages") or []),
            key=value["key"],
            remote_path=value.get("remote_path"),
            read_only_prefixes=tuple(value.get("read_only_prefixes") or ()),
            audit=value.get("audit", True),
            keep_versions=value.get("keep_versions", False),
        )
        # Defer pulling artifacts/audit/versions from the remote until they are
        # actually needed; ``messages`` already round-tripped inline.
        store._needs_remote_hydration = store.remote_path is not None
        return store

    async def _ensure_hydrated(self) -> None:
        """Populate the local working root from ``remote_path`` if still pending.

        No-op for stores produced locally and for keyed stores that have already
        been hydrated. The inline transcript is authoritative and never clobbered
        by the downloaded ``messages.json``.
        """
        if not getattr(self, "_needs_remote_hydration", False):
            return
        remote_path = self._require_remote_path()
        if await self._remote_store_exists(remote_path):
            await Dir.from_existing_remote(remote_path).download(local_path=str(self._root))
        self._needs_remote_hydration = False

    # ------------------------------------------------------------------
    # Keyed remote stores
    # ------------------------------------------------------------------

    @classmethod
    def remote_path_for_key(
        cls,
        *,
        key: str,
        org: str | None = None,
        project: str | None = None,
        domain: str | None = None,
    ) -> str:
        """Return the deterministic blob-store path for a keyed memory store.

        The path is rooted at the active raw-data bucket/storage root, excluding
        bucket-internal sharding and run-specific prefixes::

            {storage_root}/agents/memory-store/v0/{org}/{project}/{domain}/{key}

        The ``agents/memory-store`` prefix and ``v0`` version come from
        :data:`_MEMORY_NAMESPACE` / :data:`_MEMORY_SCHEMA_VERSION`.
        """
        key = _ensure_namespace_segment(key, name="key")
        if org is None:
            org = _current_org()
        if project is None or domain is None:
            import flyte

            project = project or flyte.current_project()
            domain = domain or flyte.current_domain()

        try:
            storage_root = _memory_storage_root_from_context()
        except Exception as exc:
            raise MemoryStoreError(
                "MemoryStore.create/get_or_create/exists require a raw_data_path in the Flyte context"
            ) from exc

        remote_path = _join_remote_path(
            storage_root, *_MEMORY_NAMESPACE, _MEMORY_SCHEMA_VERSION, org, project, domain, key
        )
        logger.debug("MemoryStore %r remote path: %s", key, remote_path)
        return remote_path

    @staticmethod
    async def _remote_store_exists(remote_path: str) -> bool:
        """Return whether a persisted memory store exists at ``remote_path``.

        Object stores often do not create durable directory marker objects, so
        checking the prefix itself can return false even after
        ``Dir.from_local`` uploaded files below it. ``messages.json`` is the
        MemoryStore sentinel because every save flushes it before upload.
        """
        return await storage.exists(remote_path) or await storage.exists(_join_remote_path(remote_path, MESSAGES_PATH))

    @syncify
    @classmethod
    async def create(
        cls,
        *,
        key: str,
        org: str | None = None,
        project: str | None = None,
        domain: str | None = None,
        read_only_prefixes: tuple[str, ...] = (),
        audit: bool = True,
        keep_versions: bool = False,
    ) -> "MemoryStore":
        """Create a new keyed memory store at its deterministic remote path.

        Call synchronously via ``MemoryStore.create(...)``; in async contexts use
        ``MemoryStore.create.aio(...)``.

        Raises :class:`MemoryStoreError` if the keyed blob-store path already
        exists. This preserves the explicit "create means new" contract while
        keeping subsequent saves deterministic via :meth:`save`.
        """
        remote_path = cls.remote_path_for_key(key=key, org=org, project=project, domain=domain)
        if await cls._remote_store_exists(remote_path):
            raise MemoryStoreError(f"MemoryStore {key!r} already exists at {remote_path}")

        store = cls(
            key=key,
            remote_path=remote_path,
            read_only_prefixes=read_only_prefixes,
            audit=audit,
            keep_versions=keep_versions,
        )
        # ``save`` is itself syncified; from this coroutine we drive its async form.
        await store.save.aio()
        return store

    @syncify
    @classmethod
    async def get_or_create(
        cls,
        *,
        key: str,
        org: str | None = None,
        project: str | None = None,
        domain: str | None = None,
        read_only_prefixes: tuple[str, ...] = (),
        audit: bool = True,
        keep_versions: bool = False,
    ) -> "MemoryStore":
        """Load a keyed memory store if present, otherwise create it.

        Call synchronously via ``MemoryStore.get_or_create(...)``; in async contexts
        use ``MemoryStore.get_or_create.aio(...)``.
        """
        remote_path = cls.remote_path_for_key(key=key, org=org, project=project, domain=domain)
        if await cls._remote_store_exists(remote_path):
            return await cls._load_from_dir(
                Dir.from_existing_remote(remote_path),
                key=key,
                remote_path=remote_path,
                read_only_prefixes=read_only_prefixes,
                audit=audit,
                keep_versions=keep_versions,
            )
        return await cls.create.aio(
            key=key,
            org=org,
            project=project,
            domain=domain,
            read_only_prefixes=read_only_prefixes,
            audit=audit,
            keep_versions=keep_versions,
        )

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
    # Internal path computation
    # ------------------------------------------------------------------

    def _abs(self, rel_path: str) -> pathlib.Path:
        """Resolve ``rel_path`` to an absolute path under ``self._root``.

        Defends against path traversal *and* symlink escape: the candidate
        is resolved with :meth:`pathlib.Path.resolve` (which collapses
        ``..`` and follows existing symlinks) and then verified to live
        under :attr:`_root_real`. Any escape — including via a malicious
        symlink installed inside the memory directory after download — is
        rejected.
        """
        rel = _ensure_relative_posix(rel_path)
        candidate = (self._root / rel).resolve()
        if candidate != self._root_real and self._root_real not in candidate.parents:
            raise MemoryStoreError(f"Path {rel!r} resolves outside the memory root")
        return candidate

    def _meta_path(self, rel_path: str) -> pathlib.Path:
        rel = _ensure_relative_posix(rel_path)
        return self._root / "meta" / (_encode_filename(rel) + ".json")

    def _versions_dir(self, rel_path: str) -> pathlib.Path:
        rel = _ensure_relative_posix(rel_path)
        return self._root / "versions" / _encode_filename(rel)

    def _audit_path(self) -> pathlib.Path:
        return self._root / _AUDIT_LOG_PATH

    def _assert_can_write(self, rel: str) -> None:
        if rel == MESSAGES_PATH:
            raise AccessDenied("messages.json is managed by MemoryStore; mutate via append() / extend().")
        if any(rel.startswith(p) for p in _INTERNAL_PREFIXES):
            raise AccessDenied(f"Writes to {rel!r} are not allowed (reserved prefix)")
        if any(rel.startswith(p) for p in self.read_only_prefixes):
            raise AccessDenied(f"Writes to {rel!r} are not allowed (read-only prefix)")

    # ------------------------------------------------------------------
    # Path-addressed reads
    # ------------------------------------------------------------------

    exists = _MemoryStoreExists()

    def _path_exists(self, rel_path: str) -> bool:
        """Return ``True`` if a memory file exists at ``rel_path``."""
        return self._abs(rel_path).exists()

    @syncify
    async def read_text(self, rel_path: str, default: str = "") -> str:
        """Return the UTF-8 contents of ``rel_path`` (or ``default`` if missing).

        Sync-by-default (``memory.read_text(...)``) with an ``.aio(...)`` companion.
        """
        await self._ensure_hydrated()
        try:
            return self._abs(rel_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return default

    @syncify
    async def read_json(self, rel_path: str, default: Any = None) -> Any:
        """Return the JSON-decoded contents of ``rel_path`` (or ``default`` if empty/missing).

        Sync-by-default (``memory.read_json(...)``) with an ``.aio(...)`` companion.
        """
        text = await self.read_text.aio(rel_path, default="")
        if not text.strip():
            return default
        return json.loads(text)

    def list_paths(self, prefix: str = "") -> list[str]:
        """List memory file paths under ``prefix`` (POSIX-relative, sorted).

        Internal bookkeeping (``audit/``, ``meta/``, ``versions/``) and the
        conversation transcript (``messages.json``) are excluded. Symlinked
        files are also skipped — both for safety (they can point outside
        the root) and to keep the listing deterministic.
        """
        prefix_norm = "" if not prefix else _ensure_relative_posix(prefix)
        base = self._root / prefix_norm if prefix_norm else self._root
        if not base.exists():
            return []

        root_str = os.fspath(self._root)
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            # Internal directories live exclusively at the top level of
            # ``self._root``; prune them once at that depth so we never
            # descend into ``audit/``, ``meta/``, or ``versions/``.
            if dirpath == root_str:
                dirnames[:] = [d for d in dirnames if d not in _INTERNAL_DIRS]
            for fname in filenames:
                full = pathlib.Path(dirpath) / fname
                if full.is_symlink():
                    continue
                rel = full.relative_to(self._root).as_posix()
                if rel == MESSAGES_PATH:
                    continue
                out.append(rel)
        out.sort()
        return out

    # ------------------------------------------------------------------
    # Metadata + sha helpers
    # ------------------------------------------------------------------

    @syncify
    async def get_meta(self, rel_path: str) -> MemoryMeta | None:
        """Return the :class:`MemoryMeta` sidecar for ``rel_path`` if present.

        Sync-by-default (``memory.get_meta(...)``) with an ``.aio(...)`` companion.
        """
        await self._ensure_hydrated()
        mp = self._meta_path(rel_path)
        if not mp.exists():
            return None
        try:
            raw = json.loads(mp.read_text(encoding="utf-8"))
            return MemoryMeta(**raw)
        except Exception as exc:
            raise MemoryStoreError(f"Failed to read meta for {rel_path!r}: {exc}") from exc

    @syncify
    async def current_sha(self, rel_path: str) -> str:
        """Return the sha256 of ``rel_path`` (empty string if it does not exist).

        Sync-by-default (``memory.current_sha(...)``) with an ``.aio(...)`` companion.
        """
        meta = await self.get_meta.aio(rel_path)
        if meta is not None:
            return meta.sha256
        if not self._path_exists(rel_path):
            return ""
        # No sidecar (e.g. legacy entry written outside MemoryStore): stream-hash
        # the file so very large blobs do not pull their full bytes into RAM.
        return _sha256_file(self._abs(rel_path))

    # ------------------------------------------------------------------
    # Path-addressed writes (audited + optionally versioned)
    # ------------------------------------------------------------------

    @syncify
    async def write_text(
        self,
        rel_path: str,
        content: str,
        *,
        actor: str = "agent",
        reason: str = "",
        expected_sha: str | None = None,
    ) -> MemoryMeta:
        """Write ``content`` to ``rel_path`` with optional concurrency + audit + versioning.

        Sync-by-default (``memory.write_text(...)``) with an ``.aio(...)`` companion.

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
        await self._ensure_hydrated()
        rel = _ensure_relative_posix(rel_path)
        self._assert_can_write(rel)

        p = self._abs(rel)
        old_sha = await self.current_sha.aio(rel)
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
        meta_p.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

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

    @syncify
    async def write_json(
        self,
        rel_path: str,
        obj: Any,
        *,
        actor: str = "agent",
        reason: str = "",
        expected_sha: str | None = None,
    ) -> MemoryMeta:
        """JSON-encode ``obj`` and write it via :meth:`write_text`.

        Sync-by-default (``memory.write_json(...)``) with an ``.aio(...)`` companion.
        """
        content = json.dumps(obj, indent=2, sort_keys=True, default=str)
        return await self.write_text.aio(rel_path, content, actor=actor, reason=reason, expected_sha=expected_sha)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _append_audit(self, event: dict[str, Any]) -> None:
        ap = self._audit_path()
        ap.parent.mkdir(parents=True, exist_ok=True)
        with ap.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    def audit_tail_sync(self, n: int = 20) -> list[dict[str, Any]]:
        """Synchronous variant of :meth:`audit_tail`."""
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

    async def audit_tail(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the last ``n`` audit events (most recent last).

        Returns an empty list when auditing is disabled or the log does not
        exist yet.
        """
        return await asyncio.to_thread(self.audit_tail_sync, n)

    # ------------------------------------------------------------------
    # Sync to / from flyte.io.Dir
    # ------------------------------------------------------------------

    def flush_messages_sync(self) -> None:
        """Synchronous variant of :meth:`flush_messages`."""
        p = self._root / MESSAGES_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.messages, default=str, indent=2), encoding="utf-8")

    async def flush_messages(self) -> None:
        """Persist the live transcript to ``messages.json`` under the working root."""
        await asyncio.to_thread(self.flush_messages_sync)

    @syncify
    async def save(self) -> Dir:
        """Serialize this memory to its deterministic keyed remote path.

        Call synchronously via ``memory.save(...)``; in async contexts use
        ``memory.save.aio(...)``.

        Flushes the conversation transcript to ``messages.json`` under the working
        root, then uploads the whole root (live files plus audit log, metadata
        sidecars, and any version snapshots) to :attr:`remote_path` (resolved
        from :attr:`key` if not already set).
        """
        remote_destination = self._require_remote_path()
        # Pull any remote artifacts/audit/versions into the local root before we
        # re-upload, so a store received as a task input does not overwrite the
        # keyed remote with an empty working directory.
        await self._ensure_hydrated()
        await self.flush_messages()

        # fsspec nests ``put(local_dir, dest)`` under ``dest/<basename>/`` whenever ``dest``
        # already exists (local and remote alike). Trailing slashes on both sides copy the
        # working root's *contents* into the destination instead, keeping ``messages.json`` at
        # a stable path the next run can reload. Existing files are overwritten in place.
        from_path = os.path.join(str(self._root), "")
        to_path = remote_destination.rstrip("/") + "/"
        await storage.put(from_path, to_path, recursive=True)
        return Dir.from_existing_remote(remote_destination)

    @classmethod
    async def _load_from_dir(
        cls,
        dir: Dir,
        *,
        key: str,
        remote_path: str | None = None,
        read_only_prefixes: tuple[str, ...] = (),
        audit: bool = True,
        keep_versions: bool = False,
    ) -> "MemoryStore":
        """Restore memory previously written by :meth:`save`.

        Downloads ``dir`` into a fresh local working directory and
        re-hydrates the conversation transcript. Subsequent writes are
        appended in the local copy and re-uploaded by the next
        :meth:`save` call. The local working directory is cleaned
        up automatically when the returned :class:`MemoryStore` is garbage
        collected.
        """
        local_root = pathlib.Path(tempfile.mkdtemp(prefix="flyte_agent_mem_"))
        await dir.download(local_path=str(local_root))
        store = cls(
            root=local_root,
            key=key,
            remote_path=remote_path,
            read_only_prefixes=read_only_prefixes,
            audit=audit,
            keep_versions=keep_versions,
        )
        # The constructor only registers cleanup for tempdirs *it* creates.
        # We allocated this one ourselves, so attach the same finalizer.
        weakref.finalize(store, _cleanup_temp_root, local_root)
        return store
