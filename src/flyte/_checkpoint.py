"""
Task checkpointing against Flyte checkpoint object-store prefixes (v2 SDK).

:class:`AsyncCheckpoint` uses ``flyte.storage`` for I/O.

- **Async (recommended in async tasks):** ``await checkpoint.load.aio()``, ``await checkpoint.save.aio(...)``.
- **Sync in ordinary task code:** ``checkpoint.load()``, ``checkpoint.save(...)`` run the async storage path on the
  global :mod:`flyte.syncify` background loop (same pattern as other SDK async I/O).

- **Previous-checkpoint URI repair:** For remote prev URIs (e.g. ``s3://``), if the path uses the **current**
  attempt directory (``…/{run}/{action}/{n}/…``), :class:`~AsyncCheckpoint` rewrites *n* to *n-1* when *n > 1*,
  using :func:`flyte.ctx` ``action.run_name`` and ``action.name`` (Union executor v2). If ``FLYTE_ATTEMPT_NUMBER``
  is ``>= 1``, the attempt directory integer must match the buggy current-attempt value (``FLYTE_ATTEMPT_NUMBER``
  or ``FLYTE_ATTEMPT_NUMBER + 1``). Local ``file:`` paths are not modified.

Remote checkpoint URIs are a **single object** (e.g. ``.../_flytecheckpoints``). :meth:`~AsyncCheckpoint.save`
uploads a **file** as-is; a **directory** is stored as a gzip-compressed tar. :meth:`~AsyncCheckpoint.load`
downloads that object, extracts a tar into :attr:`~AsyncCheckpoint.path`, or moves a single downloaded file to
``path / "payload"``. For a **single raw blob**, :meth:`~AsyncCheckpoint.save` accepts ``bytes``; after
:meth:`~AsyncCheckpoint.load`, that blob is available at ``checkpoint.path / "payload"`` when present.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import sys
import tarfile
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

import flyte.storage as storage
from flyte._logging import logger
from flyte.storage._parallel_reader import DownloadQueueEmpty
from flyte.syncify import syncify

CHECKPOINT_CACHE_KEY = "__flyte_sync_checkpoint__"

# Basename under `AsyncCheckpoint.path` when the remote checkpoint is a single file (not a tarball).
_PAYLOAD_BASENAME = "payload"

# Object-store schemes for which Union executor may pass a prev-checkpoint URI with the wrong attempt segment.
_REMOTE_CHECKPOINT_SCHEMES: tuple[str, ...] = (
    "s3://",
    "gs://",
    "gcs://",
    "s3a://",
    "abfss://",
    "az://",
)


def repair_union_prev_checkpoint_uri(
    prev_uri: str,
    *,
    run_name: str | None = None,
    action_name: str | None = None,
) -> str:
    """
    Fix **prev-checkpoint** URIs when the executor passes the **current** attempt directory segment
    (``…/{run_name}/{action}/{n}/…``) instead of the prior attempt (``n-1``).

    Flyte executor v2 encodes ``n`` as ``GetAttempts()+1`` in the raw-data prefix; the checkpoint
    path for the previous attempt must use ``n-1`` when ``n > 1``. When ``run_name`` / ``action_name``
    are omitted, they are read from :func:`flyte.ctx` (``action.run_name``, ``action.name``).

    If there is no task context, the pattern does not match, or ``n <= 1``, *prev_uri* is returned unchanged.

    When ``FLYTE_ATTEMPT_NUMBER`` is set and ``>= 1``, the path segment *n* must match the known executor bug
    (current attempt directory): either ``n == FLYTE_ATTEMPT_NUMBER`` (1-based attempt matching the directory)
    or ``n == FLYTE_ATTEMPT_NUMBER + 1`` (0-based :attr:`flyte.models.TaskContext.attempt_number` with a
    ``GetAttempts()+1``-style directory). Otherwise the URI is left unchanged so a correct prev path is not rewritten.
    When the variable is unset or ``< 1``, only ``n > 1`` is required (backward compatible).

    NOTE: This function will be removed once the backend is updated to use the correct attempt number.
    """
    if run_name is None or action_name is None:
        from flyte._context import ctx

        tctx = ctx()
        if tctx is None:
            return prev_uri
        run_name = tctx.action.run_name
        action_name = tctx.action.name
    if not run_name or not action_name:
        return prev_uri
    pattern = re.compile(
        rf"(/{re.escape(run_name)}/{re.escape(action_name)}/)(\d+)(/)",
    )
    m = pattern.search(prev_uri)
    if not m:
        return prev_uri
    n = int(m.group(2))
    if n <= 1:
        return prev_uri
    flyte_attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    if flyte_attempt >= 1:
        # Executor bug: prev URI uses the current attempt directory *n*; correct previous dir is *n - 1*,
        # which equals flyte_attempt - 1 when flyte/segment share the same numbering, or flyte_attempt when
        # flyte is 0-based (segment = flyte + 1). Only rewrite when *n* matches one of those bug shapes.
        if not (n == flyte_attempt or n == flyte_attempt + 1):
            return prev_uri
    start, end = m.span(2)
    return f"{prev_uri[:start]}{n - 1}{prev_uri[end:]}"


def _is_recoverable_checkpoint_load_error(exc: BaseException) -> bool:
    """
    True for missing/empty checkpoint data. Obstore parallel download raises
    :class:`DownloadQueueEmpty` inside :class:`asyncio.TaskGroup` on Python 3.11+,
    which surfaces as :class:`BaseExceptionGroup` — a plain ``except DownloadQueueEmpty``
    does not catch that.
    """
    try:
        from builtins import BaseExceptionGroup
    except ImportError:
        # Python < 3.11: no task-group wrapped storage errors; isinstance never matches.
        BaseExceptionGroup = type(None)  # type: ignore

    if isinstance(exc, (AssertionError, FileNotFoundError, OSError, DownloadQueueEmpty)):
        return True
    if sys.version_info >= (3, 11) and isinstance(exc, BaseExceptionGroup):
        return all(_is_recoverable_checkpoint_load_error(e) for e in exc.exceptions)
    return False


def _clear_directory_contents(directory: pathlib.Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _tar_directory_to_file(source_dir: pathlib.Path, tar_path: pathlib.Path) -> None:
    """Write a gzip-compressed tar of *immediate* children of ``source_dir`` to ``tar_path``."""
    with tarfile.open(tar_path, "w:gz") as tar:
        for child in sorted(source_dir.iterdir()):
            tar.add(child, arcname=child.name, recursive=True)


def _extract_tarball_or_move(archive: pathlib.Path, dest: pathlib.Path) -> None:
    """Tar archive: extract into ``dest``. Single file: rename into ``dest / payload`` (no extra copy on same FS)."""
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(path=dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(archive, dest / _PAYLOAD_BASENAME)


def _new_checkpoint_download_temp_path() -> pathlib.Path:
    fd, tmp_name = tempfile.mkstemp(prefix="flyte-cpdl-", suffix=".bin")
    os.close(fd)
    return pathlib.Path(tmp_name)


class Checkpoint(ABC):
    """
    Base type for task checkpoint helpers. Subclasses load prior checkpoint data from
    ``prev_checkpoint`` into a local workspace and upload new state to ``checkpoint_path``.
    """

    @property
    @abstractmethod
    def path(self) -> pathlib.Path:
        """Local directory for reading and writing checkpoint files (your format)."""

    @abstractmethod
    def prev_exists(self) -> bool:
        """Whether the runtime provided a previous-checkpoint prefix (retry / resume)."""

    @abstractmethod
    def save(self, data: pathlib.Path | str | bytes) -> None:
        """Save checkpoint data to the remote destination."""

    @abstractmethod
    def load(self) -> Optional[pathlib.Path]:
        """Load checkpoint data from the remote source."""


class AsyncCheckpoint(Checkpoint):
    """
    Checkpoint helper using ``flyte.storage``. Remote paths are a **single object**.

    Use :meth:`load` and :meth:`save` (blocking wrappers), or ``.load.aio()`` and ``.save.aio()`` inside async code.

    Saving a **directory** uploads a gzip tar of its top-level entries; saving a **file** or **bytes** uploads
    that payload directly. After :meth:`load`, a non-tar remote object appears as ``path / "payload"``; tarball
    checkpoints unpack under :attr:`path` (the return value of :meth:`load` is still ``path / "payload"`` and may
    not exist as a file when the checkpoint was a directory archive—use :attr:`path` and your layout in that case).
    """

    def __init__(self, checkpoint_dest: str, checkpoint_src: str | None = None):
        self._checkpoint_dest = checkpoint_dest
        self._checkpoint_src: str | None = None
        if checkpoint_src is not None and (src := checkpoint_src.strip().strip('"')) != "":
            if src.startswith(_REMOTE_CHECKPOINT_SCHEMES):
                src = repair_union_prev_checkpoint_uri(src)
            self._checkpoint_src = src
        self._td = tempfile.TemporaryDirectory(prefix="flyte-cp-")
        self._restored = False
        self._had_remote_prev = False

    def __del__(self) -> None:
        td = getattr(self, "_td", None)
        if td is not None:
            td.cleanup()

    @property
    def remote_destination(self) -> str:
        """Object-store prefix where :meth:`save` writes."""
        return self._checkpoint_dest

    @property
    def remote_source(self) -> Optional[str]:
        """Object-store prefix for the previous attempt's checkpoint, if any."""
        return self._checkpoint_src

    @property
    def path(self) -> pathlib.Path:
        """
        Local directory for reading checkpoint files.
        """
        p = pathlib.Path(self._td.name)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def _payload_path(self) -> pathlib.Path:
        return self.path / _PAYLOAD_BASENAME

    def prev_exists(self) -> bool:
        return self._checkpoint_src is not None

    def _load_return_path(self) -> Optional[pathlib.Path]:
        """Value returned from :meth:`load` after a successful download (see ``AsyncCheckpoint`` class docstring)."""
        if not self._had_remote_prev:
            return None
        return self._payload_path

    @syncify
    async def load(self) -> Optional[pathlib.Path]:
        if not self.prev_exists():
            return None
        if self._restored:
            return self._load_return_path()

        prev_uri = self._checkpoint_src
        assert prev_uri is not None

        download_path = _new_checkpoint_download_temp_path()
        try:
            _clear_directory_contents(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
            await storage.get(prev_uri, download_path, recursive=False)
            if download_path.stat().st_size == 0:
                self._had_remote_prev = False
            else:
                _extract_tarball_or_move(download_path, self.path)
                self._had_remote_prev = True
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            if not _is_recoverable_checkpoint_load_error(e):
                raise
            logger.debug("Checkpoint load skipped or failed for %s: %s", prev_uri, e)
            self._had_remote_prev = False
        finally:
            download_path.unlink(missing_ok=True)

        self._restored = True
        return self._load_return_path()

    async def _save_directory_as_tarball(self, src: pathlib.Path) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix="flyte-cptar-", suffix=".tar.gz")
        os.close(fd)
        tar_path = pathlib.Path(tmp_name)
        try:
            _tar_directory_to_file(src, tar_path)
            await storage.put(str(tar_path), self._checkpoint_dest, recursive=False)
        finally:
            tar_path.unlink(missing_ok=True)

    @syncify
    async def save(self, data: pathlib.Path | str | bytes) -> None:
        if isinstance(data, bytes):
            await storage.put_stream(data, to_path=self._checkpoint_dest)
            return

        src = pathlib.Path(data) if isinstance(data, str) else data
        if not src.exists():
            raise FileNotFoundError(f"Checkpoint source path does not exist: {src}")
        if src.is_dir():
            await self._save_directory_as_tarball(src)
        else:
            await storage.put(str(src), self._checkpoint_dest, recursive=False)
