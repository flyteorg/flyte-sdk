"""
Task checkpointing against Flyte checkpoint object-store prefixes (v2 SDK).

:class:`AsyncCheckpoint` uses ``flyte.storage`` for I/O.

- **Async (recommended in async tasks):** ``await checkpoint.load.aio()``, ``await checkpoint.save.aio(...)``.
- **Sync via syncify:** ``checkpoint.load()``, ``checkpoint.save()`` run the async obstore path on the
  global :mod:`flyte.syncify` background loop (same pattern as other SDK async I/O).
- **Pure sync fsspec I/O:** :meth:`~AsyncCheckpoint.load_sync` and :meth:`~AsyncCheckpoint.save_sync` use
  synchronous ``FileSystem.get`` / ``put`` like :meth:`flyte.io.File.download_sync`, avoiding the
  async obstore bypass and syncify thread.

- **Previous-checkpoint URI repair:** For remote prev URIs (e.g. ``s3://``), if the path uses the **current**
  attempt directory (``…/{run}/{action}/{n}/…``), :class:`~AsyncCheckpoint` rewrites *n* to *n-1* when *n > 1*,
  using :func:`flyte.ctx` ``action.run_name`` and ``action.name`` (Union executor v2). If ``FLYTE_ATTEMPT_NUMBER``
  is ``>= 1``, the attempt directory integer must match the buggy current-attempt value (``FLYTE_ATTEMPT_NUMBER``
  or ``FLYTE_ATTEMPT_NUMBER + 1``). Local ``file:`` paths are not modified.

Remote checkpoint URIs are a **single object** (e.g. ``.../_flytecheckpoints``). :meth:`~AsyncCheckpoint.save`
uploads a **file** as-is; a **directory** is stored as a gzip-compressed tar. :meth:`~AsyncCheckpoint.load`
downloads that object and extracts a tar into :attr:`~AsyncCheckpoint.path`, or copies bytes to an internal
single-file name under :attr:`~AsyncCheckpoint.path`. Use :meth:`~AsyncCheckpoint.read_payload_bytes` /
:meth:`~AsyncCheckpoint.save_payload_sync` (or :meth:`~AsyncCheckpoint.save_payload`) for that blob shape.
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
from flyte.storage._storage import strip_file_header
from flyte.syncify import syncify

CHECKPOINT_CACHE_KEY = "__flyte_sync_checkpoint__"

# Basename under :attr:`AsyncCheckpoint.path` when the remote checkpoint is a single file (not a tarball).
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

    Union executor v2 encodes ``n`` as ``GetAttempts()+1`` in the raw-data prefix; the checkpoint
    path for the previous attempt must use ``n-1`` when ``n > 1``. When ``run_name`` / ``action_name``
    are omitted, they are read from :func:`flyte.ctx` (``action.run_name``, ``action.name``).

    If there is no task context, the pattern does not match, or ``n <= 1``, *prev_uri* is returned unchanged.

    When ``FLYTE_ATTEMPT_NUMBER`` is set and ``>= 1``, the path segment *n* must match the known executor bug
    (current attempt directory): either ``n == FLYTE_ATTEMPT_NUMBER`` (1-based attempt matching the directory)
    or ``n == FLYTE_ATTEMPT_NUMBER + 1`` (0-based :attr:`flyte.models.TaskContext.attempt_number` with a
    ``GetAttempts()+1``-style directory). Otherwise the URI is left unchanged so a correct prev path is not rewritten.
    When the variable is unset or ``< 1``, only ``n > 1`` is required (backward compatible).
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
    """If ``archive`` is a tar (e.g. our directory checkpoint), extract into ``dest``; else copy as single file."""
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(path=dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(archive, dest / _PAYLOAD_BASENAME)


def _storage_get_sync(from_uri: str, to_path: pathlib.Path, *, recursive: bool = False) -> None:
    """
    Download a single object using synchronous fsspec I/O (cf. :meth:`flyte.io.File.download_sync`).
    """
    fs = storage.get_underlying_filesystem(path=from_uri)
    if "file" in fs.protocol:
        src = pathlib.Path(strip_file_header(from_uri))
        to_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, to_path)
        return
    fs.get(str(from_uri), str(to_path), recursive=recursive)


def _storage_put_sync(from_local: str | bytes, to_uri: str, *, recursive: bool = False) -> None:
    """
    Upload using synchronous fsspec I/O (cf. sync paths in ``flyte.storage``).

    Supports both:
    - a local path / URI string (e.g. ``/tmp/x`` or ``file:///tmp/x``)
    - an in-memory :class:`io.BytesIO` buffer (written to a temporary file for upload)
    """
    tmp_path: str | None = None
    try:
        if isinstance(from_local, bytes):
            fd, tmp_path = tempfile.mkstemp(prefix="flyte-cptemp-", suffix=".bin")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(from_local)
            from_local_clean = tmp_path
        else:
            from_local_clean = strip_file_header(from_local)

        fs = storage.get_underlying_filesystem(path=to_uri)
        if "file" in fs.protocol:
            dest = pathlib.Path(strip_file_header(to_uri))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(from_local_clean, dest)
            return

        fs.put(from_local_clean, to_uri, recursive=recursive)
    finally:
        if tmp_path is not None:
            pathlib.Path(tmp_path).unlink(missing_ok=True)


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


class AsyncCheckpoint(Checkpoint):
    """
    Checkpoint helper using ``flyte.storage``. Remote paths are a **single object**.

    Use :meth:`load` / :meth:`save` (syncify or ``.aio()``), or :meth:`load_sync` / :meth:`save_sync`
    for blocking fsspec I/O. Saving a **directory** uploads a gzip tar of its top-level entries;
    saving a **file** uploads it directly. After load, a non-archive remote object is available via
    :meth:`read_payload_bytes` (and can be updated with :meth:`save_payload_sync` / :meth:`save_payload`).
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
        self._td.cleanup()

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

    def prev_exists(self) -> bool:
        return self._checkpoint_src is not None

    @syncify
    async def load(self) -> Optional[pathlib.Path]:
        if not self.prev_exists():
            return None
        if self._restored:
            return self.path / _PAYLOAD_BASENAME if self._had_remote_prev else None

        assert self._checkpoint_src is not None
        fd, tmp_name = tempfile.mkstemp(prefix="flyte-cpdl-", suffix=".bin")
        os.close(fd)
        dl = pathlib.Path(tmp_name)
        try:
            _clear_directory_contents(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
            await storage.get(self._checkpoint_src, dl, recursive=False)
            if dl.stat().st_size == 0:
                self._had_remote_prev = False
            else:
                _extract_tarball_or_move(dl, self.path)
                self._had_remote_prev = True
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            if not _is_recoverable_checkpoint_load_error(e):
                raise
            logger.debug(f"Checkpoint load skipped or failed for {self._checkpoint_src}: {e}")
            self._had_remote_prev = False
        finally:
            dl.unlink(missing_ok=True)

        self._restored = True
        return self.path / _PAYLOAD_BASENAME if self._had_remote_prev else None

    @syncify
    async def save(self, data: pathlib.Path | str | bytes) -> None:

        if isinstance(data, bytes):
            await storage.put_stream(data, to_path=self._checkpoint_dest)
            return

        src = pathlib.Path(data) if isinstance(data, str) else data
        if not src.exists():
            raise FileNotFoundError(f"Checkpoint source path does not exist: {src}")
        if src.is_dir():
            fd, tmp_name = tempfile.mkstemp(prefix="flyte-cptar-", suffix=".tar.gz")
            os.close(fd)
            tar_path = pathlib.Path(tmp_name)
            try:
                _tar_directory_to_file(src, tar_path)
                await storage.put(str(tar_path), self._checkpoint_dest, recursive=False)
            finally:
                tar_path.unlink(missing_ok=True)
        else:
            await storage.put(str(src), self._checkpoint_dest, recursive=False)
