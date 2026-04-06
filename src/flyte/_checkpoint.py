"""
Task checkpointing against Flyte checkpoint object-store prefixes (v2 SDK).

`flyte.Checkpoint` uses `flyte.io.File` for single-object **reads and writes** (download / download_sync,
open / open_sync in ``"wb"`` mode) so explicit `file://` checkpoint URIs work without calling
`flyte.init` (unlike `flyte.io.File.from_local` / `from_local_sync`, which require initialization).

- **Async tasks:** `await checkpoint.load()`, `await checkpoint.save(...)`.
- **Sync tasks:** `checkpoint.load_sync()`, `checkpoint.save_sync(...)` (same semantics as
  `flyte.io.File.download_sync` / sync uploads — no `flyte.syncify`).

`flyte.models` exposes `flyte.models.CheckpointPaths`, a small value type with the platform
`checkpoint_path` and `prev_checkpoint_path` strings. `flyte.Checkpoint` is the runtime helper
that downloads the previous attempt's **blob** into a temp workspace and uploads a new blob to the
task output prefix (including tarball encoding when you save a directory).

- **Previous-checkpoint URI repair:** For remote prev URIs (e.g. `s3://`), if the path uses the **current**
  attempt directory (`…/{run}/{action}/{n}/…`), `flyte.Checkpoint` rewrites *n* to *n-1* when *n > 1*,
  using `flyte.ctx` `action.run_name` and `action.name` (Union executor v2). If `FLYTE_ATTEMPT_NUMBER`
  is `>= 1`, the attempt directory integer must match the buggy current-attempt value (`FLYTE_ATTEMPT_NUMBER`
  or `FLYTE_ATTEMPT_NUMBER + 1`). Local `file:` paths are not modified.

Remote checkpoint URIs are a **single object** (e.g. `.../_flytecheckpoints`). `flyte.Checkpoint.save`
uploads a **file** as-is; a **directory** is stored as a gzip-compressed tar. `flyte.Checkpoint.load`
downloads that object, extracts a tar into `flyte.Checkpoint.path`, or moves a single downloaded file to
`path / "payload"`. `flyte.Checkpoint.load` returns `path / "payload"` when that file exists, else
`flyte.Checkpoint.path` (restored directory tree). For a **single raw blob**, `flyte.Checkpoint.save`
accepts `bytes`; after `flyte.Checkpoint.load`, that blob is available at `checkpoint.path / "payload"`.
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
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import aiofiles

from flyte._logging import logger
from flyte.io import File
from flyte.storage._parallel_reader import DownloadQueueEmpty
from flyte.storage._storage import get_underlying_filesystem, strip_file_header

CHECKPOINT_CACHE_KEY = "__flyte_sync_checkpoint__"

# Chunk size for streaming a local file into `File.open("wb")` (matches `flyte.io._file`).
_IO_CHUNK = 8 * 1024 * 1024

# Basename under `Checkpoint.path` when the remote checkpoint is a single file (not a tarball).
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


def latest_checkpoint(
    root: pathlib.Path,
    glob_pattern: str = "**/last.ckpt",
    *,
    key: Callable[[pathlib.Path], Any] | None = None,
) -> pathlib.Path | None:
    """
    Return the file under *root* matching *glob_pattern* with the largest ``key(path)``, or ``None``.

    By default *key* is ``lambda p: p.stat().st_mtime`` (newest modification time wins). Pass *key* to
    rank matches another way (e.g. parse a step from the filename).

    For example, the Lightning framework would use ``**/last.ckpt`` under the tree restored by
    `flyte.Checkpoint.load_sync` / `flyte.Checkpoint.load`. Pass a different *glob_pattern* for other
    layouts (e.g. ``"**/*.ckpt"``).
    """
    root = pathlib.Path(root)
    if glob_pattern.startswith("**/"):
        matches = list(root.rglob(glob_pattern[3:]))
    else:
        matches = list(root.glob(glob_pattern))
    if not matches:
        return None
    sort_key = key if key is not None else (lambda p: p.stat().st_mtime)
    matches.sort(key=sort_key, reverse=True)
    return matches[0]


def repair_union_prev_checkpoint_uri(
    prev_uri: str,
    *,
    run_name: str | None = None,
    action_name: str | None = None,
) -> str:
    """
    Fix **prev-checkpoint** URIs when the executor passes the **current** attempt directory segment
    (`…/{run_name}/{action}/{n}/…`) instead of the prior attempt (`n-1`).

    Flyte executor v2 encodes *n* as `GetAttempts()+1` in the raw-data prefix; the checkpoint
    path for the previous attempt must use `n-1` when `n > 1`. When *run_name* / *action_name*
    are omitted, they are read from `flyte.ctx` (`action.run_name`, `action.name`).

    If there is no task context, the pattern does not match, or `n <= 1`, *prev_uri* is returned unchanged.

    When `FLYTE_ATTEMPT_NUMBER` is set and `>= 1`, the path segment *n* must match the known executor bug
    (current attempt directory): either `n == FLYTE_ATTEMPT_NUMBER` (1-based attempt matching the directory)
    or `n == FLYTE_ATTEMPT_NUMBER + 1` (0-based `flyte.models.TaskContext.attempt_number` with a
    `GetAttempts()+1`-style directory). Otherwise the URI is left unchanged so a correct prev path is not rewritten.
    When the variable is unset or `< 1`, only `n > 1` is required (backward compatible).

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
    `flyte.storage._parallel_reader.DownloadQueueEmpty` inside `asyncio.TaskGroup` on Python 3.11+,
    which surfaces as `builtins.BaseExceptionGroup` — a plain `except DownloadQueueEmpty`
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
    """Write a gzip-compressed tar of *immediate* children of `source_dir` to `tar_path`."""
    with tarfile.open(tar_path, "w:gz") as tar:
        for child in sorted(source_dir.iterdir()):
            tar.add(child, arcname=child.name, recursive=True)


def _extract_tarball_or_move(archive: pathlib.Path, dest: pathlib.Path) -> bool:
    """
    Tar archive: extract into `dest`. Single file: rename into `dest / payload` (no extra copy on same FS).
    Returns True if the archive was a tarball, False if it was a single file.
    """
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(path=dest)
        return True
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(archive, dest / _PAYLOAD_BASENAME)
    return False


def _new_checkpoint_download_temp_path() -> pathlib.Path:
    fd, tmp_name = tempfile.mkstemp(prefix="flyte-cpdl-", suffix=".bin")
    os.close(fd)
    return pathlib.Path(tmp_name)


def _uri_for_file_io(uri: str) -> str:
    """Strip `file://` so `flyte.io.File` local I/O uses paths the filesystem stack accepts."""
    return strip_file_header(uri) if uri.startswith("file:") else uri


def _ensure_local_checkpoint_parent(path: str) -> None:
    """Create parent dir when the checkpoint URI is on the local filesystem."""
    fs = get_underlying_filesystem(path=path)
    if "file" in fs.protocol:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


async def _upload_checkpoint_bytes(data: bytes, dest_uri: str) -> None:
    path = _uri_for_file_io(dest_uri)
    _ensure_local_checkpoint_parent(path)
    async with File(path=path).open("wb") as fh:
        await fh.write(data)  # type: ignore[union-attr]


def _upload_checkpoint_bytes_sync(data: bytes, dest_uri: str) -> None:
    path = _uri_for_file_io(dest_uri)
    _ensure_local_checkpoint_parent(path)
    with File(path=path).open_sync("wb") as fh:
        fh.write(data)


async def _upload_checkpoint_file(from_local: str, dest_uri: str) -> None:
    local = strip_file_header(from_local)
    path = _uri_for_file_io(dest_uri)
    _ensure_local_checkpoint_parent(path)
    async with File(path=path).open("wb") as out:
        async with aiofiles.open(local, "rb") as inp:
            while True:
                chunk = await inp.read(_IO_CHUNK)
                if not chunk:
                    break
                await out.write(chunk)  # type: ignore[union-attr]


def _upload_checkpoint_file_sync(from_local: str, dest_uri: str) -> None:
    local = strip_file_header(from_local)
    path = _uri_for_file_io(dest_uri)
    _ensure_local_checkpoint_parent(path)
    with File(path=path).open_sync("wb") as out:
        with open(local, "rb") as inp:
            shutil.copyfileobj(inp, out)


class BaseCheckpoint(ABC):
    """
    Base type for task checkpoint helpers. Subclasses load prior checkpoint data from
    `prev_checkpoint` into a local workspace and upload new state to `checkpoint_path`.
    """

    @property
    @abstractmethod
    def path(self) -> pathlib.Path:
        """Local directory for reading and writing checkpoint files (your format)."""

    @abstractmethod
    def prev_exists(self) -> bool:
        """Whether the runtime provided a previous-checkpoint prefix (retry / resume)."""

    @abstractmethod
    async def load(self) -> Optional[pathlib.Path]:
        """Load checkpoint data from the remote source (async)."""

    @abstractmethod
    async def save(self, data: pathlib.Path | str | bytes) -> None:
        """Save checkpoint data to the remote destination (async)."""

    @abstractmethod
    def load_sync(self) -> Optional[pathlib.Path]:
        """Load checkpoint data from the remote source (sync task code)."""

    @abstractmethod
    def save_sync(self, data: pathlib.Path | str | bytes) -> None:
        """Save checkpoint data to the remote destination (sync task code)."""


class Checkpoint(BaseCheckpoint):
    """
    Checkpoint helper using `flyte.io.File` for all checkpoint blob I/O (load/save, async and sync).
    Remote paths are a **single object**.

    In async tasks use `flyte.Checkpoint.load` and `flyte.Checkpoint.save`. In ordinary `def` tasks use
    `flyte.Checkpoint.load_sync` and `flyte.Checkpoint.save_sync`.

    Saving a **directory** uploads a gzip tar of its top-level entries; saving a **file** or **bytes** uploads
    that payload directly. After `flyte.Checkpoint.load` / `flyte.Checkpoint.load_sync`, a non-tar remote object
    appears as `path / "payload"` and those methods return that path. Tarball checkpoints unpack under
    `flyte.Checkpoint.path`;
    they return `flyte.Checkpoint.path` so callers can scan the restored tree (e.g. `pathlib.Path.rglob`).
    """

    def __init__(self, checkpoint_dest: str, checkpoint_src: str | None = None):
        self._checkpoint_dest = checkpoint_dest
        self._checkpoint_src: str | None = None
        if checkpoint_src is not None and (src := checkpoint_src.strip().strip('"')) != "":
            if src.startswith(_REMOTE_CHECKPOINT_SCHEMES):
                src = repair_union_prev_checkpoint_uri(src)
            self._checkpoint_src = src
        # Temp workspace: extracted checkpoint tree and single-file `payload` live under this directory.
        self._td = tempfile.TemporaryDirectory(prefix="flyte-cp-")
        self._restored = False
        self._is_tarball = False
        self._had_remote_prev = False

    def __del__(self) -> None:
        td = getattr(self, "_td", None)
        if td is not None:
            td.cleanup()

    @property
    def remote_destination(self) -> str:
        """Object-store prefix where `flyte.Checkpoint.save` writes."""
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

    def _load_return_path(self, is_tarball: bool = False) -> Optional[pathlib.Path]:
        """Value returned from `flyte.Checkpoint.load` after a successful download (see class docstring)."""
        if not self._had_remote_prev:
            return None
        # Directory archives unpack under path; only single-file checkpoints use path / "payload".
        if is_tarball:
            return self.path
        return self._payload_path

    async def _with_prev_download(
        self,
        prev_uri: str,
        fetch: Callable[[pathlib.Path], Awaitable[None]],
    ) -> bool:
        """Clear workspace, run async *fetch* into a temp file, ingest tarball or payload. Returns is_tarball."""
        download_path = _new_checkpoint_download_temp_path()
        is_tarball = False
        try:
            _clear_directory_contents(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
            await fetch(download_path)
            if download_path.stat().st_size == 0:
                self._had_remote_prev = False
            else:
                is_tarball = _extract_tarball_or_move(download_path, self.path)
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
        return is_tarball

    def _with_prev_download_sync(self, prev_uri: str, fetch: Callable[[pathlib.Path], None]) -> bool:
        """Sync variant of `Checkpoint._with_prev_download`."""
        download_path = _new_checkpoint_download_temp_path()
        is_tarball = False
        try:
            _clear_directory_contents(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
            fetch(download_path)
            if download_path.stat().st_size == 0:
                self._had_remote_prev = False
            else:
                is_tarball = _extract_tarball_or_move(download_path, self.path)
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
        return is_tarball

    async def load(self) -> Optional[pathlib.Path]:
        if not self.prev_exists():
            return None
        if self._restored:
            return self._load_return_path(is_tarball=self._is_tarball)

        prev_uri = self._checkpoint_src
        if prev_uri is None:
            return None

        async def _fetch(dp: pathlib.Path) -> None:
            await File(path=_uri_for_file_io(prev_uri)).download(dp)

        is_tarball = await self._with_prev_download(prev_uri, _fetch)
        self._restored = True
        self._is_tarball = is_tarball
        return self._load_return_path(is_tarball=is_tarball)

    def load_sync(self) -> Optional[pathlib.Path]:
        if not self.prev_exists():
            return None
        if self._restored:
            return self._load_return_path(is_tarball=self._is_tarball)

        prev_uri = self._checkpoint_src
        if prev_uri is None:
            return None

        def _fetch(dp: pathlib.Path) -> None:
            File(path=_uri_for_file_io(prev_uri)).download_sync(dp)

        is_tarball = self._with_prev_download_sync(prev_uri, _fetch)
        self._restored = True
        self._is_tarball = is_tarball
        return self._load_return_path(is_tarball=is_tarball)

    async def _save_directory_as_tarball(self, src: pathlib.Path) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix="flyte-cptar-", suffix=".tar.gz")
        os.close(fd)
        tar_path = pathlib.Path(tmp_name)
        try:
            _tar_directory_to_file(src, tar_path)
            await _upload_checkpoint_file(str(tar_path), self._checkpoint_dest)
        finally:
            tar_path.unlink(missing_ok=True)

    def _save_directory_as_tarball_sync(self, src: pathlib.Path) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix="flyte-cptar-", suffix=".tar.gz")
        os.close(fd)
        tar_path = pathlib.Path(tmp_name)
        try:
            _tar_directory_to_file(src, tar_path)
            _upload_checkpoint_file_sync(str(tar_path), self._checkpoint_dest)
        finally:
            tar_path.unlink(missing_ok=True)

    async def _save_local_path(self, src: pathlib.Path) -> None:
        if src.is_dir():
            await self._save_directory_as_tarball(src)
        else:
            await _upload_checkpoint_file(str(src), self._checkpoint_dest)

    def _save_local_path_sync(self, src: pathlib.Path) -> None:
        if src.is_dir():
            self._save_directory_as_tarball_sync(src)
        else:
            _upload_checkpoint_file_sync(str(src), self._checkpoint_dest)

    async def save(self, data: pathlib.Path | str | bytes) -> None:
        if isinstance(data, bytes):
            await _upload_checkpoint_bytes(data, self._checkpoint_dest)
            return

        src = pathlib.Path(data) if isinstance(data, str) else data
        if not src.exists():
            raise FileNotFoundError(f"Checkpoint source path does not exist: {src}")
        await self._save_local_path(src)

    def save_sync(self, data: pathlib.Path | str | bytes) -> None:
        if isinstance(data, bytes):
            _upload_checkpoint_bytes_sync(data, self._checkpoint_dest)
            return

        src = pathlib.Path(data) if isinstance(data, str) else data
        if not src.exists():
            raise FileNotFoundError(f"Checkpoint source path does not exist: {src}")
        self._save_local_path_sync(src)
