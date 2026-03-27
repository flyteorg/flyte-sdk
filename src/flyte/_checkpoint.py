"""
Task checkpointing against Flyte checkpoint object-store prefixes (v2 SDK).

:class:`AsyncCheckpoint` uses ``flyte.storage`` for I/O. :meth:`~AsyncCheckpoint.load` and
:meth:`~AsyncCheckpoint.save` are wrapped with :func:`flyte.syncify.syncify`, so you can:

- call them **synchronously**: ``checkpoint.load()``, ``checkpoint.save()``;
- call them **asynchronously**: ``await checkpoint.load.aio()``, ``await checkpoint.save.aio(...)``.

From async task code, prefer ``await checkpoint.load.aio()`` / ``await checkpoint.save.aio(...)``
so work runs on the caller's event loop when using ``.aio()``; bare synchronous ``load()`` /
``save()`` run the async implementation on syncify's background loop.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

import flyte.storage as storage
from flyte._logging import logger
from flyte.storage._parallel_reader import DownloadQueueEmpty
from flyte.syncify import syncify

_CHECKPOINT_CACHE_KEY = "__flyte_sync_checkpoint__"


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
    Checkpoint helper using ``flyte.storage``. Use :meth:`load` and :meth:`save` (sync or
    ``.aio()``) to download / upload checkpoint data.
    """

    def __init__(self, checkpoint_dest: str, checkpoint_src: Optional[str] = None):
        self._checkpoint_dest = checkpoint_dest
        src = checkpoint_src.strip() if checkpoint_src else ""
        self._checkpoint_src: Optional[str] = src or None
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
        Local working directory for checkpoint files.

        After :meth:`load`, object-store layouts may appear under one or more
        subdirectories (mirroring the remote prefix). Use ``path.rglob(...)`` or a
        stable relative layout when reading files back.
        """
        p = pathlib.Path(self._td.name)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def prev_exists(self) -> bool:
        return self._checkpoint_src is not None

    @syncify
    async def load(self) -> Optional[pathlib.Path]:
        if self._checkpoint_src is None:
            return None
        if self._restored:
            return self.path if self._had_remote_prev else None

        dest = self.path

        try:
            _clear_directory_contents(dest)
            dest.mkdir(parents=True, exist_ok=True)
            await storage.get(self._checkpoint_src, dest, recursive=True)
            self._had_remote_prev = True
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            if not _is_recoverable_checkpoint_load_error(e):
                raise
            logger.debug("Checkpoint load skipped or failed for %s: %s", self._checkpoint_src, e)
            self._had_remote_prev = False

        self._restored = True
        return self.path if self._had_remote_prev else None

    @syncify
    async def save(self, local_path: Optional[pathlib.Path | str] = None) -> None:
        src = pathlib.Path(local_path) if local_path is not None else self.path
        if not src.exists():
            raise FileNotFoundError(f"Checkpoint source path does not exist: {src}")
        await storage.put(str(src), self._checkpoint_dest, recursive=src.is_dir())


def task_checkpoint_cache_key() -> str:
    """Key used to stash the lazily built :class:`AsyncCheckpoint` on :attr:`TaskContext.data`."""
    return _CHECKPOINT_CACHE_KEY
