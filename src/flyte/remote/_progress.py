"""
Optional progress reporting for uploads that push bytes off the local machine.

A CLI command like `flyte create artifact` wants to draw a live progress bar while a
(potentially multi-gigabyte) file is hashed and then PUT to a signed URL. The upload
sits several layers down -- CLI -> `Artifact.create` -> `File.lazy_uploader` ->
`upload_file` -> `_upload_with_retry` -- so rather than thread a callback through every
signature, a caller installs a handler with `report_uploads()` and the upload internals
report to whatever is installed.

The handler is a module global rather than a `ContextVar` on purpose: uploads run on
syncify's background event loop thread, which does not inherit the calling thread's
context. Nothing installs a handler by default, so the reporting calls are a couple of
None checks on the normal path.

Handlers are display-only. Every callback is invoked defensively: a broken handler
degrades the progress bar, it must never fail the upload.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Literal, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiofiles.threadpool.binary import AsyncBufferedReader

__all__ = ["CHUNK_SIZE", "UploadPhase", "UploadProgressHandler", "report_uploads"]

# Bytes read (and reported) per tick. Big enough that the per-chunk bookkeeping is
# noise next to the IO, small enough that the bar moves smoothly on a slow link.
CHUNK_SIZE = 1024 * 1024

UploadPhase = Literal["hashing", "uploading"]


@runtime_checkable
class UploadProgressHandler(Protocol):
    """Receives byte-level progress for a single local file moving to blob storage."""

    def start(self, key: str, *, name: str, phase: UploadPhase, total: int) -> None:
        """Begin (or restart, when an upload is retried) tracking `key`."""

    def advance(self, key: str, size: int) -> None:
        """`size` more bytes of `key` have been hashed or sent."""

    def finish(self, key: str, *, failed: bool = False) -> None:
        """`key` reached the end of its phase, either completely or by raising."""


class _Installed:
    """Holds the process-wide handler (an attribute, so nothing needs `global`)."""

    handler: Optional[UploadProgressHandler] = None


@contextmanager
def report_uploads(handler: UploadProgressHandler) -> Iterator[UploadProgressHandler]:
    """
    Install `handler` as the process-wide upload progress handler for the duration of
    the block, restoring whatever was installed before on the way out.
    """
    previous = _Installed.handler
    _Installed.handler = handler
    try:
        yield handler
    finally:
        _Installed.handler = previous


def current_handler() -> Optional[UploadProgressHandler]:
    """The installed handler, or None when nobody is watching."""
    return _Installed.handler


def _safe(method: str, *args: Any, **kwargs: Any) -> None:
    handler = _Installed.handler
    if handler is None:
        return
    try:
        getattr(handler, method)(*args, **kwargs)
    except Exception as e:  # pragma: no cover - defensive, a bad display must not fail an upload
        from flyte._logging import logger

        logger.debug(f"Upload progress handler raised in {method}: {e}")


def report_start(key: str, *, name: str, phase: UploadPhase, total: int) -> None:
    _safe("start", key, name=name, phase=phase, total=total)


def report_advance(key: str, size: int) -> None:
    _safe("advance", key, size)


def report_finish(key: str, *, failed: bool = False) -> None:
    _safe("finish", key, failed=failed)


def hash_key(file_path: str | os.PathLike) -> str:
    return f"hash:{os.fspath(file_path)}"


def upload_key(file_path: str | os.PathLike) -> str:
    return f"upload:{os.fspath(file_path)}"


async def stream_file(
    file: AsyncBufferedReader,
    *,
    key: str,
    name: str,
    total: int,
    chunk_size: int = CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """
    Yield `file` in chunks, reporting progress as the HTTP client drains them.

    Each chunk is counted *after* it is yielded back, i.e. when the client comes back
    for the next one, so the bar tracks bytes handed to the socket rather than bytes
    read ahead into memory.
    """
    report_start(key, name=name, phase="uploading", total=total)
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            yield chunk
            report_advance(key, len(chunk))
    except BaseException:
        report_finish(key, failed=True)
        raise
    report_finish(key)
