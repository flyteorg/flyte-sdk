"""Sharded JSONL directory type.

A JsonlDir is a directory of JSONL shard files (``part-00000.jsonl``,
``part-00001.jsonl``, etc.) that supports:

- **Writing**: automatic shard rotation by record count or byte size
- **Reading**: transparent iteration across all shards in sorted order,
  with optional one-shard-ahead prefetching for overlapping I/O
- **Compression**: per-shard zstd via ``.jsonl.zst`` extension
- **Append**: detects existing shards and continues from the next index
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Any, AsyncGenerator, Generator, Literal

import orjson
from ._jsonl_file import _DEFAULT_FLUSH_BYTES, _ZSTD_EXTENSIONS, ErrorHandler, JsonlFile

from flyte.io._dir import Dir

logger = logging.getLogger(__name__)

# All recognized JSONL extensions — plain + compressed variants.
_JSONL_EXTENSIONS = (".jsonl", *_ZSTD_EXTENSIONS)

# Default shard size: 256 MB uncompressed
_DEFAULT_MAX_BYTES_PER_SHARD = 256 << 20

# Matches shard filenames like "part-00042.jsonl" or "part-00042.jsonl.zst"
# and captures the numeric index.
_SHARD_INDEX_RE = re.compile(r"part-(\d+)\.jsonl(?:\.zst(?:d)?)?$", re.IGNORECASE)

# Sentinel object pushed into the prefetch queue to signal "this shard is done".
_PREFETCH_SENTINEL = object()

# Throughput buffer: how many records to read ahead into the prefetch queue.
# Caps memory at ~8 MB (assuming ~1 KB per record).
_DEFAULT_PREFETCH_QUEUE_SIZE = 8192


def _is_jsonl_path(path: str) -> bool:
    """Return True if *path* ends with a recognized JSONL extension."""
    p = path.lower()
    return any(p.endswith(ext) for ext in _JSONL_EXTENSIONS)


def _shard_name(index: int, extension: str) -> str:
    """Build a shard filename: ``part-00042.jsonl``."""
    return f"part-{index:05d}{extension}"


def _parse_shard_index(path: str) -> int | None:
    """Extract the numeric index from a shard filename or None if not a shard."""
    m = _SHARD_INDEX_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


async def _prefetch_shard(
    shard: JsonlFile,
    on_error: Literal["raise", "skip"] | ErrorHandler,
    queue: asyncio.Queue,
) -> None:
    """Background task: stream records from *shard* into a bounded *queue*.

    Overlaps next-shard I/O with current-shard processing so the consumer
    is never stalled waiting for network reads. The queue's ``maxsize`` is a
    memory safety bound, not a back-pressure mechanism — the consumer drains
    one record at a time. A ``_PREFETCH_SENTINEL`` is always pushed at the
    end (even on error) so the consumer knows to stop draining.
    """
    try:
        async for record in shard.iter_records(on_error=on_error):
            await queue.put(record)
    except Exception:
        await queue.put(_PREFETCH_SENTINEL)
        raise
    else:
        await queue.put(_PREFETCH_SENTINEL)


class _BaseJsonlDirWriter:
    """Shared state and rotation logic for async/sync shard writers.

    Tracks three counters per shard:
    - ``_index``   — current shard number (part-NNNNN)
    - ``_records`` — records written to the current shard
    - ``_bytes``   — uncompressed bytes written to the current shard

    ``_writer`` / ``_ctx`` hold the underlying ``JsonlFile`` writer and its
    context manager so we can close+reopen on rotation.
    """

    def __init__(
        self,
        dir_path: str,
        start_index: int = 0,
        shard_extension: str = ".jsonl",
        max_records_per_shard: int | None = None,
        max_bytes_per_shard: int = _DEFAULT_MAX_BYTES_PER_SHARD,
        flush_bytes: int = _DEFAULT_FLUSH_BYTES,
        compression_level: int = 3,
    ):
        self._dir_path = dir_path
        self._shard_extension = shard_extension
        self._max_records = max_records_per_shard
        self._max_bytes = max_bytes_per_shard
        self._flush_bytes = flush_bytes
        self._compression_level = compression_level

        self._index = start_index
        self._records = 0
        self._bytes = 0

        # Active shard writer and its context manager (None when no shard is open)
        self._writer = None
        self._ctx = None

    def _path(self) -> str:
        """Full path for the current shard file."""
        return os.path.join(
            self._dir_path,
            _shard_name(self._index, self._shard_extension),
        )

    def _should_rotate(self, size: int) -> bool:
        """Check if adding *size* bytes would exceed shard thresholds."""
        if self._writer is None:
            return False

        if self._max_records is not None and self._records >= self._max_records:
            return True

        if self._bytes + size > self._max_bytes:
            return True

        return False


class JsonlDirWriter(_BaseJsonlDirWriter):
    """Async sharded JSONL writer.

    Serializes each record with orjson, then delegates to ``_write_bytes``
    which handles rotation and passes pre-serialized bytes to the underlying
    ``JsonlWriter.write_raw()`` (avoiding double-serialization).
    """

    async def write(self, record: dict[str, Any]) -> None:
        data = orjson.dumps(record) + b"\n"
        await self._write_bytes(data)

    async def write_many(self, records) -> None:
        for r in records:
            await self._write_bytes(orjson.dumps(r) + b"\n")

    async def close(self) -> None:
        if self._ctx:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
            self._writer = None

    async def _write_bytes(self, data: bytes) -> None:
        """Write pre-serialized bytes, rotating the shard if needed."""
        if self._should_rotate(len(data)):
            await self.close()  # flush + close current shard
            self._index += 1  # advance to next part-NNNNN

        if self._writer is None:
            await self._open()  # lazily open first (or next) shard

        await self._writer.write_raw(data)

        self._records += 1
        self._bytes += len(data)

    async def _open(self) -> None:
        """Open a JsonlFile writer for the current shard index."""
        ctx = JsonlFile(path=self._path()).writer(
            flush_bytes=self._flush_bytes,
            compression_level=self._compression_level,
        )
        self._ctx = ctx
        self._writer = await ctx.__aenter__()

        self._records = 0
        self._bytes = 0


class JsonlDirWriterSync(_BaseJsonlDirWriter):
    """Sync sharded JSONL writer. Same logic as :class:`JsonlDirWriter`."""

    def write(self, record: dict[str, Any]) -> None:
        self._write_bytes(orjson.dumps(record) + b"\n")

    def write_many(self, records) -> None:
        for r in records:
            self._write_bytes(orjson.dumps(r) + b"\n")

    def close(self) -> None:
        if self._ctx:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            self._writer = None

    def _write_bytes(self, data: bytes) -> None:
        if self._should_rotate(len(data)):
            self.close()
            self._index += 1

        if self._writer is None:
            self._open()

        self._writer.write_raw(data)

        self._records += 1
        self._bytes += len(data)

    def _open(self) -> None:
        ctx = JsonlFile(path=self._path()).writer_sync(
            flush_bytes=self._flush_bytes,
            compression_level=self._compression_level,
        )
        self._ctx = ctx
        self._writer = ctx.__enter__()

        self._records = 0
        self._bytes = 0


class JsonlDir(Dir):
    """A directory of sharded JSONL files.

    Provides transparent iteration across shards on read and automatic shard
    rotation on write. Inherits all :class:`Dir` capabilities (remote storage,
    walk, download, etc.).

    Shard files are named ``part-00000.jsonl`` (or ``.jsonl.zst`` for
    compressed shards), zero-padded to 5 digits and sorted alphabetically
    on read. Mixed compression within a single directory is supported.

    Example (Async read)::

        @env.task
        async def process(d: JsonlDir):
            async for record in d.iter_records():
                print(record)

    Example (Async write)::

        @env.task
        async def create() -> JsonlDir:
            base_dir = os.path.join(flyte.ctx().run_base_dir, "output_shards")
            d = JsonlDir.from_existing_remote(base_dir)
            async with d.writer(max_records_per_shard=1000) as w:
                for i in range(5000):
                    await w.write({"id": i})
            return d
    """

    format: str = "jsonl"

    async def _get_sorted_shards(self) -> list[JsonlFile]:
        """Walk the directory and return JSONL files sorted by filename."""
        shards = [
            JsonlFile(path=f.path)
            async for f in self.walk(recursive=False)
            if _is_jsonl_path(f.path)
        ]
        shards.sort(key=lambda s: os.path.basename(s.path))
        return shards

    def _get_sorted_shards_sync(self) -> list[JsonlFile]:
        """Sync variant of :meth:`_get_sorted_shards`."""
        shards = [
            JsonlFile(path=f.path)
            for f in self.walk_sync(recursive=False)
            if _is_jsonl_path(f.path)
        ]
        shards.sort(key=lambda s: os.path.basename(s.path))
        return shards

    @staticmethod
    def _next_index(shards: list[JsonlFile]) -> int:
        """Find the highest existing shard index and return index + 1.

        Used by writer() to resume numbering after existing shards.
        Returns 0 for an empty directory.
        """
        max_idx = -1
        for s in shards:
            i = _parse_shard_index(s.path)
            if i is not None:
                max_idx = max(max_idx, i)
        return max_idx + 1

    async def iter_records(
        self,
        on_error: Literal["raise", "skip"] | ErrorHandler = "raise",
        prefetch: bool = True,
        queue_size: int = _DEFAULT_PREFETCH_QUEUE_SIZE,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator that yields records from all shards in sorted order.

        When *prefetch* is True (default), the next shard is read into a
        bounded queue concurrently while the current shard is being yielded.
        This overlaps network I/O with processing without buffering more
        than one shard in memory.

        Args:
            on_error: ``"raise"`` (default), ``"skip"``, or a callable
                ``(line_number, raw_line, exception) -> None``.
            prefetch: Overlap next-shard network I/O with current-shard
                processing for higher throughput.
            queue_size: Memory safety bound on the read-ahead buffer
                (default 8192).
        """
        shards = await self._get_sorted_shards()
        if not shards:
            return

        # Fast path: no prefetch needed for a single shard or when disabled.
        if not prefetch or len(shards) == 1:
            for s in shards:
                async for r in s.iter_records(on_error=on_error):
                    yield r
            return

        # Prefetch-one-ahead strategy (throughput optimization):
        #
        #   Iteration 0 (first shard): stream directly (no queue yet).
        #   At the end of each iteration: kick off a background task that
        #     reads the next shard into an asyncio.Queue so the next
        #     shard's network I/O overlaps with consumer processing.
        #   Next iteration: drain the queue (yields records as they arrive),
        #     then await the task to propagate any exception.
        #
        # The queue keeps the consumer fed without stalling on I/O.
        # maxsize is a memory safety bound — at most one shard's worth of
        # parsed records sits in the queue, capped by queue_size.

        task = None
        queue = None

        try:
            for i, shard in enumerate(shards):
                if queue:
                    # Drain records that the background task put into the queue.
                    while True:
                        item = await queue.get()
                        if item is _PREFETCH_SENTINEL:
                            break
                        yield item

                    # Await the task to propagate any exception it raised.
                    await task
                    queue = task = None
                else:
                    # First shard — no prefetch available yet, so stream directly.
                    async for r in shard.iter_records(on_error=on_error):
                        yield r

                # Start prefetching the next shard (if any) while the caller
                # processes the records we just yielded.
                if i + 1 < len(shards):
                    queue = asyncio.Queue(queue_size)
                    task = asyncio.create_task(
                        _prefetch_shard(shards[i + 1], on_error, queue)
                    )
        finally:
            # Clean up if the consumer exits early (break, exception, etc.)
            if task:
                task.cancel()
                with suppress(Exception):
                    await task

    def iter_records_sync(
        self,
        on_error: Literal["raise", "skip"] | ErrorHandler = "raise",
    ) -> Generator[dict[str, Any], None, None]:
        """Sync generator that yields records from all shards in sorted order."""
        for s in self._get_sorted_shards_sync():
            yield from s.iter_records_sync(on_error=on_error)

    @asynccontextmanager
    async def writer(
        self,
        shard_extension: str = ".jsonl",
        max_records_per_shard: int | None = None,
        max_bytes_per_shard: int = _DEFAULT_MAX_BYTES_PER_SHARD,
        flush_bytes: int = _DEFAULT_FLUSH_BYTES,
        compression_level: int = 3,
    ) -> AsyncGenerator[JsonlDirWriter, None]:
        """Async context manager returning a :class:`JsonlDirWriter`.

        Scans the directory for existing shards and starts writing from the
        next available index, so appending to an existing directory is safe.

        Args:
            shard_extension: File extension (e.g. ``.jsonl`` or ``.jsonl.zst``).
            max_records_per_shard: Roll after this many records (None = no limit).
            max_bytes_per_shard: Roll after this many uncompressed bytes (default 256 MB).
            flush_bytes: Buffer flush threshold in bytes (default 1 MB).
            compression_level: Zstd level (default 3, only for ``.jsonl.zst``).
        """
        # Scan existing shards so we don't overwrite them.
        start = self._next_index(await self._get_sorted_shards())

        w = JsonlDirWriter(
            self.path,
            start,
            shard_extension,
            max_records_per_shard,
            max_bytes_per_shard,
            flush_bytes,
            compression_level,
        )

        try:
            yield w
        finally:
            await w.close()

    @contextmanager
    def writer_sync(
        self,
        shard_extension: str = ".jsonl",
        max_records_per_shard: int | None = None,
        max_bytes_per_shard: int = _DEFAULT_MAX_BYTES_PER_SHARD,
        flush_bytes: int = _DEFAULT_FLUSH_BYTES,
        compression_level: int = 3,
    ) -> Generator[JsonlDirWriterSync, None, None]:
        """Sync context manager returning a :class:`JsonlDirWriterSync`.

        See :meth:`writer` for argument descriptions.
        """
        start = self._next_index(self._get_sorted_shards_sync())

        w = JsonlDirWriterSync(
            self.path,
            start,
            shard_extension,
            max_records_per_shard,
            max_bytes_per_shard,
            flush_bytes,
            compression_level,
        )

        try:
            yield w
        finally:
            w.close()
