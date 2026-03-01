from __future__ import annotations

import asyncio
import io
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Callable, Generator, Literal

import orjson
import zstandard as zstd

from flyte.io._file import File

logger = logging.getLogger(__name__)

# Default buffer flush threshold: 1 MB
_DEFAULT_FLUSH_BYTES = 1 << 20

_ZSTD_EXTENSIONS = (".jsonl.zst", ".jsonl.zstd")


def _is_zstd_path(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in _ZSTD_EXTENSIONS)


class _JsonlBuffer:
    """Shared JSONL buffering logic."""

    def __init__(self, flush_bytes: int):
        self._buf = bytearray()
        self._flush_bytes = flush_bytes

    def append(self, record: dict[str, Any]) -> bool:
        """Append record. Returns True if flush is needed."""
        self._buf += orjson.dumps(record) + b"\n"
        return len(self._buf) >= self._flush_bytes

    def append_many(self, records: list[dict[str, Any]]) -> bool:
        """Append multiple records. Returns True if flush is needed."""
        for r in records:
            self._buf += orjson.dumps(r) + b"\n"
        return len(self._buf) >= self._flush_bytes

    def data(self) -> bytearray:
        return self._buf

    def clear(self) -> None:
        self._buf = bytearray()

    def has_data(self) -> bool:
        return bool(self._buf)


class JsonlWriter:
    """Async buffered JSONL writer."""

    def __init__(self, file_handle, flush_bytes: int = _DEFAULT_FLUSH_BYTES):
        self._fh = file_handle
        self._buf = _JsonlBuffer(flush_bytes)

    async def write(self, record: dict[str, Any]) -> None:
        if self._buf.append(record):
            await self.flush()

    async def write_raw(self, data: bytes) -> None:
        self._buf.data().extend(data)
        if len(self._buf.data()) >= self._buf._flush_bytes:
            await self.flush()

    async def write_many(self, records: list[dict[str, Any]]) -> None:
        if self._buf.append_many(records):
            await self.flush()

    async def flush(self) -> None:
        if self._buf.has_data():
            await self._fh.write(self._buf.data())
            self._buf.clear()


class JsonlWriterSync:
    """Sync buffered JSONL writer."""

    def __init__(self, file_handle, flush_bytes: int = _DEFAULT_FLUSH_BYTES):
        self._fh = file_handle
        self._buf = _JsonlBuffer(flush_bytes)

    def write(self, record: dict[str, Any]) -> None:
        if self._buf.append(record):
            self.flush()

    def write_raw(self, data: bytes) -> None:
        self._buf.data().extend(data)
        if len(self._buf.data()) >= self._buf._flush_bytes:
            self.flush()

    def write_many(self, records: list[dict[str, Any]]) -> None:
        if self._buf.append_many(records):
            self.flush()

    def flush(self) -> None:
        if self._buf.has_data():
            self._fh.write(self._buf.data())
            self._buf.clear()


ErrorHandler = Callable[[int, bytes, Exception], None]


def _default_error_handler(line_number: int, raw_line: bytes, exc: Exception) -> None:
    logger.warning(
        "Error parsing line %d: %s",
        line_number,
        raw_line.decode("utf-8", "replace"),
        exc_info=exc,
    )


def _parse_line(
    line: bytes, line_number: int, handler: ErrorHandler | None
) -> dict[str, Any] | None:
    """Parse a single JSONL line, delegating errors to *handler*.

    Returns the parsed dict or ``None`` if the line was skipped.
    """
    line = bytes(line).rstrip(b"\r\n")
    if not line or line.isspace():
        return None
    if handler is None:
        return orjson.loads(line)
    try:
        return orjson.loads(line)
    except Exception as exc:
        handler(line_number, line, exc)
        return None


class JsonlFile(File):
    """
    A file type for JSONL (JSON Lines) files, backed by ``orjson`` for fast
    serialisation.

    Provides streaming read and write methods that process one record at a time
    without loading the entire file into memory. Inherits all :class:`File`
    capabilities (remote storage, upload/download, etc.).

    Supports zstd-compressed files transparently via extension detection
    (``.jsonl.zst`` / ``.jsonl.zstd``).

    Example (Async read - compressed or uncompressed):

    ```python
    @env.task
    async def process(f: JsonlFile):
        async for record in f.iter_records():
            print(record)
    ```

    Example (Async write - compressed or uncompressed):

    ```python
    @env.task
    async def create() -> JsonlFile:
        f = JsonlFile.new_remote("data.jsonl")
        async with f.writer() as w:
            await w.write({"key": "value"})
        return f
    ```

    Example (Sync write - compressed or uncompressed):

    ```python
    @env.task
    async def create() -> JsonlFile:
        f = JsonlFile.new_remote("data.jsonl")
        with f.writer_sync() as w:
            w.write({"key": "value"})
        return f
    ```
    """

    format: str = "jsonl"

    async def iter_records(
        self,
        on_error: Literal["raise", "skip"] | ErrorHandler = "raise",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator that yields parsed dicts line by line."""
        handler = self._resolve_error_handler(on_error)

        line_number = 0
        async for line in self._iter_lines_async():
            line_number += 1
            record = _parse_line(line, line_number, handler)
            if record is not None:
                yield record

    async def _iter_lines_async(self) -> AsyncGenerator[bytes, None]:
        """Yield raw lines (bytes) from the file, handling compression."""
        if not _is_zstd_path(self.path):
            async with self.open("rb") as fh:
                while True:
                    line = await fh.readline()
                    if not line:
                        break
                    yield line
            return

        # ZSTD path — streaming decompression
        # Decompression is kept inline (not offloaded to a thread) because
        # at 64KB chunks it completes in microseconds — less than the cost
        # of a thread switch.
        dctx = zstd.ZstdDecompressor()

        async with self.open("rb") as fh:
            streamer = dctx.decompressobj()
            buffer = bytearray()

            while True:
                chunk = await fh.read(1 << 16)  # 64 KB
                if not chunk:
                    break

                buffer += streamer.decompress(chunk)

                while b"\n" in buffer:
                    idx = buffer.index(b"\n")
                    line = bytes(buffer[: idx + 1])
                    del buffer[: idx + 1]
                    yield line

            # leftover data
            if buffer:
                yield bytes(buffer)

    def iter_records_sync(
        self,
        on_error: Literal["raise", "skip"] | ErrorHandler = "raise",
    ) -> Generator[dict[str, Any], None, None]:
        """Sync generator that yields parsed dicts line by line."""
        handler = self._resolve_error_handler(on_error)

        for line_number, line in enumerate(self._iter_lines_sync(), 1):
            record = _parse_line(line, line_number, handler)
            if record is not None:
                yield record

    def _iter_lines_sync(self) -> Generator[bytes, None, None]:
        """Yield raw lines (bytes) from the file, handling compression."""
        if _is_zstd_path(self.path):
            dctx = zstd.ZstdDecompressor()
            with self.open_sync("rb") as fh:
                reader = io.BufferedReader(dctx.stream_reader(fh))
                yield from reader
        else:
            with self.open_sync("rb") as fh:
                yield from fh

    @asynccontextmanager
    async def writer(
        self,
        flush_bytes: int = _DEFAULT_FLUSH_BYTES,
        compression_level: int = 3,
    ) -> AsyncGenerator[JsonlWriter, None]:
        """Async context manager returning a :class:`JsonlWriter` for streaming writes.

        If the file path ends in ``.jsonl.zst``, output is zstd-compressed.

        Args:
            flush_bytes: Buffer flush threshold in bytes (default 1 MB).
            compression_level: Zstd compression level (default 3). Only used
                for ``.jsonl.zst`` paths. Higher = smaller files, slower writes.
        """
        if _is_zstd_path(self.path):
            async with self._writer_zstd(flush_bytes, compression_level) as w:
                yield w
            return

        async with self.open("wb") as fh:
            w = JsonlWriter(fh, flush_bytes=flush_bytes)
            try:
                yield w
            finally:
                await w.flush()

    @asynccontextmanager
    async def _writer_zstd(
        self, flush_bytes: int, compression_level: int
    ) -> AsyncGenerator[JsonlWriter, None]:
        """Buffer uncompressed JSONL, compress with zstd on flush, write to storage."""
        cctx = zstd.ZstdCompressor(level=compression_level)
        compressor = cctx.compressobj()

        async with self.open("wb") as fh:

            class _ZstdJsonlWriter(JsonlWriter):
                async def flush(self) -> None:
                    if self._buf.has_data():
                        compressed = await asyncio.to_thread(
                            compressor.compress, bytes(self._buf.data())
                        )
                        if compressed:
                            await fh.write(compressed)
                        self._buf.clear()

            w = _ZstdJsonlWriter(fh, flush_bytes=flush_bytes)
            try:
                yield w
            finally:
                await w.flush()
                final = await asyncio.to_thread(compressor.flush)
                if final:
                    await fh.write(final)

    @contextmanager
    def writer_sync(
        self,
        flush_bytes: int = _DEFAULT_FLUSH_BYTES,
        compression_level: int = 3,
    ) -> Generator[JsonlWriterSync, None, None]:
        """Sync context manager returning a :class:`JsonlWriterSync` for streaming writes.

        If the file path ends in ``.jsonl.zst``, output is zstd-compressed.

        Args:
            flush_bytes: Buffer flush threshold in bytes (default 1 MB).
            compression_level: Zstd compression level (default 3). Only used
                for ``.jsonl.zst`` paths. Higher = smaller files, slower writes.
        """
        if _is_zstd_path(self.path):
            with self._writer_zstd_sync(flush_bytes, compression_level) as w:
                yield w
            return

        with self.open_sync("wb") as fh:
            w = JsonlWriterSync(fh, flush_bytes=flush_bytes)
            try:
                yield w
            finally:
                w.flush()

    @contextmanager
    def _writer_zstd_sync(
        self, flush_bytes: int, compression_level: int
    ) -> Generator[JsonlWriterSync, None, None]:
        """Sync zstd-compressed writer using zstd.stream_writer."""
        with self.open_sync("wb") as fh:
            cctx = zstd.ZstdCompressor(level=compression_level)
            with cctx.stream_writer(fh) as zst_fh:
                w = JsonlWriterSync(zst_fh, flush_bytes=flush_bytes)
                try:
                    yield w
                finally:
                    w.flush()

    @staticmethod
    def _resolve_error_handler(
        on_error: Literal["raise", "skip"] | ErrorHandler,
    ) -> ErrorHandler | None:
        """Return ``None`` for raise (fast path) or a callable handler."""
        if on_error == "raise":
            return None
        if on_error == "skip":
            return _default_error_handler
        if callable(on_error):
            return on_error
        raise ValueError(
            f"on_error must be 'raise', 'skip', or a callable, got {on_error!r}"
        )
