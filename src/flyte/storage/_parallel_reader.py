from __future__ import annotations

import asyncio
import dataclasses
import io
import os
import pathlib
import sys
import tempfile
import typing
from typing import Any, Hashable, Protocol

import aiofiles
import aiofiles.os
import obstore

if typing.TYPE_CHECKING:
    from obstore import Bytes, ObjectMeta
    from obstore.store import ObjectStore

CHUNK_SIZE = int(os.getenv("FLYTE_IO_CHUNK_SIZE", str(16 * 1024 * 1024)))
MAX_CONCURRENCY = int(os.getenv("FLYTE_IO_MAX_CONCURRENCY", str(32)))


class DownloadQueueEmpty(RuntimeError):
    pass


class BufferProtocol(Protocol):
    async def write(self, offset, length, value: Bytes) -> None: ...

    async def read(self) -> memoryview: ...

    @property
    def complete(self) -> bool: ...


@dataclasses.dataclass
class _MemoryBuffer:
    arr: bytearray
    pending: int
    _closed: bool = False

    async def write(self, offset: int, length: int, value: Bytes) -> None:
        self.arr[offset : offset + length] = memoryview(value)
        self.pending -= length

    async def read(self) -> memoryview:
        return memoryview(self.arr)

    @property
    def complete(self) -> bool:
        return self.pending == 0

    @classmethod
    def new(cls, size):
        return cls(arr=bytearray(size), pending=size)


@dataclasses.dataclass
class _FileBuffer:
    path: pathlib.Path
    pending: int
    _handle: io.FileIO | None = None
    _closed: bool = False

    async def write(self, offset: int, length: int, value: Bytes) -> None:
        async with aiofiles.open(self.path, mode="r+b") as f:
            await f.seek(offset)
            await f.write(value)
        self.pending -= length

    async def read(self) -> memoryview:
        async with aiofiles.open(self.path, mode="rb") as f:
            return memoryview(await f.read())

    @property
    def complete(self) -> bool:
        return self.pending == 0

    @classmethod
    def new(cls, path: pathlib.Path, size: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return cls(path=path, pending=size)


@dataclasses.dataclass
class Chunk:
    offset: int
    length: int


@dataclasses.dataclass
class Source:
    id: Hashable
    path: pathlib.Path  # Should be str, represents the fully qualified prefix of a file (no bucket)
    length: int
    metadata: Any | None = None


@dataclasses.dataclass
class DownloadTask:
    source: Source
    chunk: Chunk
    target: pathlib.Path | None = None


class ObstoreParallelReader:
    def __init__(
        self,
        store: ObjectStore,
        *,
        chunk_size=CHUNK_SIZE,
        max_concurrency=MAX_CONCURRENCY,
    ):
        self._store = store
        self._chunk_size = chunk_size
        self._max_concurrency = max_concurrency

    def _chunks(self, size) -> typing.Iterator[tuple[int, int]]:
        cs = self._chunk_size
        for offset in range(0, size, cs):
            length = min(cs, size - offset)
            yield offset, length

    async def _as_completed(self, gen: typing.AsyncGenerator[DownloadTask, None], transformer=None):
        inq: asyncio.Queue = asyncio.Queue(self._max_concurrency * 2)
        outq: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        done = asyncio.Event()

        active: dict[Hashable, _FileBuffer | _MemoryBuffer] = {}

        async def _fill():
            # Helper function to fill the input queue, this is because the generator is async because it does list/head
            # calls on the object store which are async.
            try:
                counter = 0
                async for task in gen:
                    if task.source.id not in active:
                        active[task.source.id] = (
                            _FileBuffer.new(task.target, task.source.length)
                            if task.target is not None
                            else _MemoryBuffer.new(task.source.length)
                        )
                    await inq.put(task)
                    counter += 1
                await inq.put(sentinel)
                if counter == 0:
                    raise DownloadQueueEmpty
            except asyncio.CancelledError:
                # document why we need to swallow this
                pass

        async def _worker():
            try:
                while not done.is_set():
                    task = await inq.get()
                    if task is sentinel:
                        inq.put_nowait(sentinel)
                        break
                    chunk_source_offset = task.chunk.offset
                    buf = active[task.source.id]
                    data_to_write = await obstore.get_range_async(
                        self._store,
                        str(task.source.path),
                        start=chunk_source_offset,
                        end=chunk_source_offset + task.chunk.length,
                    )
                    await buf.write(
                        task.chunk.offset,
                        task.chunk.length,
                        data_to_write,
                    )
                    if not buf.complete:
                        continue
                    if transformer is not None:
                        result = await transformer(buf)
                    elif task.target is not None:
                        result = task.target
                    else:
                        result = task.source
                    outq.put_nowait((task.source.id, result))
                    del active[task.source.id]
            except asyncio.CancelledError:
                pass
            finally:
                done.set()

        # Yield results as they are completed
        if sys.version_info >= (3, 11):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_fill())
                for _ in range(self._max_concurrency):
                    tg.create_task(_worker())
                while not done.is_set():
                    yield await outq.get()
        else:
            fill_task = asyncio.create_task(_fill())
            worker_tasks = [asyncio.create_task(_worker()) for _ in range(self._max_concurrency)]
            try:
                while not done.is_set():
                    yield await outq.get()
            except Exception as e:
                if not fill_task.done():
                    fill_task.cancel()
                for wt in worker_tasks:
                    if not wt.done():
                        wt.cancel()
                raise e
            finally:
                await asyncio.gather(fill_task, *worker_tasks, return_exceptions=True)

        # Drain the output queue
        try:
            while True:
                yield outq.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def download_files(
        self, src_prefix: pathlib.Path, target_prefix: pathlib.Path, *paths, destination_file_name: str | None = None
    ) -> None:
        """
        src_prefix: Prefix you want to download from in the object store, not including the bucket name, nor file name.
                    Should be replaced with string
        target_prefix: Local directory to download to
        paths: Specific paths (relative to src_prefix) to download. If empty, download everything
        """

        async def _list_downloadable() -> typing.AsyncGenerator[ObjectMeta, None]:
            if paths:
                # For specific file paths, use async head
                for path_ in paths:
                    path = src_prefix / path_
                    x = await obstore.head_async(self._store, str(path))
                    yield x
                return

            # Use obstore.list() for recursive listing (all files in all subdirectories)
            # obstore.list() returns an async iterator that yields batches (lists) of objects
            async for batch in obstore.list(self._store, prefix=str(src_prefix)):
                for obj in batch:
                    yield obj

        async def _gen(tmp_dir: str) -> typing.AsyncGenerator[DownloadTask, None]:
            async for obj in _list_downloadable():
                path = pathlib.Path(obj["path"])  # e.g. Path(prefix/file.txt), needs to be changed to str.
                size = obj["size"]
                source = Source(id=path, path=path, length=size)
                # Strip src_prefix from path for destination
                rel_path = path.relative_to(src_prefix)  # doesn't work on windows
                for offset, length in self._chunks(size):
                    yield DownloadTask(
                        source=source,
                        target=tmp_dir / rel_path,  # doesn't work on windows
                        chunk=Chunk(offset, length),
                    )

        def _transform_decorator(tmp_dir: str):
            async def _transformer(buf: _FileBuffer) -> None:
                if len(paths) == 1 and destination_file_name is not None:
                    target = target_prefix / destination_file_name
                else:
                    target = target_prefix / buf.path.relative_to(tmp_dir)
                await aiofiles.os.makedirs(target.parent, exist_ok=True)
                return await aiofiles.os.replace(buf.path, target)  # mv buf.path target

            return _transformer

        with tempfile.TemporaryDirectory() as temporary_dir:
            async for _ in self._as_completed(_gen(temporary_dir), transformer=_transform_decorator(temporary_dir)):
                pass
