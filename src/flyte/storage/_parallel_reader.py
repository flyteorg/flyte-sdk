import asyncio
import dataclasses
import io
import logging
import pathlib
import tempfile
import time
import os
from typing import Any, Hashable, Protocol
from urllib.parse import urlparse

import aiofiles
import aiofiles.os
import numpy as np
import obstore

CHUNK_SIZE = int(os.getenv("FLYTE_IO_CHUNK_SIZE", str(16 * 1024 * 1024)))
MAX_CONCURRENCY = int(os.getenv("FLYTE_IO_MAX_CONCURRENCY", str(32)))


class DownloadQueueEmpty(RuntimeError):
    pass


# def obstore_from_url(url, **kwargs):
#     for maybe_store in (
#         obstore.store.S3Store,
#         obstore.store.GCSStore,
#         obstore.store.AzureStore,
#     ):
#         try:
#             return maybe_store.from_url(url, **kwargs)
#         except obstore.exceptions.ObstoreError:
#             pass
#     raise ValueError(f"Could not find valid store for URL: {url}. Must be an S3, GCS, or Azure URI")


# def prefix_exists(url: str) -> bool:
#     store = obstore_from_url(url)
#     prefix = urlparse(url).path.lstrip("/")
#     for _ in obstore.list(store, prefix, chunk_size=1):
#         return True
#     return False


class BufferProtocol(Protocol):
    async def write(self, offset, length, value) -> None: ...

    async def read(self) -> memoryview: ...

    @property
    def complete(self) -> bool: ...


@dataclasses.dataclass
class _MemoryBuffer:
    arr: np.ndarray
    pending: int
    _closed: bool = False

    async def write(self, offset, length, value) -> None:
        self.arr[offset : offset + length] = value
        self.pending -= length

    async def read(self) -> memoryview:
        return memoryview(self.arr)

    @property
    def complete(self) -> bool:
        return self.pending == 0

    @classmethod
    def new(cls, size):
        return cls(arr=np.empty(size, dtype=np.uint8), pending=size)


@dataclasses.dataclass
class _FileBuffer:
    path: pathlib.Path
    pending: int
    _handle: io.FileIO | None = None
    _closed: bool = False

    async def write(self, offset, length, value) -> None:
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
    def new(cls, path, size):
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
    path: pathlib.Path
    length: int
    offset: int = 0
    metadata: Any | None = None


@dataclasses.dataclass
class DownloadTask:
    source: Source
    chunk: Chunk
    target: pathlib.Path | None = None


class ObstoreParallelReader:
    def __init__(
        self,
        store,
        *,
        chunk_size=CHUNK_SIZE,
        max_concurrency=MAX_CONCURRENCY,
    ):
        self._store = store
        self._chunk_size = chunk_size
        self._max_concurrency = max_concurrency

    def _chunks(self, size):
        offsets = np.arange(0, size, self._chunk_size)
        lengths = np.minimum(self._chunk_size, size - offsets)
        return zip(offsets, lengths)

    async def _as_completed(self, gen, transformer=None):
        inq = asyncio.Queue(self._max_concurrency * 2)
        outq = asyncio.Queue()
        sentinel = object()
        done = asyncio.Event()

        active = {}

        async def _fill():
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
                pass

        async def _worker():
            try:
                while not done.is_set():
                    task = await inq.get()
                    if task is sentinel:
                        inq.put_nowait(sentinel)
                        break
                    chunk_source_offset = task.chunk.offset + task.source.offset
                    buf = active[task.source.id]
                    await buf.write(
                        task.chunk.offset,
                        task.chunk.length,
                        await obstore.get_range_async(
                            self._store,
                            str(task.source.path),
                            start=chunk_source_offset,
                            end=chunk_source_offset + task.chunk.length,
                        ),
                    )
                    if not buf.complete:
                        continue
                    if transformer is not None:
                        result = await transformer(task.source, buf)
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
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_fill())
            for _ in range(self._max_concurrency):
                tg.create_task(_worker())
            while not done.is_set():
                yield await outq.get()

        # Drain the output queue
        try:
            while True:
                yield outq.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def download_files(self, src_prefix, target_prefix, *paths, include=None, exclude=None):
        def _keep(path):
            if include is not None and not any(path.match(i) for i in include):
                return False
            if exclude is not None and any(path.match(e) for e in exclude):
                return False
            return True

        async def _list_downloadable():
            if paths:
                for path_ in paths:
                    path = src_prefix / path_
                    if _keep(path):
                        yield await obstore.head_async(self._store, str(path))
                return

            list_result = await obstore.list_with_delimiter_async(self._store, prefix=str(src_prefix))
            for obj in list_result["objects"]:
                path = pathlib.Path(obj["path"])
                if _keep(path):
                    yield obj

        async def _gen(tmpdir):
            async for obj in _list_downloadable():
                path = pathlib.Path(obj["path"])
                size = obj["size"]
                source = Source(id=path, path=path, length=size)
                # Strip src_prefix from path for destination
                rel_path = path.relative_to(src_prefix)
                for offset, length in self._chunks(size):
                    yield DownloadTask(
                        source=source,
                        target=tmpdir / rel_path,
                        chunk=Chunk(offset, length),
                    )

        def _transform_decorator(tmpdir):
            async def _transformer(source: Source, buf: BufferProtocol) -> None:
                target = target_prefix / buf.path.relative_to(tmpdir)
                await aiofiles.os.makedirs(target.parent, exist_ok=True)
                return await aiofiles.os.replace(buf.path, target)

            return _transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            async for _ in self._as_completed(_gen(tmpdir), transformer=_transform_decorator(tmpdir)):
                pass

    async def get_ranges(self, gen, transformer=None):
        async def _gen():
            async for source in gen:
                for offset, length in self._chunks(source.length):
                    yield DownloadTask(source=source, chunk=Chunk(offset, length))

        async for result in self._as_completed(_gen(), transformer=transformer):
            yield result
