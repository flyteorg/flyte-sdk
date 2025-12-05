import asyncio
import dataclasses
import io
import json
import logging
import pathlib
import struct
import tempfile
import time
from collections import defaultdict
from typing import Any, Hashable, Protocol
from urllib.parse import urlparse

import aiofiles
import aiofiles.os
import numpy as np
import obstore
import pydantic
from typing_extensions import Annotated

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError("torch is not installed. Please install 'torch', to use the model loader.")


from flyte.app.extras._model_loader.config import (
    CHUNK_SIZE,
    MAX_CONCURRENCY,
)

logger = logging.getLogger(__name__)

LITTLE_ENDIAN_LONG_LONG_STRUCT_FORMAT = "<Q"

SAFETENSORS_FORMAT_KEY = "format"
SAFETENSORS_FORMAT_VALUE = "pt"
SAFETENSORS_SUFFIX = ".safetensors"
SAFETENSORS_DEFAULT_PATTERN = f"*{SAFETENSORS_SUFFIX}"
SAFETENSORS_SHARDED_PATTERN = f"model-rank-{{rank}}-part-*{SAFETENSORS_SUFFIX}"
SAFETENSORS_INTERNAL_METADATA_KEY = "__metadata__"
SAFETENSORS_INDEX_PATH = "model.safetensors.index.json"
SAFETENSORS_HEADER_BUFFER_SIZE = 8
SAFETENSORS_TO_TORCH_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E4M3": torch.float8_e4m3fn,
}


class DownloadQueueEmpty(RuntimeError):
    pass


def prefetch(remote_model_path, local_model_path, exclude_safetensors=True):
    logger.info(f"Pre-fetching model artifacts from {remote_model_path} to {local_model_path}...")
    if exclude_safetensors:
        logger.info(f"Deferring download of safetensor files from {remote_model_path}")
    start = time.perf_counter()
    parsed_url = urlparse(remote_model_path)
    src_prefix = pathlib.Path(parsed_url.path.lstrip("/"))
    target_prefix = pathlib.Path(local_model_path)
    store = obstore_from_url(remote_model_path)
    reader = ObstoreParallelReader(store)
    try:
        asyncio.run(
            reader.download_files(
                src_prefix,
                target_prefix,
                exclude=[SAFETENSORS_DEFAULT_PATTERN] if exclude_safetensors else None,
            )
        )
    except* DownloadQueueEmpty:
        logger.warning("No model artifacts found to pre-fetch.")
    else:
        logger.info(f"Pre-fetched model artifacts in {time.perf_counter() - start:.2f}s")


def obstore_from_url(url, **kwargs):
    for maybe_store in (
        obstore.store.S3Store,
        obstore.store.GCSStore,
        obstore.store.AzureStore,
    ):
        try:
            return maybe_store.from_url(url, **kwargs)
        except obstore.exceptions.ObstoreError:
            pass
    raise ValueError(f"Could not find valid store for URL: {url}. Must be an S3, GCS, or Azure URI")


def prefix_exists(url: str) -> bool:
    store = obstore_from_url(url)
    prefix = urlparse(url).path.lstrip("/")
    for _ in obstore.list(store, prefix, chunk_size=1):
        return True
    return False


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


def _dtype_to_torch_dtype(dtype: str) -> torch.dtype:
    try:
        return SAFETENSORS_TO_TORCH_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype}")


class TensorMetadata(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: str
    shape: list[int]
    dtype: Annotated[torch.dtype, pydantic.BeforeValidator(_dtype_to_torch_dtype)]
    data_offsets: tuple[int, int]

    @pydantic.computed_field
    @property
    def size(self) -> int:
        start, end = self.data_offsets
        return end - start

    @pydantic.computed_field
    @property
    def length(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def __len__(self):
        return self.length


class SafeTensorsMetadata(pydantic.BaseModel):
    path: str
    data_start: int
    tensors: list[TensorMetadata]


class SafeTensorsStreamer:
    def __init__(
        self,
        remote_path,
        local_path,
        chunk_size=CHUNK_SIZE,
        max_concurrency=MAX_CONCURRENCY,
        rank=0,
        tensor_parallel_size=1,
        store_kwargs=None,
    ):
        self._store = obstore_from_url(remote_path, **(store_kwargs or {}))
        parsed_url = urlparse(remote_path)
        self._bucket = parsed_url.netloc
        self._prefix = pathlib.Path(parsed_url.path.lstrip("/"))
        self._local_path = pathlib.Path(local_path)
        self._reader = ObstoreParallelReader(self._store, chunk_size=chunk_size, max_concurrency=max_concurrency)
        self._rank = rank
        self._tensor_parallel_size = tensor_parallel_size

    async def _parse_safetensors_metadata(self, path):
        header_len = await obstore.get_range_async(self._store, str(path), start=0, end=SAFETENSORS_HEADER_BUFFER_SIZE)
        header_size = struct.unpack(
            LITTLE_ENDIAN_LONG_LONG_STRUCT_FORMAT,
            header_len,
        )[0]
        header_data = json.loads(
            (
                await obstore.get_range_async(
                    self._store,
                    str(path),
                    start=SAFETENSORS_HEADER_BUFFER_SIZE,
                    end=SAFETENSORS_HEADER_BUFFER_SIZE + header_size,
                )
            ).to_bytes()
        )
        if (
            format := header_data.pop(SAFETENSORS_INTERNAL_METADATA_KEY, {}).get(SAFETENSORS_FORMAT_KEY)
        ) and format != SAFETENSORS_FORMAT_VALUE:
            raise ValueError(f"Unsupported format: {format}")
        return SafeTensorsMetadata(
            path=str(path),
            data_start=SAFETENSORS_HEADER_BUFFER_SIZE + header_size,
            tensors=[TensorMetadata.model_validate({"name": k, **v}) for k, v in header_data.items()],
        )

    async def _list_safetensors_files_with_index(self):
        # Get index of expected tensors if it exists
        weight_map_resp = await obstore.get_async(self._store, str(self._prefix / SAFETENSORS_INDEX_PATH))
        weight_map_bytes = bytes(await weight_map_resp.bytes_async())
        tensor_to_path_map = json.loads(weight_map_bytes)["weight_map"]

        # Create index for path -> tensors
        index = defaultdict(set)
        for tensor, path in tensor_to_path_map.items():
            index[path].add(tensor)

        return index.items()

    async def _load_safetensors_metadata_from_index(self):
        for path, expected in await self._list_safetensors_files_with_index():
            stm = await self._parse_safetensors_metadata(self._prefix / path)
            # Keep only the tensors we expect (should already be deduplicated)
            keep = {tm.name: tm for tm in filter(lambda tm: tm.name in expected, stm.tensors)}
            # We have missing tensors at the path. Bail out!
            if missing := expected - keep.keys():
                raise ValueError(f"Missing {len(missing)} tensors at {path!r}: {' '.join(missing)}")
            stm.tensors = list(keep.values())
            yield stm

    async def _list_safetensors_files_with_pattern(self, pattern):
        paths = set()
        list_result = await obstore.list_with_delimiter_async(self._store, prefix=str(self._prefix))
        for obj in list_result["objects"]:
            path = pathlib.Path(obj["path"])
            if path.match(pattern):
                paths.add(path)
        if not paths:
            raise ValueError(f"No files found matching pattern: {pattern}")
        return paths

    async def _load_safetensors_metadata_with_pattern(self, pattern):
        seen = set()
        stms = await asyncio.gather(
            *(
                self._parse_safetensors_metadata(path)
                for path in await self._list_safetensors_files_with_pattern(pattern)
            )
        )
        for stm in stms:
            stm.tensors = list(
                filter(
                    lambda tm: tm.name not in seen and not seen.add(tm.name),
                    stm.tensors,
                )
            )
            yield stm

    async def _load_safetensors_metadata(self):
        # When using tensor parallelism, we can't rely on the index. Fallback to using a pattern.
        if self._tensor_parallel_size > 1:
            async for stm in self._load_safetensors_metadata_with_pattern(
                SAFETENSORS_SHARDED_PATTERN.format(rank=self._rank)
            ):
                yield stm
            return

        # No tensor parallelism. Try to use the index first, then fallback to a pattern.
        try:
            async for stm in self._load_safetensors_metadata_from_index():
                yield stm
        except (
            obstore.exceptions.NotFoundError,
            json.decoder.JSONDecodeError,
            FileNotFoundError,
            KeyError,
        ):
            async for stm in self._load_safetensors_metadata_with_pattern(SAFETENSORS_DEFAULT_PATTERN):
                yield stm

    async def _get_tensors_async(self):
        async def _to_tensor(source: Source, buf: BufferProtocol) -> torch.Tensor:
            return torch.frombuffer(
                await buf.read(),
                dtype=source.metadata.dtype,
                count=len(source.metadata),
                offset=0,
            ).view(source.metadata.shape)

        async def _gen():
            async for stm in self._load_safetensors_metadata():
                for tm in stm.tensors:
                    yield Source(
                        id=tm.name,
                        path=stm.path,
                        length=tm.size,
                        offset=stm.data_start + tm.data_offsets[0],
                        metadata=tm,
                    )

        # Yield tensors as they are downloaded
        async for name, tensor in self._reader.get_ranges(_gen(), transformer=_to_tensor):
            yield name, tensor

    def get_tensors(self):
        logger.info("Streaming tensors...")
        start = time.perf_counter()
        counter = 0
        gen = self._get_tensors_async()
        with asyncio.Runner() as runner:
            try:
                while True:
                    yield runner.run(gen.__anext__())
                    counter += 1
            except StopAsyncIteration:
                pass
        logger.info(f"Streamed {counter} tensors in {time.perf_counter() - start:.2f}s")
