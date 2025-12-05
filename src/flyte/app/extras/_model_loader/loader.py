import asyncio
import json
import logging
import pathlib
import struct
import time
import typing
from collections import defaultdict

import obstore
import pydantic
from obstore.store import ObjectStore
from typing_extensions import Annotated

from flyte.storage._parallel_reader import (
    BufferProtocol,
    Chunk,
    DownloadQueueEmpty,
    DownloadTask,
    ObstoreParallelReader,
    Source,
)
from flyte.storage._storage import get_underlying_filesystem

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


async def prefetch(remote_model_path, local_model_path, exclude_safetensors=True):
    from flyte.storage._storage import _get_obstore_bypass

    logger.info(f"Pre-fetching model artifacts from {remote_model_path} to {local_model_path}...")
    if exclude_safetensors:
        logger.info(f"Deferring download of safetensor files from {remote_model_path}")
    start = time.perf_counter()

    try:
        # Exclude safetensors if model streaming is enabled, which will be handled by the flyte vllm model loader
        await _get_obstore_bypass(
            remote_model_path,
            local_model_path,
            recursive=True,
            exclude=[SAFETENSORS_DEFAULT_PATTERN] if exclude_safetensors else None,
        )
    except DownloadQueueEmpty:
        logger.warning("No model artifacts found to pre-fetch.")
    else:
        logger.info(f"Pre-fetched model artifacts in {time.perf_counter() - start:.2f}s")


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

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def size(self) -> int:
        start, end = self.data_offsets
        return end - start

    @pydantic.computed_field  # type: ignore[prop-decorator]
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
        fs = get_underlying_filesystem(path=remote_path)
        bucket, prefix = fs._split_path(remote_path)  # pylint: disable=W0212

        self._store: ObjectStore = fs._construct_store(bucket)
        self._bucket = bucket
        self._prefix: pathlib.Path = pathlib.Path(prefix)
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
            stm.tensors = list[TensorMetadata](
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
            json.decoder.JSONDecodeError,
            FileNotFoundError,
            KeyError,
        ):
            async for stm in self._load_safetensors_metadata_with_pattern(SAFETENSORS_DEFAULT_PATTERN):
                yield stm

    async def _get_tensors_async(self) -> typing.AsyncGenerator[tuple[str, torch.Tensor], None]:
        async def _to_tensor(buf: BufferProtocol, source: Source) -> torch.Tensor:
            assert isinstance(source.metadata, TensorMetadata)
            return torch.frombuffer(
                await buf.read(),
                dtype=source.metadata.dtype,
                count=len(source.metadata),
                offset=0,
            ).view(source.metadata.shape)

        async def _gen() -> typing.AsyncGenerator[DownloadTask, None]:
            async for stm in self._load_safetensors_metadata():
                for tm in stm.tensors:
                    source = Source(
                        id=tm.name,
                        path=stm.path,
                        length=tm.size,
                        offset=stm.data_start + tm.data_offsets[0],
                        metadata=tm,
                    )
                    for offset, length in self._reader._chunks(tm.size):
                        yield DownloadTask(
                            source=source,
                            chunk=Chunk(offset, length),
                        )

        # Yield tensors as they are downloaded
        async for result in self._reader._as_completed(_gen(), transformer=_to_tensor):
            yield result

    def get_tensors(self) -> typing.Generator[tuple[str, torch.Tensor], None, None]:
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
