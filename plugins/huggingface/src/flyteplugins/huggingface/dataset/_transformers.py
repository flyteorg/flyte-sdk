from __future__ import annotations

import pathlib
import typing
from pathlib import Path

import datasets
from fsspec.core import strip_protocol
import pyarrow as pa
import pyarrow.parquet as pq
from flyteidl2.core import literals_pb2, types_pb2

import flyte.storage as storage
from flyte._logging import logger
from flyte.io import PARQUET, DataFrame
from flyte.io.extend import DataFrameDecoder, DataFrameEncoder

from ._io import ensure_hf_cached, join_uri_path, list_parquet_files, run_sync_io
from ._source import HFSource

_ROWS_PER_SHARD = 100_000


def _write_dataset(df: datasets.Dataset, path: str, filesystem) -> None:
    table = df.data.table
    writer = pq.ParquetWriter(strip_protocol(path), table.schema, filesystem=filesystem)

    try:
        for batch in table.to_batches(max_chunksize=10_000):
            writer.write_batch(batch)
    finally:
        writer.close()


def _read_parquet_files(
    parquet_files: list[str],
    columns: list[str] | None,
) -> pa.Table:
    tables = [pq.read_table(strip_protocol(f), columns=columns) for f in parquet_files]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def _batch_to_rows(batch: pa.RecordBatch) -> list[dict]:
    col_lists = {
        name: col.to_pylist() for name, col in zip(batch.schema.names, batch.columns)
    }
    return [
        {name: col_lists[name][i] for name in batch.schema.names}
        for i in range(batch.num_rows)
    ]


def _write_iterable_dataset(ds: datasets.IterableDataset, uri: str, filesystem) -> None:
    file_idx = 0
    rows_in_shard = 0
    writer: pq.ParquetWriter | None = None

    for batch in ds.iter(batch_size=10_000):
        table = pa.table(batch)

        if writer is None:
            shard_path = join_uri_path(uri, f"{file_idx:05}.parquet")
            writer = pq.ParquetWriter(
                strip_protocol(shard_path),
                table.schema,
                filesystem=filesystem,
            )

        for arrow_batch in table.to_batches():
            writer.write_batch(arrow_batch)

        rows_in_shard += len(table)

        if rows_in_shard >= _ROWS_PER_SHARD:
            writer.close()
            writer = None

            file_idx += 1
            rows_in_shard = 0

    if writer is not None:
        writer.close()


def _requested_columns(
    current_task_metadata: literals_pb2.StructuredDatasetMetadata,
) -> list[str] | None:
    if (
        current_task_metadata.structured_dataset_type
        and current_task_metadata.structured_dataset_type.columns
    ):
        return [c.name for c in current_task_metadata.structured_dataset_type.columns]
    return None


async def _localize_parquet_files(parquet_files: list[str]) -> list[str]:
    """Download remote parquet files before handing them to PyArrow."""
    local_files: list[str] = []

    for file_path in parquet_files:
        if storage.is_remote(file_path):
            local_dir = storage.get_random_local_directory()
            local_name = pathlib.PurePosixPath(file_path.rstrip("/")).name
            local_path = str(local_dir / local_name)
            logger.info(f"Downloading remote parquet shard {file_path} to {local_path}")
            local_files.append(await storage.get(file_path, local_path))
        else:
            local_files.append(strip_protocol(file_path))
    return local_files


class HuggingFaceDatasetToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(datasets.Dataset, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        val = dataframe.val

        if val is None and dataframe.uri:
            structured_dataset_type.format = PARQUET
            return literals_pb2.StructuredDataset(
                uri=typing.cast(str, dataframe.uri),
                metadata=literals_pb2.StructuredDatasetMetadata(
                    structured_dataset_type=structured_dataset_type
                ),
            )

        if not dataframe.uri:
            from flyte._context import internal_ctx

            uri = str(internal_ctx().raw_data.get_random_remote_path())
        else:
            uri = typing.cast(str, dataframe.uri)

        if not storage.is_remote(uri):
            Path(uri).mkdir(parents=True, exist_ok=True)

        path = join_uri_path(uri, f"{0:05}.parquet")
        df = typing.cast(datasets.Dataset, val)

        filesystem = storage.get_underlying_filesystem(path=path)
        logger.info(
            f"Writing Hugging Face Dataset output to "
            f"{'remote' if storage.is_remote(uri) else 'local'} parquet directory {uri}"
        )
        await run_sync_io(
            "write HuggingFace dataset", _write_dataset, df, path, filesystem
        )

        structured_dataset_type.format = PARQUET
        return literals_pb2.StructuredDataset(
            uri=uri,
            metadata=literals_pb2.StructuredDatasetMetadata(
                structured_dataset_type=structured_dataset_type
            ),
        )


class ParquetToHuggingFaceDatasetDecodingHandler(DataFrameDecoder):
    def __init__(self, protocol: str | None = None):
        super().__init__(datasets.Dataset, protocol, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> datasets.Dataset:
        uri = flyte_value.uri

        if uri.startswith("hf://"):
            source_uri = uri
            try:
                uri = await ensure_hf_cached(HFSource.from_hf_uri(uri))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to materialize Hugging Face dataset from {uri}: "
                    f"{type(e).__name__}: {e!r}"
                ) from e
            logger.info(
                f"Resolved Hugging Face source {source_uri} to "
                f"{'remote' if storage.is_remote(uri) else 'local'} parquet directory {uri}"
            )

        filesystem = storage.get_underlying_filesystem(path=uri)
        logger.info(
            f"Reading Hugging Face Dataset parquet from "
            f"{'remote' if storage.is_remote(uri) else 'local'} directory {uri}"
        )

        parquet_files = await list_parquet_files(uri, filesystem)
        parquet_files = await _localize_parquet_files(parquet_files)

        table = await run_sync_io(
            "read parquet files",
            _read_parquet_files,
            parquet_files,
            _requested_columns(current_task_metadata),
        )

        return datasets.Dataset(table)


class HFToHuggingFaceDatasetDecodingHandler(ParquetToHuggingFaceDatasetDecodingHandler):
    def __init__(self):
        super().__init__("hf")


class HuggingFaceIterableDatasetToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(datasets.IterableDataset, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        val = dataframe.val

        if val is None and dataframe.uri:
            structured_dataset_type.format = PARQUET
            return literals_pb2.StructuredDataset(
                uri=typing.cast(str, dataframe.uri),
                metadata=literals_pb2.StructuredDatasetMetadata(
                    structured_dataset_type=structured_dataset_type
                ),
            )

        if not dataframe.uri:
            from flyte._context import internal_ctx

            uri = str(internal_ctx().raw_data.get_random_remote_path())
        else:
            uri = typing.cast(str, dataframe.uri)

        if not storage.is_remote(uri):
            Path(uri).mkdir(parents=True, exist_ok=True)

        ds = typing.cast(datasets.IterableDataset, val)
        filesystem = storage.get_underlying_filesystem(path=uri)
        logger.info(
            f"Writing Hugging Face IterableDataset output to "
            f"{'remote' if storage.is_remote(uri) else 'local'} parquet directory {uri}"
        )
        await run_sync_io(
            "write HuggingFace iterable dataset",
            _write_iterable_dataset,
            ds,
            uri,
            filesystem,
        )

        structured_dataset_type.format = PARQUET
        return literals_pb2.StructuredDataset(
            uri=uri,
            metadata=literals_pb2.StructuredDatasetMetadata(
                structured_dataset_type=structured_dataset_type
            ),
        )


class ParquetToHuggingFaceIterableDatasetDecodingHandler(DataFrameDecoder):
    def __init__(self, protocol: str | None = None):
        super().__init__(datasets.IterableDataset, protocol, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> datasets.IterableDataset:
        uri = flyte_value.uri

        if uri.startswith("hf://"):
            source_uri = uri
            try:
                uri = await ensure_hf_cached(HFSource.from_hf_uri(uri))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to materialize Hugging Face dataset from {uri}: "
                    f"{type(e).__name__}: {e!r}"
                ) from e
            logger.info(
                f"Resolved Hugging Face source {source_uri} to "
                f"{'remote' if storage.is_remote(uri) else 'local'} parquet directory {uri}"
            )

        filesystem = storage.get_underlying_filesystem(path=uri)
        logger.info(
            f"Reading Hugging Face IterableDataset parquet from "
            f"{'remote' if storage.is_remote(uri) else 'local'} directory {uri}"
        )
        parquet_files = await list_parquet_files(uri, filesystem)
        parquet_files = await _localize_parquet_files(parquet_files)
        columns = _requested_columns(current_task_metadata)

        def _gen() -> typing.Iterator[dict]:
            for file_path in parquet_files:
                pf = pq.ParquetFile(strip_protocol(file_path))
                for batch in pf.iter_batches(batch_size=10_000, columns=columns):
                    yield from _batch_to_rows(batch)

        return datasets.IterableDataset.from_generator(_gen)


class HFToHuggingFaceIterableDatasetDecodingHandler(
    ParquetToHuggingFaceIterableDatasetDecodingHandler
):
    def __init__(self):
        super().__init__("hf")
