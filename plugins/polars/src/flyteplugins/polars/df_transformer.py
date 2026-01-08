import os
import typing
from pathlib import Path

from flyteidl2.core import literals_pb2, types_pb2
from fsspec.core import strip_protocol

import flyte.storage as storage
from flyte._logging import logger
from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import (
    PARQUET,
    DataFrame,
    DataFrameDecoder,
    DataFrameEncoder,
    DataFrameTransformerEngine,
)

if typing.TYPE_CHECKING:
    import polars as pl
else:
    pl = lazy_module("polars")


def get_polars_storage_options(uri: str, anonymous: bool = False) -> typing.Optional[typing.Dict]:
    """Get storage options for polars based on the URI protocol."""
    if uri.startswith("s3://") or uri.startswith("s3a://") or uri.startswith("s3n://"):
        return storage.get_configured_fsspec_kwargs("s3", anonymous=anonymous)
    return {}


class PolarsToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(pl.DataFrame, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        if not dataframe.uri:
            from flyte._context import internal_ctx

            ctx = internal_ctx()
            uri = str(ctx.raw_data.get_random_remote_path())
        else:
            uri = typing.cast(str, dataframe.uri)

        if not storage.is_remote(uri):
            Path(uri).mkdir(parents=True, exist_ok=True)
        path = os.path.join(uri, f"{0:05}")
        df = typing.cast(pl.DataFrame, dataframe.val)

        # Polars uses different storage options handling
        storage_options = get_polars_storage_options(uri=path)
        if storage_options:
            # For remote storage, polars uses cloudpathlib or fsspec
            # We'll use the filesystem approach similar to arrow
            filesystem = storage.get_underlying_filesystem(path=path)
            if filesystem is not None:
                # Write to a local temp file first, then upload
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    df.write_parquet(tmp_path)
                    # Upload using the filesystem
                    with filesystem.open(strip_protocol(path), "wb") as f:
                        with open(tmp_path, "rb") as local_f:
                            f.write(local_f.read())
                    os.unlink(tmp_path)
            else:
                df.write_parquet(path, storage_options=storage_options)
        else:
            df.write_parquet(path)

        structured_dataset_type.format = PARQUET
        return literals_pb2.StructuredDataset(
            uri=uri, metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type)
        )


class ParquetToPolarsDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pl.DataFrame, None, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pl.DataFrame":
        uri = flyte_value.uri
        columns = None
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]

        # Handle directory-based parquet storage (similar to pandas handler)
        # Check if URI points to a directory with parquet files
        parquet_path = uri
        if not storage.is_remote(uri):
            # For local paths, check if it's a directory and find the parquet file
            if Path(uri).is_dir():
                parquet_path = os.path.join(uri, f"{0:05}")
            else:
                parquet_path = uri
        else:
            # For remote paths, polars read_parquet can handle directory URIs
            parquet_path = uri

        storage_options = get_polars_storage_options(uri=parquet_path)
        try:
            if storage_options:
                filesystem = storage.get_underlying_filesystem(path=parquet_path)
                if filesystem is not None:
                    # Read from remote storage
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        with filesystem.open(strip_protocol(parquet_path), "rb") as remote_f:
                            with open(tmp_path, "wb") as local_f:
                                local_f.write(remote_f.read())
                        df = pl.read_parquet(tmp_path, columns=columns)
                        os.unlink(tmp_path)
                        return df
                else:
                    return pl.read_parquet(parquet_path, columns=columns, storage_options=storage_options)
            else:
                return pl.read_parquet(parquet_path, columns=columns)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                storage_options = get_polars_storage_options(uri=parquet_path, anonymous=True)
                filesystem = storage.get_underlying_filesystem(path=parquet_path, anonymous=True)
                if filesystem is not None:
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        with filesystem.open(strip_protocol(parquet_path), "rb") as remote_f:
                            with open(tmp_path, "wb") as local_f:
                                local_f.write(remote_f.read())
                        df = pl.read_parquet(tmp_path, columns=columns)
                        os.unlink(tmp_path)
                        return df
                return pl.read_parquet(parquet_path, columns=columns, storage_options=storage_options)
            else:
                raise


class PolarsLazyFrameToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(pl.LazyFrame, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        if not dataframe.uri:
            from flyte._context import internal_ctx

            ctx = internal_ctx()
            uri = str(ctx.raw_data.get_random_remote_path())
        else:
            uri = typing.cast(str, dataframe.uri)

        if not storage.is_remote(uri):
            Path(uri).mkdir(parents=True, exist_ok=True)
        path = os.path.join(uri, f"{0:05}")
        lazy_df = typing.cast(pl.LazyFrame, dataframe.val)

        # Collect the lazy frame to eager before writing
        df = lazy_df.collect()

        # Polars uses different storage options handling
        storage_options = get_polars_storage_options(uri=path)
        if storage_options:
            # For remote storage, polars uses cloudpathlib or fsspec
            # We'll use the filesystem approach similar to arrow
            filesystem = storage.get_underlying_filesystem(path=path)
            if filesystem is not None:
                # Write to a local temp file first, then upload
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    df.write_parquet(tmp_path)
                    # Upload using the filesystem
                    with filesystem.open(strip_protocol(path), "wb") as f:
                        with open(tmp_path, "rb") as local_f:
                            f.write(local_f.read())
                    os.unlink(tmp_path)
            else:
                df.write_parquet(path, storage_options=storage_options)
        else:
            df.write_parquet(path)

        structured_dataset_type.format = PARQUET
        return literals_pb2.StructuredDataset(
            uri=uri, metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type)
        )


class ParquetToPolarsLazyFrameDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pl.LazyFrame, None, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pl.LazyFrame":
        uri = flyte_value.uri
        columns = None
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]

        # Handle directory-based parquet storage
        # For LazyFrame, we can use scan_parquet with directory or file path
        # Polars scan_parquet can handle directory paths with glob patterns
        parquet_path = uri
        if not storage.is_remote(uri):
            # For local paths, check if it's a directory
            if Path(uri).is_dir():
                # Use glob pattern for directory
                parquet_path = os.path.join(uri, "*.parquet")
            else:
                parquet_path = uri
        else:
            # For remote paths, polars scan_parquet can handle directory URIs
            parquet_path = uri

        storage_options = get_polars_storage_options(uri=parquet_path)
        try:
            if storage_options:
                filesystem = storage.get_underlying_filesystem(path=parquet_path)
                if filesystem is not None:
                    # Read from remote storage as lazy frame
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        # For remote directories, we might need to handle multiple files
                        # For now, try to read the first file (00000)
                        remote_path = strip_protocol(parquet_path)
                        if not remote_path.endswith(".parquet"):
                            # It's a directory, try to read the first file
                            remote_path = os.path.join(remote_path, f"{0:05}")
                        with filesystem.open(remote_path, "rb") as remote_f:
                            with open(tmp_path, "wb") as local_f:
                                local_f.write(remote_f.read())
                        lazy_df = pl.scan_parquet(tmp_path, columns=columns)
                        os.unlink(tmp_path)
                        return lazy_df
                else:
                    return pl.scan_parquet(parquet_path, columns=columns, storage_options=storage_options)
            else:
                return pl.scan_parquet(parquet_path, columns=columns)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                storage_options = get_polars_storage_options(uri=parquet_path, anonymous=True)
                filesystem = storage.get_underlying_filesystem(path=parquet_path, anonymous=True)
                if filesystem is not None:
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        # For remote directories, try to read the first file (00000)
                        remote_path = strip_protocol(parquet_path)
                        if not remote_path.endswith(".parquet") and not remote_path.endswith("*"):
                            # It's a directory, try to read the first file
                            remote_path = os.path.join(remote_path, f"{0:05}")
                        with filesystem.open(remote_path, "rb") as remote_f:
                            with open(tmp_path, "wb") as local_f:
                                local_f.write(remote_f.read())
                        lazy_df = pl.scan_parquet(tmp_path, columns=columns)
                        os.unlink(tmp_path)
                        return lazy_df
                return pl.scan_parquet(parquet_path, columns=columns, storage_options=storage_options)
            else:
                raise


# Register handlers
DataFrameTransformerEngine.register(PolarsToParquetEncodingHandler(), default_format_for_type=True)
DataFrameTransformerEngine.register(ParquetToPolarsDecodingHandler(), default_format_for_type=True)
DataFrameTransformerEngine.register(PolarsLazyFrameToParquetEncodingHandler(), default_format_for_type=True)
DataFrameTransformerEngine.register(ParquetToPolarsLazyFrameDecodingHandler(), default_format_for_type=True)
