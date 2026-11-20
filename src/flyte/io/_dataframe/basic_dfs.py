import typing
from pathlib import Path
from typing import TypeVar

from flyteidl2.core import literals_pb2, types_pb2
from fsspec.core import split_protocol, strip_protocol

import flyte.storage as storage
from flyte._logging import logger
from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import (
    CSV,
    PARQUET,
    DataFrame,
    DataFrameDecoder,
    DataFrameEncoder,
)

if typing.TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
else:
    pd = lazy_module("pandas")
    pa = lazy_module("pyarrow")

T = TypeVar("T")


def _get_pandas_path_and_filesystem(uri: str, anonymous: bool = False) -> typing.Tuple[str, typing.Any]:
    """
    Return the path and filesystem to hand to pandas for `uri`.

    For S3 URIs this is the protocol-less path plus Flyte's own filesystem. Flyte's configured kwargs
    are obstore-shaped, so they must not be handed to pandas as `storage_options`: pandas would resolve
    "s3" through the global fsspec registry, which may hold s3fs when it is installed alongside Flyte.

    For anything else this is `(uri, None)`, which lets pandas resolve the path itself.
    """
    if uri.startswith("s3://"):
        filesystem = storage.get_underlying_filesystem(path=uri, anonymous=anonymous)
        return filesystem._strip_protocol(uri), filesystem
    return uri, None


class PandasToCSVEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(pd.DataFrame, None, CSV)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        if not dataframe.uri:
            from flyte._context import internal_ctx

            ctx = internal_ctx()
            uri = ctx.raw_data.get_random_remote_path()
        else:
            uri = typing.cast(str, dataframe.uri)

        if not storage.is_remote(uri):
            Path(uri).mkdir(parents=True, exist_ok=True)
        csv_file = storage.join(uri, "data.csv")
        df = typing.cast(pd.DataFrame, dataframe.val)
        path, filesystem = _get_pandas_path_and_filesystem(csv_file)
        if filesystem is not None:
            with filesystem.open(path, "wb") as f:
                df.to_csv(f, index=False, mode="wb", lineterminator="\n")
        else:
            df.to_csv(path, index=False, lineterminator="\n")
        structured_dataset_type.format = CSV
        return literals_pb2.StructuredDataset(
            uri=uri, metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type)
        )


class CSVToPandasDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pd.DataFrame, None, CSV)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pd.DataFrame":
        uri = flyte_value.uri
        columns = None

        csv_file = uri
        if not uri.endswith(".csv"):
            csv_file = storage.join(uri, "data.csv")

        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
        try:
            import io

            # The pattern used here is a bit wonky because of obstore issues with csv, getting early eof error.
            buf = io.BytesIO()
            async for chunk in storage.get_stream(csv_file):
                buf.write(chunk)
            buf.seek(0)
            df = pd.read_csv(buf)
            return df

        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                path, filesystem = _get_pandas_path_and_filesystem(csv_file, anonymous=True)
                if filesystem is not None:
                    with filesystem.open(path, "rb") as f:
                        return pd.read_csv(f, usecols=columns)
                return pd.read_csv(path, usecols=columns)
            else:
                raise


class PandasToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(pd.DataFrame, None, PARQUET)

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
        path, filesystem = _get_pandas_path_and_filesystem(storage.join(uri, f"{0:05}"))
        df = typing.cast(pd.DataFrame, dataframe.val)
        df.to_parquet(
            path,
            coerce_timestamps="us",
            allow_truncated_timestamps=False,
            filesystem=filesystem,
        )

        structured_dataset_type.format = PARQUET
        return literals_pb2.StructuredDataset(
            uri=uri, metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type)
        )


class ParquetToPandasDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pd.DataFrame, None, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pd.DataFrame":
        uri = flyte_value.uri
        columns = None
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
        try:
            path, filesystem = _get_pandas_path_and_filesystem(uri)
            return pd.read_parquet(path, columns=columns, filesystem=filesystem)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                path, filesystem = _get_pandas_path_and_filesystem(uri, anonymous=True)
                return pd.read_parquet(path, columns=columns, filesystem=filesystem)
            else:
                raise


class ArrowToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(pa.Table, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        import pyarrow.parquet as pq

        if not dataframe.uri:
            from flyte._context import internal_ctx

            ctx = internal_ctx()
            uri = ctx.raw_data.get_random_remote_path()
        else:
            uri = typing.cast(str, dataframe.uri)

        if not storage.is_remote(uri):
            Path(uri).mkdir(parents=True, exist_ok=True)
        path = storage.join(uri, f"{0:05}")
        filesystem = storage.get_underlying_filesystem(path=path)
        pq.write_table(dataframe.val, strip_protocol(path), filesystem=filesystem)
        return literals_pb2.StructuredDataset(
            uri=uri,
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
        )


class ParquetToArrowDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pa.Table, None, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pa.Table":
        import pyarrow.parquet as pq

        path = flyte_value.uri
        if not storage.is_remote(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            _, path = split_protocol(path)

        columns = None
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
        try:
            return pq.read_table(path, columns=columns)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                fs = storage.get_underlying_filesystem(path=path, anonymous=True)
                if fs is not None:
                    return pq.read_table(path, filesystem=fs, columns=columns)
                return None
            else:
                raise
