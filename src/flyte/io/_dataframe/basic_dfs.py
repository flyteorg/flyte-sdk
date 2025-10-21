import os
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


def get_pandas_storage_options(uri: str, anonymous: bool = False) -> typing.Optional[typing.Dict]:
    from pandas.io.common import is_fsspec_url  # type: ignore

    if is_fsspec_url(uri):
        if uri.startswith("s3"):
            return storage.get_configured_fsspec_kwargs("s3", anonymous=anonymous)
        return {}

    # Pandas does not allow storage_options for non-fsspec paths e.g. local.
    return None


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
        df.to_csv(
            csv_file,
            index=False,
            storage_options=get_pandas_storage_options(uri=csv_file),
        )
        structured_dataset_type.format = CSV
        return literals_pb2.StructuredDataset(
            uri=uri, metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type)
        )


class CSVToPandasDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pd.DataFrame, None, CSV)

    async def decode(
        self,
        proto_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pd.DataFrame":
        uri = proto_value.uri
        columns = None
        kwargs = get_pandas_storage_options(uri=uri)
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
                kwargs = get_pandas_storage_options(uri=uri, anonymous=True)
                return pd.read_csv(csv_file, usecols=columns, storage_options=kwargs)
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
        path = os.path.join(uri, f"{0:05}")
        df = typing.cast(pd.DataFrame, dataframe.val)
        df.to_parquet(
            path,
            coerce_timestamps="us",
            allow_truncated_timestamps=False,
            storage_options=get_pandas_storage_options(uri=path),
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
        kwargs = get_pandas_storage_options(uri=uri)
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
        try:
            return pd.read_parquet(uri, columns=columns, storage_options=kwargs)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                kwargs = get_pandas_storage_options(uri=uri, anonymous=True)
                return pd.read_parquet(uri, columns=columns, storage_options=kwargs)
            else:
                raise


class ArrowToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(pa.Table, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        dataframe_type: types_pb2.StructuredDatasetType,
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
        path = os.path.join(uri, f"{0:05}")
        filesystem = storage.get_underlying_filesystem(path=path)
        pq.write_table(dataframe.val, strip_protocol(path), filesystem=filesystem)
        return literals_pb2.StructuredDataset(uri=uri, metadata=literals_pb2.StructuredDatasetMetadata(dataframe_type))


class ParquetToArrowDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pa.Table, None, PARQUET)

    async def decode(
        self,
        proto_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pa.Table":
        import pyarrow.parquet as pq

        uri = proto_value.uri
        if not storage.is_remote(uri):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        _, path = split_protocol(uri)

        columns = None
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
        try:
            return pq.read_table(path, columns=columns)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                fs = storage.get_underlying_filesystem(path=uri, anonymous=True)
                if fs is not None:
                    return pq.read_table(path, filesystem=fs, columns=columns)
                return None
            else:
                raise
