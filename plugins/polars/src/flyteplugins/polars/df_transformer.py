import functools
import os
import typing
from pathlib import Path

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
from flyteidl2.core import literals_pb2, types_pb2

if typing.TYPE_CHECKING:
    import polars as pl
else:
    pl = lazy_module("polars")


def get_polars_storage_options(protocol: typing.Optional[str], anonymous: bool = False) -> typing.Dict[str, str]:
    """
    Get storage options in a format compatible with Polars.

    Polars requires storage_options to be a flat dict with string keys and values,
    unlike fsspec which accepts nested dicts and complex objects.
    """
    from flyte._initialize import get_storage
    from flyte.errors import InitializationError

    if not protocol:
        return {}

    try:
        storage_config = get_storage()
    except InitializationError:
        storage_config = None

    match protocol:
        case "s3":
            from flyte.storage import S3

            if storage_config and isinstance(storage_config, S3):
                s3_config = storage_config
            else:
                s3_config = S3.auto()

            opts: typing.Dict[str, str] = {}
            if s3_config.access_key_id:
                opts["aws_access_key_id"] = s3_config.access_key_id
            if s3_config.secret_access_key:
                opts["aws_secret_access_key"] = s3_config.secret_access_key
            if s3_config.region:
                opts["aws_region"] = s3_config.region
            if s3_config.endpoint:
                opts["aws_endpoint_url"] = s3_config.endpoint
            if anonymous:
                opts["aws_skip_signature"] = "true"
            return opts

        case "gs":
            # GCS typically uses application default credentials
            # Polars supports this automatically
            return {}

        case "abfs" | "abfss":
            from flyte.storage import ABFS

            if storage_config and isinstance(storage_config, ABFS):
                abfs_config = storage_config
            else:
                abfs_config = ABFS.auto()

            opts = {}
            if abfs_config.account_name:
                opts["azure_storage_account_name"] = abfs_config.account_name
            if abfs_config.account_key:
                opts["azure_storage_account_key"] = abfs_config.account_key
            if abfs_config.tenant_id:
                opts["azure_storage_tenant_id"] = abfs_config.tenant_id
            if abfs_config.client_id:
                opts["azure_storage_client_id"] = abfs_config.client_id
            if abfs_config.client_secret:
                opts["azure_storage_client_secret"] = abfs_config.client_secret
            return opts

        case _:
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
        path = os.path.join(uri, f"{0:05}.parquet")
        df = typing.cast(pl.DataFrame, dataframe.val)

        # Polars requires flat string key-value storage options
        filesystem = storage.get_underlying_filesystem(path=path)
        storage_options = get_polars_storage_options(protocol=filesystem.protocol)
        df.write_parquet(path, storage_options=storage_options or None)

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

        parquet_path = os.path.join(uri, f"{0:05}.parquet")
        filesystem = storage.get_underlying_filesystem(path=parquet_path)
        storage_options = get_polars_storage_options(protocol=filesystem.protocol)
        try:
            return pl.read_parquet(parquet_path, columns=columns, storage_options=storage_options or None)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                storage_options = get_polars_storage_options(protocol=filesystem.protocol, anonymous=True)
                return pl.read_parquet(parquet_path, columns=columns, storage_options=storage_options or None)
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
        path = f"{os.path.join(uri, f'{0:05}')}.parquet"
        lazy_df = typing.cast(pl.LazyFrame, dataframe.val)

        # Use sink_parquet for efficient lazy writing
        filesystem = storage.get_underlying_filesystem(path=uri)
        path = path + filesystem.sep
        storage_options = get_polars_storage_options(protocol=filesystem.protocol)

        # TODO: support partitioning, which will entail user-defined behavior
        lazy_df.sink_parquet(path, storage_options=storage_options or None)

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

        parquet_path = os.path.join(uri, f"{0:05}.parquet")

        filesystem = storage.get_underlying_filesystem(path=parquet_path)
        storage_options = get_polars_storage_options(protocol=filesystem.protocol)
        try:
            # TODO: support partitioning, which will entail user-defined behavior
            lf = pl.scan_parquet(parquet_path, storage_options=storage_options or None)
            if columns:
                lf = lf.select(*columns)
            return lf
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous S3 access")
                storage_options = get_polars_storage_options(protocol=filesystem.protocol, anonymous=True)
                lf = pl.scan_parquet(parquet_path, storage_options=storage_options or None)
                if columns:
                    lf = lf.select(*columns)
                return lf
            else:
                raise


@functools.lru_cache(maxsize=None)
def register_polars_df_transformers():
    """Register Polars DataFrame encoders and decoders with the DataFrameTransformerEngine.

    This function is called automatically via the flyte.plugins.types entry point
    when flyte.init() is called with load_plugin_type_transformers=True (the default).
    """
    DataFrameTransformerEngine.register(PolarsToParquetEncodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(ParquetToPolarsDecodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(PolarsLazyFrameToParquetEncodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(ParquetToPolarsLazyFrameDecodingHandler(), default_format_for_type=True)


# Also register at module import time for backwards compatibility
register_polars_df_transformers()
