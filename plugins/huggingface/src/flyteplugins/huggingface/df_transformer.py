import functools
import os
import typing
from pathlib import Path

import flyte.storage as storage
from flyte._logging import logger
from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import PARQUET, DataFrame
from flyte.io.extend import (
    DataFrameDecoder,
    DataFrameEncoder,
    DataFrameTransformerEngine,
)
from flyteidl2.core import literals_pb2, types_pb2

if typing.TYPE_CHECKING:
    import datasets
else:
    datasets = lazy_module("datasets")


def get_hf_storage_options(protocol: typing.Optional[str], anonymous: bool = False) -> typing.Dict[str, typing.Any]:
    """Get fsspec-compatible storage options for HuggingFace datasets.

    HuggingFace datasets uses fsspec for remote I/O, so storage options
    follow the fsspec/s3fs/gcsfs/adlfs conventions.
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

            opts: typing.Dict[str, typing.Any] = {}
            if s3_config.access_key_id:
                opts["key"] = s3_config.access_key_id
            if s3_config.secret_access_key:
                opts["secret"] = s3_config.secret_access_key
            if s3_config.endpoint:
                opts["client_kwargs"] = {"endpoint_url": s3_config.endpoint}
            if anonymous:
                opts["anon"] = "true"
            return opts

        case "gs":
            return {}

        case "abfs" | "abfss":
            from flyte.storage import ABFS

            if storage_config and isinstance(storage_config, ABFS):
                abfs_config = storage_config
            else:
                abfs_config = ABFS.auto()

            opts = {}
            if abfs_config.account_name:
                opts["account_name"] = abfs_config.account_name
            if abfs_config.account_key:
                opts["account_key"] = abfs_config.account_key
            if abfs_config.tenant_id:
                opts["tenant_id"] = abfs_config.tenant_id
            if abfs_config.client_id:
                opts["client_id"] = abfs_config.client_id
            if abfs_config.client_secret:
                opts["client_secret"] = abfs_config.client_secret
            return opts

        case _:
            return {}


class HuggingFaceDatasetToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(datasets.Dataset, None, PARQUET)

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
        df = typing.cast(datasets.Dataset, dataframe.val)

        if structured_dataset_type.columns:
            columns = [c.name for c in structured_dataset_type.columns]
            existing = set(df.features.keys())
            to_remove = [c for c in existing if c not in columns]
            if to_remove:
                df = df.remove_columns(to_remove)

        filesystem = storage.get_underlying_filesystem(path=path)
        storage_options = get_hf_storage_options(protocol=filesystem.protocol)
        df.to_parquet(path, storage_options=storage_options or None)

        structured_dataset_type.format = PARQUET
        return literals_pb2.StructuredDataset(
            uri=uri, metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type)
        )


class ParquetToHuggingFaceDatasetDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(datasets.Dataset, None, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "datasets.Dataset":
        uri = flyte_value.uri
        columns = None
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]

        parquet_path = os.path.join(uri, f"{0:05}.parquet")
        filesystem = storage.get_underlying_filesystem(path=parquet_path)
        storage_options = get_hf_storage_options(protocol=filesystem.protocol)
        try:
            return datasets.Dataset.from_parquet(parquet_path, columns=columns, storage_options=storage_options or None)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous access")
                storage_options = get_hf_storage_options(protocol=filesystem.protocol, anonymous=True)
                return datasets.Dataset.from_parquet(
                    parquet_path, columns=columns, storage_options=storage_options or None
                )
            else:
                raise


@functools.lru_cache(maxsize=None)
def register_huggingface_df_transformers():
    """Register Hugging Face Dataset encoders and decoders with the DataFrameTransformerEngine.

    This function is called automatically via the flyte.plugins.types entry point
    when flyte.init() is called with load_plugin_type_transformers=True (the default).
    """
    DataFrameTransformerEngine.register(HuggingFaceDatasetToParquetEncodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(ParquetToHuggingFaceDatasetDecodingHandler(), default_format_for_type=True)


# Also register at module import time for backwards compatibility
register_huggingface_df_transformers()
