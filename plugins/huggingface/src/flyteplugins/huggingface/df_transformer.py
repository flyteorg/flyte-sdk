import functools
import os
import typing
from pathlib import Path

import flyte.storage as storage
from flyte import logger
from flyte._utils import lazy_module
from flyte.io import PARQUET, DataFrame
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


def _get_storage_options(protocol: typing.Optional[str], anonymous: bool = False) -> typing.Dict[str, typing.Any]:
    if not protocol:
        return {}
    return storage.get_configured_fsspec_kwargs(protocol=protocol, anonymous=anonymous)


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

        filesystem = storage.get_underlying_filesystem(path=path)
        storage_options = _get_storage_options(protocol=filesystem.protocol)
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
        storage_options = _get_storage_options(protocol=filesystem.protocol)
        try:
            return datasets.Dataset.from_parquet(parquet_path, columns=columns, storage_options=storage_options or None)
        except Exception as exc:
            if exc.__class__.__name__ == "NoCredentialsError":
                logger.debug("S3 source detected, attempting anonymous access")
                storage_options = _get_storage_options(protocol=filesystem.protocol, anonymous=True)
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


register_huggingface_df_transformers()
