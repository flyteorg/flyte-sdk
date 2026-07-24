import functools
import typing
from pathlib import Path

import flyte.storage as storage
from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import DataFrame
from flyte.io.extend import (
    DataFrameDecoder,
    DataFrameEncoder,
    DataFrameTransformerEngine,
)
from flyteidl2.core import literals_pb2, types_pb2

if typing.TYPE_CHECKING:
    import pyarrow as pa

    import lance
else:
    lance = lazy_module("lance")
    pa = lazy_module("pyarrow")

# Format string registered with the DataFrame type system. A DataFrame with this
# format is backed by a Lance dataset directory on (possibly remote) storage.
LANCE = "lance"


def get_lance_storage_options(protocol: typing.Optional[str]) -> typing.Dict[str, str]:
    """
    Build a flat string storage_options dict for Lance from Flyte's storage config.

    Lance is backed by the same object_store crate as Polars, so the keys mirror the
    Polars ones closely. The one difference for S3 is the endpoint key: Lance uses
    "aws_endpoint" (Polars uses "aws_endpoint_url").
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
                opts["aws_endpoint"] = s3_config.endpoint
                # Custom endpoints (e.g. a local MinIO) are usually plain HTTP; the
                # object_store client refuses HTTP unless explicitly allowed.
                if s3_config.endpoint.startswith("http://"):
                    opts["aws_allow_http"] = "true"
            return opts

        case "gs" | "gcs":
            # GCS typically uses application default credentials, which the
            # object_store client picks up automatically.
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


def _resolve_write_uri(dataframe: DataFrame) -> str:
    """Pick the URI to write to: an explicit one on the DataFrame, or a fresh
    Flyte-managed path from the task's raw-data prefix (local temp or blob store)."""
    if dataframe.uri:
        return typing.cast(str, dataframe.uri)

    from flyte._context import internal_ctx

    return str(internal_ctx().raw_data.get_random_remote_path())


def _storage_options_for(uri: str) -> typing.Optional[typing.Dict[str, str]]:
    """Storage options for a URI, or None for local paths (Lance wants None locally)."""
    if not storage.is_remote(uri):
        return None
    filesystem = storage.get_underlying_filesystem(path=uri)
    protocol = filesystem.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    return get_lance_storage_options(protocol=protocol) or None


class LanceToLanceDatasetDecodingHandler(DataFrameDecoder):
    """Decode a "lance" DataFrame into a live lance.LanceDataset handle.

    This is the streaming path: the dataset is opened lazily (manifest only) and
    the caller streams from it via scanner()/take()/to_batches(). Column subsetting
    is intentionally left to the caller's scanner(columns=...), which is more
    flexible for streaming than eagerly subsetting here.
    """

    def __init__(self):
        super().__init__(lance.LanceDataset, None, LANCE)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "lance.LanceDataset":
        uri = flyte_value.uri
        return lance.dataset(uri, storage_options=_storage_options_for(uri))


class LanceDatasetToLanceEncodingHandler(DataFrameEncoder):
    """Encode a lance.LanceDataset by copying it into Flyte-managed storage.

    The source dataset is streamed fragment-by-fragment via its RecordBatchReader,
    so this does not materialize the whole dataset in memory.
    """

    def __init__(self):
        super().__init__(lance.LanceDataset, None, LANCE)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        uri = _resolve_write_uri(dataframe)
        if not storage.is_remote(uri):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)

        source = typing.cast("lance.LanceDataset", dataframe.val)
        reader = source.scanner().to_reader()
        lance.write_dataset(reader, uri, storage_options=_storage_options_for(uri))

        structured_dataset_type.format = LANCE
        return literals_pb2.StructuredDataset(
            uri=uri,
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
        )


class ArrowToLanceEncodingHandler(DataFrameEncoder):
    """Encode an in-memory pyarrow.Table as a Lance dataset.

    Not registered as the default for pyarrow.Table (that stays Parquet); opt in
    with an explicit "lance" format, e.g. Annotated[DataFrame, "lance"].
    """

    def __init__(self):
        super().__init__(pa.Table, None, LANCE)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        uri = _resolve_write_uri(dataframe)
        if not storage.is_remote(uri):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)

        table = typing.cast("pa.Table", dataframe.val)
        lance.write_dataset(table, uri, storage_options=_storage_options_for(uri))

        structured_dataset_type.format = LANCE
        return literals_pb2.StructuredDataset(
            uri=uri,
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
        )


class LanceToArrowDecodingHandler(DataFrameDecoder):
    """Decode a "lance" DataFrame eagerly into a pyarrow.Table.

    This materializes the whole dataset in memory. For large or multimodal datasets
    (e.g. image bytes) decode to lance.LanceDataset and stream instead.
    """

    def __init__(self):
        super().__init__(pa.Table, None, LANCE)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pa.Table":
        uri = flyte_value.uri
        columns = None
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
        return lance.dataset(uri, storage_options=_storage_options_for(uri)).to_table(columns=columns)


@functools.lru_cache(maxsize=None)
def register_lance_df_transformers():
    """Register Lance DataFrame encoders and decoders with the DataFrameTransformerEngine.

    This function is called automatically via the flyte.plugins.types entry point
    when flyte.init() is called with load_plugin_type_transformers=True (the default).
    """
    DataFrameTransformerEngine.register(LanceDatasetToLanceEncodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(LanceToLanceDatasetDecodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(ArrowToLanceEncodingHandler())
    DataFrameTransformerEngine.register(LanceToArrowDecodingHandler())


# Also register at module import time for backwards compatibility
register_lance_df_transformers()
