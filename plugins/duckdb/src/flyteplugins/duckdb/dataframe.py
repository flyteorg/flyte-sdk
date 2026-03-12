import typing

from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import DataFrame, DataFrameDecoder, DataFrameEncoder
from flyteidl2.core import literals_pb2, types_pb2

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    pd = lazy_module("pandas")

DUCKDB = "duckdb"
PROTOCOL_PREFIX = "duckdb://"


def _read_from_duckdb(
    flyte_value: literals_pb2.StructuredDataset,
    current_task_metadata: literals_pb2.StructuredDatasetMetadata,
) -> "pd.DataFrame":
    uri = flyte_value.uri
    if not uri:
        raise ValueError("flyte_value.uri cannot be empty.")

    parquet_path = uri.removeprefix(PROTOCOL_PREFIX)
    return pd.read_parquet(parquet_path)


def _write_to_duckdb(dataframe: DataFrame):
    if not dataframe.uri:
        raise ValueError("dataframe.uri cannot be None.")

    uri = typing.cast(str, dataframe.uri)
    parquet_path = uri.removeprefix(PROTOCOL_PREFIX)
    df = typing.cast("pd.DataFrame", dataframe.val)
    df.to_parquet(parquet_path)


class PandasToDuckDBEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(pd.DataFrame, DUCKDB, "")

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        _write_to_duckdb(dataframe)
        return literals_pb2.StructuredDataset(
            uri=typing.cast(str, dataframe.uri),
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
        )


class DuckDBToPandasDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pd.DataFrame, DUCKDB, "")

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pd.DataFrame":
        return _read_from_duckdb(flyte_value, current_task_metadata)
