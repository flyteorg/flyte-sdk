import os
import re
import typing

from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import DataFrame, DataFrameDecoder, DataFrameEncoder
from flyteidl2.core import literals_pb2, types_pb2

from ._crypto import get_private_key

if typing.TYPE_CHECKING:
    import pandas as pd

    import snowflake.connector
else:
    pd = lazy_module("pandas")

SNOWFLAKE = "snowflake"
PROTOCOL_SEP = "\\/|://|:"


def _get_connection(
    user: str,
    account: str,
    database: str,
    schema: str,
    warehouse: str,
) -> "snowflake.connector.SnowflakeConnection":
    """Create a Snowflake connection using environment-provided credentials."""
    import snowflake.connector

    conn_params: dict[str, typing.Any] = {
        "user": user,
        "account": account,
        "database": database,
        "schema": schema,
        "warehouse": warehouse,
    }

    # The secrets will be injected as environment variables.
    private_key_content = os.environ.get("SNOWFLAKE_PRIVATE_KEY")
    if private_key_content:
        private_key_passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
        conn_params["private_key"] = get_private_key(private_key_content, private_key_passphrase)

    return snowflake.connector.connect(**conn_params)


def _write_to_sf(dataframe: DataFrame):
    if not dataframe.uri:
        raise ValueError("dataframe.uri cannot be None.")

    from snowflake.connector.pandas_tools import write_pandas

    uri = typing.cast(str, dataframe.uri)
    _, user, account, warehouse, database, schema, table = re.split(PROTOCOL_SEP, uri)
    df = typing.cast("pd.DataFrame", dataframe.val)

    conn = _get_connection(user, account, database, schema, warehouse)
    write_pandas(conn, df, table)


def _read_from_sf(
    flyte_value: literals_pb2.StructuredDataset,
    current_task_metadata: literals_pb2.StructuredDatasetMetadata,
) -> "pd.DataFrame":
    uri = flyte_value.uri
    if not uri:
        raise ValueError("flyte_value.uri cannot be empty.")

    _, user, account, warehouse, database, schema, query_id = re.split(PROTOCOL_SEP, uri)

    conn = _get_connection(user, account, database, schema, warehouse)
    cs = conn.cursor()
    cs.get_results_from_sfqid(query_id)
    return cs.fetch_pandas_all()


class PandasToSnowflakeEncodingHandlers(DataFrameEncoder):
    def __init__(self):
        super().__init__(pd.DataFrame, SNOWFLAKE, "")

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        _write_to_sf(dataframe)
        return literals_pb2.StructuredDataset(
            uri=typing.cast(str, dataframe.uri),
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
        )


class SnowflakeToPandasDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pd.DataFrame, SNOWFLAKE, "")

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pd.DataFrame":
        return _read_from_sf(flyte_value, current_task_metadata)
