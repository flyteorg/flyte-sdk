import os
import re
import typing
from typing import Optional

import snowflake.connector
from flyteidl2.core import literals_pb2, types_pb2
from snowflake.connector.pandas_tools import write_pandas

from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import DataFrame, DataFrameDecoder, DataFrameEncoder

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    pd = lazy_module("pandas")

SNOWFLAKE = "snowflake"
PROTOCOL_SEP = "\\/|://|:"


def _get_private_key(
    private_key_content: str, private_key_passphrase: Optional[str] = None
) -> bytes:
    """Decode a PEM private key and return it in DER format."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    private_key_bytes = private_key_content.strip().encode()
    password = private_key_passphrase.encode() if private_key_passphrase else None

    private_key = serialization.load_pem_private_key(
        private_key_bytes,
        password=password,
        backend=default_backend(),
    )

    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _get_connection(
    user: str,
    account: str,
    database: str,
    schema: str,
    warehouse: str,
) -> snowflake.connector.SnowflakeConnection:
    """Create a Snowflake connection using environment-provided credentials."""
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
        conn_params["private_key"] = _get_private_key(
            private_key_content, private_key_passphrase
        )

    return snowflake.connector.connect(**conn_params)


def _write_to_sf(dataframe: DataFrame):
    if not dataframe.uri:
        raise ValueError("dataframe.uri cannot be None.")

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

    _, user, account, warehouse, database, schema, query_id = re.split(
        PROTOCOL_SEP, uri
    )

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
            metadata=literals_pb2.StructuredDatasetMetadata(
                structured_dataset_type=structured_dataset_type
            ),
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
