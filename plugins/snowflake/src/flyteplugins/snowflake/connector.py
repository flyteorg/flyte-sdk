import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from async_lru import alru_cache
from flyte import logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyte.connectors.utils import convert_to_flyte_phase
from flyte.io import DataFrame
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf import json_format
from snowflake import connector

TASK_TYPE = "snowflake"


@dataclass
class SnowflakeJobMetadata(ResourceMeta):
    """
    Metadata for a Snowflake query job.

    Attributes:
        account: Snowflake account identifier.
        user: Snowflake user name.
        database: Snowflake database name.
        schema: Snowflake schema name.
        warehouse: Snowflake warehouse name.
        query_id: Unique identifier for the submitted query.
        has_output: Indicates if the query produces output.
        connection_kwargs: Additional connection parameters.
    """

    account: str
    user: str
    database: str
    schema: str
    warehouse: str
    query_id: str
    has_output: bool
    connection_kwargs: Optional[Dict[str, Any]] = None


def _get_private_key(private_key_content: str, private_key_passphrase: Optional[str] = None) -> bytes:
    """
    Decode the private key from the secret and return it in DER format.

    Args:
        private_key_content: The private key content in PEM format.
        private_key_passphrase: The passphrase for the private key, if any.

    Returns:
        The private key in DER format.
    """
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


@alru_cache
async def _get_snowflake_connection(
    account: str,
    user: str,
    database: str,
    schema: str,
    warehouse: str,
    private_key_content: Optional[str] = None,
    private_key_passphrase: Optional[str] = None,
    **connection_kwargs,
) -> connector.SnowflakeConnection:
    """
    Create and return a Snowflake connection.

    Supports private key authentication (recommended) and other auth methods via `connection_kwargs`.
    See: https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api

    Args:
        account: Snowflake account identifier.
        user: Snowflake user name.
        database: Snowflake database name.
        schema: Snowflake schema name.
        warehouse: Snowflake warehouse name.
        private_key_content: The private key content in PEM format, if any.
        private_key_passphrase: The passphrase for the private key, if any.
        **connection_kwargs: Additional connection parameters.

    Returns:
        An active Snowflake connection.
    """

    def _create_connection():
        connection_params = {
            "account": account,
            "user": user,
            "database": database,
            "schema": schema,
            "warehouse": warehouse,
            **connection_kwargs,
        }

        # Add private key authentication if provided
        if private_key_content:
            private_key = _get_private_key(private_key_content, private_key_passphrase)
            connection_params["private_key"] = private_key

        # Let Snowflake connector validate authentication requirements
        return connector.connect(**connection_params)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _create_connection)


def _construct_query_link(account: str, query_id: str) -> str:
    """Construct a Snowflake console link for the query.

    Args:
        account: Snowflake account identifier.
        query_id: Unique identifier for the submitted query.

    Returns:
        URL to the Snowflake query dashboard for the given query ID.
    """
    if "-" in account:
        parts = account.split("-", 1)
        if len(parts) == 2:
            org_name, account_name = parts
            base_url = f"https://app.snowflake.com/{org_name}/{account_name}"
        else:
            base_url = f"https://app.snowflake.com/{account}"
    else:
        # Simple account locator
        base_url = f"https://app.snowflake.com/{account}"

    return f"{base_url}/#/compute/history/queries/{query_id}/detail"


def _expand_batch_query(query: str, inputs: Dict[str, List[Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Expand a parameterized INSERT query with list inputs into a multi-row VALUES statement.

    Example:
        query = "INSERT INTO t (a, b) VALUES (%(a)s, %(b)s)"
        inputs = {"a": [1, 2], "b": ["x", "y"]}

    Produces:
        "INSERT INTO t (a, b) VALUES (%(a_0)s, %(b_0)s), (%(a_1)s, %(b_1)s)"
        flat_params = {"a_0": 1, "b_0": "x", "a_1": 2, "b_1": "y"}

    Constraints:
    - Query must contain exactly one VALUES (...) clause
    - All input lists must have the same non-zero length
    """
    keys = list(inputs.keys())

    lengths = {len(v) for v in inputs.values()}
    if len(lengths) != 1:
        raise ValueError("All batch input lists must have the same length.")

    n_rows = lengths.pop()
    if n_rows == 0:
        raise ValueError("Batch inputs must contain at least one row.")

    # Find all VALUES (...) clauses (case-insensitive)
    # The inner group handles %(...)s placeholders which contain literal parentheses
    matches = list(re.finditer(r"VALUES\s*\(((?:[^()]*|\([^)]*\))*)\)", query, re.IGNORECASE))
    if not matches:
        raise ValueError("Batch inputs require a query template with a VALUES (...) clause.")
    if len(matches) > 1:
        raise ValueError("Batch query expansion supports exactly one VALUES (...) clause.")

    match = matches[0]
    values_content = match.group(1)

    value_rows = []
    flat_params: Dict[str, Any] = {}

    for i in range(n_rows):
        row_content = values_content
        for key in keys:
            row_content = re.sub(
                rf"%\({re.escape(key)}\)s",
                f"%({key}_{i})s",
                row_content,
            )
            flat_params[f"{key}_{i}"] = inputs[key][i]
        value_rows.append(f"({row_content})")

    before = query[: match.start()]
    after = query[match.end() :]

    expanded_query = before + "VALUES " + ", ".join(value_rows) + after
    return expanded_query, flat_params


class SnowflakeConnector(AsyncConnector):
    name = "Snowflake Connector"
    task_type_name = TASK_TYPE
    metadata_type = SnowflakeJobMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        inputs: Optional[Dict[str, Any]] = None,
        snowflake_private_key: Optional[str] = None,
        snowflake_private_key_passphrase: Optional[str] = None,
        **kwargs,
    ) -> SnowflakeJobMetadata:
        """
        Submit a query to Snowflake asynchronously.

        Args:
            task_template: The Flyte task template containing the SQL query and configuration.
            inputs: Optional dictionary of input parameters for parameterized queries.
            snowflake_private_key: The private key content set as a Flyte secret.
            snowflake_private_key_passphrase: The passphrase for the private key set as a Flyte secret, if any.

        Returns:
            A SnowflakeJobMetadata object containing the query ID and link to the query dashboard.
        """
        custom = json_format.MessageToDict(task_template.custom) if task_template.custom else {}

        account = custom.get("account")
        if not account:
            raise ValueError("Missing Snowflake account. Set it through task configuration.")

        user = custom.get("user")
        database = custom.get("database")
        schema = custom.get("schema", "PUBLIC")
        warehouse = custom.get("warehouse")

        if not all([user, database, warehouse]):
            raise ValueError("User, database and warehouse must be specified in the task configuration.")

        # Get additional connection parameters from custom config
        connection_kwargs = custom.get("connection_kwargs", {})

        conn = await _get_snowflake_connection(
            account=account,
            user=user,
            database=database,
            schema=schema,
            warehouse=warehouse,
            private_key_content=snowflake_private_key,
            private_key_passphrase=snowflake_private_key_passphrase,
            **connection_kwargs,
        )

        query = task_template.sql.statement
        batch = custom.get("batch", False)

        def _execute_query():
            cursor = conn.cursor()

            if batch and inputs:
                expanded_query, flat_params = _expand_batch_query(query, inputs)
                cursor.execute_async(expanded_query, flat_params)
            else:
                cursor.execute_async(query, inputs)

            query_id = cursor.sfqid
            cursor.close()

            return query_id

        loop = asyncio.get_running_loop()
        query_id = await loop.run_in_executor(None, _execute_query)

        logger.info(f"Snowflake query submitted with ID: {query_id}")

        return SnowflakeJobMetadata(
            account=account,
            user=user,
            database=database,
            schema=schema,
            warehouse=warehouse,
            query_id=query_id,
            has_output=task_template.interface.outputs is not None
            and len(task_template.interface.outputs.variables) > 0,
            connection_kwargs=connection_kwargs,
        )

    async def get(
        self,
        resource_meta: SnowflakeJobMetadata,
        snowflake_private_key: Optional[str] = None,
        snowflake_private_key_passphrase: Optional[str] = None,
        **kwargs,
    ) -> Resource:
        """
        Poll the status of a Snowflake query.

        Args:
            resource_meta: The SnowflakeJobMetadata containing the query ID.
            snowflake_private_key: The private key content set as a Flyte secret.
            snowflake_private_key_passphrase: The passphrase for the private key set as a Flyte secret, if any.

        Returns:
            A Resource object containing the query results and a link to the query dashboard.
        """
        conn = await _get_snowflake_connection(
            account=resource_meta.account,
            user=resource_meta.user,
            database=resource_meta.database,
            schema=resource_meta.schema,
            warehouse=resource_meta.warehouse,
            private_key_content=snowflake_private_key,
            private_key_passphrase=snowflake_private_key_passphrase,
            **(resource_meta.connection_kwargs or {}),
        )

        log_link = TaskLog(
            uri=_construct_query_link(resource_meta.account, resource_meta.query_id),
            name="Snowflake Dashboard",
            ready=True,
            link_type=TaskLog.DASHBOARD,
        )

        def _get_query_status():
            try:
                status = conn.get_query_status_throw_if_error(resource_meta.query_id)
                return status, None
            except Exception as e:
                return None, str(e)

        loop = asyncio.get_running_loop()
        status, error = await loop.run_in_executor(None, _get_query_status)

        if error:
            logger.error(f"Snowflake query failed: {error}")
            return Resource(phase=TaskExecution.FAILED, message=error, log_links=[log_link])

        # Map Snowflake status to Flyte phase
        # Snowflake statuses: RUNNING, SUCCESS, FAILED_WITH_ERROR, ABORTING, etc.
        cur_phase = convert_to_flyte_phase(status.name)
        outputs = None

        if cur_phase == TaskExecution.SUCCEEDED and resource_meta.has_output:
            # Construct the output URI for the results
            output_location = (
                f"snowflake://{resource_meta.user}/{resource_meta.account}/{resource_meta.warehouse}/"
                f"{resource_meta.database}/{resource_meta.schema}/{resource_meta.query_id}"
            )
            outputs = {"results": DataFrame(uri=output_location)}

        return Resource(phase=cur_phase, message=status.name, log_links=[log_link], outputs=outputs)

    async def delete(
        self,
        resource_meta: SnowflakeJobMetadata,
        snowflake_private_key: Optional[str] = None,
        snowflake_private_key_passphrase: Optional[str] = None,
        **kwargs,
    ):
        """
        Cancel a running Snowflake query.

        Args:
            resource_meta: The SnowflakeJobMetadata containing the query ID.
            snowflake_private_key: The private key content set as a Flyte secret.
            snowflake_private_key_passphrase: The passphrase for the private key set as a Flyte secret, if any.
        """
        conn = await _get_snowflake_connection(
            account=resource_meta.account,
            user=resource_meta.user,
            database=resource_meta.database,
            schema=resource_meta.schema,
            warehouse=resource_meta.warehouse,
            private_key_content=snowflake_private_key,
            private_key_passphrase=snowflake_private_key_passphrase,
            **(resource_meta.connection_kwargs or {}),
        )

        def _cancel_query():
            cursor = conn.cursor()
            try:
                cursor.execute(f"SELECT SYSTEM$CANCEL_QUERY('{resource_meta.query_id}')")
            finally:
                cursor.close()
                conn.close()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _cancel_query)

        logger.info(f"Snowflake query {resource_meta.query_id} cancelled")


ConnectorRegistry.register(SnowflakeConnector())
