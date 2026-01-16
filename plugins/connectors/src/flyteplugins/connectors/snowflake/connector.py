import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    account: str
    user: str
    database: str
    schema: str
    warehouse: str
    query_id: str
    has_output: bool
    connection_kwargs: Optional[Dict[str, Any]] = None


def _get_private_key(
    private_key_content: str, private_key_passphrase: Optional[str] = None
) -> bytes:
    """
    Decode the private key from the secret and return it in DER format.
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

    Supports private key authentication (recommended) and other auth methods via connection_kwargs.
    See: https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api
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
    """Construct a Snowflake console link for the query."""
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


class SnowflakeConnector(AsyncConnector):
    name = "Snowflake Connector"
    task_type_name = TASK_TYPE
    metadata_type = SnowflakeJobMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SnowflakeJobMetadata:
        """
        Submit a query to Snowflake asynchronously.
        """
        custom = (
            json_format.MessageToDict(task_template.custom)
            if task_template.custom
            else {}
        )

        account = custom.get("account")
        if not account:
            raise ValueError(
                f"Missing Snowflake account. Please set it through task configuration."
            )

        user = custom.get("user")
        database = custom.get("database")
        schema = custom.get("schema", "PUBLIC")
        warehouse = custom.get("warehouse")

        if not all([user, database, warehouse]):
            raise ValueError(
                "User, database and warehouse must be specified in the task configuration."
            )

        # Extract authentication secrets from kwargs
        private_key_content = kwargs.get("snowflake_private_key")
        private_key_passphrase = kwargs.get("snowflake_private_key_passphrase")

        # Get additional connection parameters from custom config
        connection_kwargs = custom.get("connection_kwargs", {})

        conn = await _get_snowflake_connection(
            account=account,
            user=user,
            database=database,
            schema=schema,
            warehouse=warehouse,
            private_key_content=private_key_content,
            private_key_passphrase=private_key_passphrase,
            **connection_kwargs,
        )

        query = task_template.sql.statement

        def _execute_query():
            cursor = conn.cursor()
            cursor.execute_async(query)
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
            and len(task_template.interface.outputs) > 0,
            connection_kwargs=connection_kwargs,
        )

    async def get(
        self,
        resource_meta: SnowflakeJobMetadata,
        **kwargs,
    ) -> Resource:
        """
        Poll the status of a Snowflake query.
        """
        # Extract authentication secrets from kwargs
        private_key_content = kwargs.get("snowflake_private_key")
        private_key_passphrase = kwargs.get("snowflake_private_key_passphrase")

        conn = await _get_snowflake_connection(
            account=resource_meta.account,
            user=resource_meta.user,
            database=resource_meta.database,
            schema=resource_meta.schema,
            warehouse=resource_meta.warehouse,
            private_key_content=private_key_content,
            private_key_passphrase=private_key_passphrase,
            **(resource_meta.connection_kwargs or {}),
        )

        log_link = TaskLog(
            uri=_construct_query_link(resource_meta.account, resource_meta.query_id),
            name="Snowflake Dashboard",
        )

        def _get_query_status():
            cursor = conn.cursor()
            try:
                status = cursor.get_query_status_throw_if_error(resource_meta.query_id)
                return status, None
            except Exception as e:
                return None, str(e)
            finally:
                cursor.close()

        loop = asyncio.get_running_loop()
        status, error = await loop.run_in_executor(None, _get_query_status)

        if error:
            logger.error(f"Snowflake query failed: {error}")
            return Resource(
                phase=TaskExecution.FAILED, message=error, log_links=[log_link]
            )

        # Map Snowflake status to Flyte phase
        # Snowflake statuses: RUNNING, SUCCESS, FAILED_WITH_ERROR, ABORTING, etc.
        cur_phase = convert_to_flyte_phase(status.name)
        outputs = None

        if cur_phase == TaskExecution.SUCCEEDED and resource_meta.has_output:
            # Construct the output URI for the results
            output_location = (
                f"snowflake://{resource_meta.account}/{resource_meta.database}/"
                f"{resource_meta.schema}/{resource_meta.query_id}"
            )
            outputs = {"results": DataFrame(uri=output_location)}

        return Resource(
            phase=cur_phase, message=status.name, log_links=[log_link], outputs=outputs
        )

    async def delete(
        self,
        resource_meta: SnowflakeJobMetadata,
        **kwargs,
    ):
        """
        Cancel a running Snowflake query.
        """
        # Extract authentication secrets from kwargs
        private_key_content = kwargs.get("snowflake_private_key")
        private_key_passphrase = kwargs.get("snowflake_private_key_passphrase")

        conn = await _get_snowflake_connection(
            account=resource_meta.account,
            user=resource_meta.user,
            database=resource_meta.database,
            schema=resource_meta.schema,
            warehouse=resource_meta.warehouse,
            private_key_content=private_key_content,
            private_key_passphrase=private_key_passphrase,
            **(resource_meta.connection_kwargs or {}),
        )

        def _cancel_query():
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"SELECT SYSTEM$CANCEL_QUERY('{resource_meta.query_id}')"
                )
            finally:
                cursor.close()
                conn.close()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _cancel_query)
        logger.info(f"Snowflake query {resource_meta.query_id} cancelled")


ConnectorRegistry.register(SnowflakeConnector())
