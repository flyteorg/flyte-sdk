import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import TaskTemplate
from flyte.models import NativeInterface, SerializationContext
from flyteidl2.core import tasks_pb2


@dataclass
class SnowflakeConfig(object):
    """
    Configure a Snowflake Task using a `SnowflakeConfig` object.

    Additional connection parameters (role, authenticator, session_parameters, etc.) can be passed
    via connection_kwargs.
    See: https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api

    Args:
        account: The Snowflake account identifier.
        database: The Snowflake database name.
        schema: The Snowflake schema name.
        warehouse: The Snowflake warehouse name.
        user: The Snowflake user name.
        connection_kwargs: Optional dictionary of additional Snowflake connection parameters.
    """

    account: str
    database: str
    schema: str
    warehouse: str
    user: str
    connection_kwargs: Optional[Dict[str, Any]] = None


class Snowflake(AsyncConnectorExecutorMixin, TaskTemplate):
    _TASK_TYPE = "snowflake"

    def __init__(
        self,
        name: str,
        query_template: str,
        plugin_config: SnowflakeConfig,
        inputs: Optional[Dict[str, Type]] = None,
        output_dataframe_type: Optional[Type] = None,
        secret_group: Optional[str] = None,
        snowflake_private_key: Optional[str] = None,
        snowflake_private_key_passphrase: Optional[str] = None,
        batch: bool = False,
        **kwargs,
    ):
        """
        Task to run parameterized SQL queries against Snowflake.

        Args:
            name: The name of this task.
            query_template: The actual query to run. This can be parameterized using Python's
                printf-style string formatting with named parameters (e.g. %(param_name)s).
            plugin_config: `SnowflakeConfig` object containing connection metadata.
            inputs: Name and type of inputs specified as a dictionary.
            output_dataframe_type: If some data is produced by this query, then you can specify the
                output dataframe type.
            secret_group: Optional group for secrets in the secret store. The environment variable
                name is auto-generated from ``{secret_group}_{key}``, uppercased with hyphens
                replaced by underscores. If omitted, the key alone is used.
            snowflake_private_key: The secret key for the Snowflake private key (key-pair auth).
            snowflake_private_key_passphrase: The secret key for the private key passphrase
                (if encrypted).
            batch: When True, list inputs are expanded into a multi-row VALUES clause. The
                query_template should contain a single ``VALUES (%(col)s, ...)`` placeholder
                and each input should be a list of equal length.

        Note: For password authentication or other auth methods, pass them via `connection_kwargs`.
        """
        outputs = None
        if output_dataframe_type is not None:
            outputs = {"results": output_dataframe_type}

        super().__init__(
            name=name,
            interface=NativeInterface(
                {k: (v, None) for k, v in inputs.items()} if inputs else {},
                outputs or {},
            ),
            task_type=self._TASK_TYPE,
            **kwargs,
        )

        self.output_dataframe_type = output_dataframe_type
        self.plugin_config = plugin_config
        self.query_template = re.sub(r"\s+", " ", query_template.replace("\n", " ").replace("\t", " ")).strip()
        self.batch = batch
        self.secret_group = secret_group
        self.snowflake_private_key = snowflake_private_key
        self.snowflake_private_key_passphrase = snowflake_private_key_passphrase

    def _to_env_var(self, key: str) -> str:
        """Generate an environment variable name from the secret group and key."""
        env_var = f"{self.secret_group}_{key}" if self.secret_group else key
        return env_var.replace("-", "_").upper()

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        config = {
            "account": self.plugin_config.account,
            "database": self.plugin_config.database,
            "schema": self.plugin_config.schema,
            "warehouse": self.plugin_config.warehouse,
            "user": self.plugin_config.user,
        }

        if self.batch:
            config["batch"] = True

        # Add additional connection parameters
        if self.plugin_config.connection_kwargs:
            config["connection_kwargs"] = self.plugin_config.connection_kwargs

        secrets = {}
        if self.snowflake_private_key is not None:
            secrets["snowflake_private_key"] = self._to_env_var(self.snowflake_private_key)
        if self.snowflake_private_key_passphrase is not None:
            secrets["snowflake_private_key_passphrase"] = self._to_env_var(self.snowflake_private_key_passphrase)
        if secrets:
            config["secrets"] = secrets

        return config

    def sql(self, sctx: SerializationContext) -> Optional[str]:
        sql = tasks_pb2.Sql(statement=self.query_template, dialect=tasks_pb2.Sql.Dialect.ANSI)
        return sql
