import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import TaskTemplate
from flyte.io import DataFrame
from flyte.models import NativeInterface, SerializationContext
from flyteidl2.core import tasks_pb2


@dataclass
class SnowflakeConfig(object):
    """
    SnowflakeConfig should be used to configure a Snowflake Task.

    Additional connection parameters (role, authenticator, session_parameters, etc.) can be passed
    via connection_kwargs.
    See: https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api
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
        output_dataframe_type: Optional[Type[DataFrame]] = None,
        snowflake_private_key: Optional[str] = None,
        snowflake_private_key_passphrase: Optional[str] = None,
        **kwargs,
    ):
        """
        To be used to query Snowflake databases.

        :param name: The name of this task, should be unique in the project
        :param query_template: The actual query to run. We use Flyte's Golang templating format for query templating.
        :param plugin_config: SnowflakeConfig object (includes connection_kwargs for additional parameters)
        :param inputs: Name and type of inputs specified as an ordered dictionary
        :param output_dataframe_type: If some data is produced by this query, then you can specify the
         output dataframe type.
        :param snowflake_private_key: The name of the secret containing the Snowflake private key for key-pair auth.
        :param snowflake_private_key_passphrase: The name of the secret containing the private key passphrase
            (if encrypted).

        Note: For password authentication or other auth methods, pass them via plugin_config.connection_kwargs.
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
            image=None,
            **kwargs,
        )
        self.output_dataframe_type = output_dataframe_type
        self.plugin_config = plugin_config
        self.query_template = re.sub(r"\s+", " ", query_template.replace("\n", " ").replace("\t", " ")).strip()
        self.snowflake_private_key = snowflake_private_key
        self.snowflake_private_key_passphrase = snowflake_private_key_passphrase

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        config = {
            "account": self.plugin_config.account,
            "database": self.plugin_config.database,
            "schema": self.plugin_config.schema,
            "warehouse": self.plugin_config.warehouse,
            "user": self.plugin_config.user,
        }

        # Add additional connection parameters
        if self.plugin_config.connection_kwargs:
            config["connection_kwargs"] = self.plugin_config.connection_kwargs

        secrets = {}
        if self.snowflake_private_key is not None:
            secrets["snowflake_private_key"] = self.snowflake_private_key
        if self.snowflake_private_key_passphrase is not None:
            secrets["snowflake_private_key_passphrase"] = self.snowflake_private_key_passphrase
        if secrets:
            config["secrets"] = secrets

        return config

    def sql(self, sctx: SerializationContext) -> Optional[str]:
        sql = tasks_pb2.Sql(statement=self.query_template, dialect=tasks_pb2.Sql.Dialect.ANSI)
        return sql
