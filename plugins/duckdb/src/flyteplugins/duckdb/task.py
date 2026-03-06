import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import TaskTemplate
from flyte.models import NativeInterface, SerializationContext
from flyteidl2.core import tasks_pb2


@dataclass
class DuckDBConfig(object):
    """
    Configure a DuckDB Task using a `DuckDBConfig` object.

    Args:
        database_path: Path to a DuckDB database file, or ":memory:" for an in-memory database.
        extensions: Optional list of DuckDB extensions to install and load before executing
            the query (e.g., ["httpfs", "parquet"]).
    """

    database_path: str = ":memory:"
    extensions: Optional[List[str]] = None


class DuckDB(AsyncConnectorExecutorMixin, TaskTemplate):
    _TASK_TYPE = "duckdb"

    def __init__(
        self,
        name: str,
        query_template: str,
        plugin_config: Optional[DuckDBConfig] = None,
        inputs: Optional[Dict[str, Type]] = None,
        output_dataframe_type: Optional[Type] = None,
        **kwargs,
    ):
        """
        Task to run SQL queries against DuckDB.

        DuckDB is an embedded analytical database (like SQLite for OLAP). Queries execute
        locally and synchronously, so no remote credentials or polling are required.

        Args:
            name: The name of this task.
            query_template: The SQL query to run. This can be parameterized using Python's
                printf-style string formatting with named parameters (e.g. %(param_name)s).
            plugin_config: Optional `DuckDBConfig` object. Defaults to in-memory database.
            inputs: Name and type of inputs specified as a dictionary.
            output_dataframe_type: If the query produces data, specify the output dataframe type.
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
        self.plugin_config = plugin_config or DuckDBConfig()
        self.query_template = re.sub(r"\s+", " ", query_template.replace("\n", " ").replace("\t", " ")).strip()

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        config = {
            "database_path": self.plugin_config.database_path,
        }

        if self.plugin_config.extensions:
            config["extensions"] = self.plugin_config.extensions

        return config

    def sql(self, sctx: SerializationContext) -> Optional[str]:
        sql = tasks_pb2.Sql(statement=self.query_template, dialect=tasks_pb2.Sql.Dialect.ANSI)
        return sql
