import json
import typing
from dataclasses import dataclass

from flyte._utils import lazy_module
from flyte.extend import TaskTemplate
from flyte.io import DataFrame
from flyte.models import NativeInterface, SerializationContext
from flyteidl2.core import tasks_pb2

if typing.TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
else:
    pd = lazy_module("pandas")
    pa = lazy_module("pyarrow")

duckdb = lazy_module("duckdb")


@dataclass
class DuckDBConfig:
    """Configuration for a DuckDB task.

    Args:
        database_path: Path to a DuckDB database file, or ":memory:" for in-memory.
        extensions: List of DuckDB extensions to install and load before query execution
            (e.g., ["httpfs", "spatial", "json"]).
    """

    database_path: str = ":memory:"
    extensions: typing.Optional[typing.List[str]] = None


class DuckDB(TaskTemplate):
    """Run SQL queries against DuckDB as a Flyte task.

    DuckDB is an embedded analytical database (like SQLite for OLAP). Queries execute
    locally and synchronously, with no remote credentials or polling required.

    Supports DataFrame inputs (registered as virtual tables in DuckDB), parameterized
    queries with ``?`` or ``$N`` placeholders, extension loading, and multi-query execution.

    Args:
        name: Task name.
        query: SQL query string or list of queries to execute in sequence. The result of
            the last query is returned. If None, must be provided at runtime via a
            ``query`` string input.
        inputs: Input name-to-type mapping. DataFrame types (``pd.DataFrame``,
            ``pa.Table``, ``flyte.io.DataFrame``) are registered as queryable virtual
            tables. ``list`` or ``str`` types are used as query parameters.
        config: Optional DuckDB configuration. Defaults to in-memory database.

    Example::

        import pandas as pd
        from flyteplugins.duckdb import DuckDB

        analyze = DuckDB(
            name="analyze",
            query="SELECT SUM(a) AS total FROM mydf",
            inputs={"mydf": pd.DataFrame},
        )
    """

    _TASK_TYPE = "duckdb"

    def __init__(
        self,
        name: str,
        query: typing.Optional[typing.Union[str, typing.List[str]]] = None,
        inputs: typing.Optional[typing.Dict[str, type]] = None,
        config: typing.Optional[DuckDBConfig] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            task_type=self._TASK_TYPE,
            image=None,
            interface=NativeInterface(
                {k: (v, None) for k, v in inputs.items()} if inputs else {},
                {"result": DataFrame},
            ),
            **kwargs,
        )
        self._query = query
        self._config = config or DuckDBConfig()

    async def execute(self, **kwargs) -> DataFrame:
        con = duckdb.connect(database=self._config.database_path)
        try:
            for ext in self._config.extensions or []:
                con.install_extension(ext)
                con.load_extension(ext)

            params = None
            query = self._query

            for key, val in kwargs.items():
                if key == "query" and isinstance(val, str):
                    query = val
                elif isinstance(val, (pd.DataFrame, pa.Table)):
                    con.register(key, val)
                elif isinstance(val, DataFrame):
                    raw = val.val
                    if raw is not None:
                        if isinstance(raw, pa.Table):
                            arrow_table = raw
                        elif isinstance(raw, pd.DataFrame):
                            arrow_table = pa.Table.from_pandas(raw)
                        else:
                            arrow_table = pa.table(raw)
                    else:
                        arrow_table = await val.open(pa.Table).all()
                    con.register(key, arrow_table)
                elif isinstance(val, list):
                    params = val
                elif isinstance(val, str):
                    params = json.loads(val)
                else:
                    raise ValueError(f"Unsupported input type for '{key}': {type(val)}")

            if query is None:
                raise ValueError("A query must be provided at task definition or at runtime via a 'query' input.")

            queries = query if isinstance(query, list) else [query]
            if not queries:
                raise ValueError("Query list must not be empty.")
            result = self._execute_queries(con, queries, params)
            return DataFrame.wrap_df(result.to_arrow_table())
        finally:
            con.close()

    def _execute_queries(self, con, queries: typing.List[str], params=None):
        """Execute queries in sequence, returning the DuckDB result of the last one.

        When params is a nested list (params[0] is a list), each parameterized query
        consumes the next element from params in order. Otherwise all parameterized
        queries share the same params list.
        """
        multiple_params = params is not None and len(params) > 0 and isinstance(params[0], list)
        counter = -1
        result = None

        for query in queries:
            has_placeholders = "?" in query or "$" in query

            if has_placeholders and params is not None:
                if multiple_params:
                    counter += 1
                    if counter >= len(params):
                        raise ValueError(f"Not enough parameter sets for parameterized query #{counter + 1}.")
                    current_params = params[counter]
                else:
                    current_params = params

                if query.lstrip().lower().startswith("insert"):
                    result = con.executemany(query, current_params)
                else:
                    result = con.execute(query, current_params)
            else:
                result = con.execute(query)

        return result

    def custom_config(self, sctx: SerializationContext) -> typing.Optional[typing.Dict[str, typing.Any]]:
        config: typing.Dict[str, typing.Any] = {"database_path": self._config.database_path}
        if self._config.extensions:
            config["extensions"] = self._config.extensions
        return config

    def sql(self, sctx: SerializationContext) -> typing.Optional[tasks_pb2.Sql]:
        if self._query is None:
            return None
        statement = self._query[-1] if isinstance(self._query, list) else self._query
        return tasks_pb2.Sql(statement=statement, dialect=tasks_pb2.Sql.Dialect.ANSI)
