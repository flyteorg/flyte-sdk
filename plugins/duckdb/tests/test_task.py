import json

import pandas as pd
import pyarrow as pa
import pytest
from flyte.io import DataFrame
from flyte.models import SerializationContext
from flyteidl2.core.tasks_pb2 import Sql

from flyteplugins.duckdb import DuckDB, DuckDBConfig

SCTX = SerializationContext(version="test")


def _make_task(**kwargs) -> DuckDB:
    defaults = {"name": "test", "query": "SELECT 1"}
    defaults.update(kwargs)
    return DuckDB(**defaults)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestDuckDBConfig:
    def test_defaults(self):
        config = DuckDBConfig()
        assert config.database_path == ":memory:"
        assert config.extensions is None

    def test_custom(self):
        config = DuckDBConfig(database_path="/tmp/test.duckdb", extensions=["httpfs", "json"])
        assert config.database_path == "/tmp/test.duckdb"
        assert config.extensions == ["httpfs", "json"]


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------


class TestDuckDBTask:
    def test_task_type(self):
        task = _make_task()
        assert task._TASK_TYPE == "duckdb"
        assert task.task_type == "duckdb"

    def test_default_config(self):
        task = _make_task()
        assert task._config.database_path == ":memory:"
        assert task._config.extensions is None

    def test_custom_config(self):
        config = DuckDBConfig(database_path="/data/test.duckdb", extensions=["json"])
        task = _make_task(config=config)
        assert task._config.database_path == "/data/test.duckdb"
        assert task._config.extensions == ["json"]

    def test_no_image(self):
        task = _make_task()
        assert task.image is None


# ---------------------------------------------------------------------------
# Serialization: custom_config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    def test_default(self):
        task = _make_task()
        config = task.custom_config(SCTX)
        assert config == {"database_path": ":memory:"}

    def test_with_extensions(self):
        task = _make_task(config=DuckDBConfig(extensions=["httpfs", "spatial"]))
        config = task.custom_config(SCTX)
        assert config == {"database_path": ":memory:", "extensions": ["httpfs", "spatial"]}

    def test_full(self):
        task = _make_task(config=DuckDBConfig(database_path="/tmp/t.duckdb", extensions=["json"]))
        config = task.custom_config(SCTX)
        assert config == {"database_path": "/tmp/t.duckdb", "extensions": ["json"]}


# ---------------------------------------------------------------------------
# Serialization: sql
# ---------------------------------------------------------------------------


class TestSql:
    def test_single_query(self):
        task = _make_task(query="SELECT * FROM users")
        sql = task.sql(SCTX)
        assert sql.statement == "SELECT * FROM users"
        assert sql.dialect == Sql.Dialect.ANSI

    def test_multi_query_returns_last(self):
        task = _make_task(query=["CREATE TABLE t (id INT)", "SELECT * FROM t"])
        sql = task.sql(SCTX)
        assert sql.statement == "SELECT * FROM t"

    def test_no_query_returns_none(self):
        task = DuckDB(name="dynamic", query=None, inputs={"query": str})
        assert task.sql(SCTX) is None


# ---------------------------------------------------------------------------
# Execute: basic queries
# ---------------------------------------------------------------------------


class TestExecute:
    @pytest.mark.asyncio
    async def test_simple_select(self):
        task = _make_task(query="SELECT 42 AS answer")
        result = await task.execute()
        df = result.val.to_pandas()
        assert df["answer"].iloc[0] == 42

    @pytest.mark.asyncio
    async def test_range_query(self):
        task = _make_task(query="SELECT * FROM range(5)")
        result = await task.execute()
        df = result.val.to_pandas()
        assert len(df) == 5

    @pytest.mark.asyncio
    async def test_no_query_raises(self):
        task = DuckDB(name="empty", query=None)
        with pytest.raises(ValueError, match="query must be provided"):
            await task.execute()

    @pytest.mark.asyncio
    async def test_empty_query_list_raises(self):
        task = DuckDB(name="empty_list", query=[])
        with pytest.raises(ValueError, match="must not be empty"):
            await task.execute()

    @pytest.mark.asyncio
    async def test_ddl_only_query_returns_empty(self):
        task = _make_task(query="CREATE TABLE t (id INTEGER)")
        result = await task.execute()
        assert result.val is not None


# ---------------------------------------------------------------------------
# Execute: DataFrame inputs
# ---------------------------------------------------------------------------


class TestDataFrameInputs:
    @pytest.mark.asyncio
    async def test_pandas_input(self):
        task = _make_task(
            query="SELECT SUM(a) AS total FROM mydf",
            inputs={"mydf": pd.DataFrame},
        )
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = await task.execute(mydf=df)
        out = result.val.to_pandas()
        assert out["total"].iloc[0] == 6

    @pytest.mark.asyncio
    async def test_arrow_input(self):
        task = _make_task(
            query="SELECT * FROM arrow_table WHERE i = 2",
            inputs={"arrow_table": pa.Table},
        )
        table = pa.table({"i": [1, 2, 3], "j": ["a", "b", "c"]})
        result = await task.execute(arrow_table=table)
        out = result.val.to_pandas()
        assert len(out) == 1
        assert out["j"].iloc[0] == "b"

    @pytest.mark.asyncio
    async def test_flyte_dataframe_input(self):
        task = _make_task(
            query="SELECT SUM(a) AS total FROM mydf",
            inputs={"mydf": DataFrame},
        )
        raw = pd.DataFrame({"a": [10, 20, 30]})
        fdf = DataFrame.from_df(raw)
        result = await task.execute(mydf=fdf)
        out = result.val.to_pandas()
        assert out["total"].iloc[0] == 60

    @pytest.mark.asyncio
    async def test_multiple_dataframe_inputs(self):
        task = _make_task(
            query="SELECT a.x, b.y FROM df_a a JOIN df_b b ON a.id = b.id",
            inputs={"df_a": pd.DataFrame, "df_b": pd.DataFrame},
        )
        df_a = pd.DataFrame({"id": [1, 2], "x": ["foo", "bar"]})
        df_b = pd.DataFrame({"id": [1, 2], "y": [100, 200]})
        result = await task.execute(df_a=df_a, df_b=df_b)
        out = result.val.to_pandas()
        assert len(out) == 2
        assert set(out["y"]) == {100, 200}


# ---------------------------------------------------------------------------
# Execute: parameterized queries
# ---------------------------------------------------------------------------


class TestInsertDetection:
    @pytest.mark.asyncio
    async def test_column_named_insert_not_treated_as_insert(self):
        """A SELECT on a column named 'insert_date' should use execute(), not executemany()."""
        task = _make_task(
            query=[
                "CREATE TABLE log (insert_date DATE, val INTEGER)",
                "INSERT INTO log VALUES ('2026-01-01', 1)",
                "SELECT insert_date FROM log WHERE val = ?",
            ],
            inputs={"params": list},
        )
        result = await task.execute(params=[1])
        out = result.val.to_pandas()
        assert len(out) == 1


class TestParameterizedQueries:
    @pytest.mark.asyncio
    async def test_positional_params(self):
        task = _make_task(
            query="SELECT * FROM range(10) WHERE range > ?",
            inputs={"params": list},
        )
        result = await task.execute(params=[5])
        out = result.val.to_pandas()
        assert len(out) == 4
        assert all(out["range"] > 5)

    @pytest.mark.asyncio
    async def test_dollar_params(self):
        task = _make_task(
            query="SELECT $1 AS col1, $2 AS col2",
            inputs={"params": list},
        )
        result = await task.execute(params=["hello", "world"])
        out = result.val.to_pandas()
        assert out["col1"].iloc[0] == "hello"
        assert out["col2"].iloc[0] == "world"

    @pytest.mark.asyncio
    async def test_json_string_params(self):
        task = _make_task(
            query="SELECT $1 AS val",
            inputs={"params": str},
        )
        result = await task.execute(params=json.dumps(["test_value"]))
        out = result.val.to_pandas()
        assert out["val"].iloc[0] == "test_value"


# ---------------------------------------------------------------------------
# Execute: multi-query
# ---------------------------------------------------------------------------


class TestMultiQuery:
    @pytest.mark.asyncio
    async def test_create_insert_select(self):
        task = _make_task(
            query=[
                "CREATE TABLE items (name VARCHAR, price INTEGER)",
                "INSERT INTO items VALUES ('apple', 1), ('banana', 2)",
                "SELECT SUM(price) AS total FROM items",
            ],
        )
        result = await task.execute()
        out = result.val.to_pandas()
        assert out["total"].iloc[0] == 3

    @pytest.mark.asyncio
    async def test_multi_query_with_multi_params(self):
        task = _make_task(
            query=[
                "CREATE TABLE items (name VARCHAR, price DECIMAL(10,2))",
                "INSERT INTO items VALUES (?, ?)",
                "SELECT $1 AS col1, $2 AS col2",
            ],
            inputs={"params": str},
        )
        params = [[["apple", 1.0], ["banana", 2.0]], ["hello", "world"]]
        result = await task.execute(params=json.dumps(params))
        out = result.val.to_pandas()
        assert out["col1"].iloc[0] == "hello"
        assert out["col2"].iloc[0] == "world"


# ---------------------------------------------------------------------------
# Execute: runtime query
# ---------------------------------------------------------------------------


class TestRuntimeQuery:
    @pytest.mark.asyncio
    async def test_query_from_input(self):
        task = DuckDB(
            name="dynamic",
            query=None,
            inputs={"mydf": pd.DataFrame, "query": str},
        )
        df = pd.DataFrame({"x": [10, 20, 30]})
        result = await task.execute(mydf=df, query="SELECT MAX(x) AS max_x FROM mydf")
        out = result.val.to_pandas()
        assert out["max_x"].iloc[0] == 30

    @pytest.mark.asyncio
    async def test_runtime_query_overrides_default(self):
        task = _make_task(
            query="SELECT 1 AS original",
            inputs={"query": str},
        )
        result = await task.execute(query="SELECT 99 AS overridden")
        out = result.val.to_pandas()
        assert out["overridden"].iloc[0] == 99


# ---------------------------------------------------------------------------
# Execute: extensions
# ---------------------------------------------------------------------------


class TestExtensions:
    @pytest.mark.asyncio
    async def test_json_extension(self):
        task = _make_task(
            query="SELECT json_extract('{\"key\": \"value\"}', '$.key') AS val",
            config=DuckDBConfig(extensions=["json"]),
        )
        result = await task.execute()
        out = result.val.to_pandas()
        assert "value" in str(out["val"].iloc[0])
