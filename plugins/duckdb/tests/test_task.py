from flyte.models import SerializationContext
from flyteidl2.core.tasks_pb2 import Sql

from flyteplugins.duckdb.task import DuckDB, DuckDBConfig

SCTX = SerializationContext(version="test")


def _make_task(**kwargs) -> DuckDB:
    defaults = {
        "name": "test",
        "query_template": "SELECT 1",
    }
    defaults.update(kwargs)
    return DuckDB(**defaults)


class TestDuckDBTask:
    def test_minimal_creation(self):
        task = _make_task()
        assert task._TASK_TYPE == "duckdb"
        assert task.query_template == "SELECT 1"
        assert task.plugin_config.database_path == ":memory:"
        assert task.plugin_config.extensions is None

    def test_custom_config(self):
        config = DuckDBConfig(database_path="/data/test.duckdb", extensions=["httpfs", "json"])
        task = _make_task(plugin_config=config)
        assert task.plugin_config.database_path == "/data/test.duckdb"
        assert task.plugin_config.extensions == ["httpfs", "json"]

    def test_whitespace_normalization(self):
        task = _make_task(
            query_template="""
                SELECT *
                FROM    users
                WHERE   id = 1
            """
        )
        assert task.query_template == "SELECT * FROM users WHERE id = 1"

    def test_tab_normalization(self):
        task = _make_task(query_template="SELECT\t*\tFROM\tusers")
        assert task.query_template == "SELECT * FROM users"


class TestCustomConfig:
    def test_default_config(self):
        task = _make_task()
        config = task.custom_config(SCTX)

        assert config["database_path"] == ":memory:"
        assert "extensions" not in config

    def test_custom_database_path(self):
        db_config = DuckDBConfig(database_path="/data/analytics.duckdb")
        task = _make_task(plugin_config=db_config)
        config = task.custom_config(SCTX)

        assert config["database_path"] == "/data/analytics.duckdb"

    def test_with_extensions(self):
        db_config = DuckDBConfig(extensions=["httpfs", "spatial"])
        task = _make_task(plugin_config=db_config)
        config = task.custom_config(SCTX)

        assert config["extensions"] == ["httpfs", "spatial"]

    def test_no_extensions_by_default(self):
        task = _make_task()
        config = task.custom_config(SCTX)

        assert "extensions" not in config

    def test_full_config(self):
        db_config = DuckDBConfig(database_path="/tmp/test.duckdb", extensions=["httpfs"])
        task = _make_task(plugin_config=db_config)
        config = task.custom_config(SCTX)

        assert config == {
            "database_path": "/tmp/test.duckdb",
            "extensions": ["httpfs"],
        }


class TestSql:
    def test_sql_returns_ansi_dialect(self):
        task = _make_task(query_template="SELECT * FROM users")
        sql = task.sql(SCTX)

        assert sql.statement == "SELECT * FROM users"
        assert sql.dialect == Sql.Dialect.ANSI

    def test_sql_with_parameterized_query(self):
        task = _make_task(query_template="SELECT * FROM users WHERE id = %(user_id)s")
        sql = task.sql(SCTX)

        assert sql.statement == "SELECT * FROM users WHERE id = %(user_id)s"
