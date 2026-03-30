# DuckDB Plugin for Flyte

Run DuckDB SQL queries as Flyte tasks with DataFrame inputs, parameterized queries, and extension support.

DuckDB is an embedded analytical database (like SQLite for OLAP). Queries execute locally and synchronously.

## Installation

```bash
pip install flyteplugins-duckdb
```

## Quick start

```python
import pandas as pd
from flyteplugins.duckdb import DuckDB

analyze = DuckDB(
    name="analyze",
    query="SELECT SUM(a) AS total FROM mydf",
    inputs={"mydf": pd.DataFrame},
)
```

## DataFrame inputs

Pass pandas DataFrames or PyArrow Tables as inputs. They are registered as virtual tables queryable by name:

```python
import pyarrow as pa

task = DuckDB(
    name="join_tables",
    query="SELECT a.name, b.total FROM users a JOIN orders b ON a.id = b.user_id",
    inputs={"users": pd.DataFrame, "orders": pa.Table},
)
```

You can also pass `flyte.io.DataFrame` for interoperability with any DataFrame type in the Flyte ecosystem.

## Parameterized queries

Use `?` or `$N` placeholders with list parameters:

```python
task = DuckDB(
    name="filtered",
    query="SELECT * FROM mydf WHERE age > ?",
    inputs={"mydf": pd.DataFrame, "params": list},
)
```

## Multiple queries

Pass a list of queries. All are executed in order and the result of the last query is returned:

```python
task = DuckDB(
    name="etl",
    query=[
        "CREATE TABLE staging AS SELECT * FROM raw WHERE active = true",
        "SELECT department, COUNT(*) AS cnt FROM staging GROUP BY department",
    ],
    inputs={"raw": pd.DataFrame},
)
```

## Runtime queries

Omit `query` and provide it at execution time via a `query` string input:

```python
task = DuckDB(
    name="dynamic",
    inputs={"mydf": pd.DataFrame, "query": str},
)
```

## Extensions

DuckDB extensions are auto-installed and loaded before query execution:

```python
from flyteplugins.duckdb import DuckDBConfig

task = DuckDB(
    name="s3_query",
    query="SELECT * FROM 's3://bucket/data.parquet' LIMIT 100",
    config=DuckDBConfig(extensions=["httpfs"]),
)
```

Common extensions: `httpfs`, `json`, `spatial`, `excel`, `parquet`.

## Configuration

```python
from flyteplugins.duckdb import DuckDBConfig

config = DuckDBConfig(
    database_path=":memory:",      # default; or "/path/to/file.duckdb"
    extensions=["httpfs", "json"],
)

task = DuckDB(name="my_task", query="SELECT 1", config=config)
```
