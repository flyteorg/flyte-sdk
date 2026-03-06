# DuckDB Plugin for Flyte

Run DuckDB SQL queries as Flyte tasks with parameterized inputs, extension support, and DataFrame output.

DuckDB is an embedded analytical database (like SQLite for OLAP). Queries execute locally and synchronously, so no remote credentials or connection setup is required.

## Installation

```bash
pip install flyteplugins-duckdb
```

## Quick start

```python
from flyteplugins.duckdb import DuckDB, DuckDBConfig

import flyte

config = DuckDBConfig()

query = DuckDB(
    name="count_rows",
    query_template="SELECT COUNT(*) AS total FROM 'data.parquet'",
    plugin_config=config,
    output_dataframe_type=pd.DataFrame,
)
```

## In-memory queries

By default, DuckDB runs in-memory. This is ideal for ad-hoc analytics and querying files directly:

```python
config = DuckDBConfig()  # defaults to database_path=":memory:"

task = DuckDB(
    name="analyze",
    query_template="SELECT * FROM 'sales.parquet' WHERE amount > 100",
    plugin_config=config,
    output_dataframe_type=pd.DataFrame,
)
```

## File-based databases

To query a persistent DuckDB database file:

```python
config = DuckDBConfig(database_path="/data/analytics.duckdb")

task = DuckDB(
    name="query_db",
    query_template="SELECT * FROM customers LIMIT 10",
    plugin_config=config,
    output_dataframe_type=pd.DataFrame,
)
```

## Parameterized queries

Use `%(name)s` placeholders and typed `inputs`:

```python
lookup = DuckDB(
    name="lookup_user",
    query_template="SELECT * FROM 'users.parquet' WHERE id = %(user_id)s",
    plugin_config=config,
    inputs={"user_id": int},
    output_dataframe_type=pd.DataFrame,
)
```

## Extensions

DuckDB supports extensions for additional functionality. Install and load them via `DuckDBConfig.extensions`:

```python
config = DuckDBConfig(extensions=["httpfs"])

task = DuckDB(
    name="query_s3",
    query_template="SELECT * FROM 's3://bucket/data.parquet' LIMIT 100",
    plugin_config=config,
    output_dataframe_type=pd.DataFrame,
)
```

Common extensions:
- `httpfs` - Read files from HTTP/S3
- `spatial` - Geospatial functions
- `json` - JSON processing
- `excel` - Read Excel files

## Reading results as DataFrames

Set `output_dataframe_type` to get query results as a pandas DataFrame:

```python
import pandas as pd

select_task = DuckDB(
    name="get_data",
    query_template="SELECT * FROM 'data.parquet'",
    plugin_config=config,
    output_dataframe_type=pd.DataFrame,
)
```

## Full example

```python
import pandas as pd
from flyteplugins.duckdb import DuckDB, DuckDBConfig

import flyte

config = DuckDBConfig(extensions=["httpfs"])

analyze_task = DuckDB(
    name="analyze_sales",
    query_template="SELECT region, SUM(amount) as total FROM 'sales.parquet' GROUP BY region",
    plugin_config=config,
    output_dataframe_type=pd.DataFrame,
)

duckdb_env = flyte.TaskEnvironment.from_task("duckdb_env", analyze_task)

env = flyte.TaskEnvironment(
    name="example_env",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-duckdb"),
    depends_on=[duckdb_env],
)


@env.task
async def main() -> float:
    df = await analyze_task()
    return df["total"].sum().item()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(main)
    print(run.url)
```
