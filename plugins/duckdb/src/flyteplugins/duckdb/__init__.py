"""
Key features:

- Run SQL queries against DuckDB (in-memory or file-based)
- Parameterized SQL queries with typed inputs
- Query Parquet, CSV, and JSON files directly
- Load DuckDB extensions (httpfs, spatial, etc.)
- Returns query results as DataFrames

Basic usage example:
```python
import flyte
from flyte.io import DataFrame
from flyteplugins.duckdb import DuckDB, DuckDBConfig

config = DuckDBConfig()

count_rows = DuckDB(
    name="count_rows",
    query_template="SELECT COUNT(*) AS total FROM 'data.parquet'",
    plugin_config=config,
    output_dataframe_type=DataFrame,
)

flyte.TaskEnvironment.from_task("duckdb_env", count_rows)

if __name__ == "__main__":
    flyte.init_from_config()

    # Run locally (connector runs in-process)
    run = flyte.with_runcontext(mode="local").run(count_rows)

    # Run remotely (connector runs on the control plane)
    run = flyte.with_runcontext(mode="remote").run(count_rows)

    print(run.url)
```
"""

from flyte.io._dataframe.dataframe import DataFrameTransformerEngine

from flyteplugins.duckdb.connector import DuckDBConnector
from flyteplugins.duckdb.dataframe import (
    DuckDBToPandasDecodingHandler,
    PandasToDuckDBEncodingHandler,
)
from flyteplugins.duckdb.task import DuckDB, DuckDBConfig

DataFrameTransformerEngine.register(PandasToDuckDBEncodingHandler())
DataFrameTransformerEngine.register(DuckDBToPandasDecodingHandler())

__all__ = ["DuckDB", "DuckDBConfig", "DuckDBConnector"]
