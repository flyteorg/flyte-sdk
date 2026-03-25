"""BigQuery connector plugin for Flyte.

This plugin provides integration between Flyte tasks and Google BigQuery,
enabling you to run parameterized SQL queries as Flyte tasks with full
observability, retries, and caching.

Key features:

- Parameterized SQL queries with typed inputs
- Returns query results as DataFrames
- Automatic links to the BigQuery job console in the Flyte UI
- Query cancellation on task abort

Basic usage example:

```python
import flyte
from flyte.io import DataFrame
from flyteplugins.bigquery import BigQueryConfig, BigQueryTask

config = BigQueryConfig(
    ProjectID="my-gcp-project",
    Location="US",
)

query_task = BigQueryTask(
    name="count_events",
    query_template="SELECT COUNT(*) AS total FROM `{ds}.events` WHERE date = @date",
    plugin_config=config,
    inputs={"date": str},
    output_dataframe_type=DataFrame[dict],
)

@flyte.task
def run_query(date: str) -> DataFrame[dict]:
    return query_task(date=date)
```
"""

from flyteplugins.bigquery.connector import BigQueryConnector
from flyteplugins.bigquery.task import BigQueryConfig, BigQueryTask

__all__ = ["BigQueryConfig", "BigQueryConnector", "BigQueryTask"]
