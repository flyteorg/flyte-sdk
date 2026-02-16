# BigQuery Plugin for Flyte

This plugin provides BigQuery integration for Flyte, enabling you to run BigQuery queries as Flyte tasks.

## Installation

```bash
pip install flyteplugins-bigquery
```

## Usage

```python
from flyteplugins.bigquery import BigQueryConfig, BigQueryTask

config = BigQueryConfig(ProjectID="my-project", Location="US")
task = BigQueryTask(
    name="my_query",
    query_template="SELECT * FROM dataset.table WHERE id = {{ .user_id }}",
    plugin_config=config,
    inputs={"user_id": int},
)
```
