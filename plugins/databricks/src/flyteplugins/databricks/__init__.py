"""Databricks connector plugin for Flyte.

This plugin provides integration between Flyte tasks and Databricks,
enabling you to run PySpark jobs on Databricks clusters as Flyte tasks
with full observability, retries, and caching.

Key features:

- Run PySpark tasks natively on Databricks clusters
- Configurable cluster spec via the Databricks Jobs API
- Automatic job lifecycle management: create, poll, cancel
- Automatic links to the Databricks job run UI in the Flyte UI

Basic usage example:

```python
import flyte
from flyteplugins.databricks import Databricks

databricks_config = Databricks(
    spark_conf={"spark.executor.memory": "4g"},
    databricks_conf={
        "run_name": "my_job",
        "new_cluster": {
            "spark_version": "13.3.x-scala2.12",
            "node_type_id": "i3.xlarge",
            "num_workers": 2,
        },
    },
    databricks_instance="myorg.cloud.databricks.com",
    databricks_token="databricks_token_secret",
)

env = flyte.TaskEnvironment(
    name="databricks_env",
    plugin_config=databricks_config,
    image=flyte.Image.from_debian_base(name="pyspark").with_pip_packages(
        "flyteplugins-databricks"
    ),
)

@env.task
def process_data(input_path: str) -> int:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(input_path)
    return df.count()
```
"""

from flyteplugins.databricks.connector import DatabricksConnector
from flyteplugins.databricks.task import Databricks

__all__ = ["Databricks", "DatabricksConnector"]
