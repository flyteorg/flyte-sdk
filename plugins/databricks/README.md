# Databricks Plugin for Flyte

This plugin provides Databricks integration for Flyte, enabling you to run Spark jobs on Databricks as Flyte tasks.

## Installation

```bash
pip install flyteplugins-databricks
```

## Usage

```python
from flyteplugins.databricks import Databricks, DatabricksConnector

@task(task_config=Databricks(
    databricks_conf={
        "run_name": "flyte databricks plugin",
        "new_cluster": {
            "spark_version": "13.3.x-scala2.12",
            "autoscale": {
                "min_workers": 1,
                "max_workers": 1,
            },
            "node_type_id": "m6i.large",
            "num_workers": 1,
            "aws_attributes": {
                "availability": "SPOT_WITH_FALLBACK",
                "instance_profile_arn": "arn:aws:iam::339713193121:instance-profile/databricks-demo",
                "ebs_volume_type": "GENERAL_PURPOSE_SSD",
                "ebs_volume_count": 1,
                "ebs_volume_size": 100,
                "first_on_demand": 1,
            },
        },
        # "existing_cluster_id": "1113-204018-tb9vr2fm", # use existing cluster id if you want
        "timeout_seconds": 3600,
        "max_retries": 1,
    },
    databricks_instance="mycompany.cloud.databricks.com",
))
def my_spark_task() -> int:
    # Your Spark code here
    return 42
```
