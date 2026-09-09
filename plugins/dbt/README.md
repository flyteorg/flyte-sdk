# flyteplugins-dbt

Run dbt CLI invocations as Flyte v2 tasks.

`DbtTask` maps one `dbtRunner.invoke(cli_args)` call to one Flyte task. CLI arguments are task inputs, so the same task definition can run different dbt commands.

Install the dbt adapter required by your project, such as `dbt-duckdb`, `dbt-bigquery`, or `dbt-snowflake`, in the task image alongside this plugin.

```python
from pathlib import Path

import flyte
from flyteplugins.dbt import DbtTask

DBT_PROJECT_DIR = "jaffle_shop"
DBT_PROFILES_DIR = "dbt-profiles"

env = flyte.TaskEnvironment(
    name="dbt",
    image=flyte.Image.from_debian_base()
    .with_requirements("requirements.txt")
    .with_source_folder(Path(DBT_PROJECT_DIR))
    .with_source_folder(Path(DBT_PROFILES_DIR)),
)

dbt_test = DbtTask(name="dbt-test", task_environment=env)


@env.task
async def main():
    return await dbt_test.aio(
        cli_args=[
            "test",
            "--project-dir",
            DBT_PROJECT_DIR,
            "--profiles-dir",
            DBT_PROFILES_DIR,
            "--profile",
            DBT_PROJECT_DIR,
        ]
    )
```

The task returns a list of `DbtNodeResult` values summarized from dbt node results. If dbt reports failure, the task raises the dbt exception when one is available.

## Event callbacks

By default, the task registers a dbt event callback that records finished dbt nodes as Flyte traces. Additional dbt callbacks can be passed to `DbtTask`.

```python
def log_dbt_node(event):
    data = getattr(event, "data", None)
    node_info = getattr(data, "node_info", None)
    if node_info is None:
        return

    node_name = getattr(node_info, "node_name", None)
    node_status = getattr(node_info, "node_status", None)
    print(f"dbt node={node_name} status={node_status}")


dbt_test = DbtTask(
    name="dbt-test",
    task_environment=env,
    callbacks=[log_dbt_node],
)
```

For remote execution, callbacks must be importable functions. Import-path strings are also supported:

```python
dbt_test = DbtTask(
    name="dbt-test",
    task_environment=env,
    callbacks=["my_project.callbacks.log_dbt_node"],
)
```

The built-in callback can be disabled:

```python
dbt_test = DbtTask(
    name="dbt-test",
    task_environment=env,
    callbacks=["my_project.callbacks.log_dbt_node"],
    include_default_callback=False,
)
```
