"""
Example: Using @mlflow_run decorator for parent/child task logging.

This example demonstrates:
- Basic mlflow_run usage on tasks
- Parent/child task relationships with run sharing
- Setting run_mode via mlflow_config() for workflow-level defaults
- Overriding run_mode per-task with decorator argument
- Traced tasks accessing parent's run
- Configuration with mlflow_config context manager
- Per-task config overrides (tags, run_mode) via context manager
- Auto-generated UI links via link_host config and Mlflow link class
- Task without @mlflow_run (no run available)
"""

import logging
from pathlib import Path

import flyte
import mlflow
from flyte._image import PythonWheels

from flyteplugins.mlflow import Mlflow, get_mlflow_run, mlflow_config, mlflow_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DATABRICKS_USERNAME = "<username>"
DATABRICKS_HOST = "<host>"

env = flyte.TaskEnvironment(
    name="mlflow-init-example",
    image=flyte.Image.from_debian_base(name="mlflow-init-example")
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-mlflow",
            pre=True,
        ),
    )
    .with_pip_packages("scikit-learn", "numpy", "mlflow[databricks]"),
    secrets=[flyte.Secret(key="databricks_token", as_env_var="DATABRICKS_TOKEN")],
    env_vars={
        "MLFLOW_TRACKING_URI": "databricks",
        "GIT_PYTHON_REFRESH": "quiet",
        "DATABRICKS_HOST": DATABRICKS_HOST,
    },
)


# Traces can access parent task's MLflow run (no @mlflow_run decorator allowed on traces)
@flyte.trace
async def traced_child_task(x: int) -> str:
    run = get_mlflow_run()

    print(f"Traced Child task - Run ID: {run.info.run_id}")
    print(f"Traced Child task - Experiment: {run.info.experiment_id}")

    # Log some metrics
    mlflow.log_metric("traced_child_metric", x * 3)

    return run.info.run_id


# This task inherits run_mode from context (no decorator arg specified)
@mlflow_run
@env.task(links=(Mlflow()))
async def grandchild_task(x: int) -> str:
    run = get_mlflow_run()

    print(f"Grandchild task - Run ID: {run.info.run_id}")

    # Log some metrics
    mlflow.log_metric("grandchild_metric", x * 4)

    return run.info.run_id


# This task overrides context's run_mode with decorator argument
# Always creates a new run regardless of context setting
@mlflow_run(run_mode="new")
@env.task(links=(Mlflow()))
async def child_task_new_run(x: int) -> str:
    run = get_mlflow_run()

    print(f"Child task (new run) - Run ID: {run.info.run_id}")

    # Log some metrics
    mlflow.log_metric("child_metric", x * 2)
    mlflow.log_param("input", x)

    # Call grandchild task - it inherits run_mode from context
    grandchild_result = await grandchild_task(x + 1)
    print(f"Grandchild result: {grandchild_result}")

    return run.info.run_id


# This task inherits run_mode from context
# Inherits run_mode from context — shares the parent's run by default
@mlflow_run
@env.task(links=(Mlflow()))
async def child_task_inherits_mode(x: int) -> str:
    run = get_mlflow_run()

    print(f"Child task (inherited mode) - Run ID: {run.info.run_id}")
    print(f"Child task (inherited mode) - Status: {run.info.status}")

    # Log some metrics
    mlflow.log_metric("child_metric", x * 2)

    return run.info.run_id


# Task without @mlflow_run - no MLflow run available
@env.task
async def task_without_mlflow_run() -> str | int:
    run = get_mlflow_run()
    print(f"Task without @mlflow_run - Run: {run}")

    return run.info.run_id if run else -1


@mlflow_run
@env.task
async def parent_task() -> str:
    run = get_mlflow_run()

    print(f"Parent task - Run ID: {run.info.run_id}")
    print(f"Parent task - Experiment: {run.info.experiment_id}")

    # Log parent metrics
    mlflow.log_metric("parent_metric", 100)
    mlflow.log_param("experiment_type", "parent_child")

    # 1. Child task with run_mode="new" (decorator override) - creates a new run
    with mlflow_config(run_name="child-run", tags={"role": "child"}):
        result1 = await child_task_new_run(5)
    print(f"Child with new run: {result1} (should differ from parent)")

    # 2. Child task inheriting run_mode from context - shares parent's run
    result2 = await child_task_inherits_mode(10)
    print(f"Child inheriting mode: {result2} (should match parent: {run.info.run_id})")

    # 3. Child task with config override but still inheriting run_mode
    with mlflow_config(tags={"role": "child-with-config"}):
        result3 = await child_task_inherits_mode(20)
    print(f"Child with config override: {result3} (should match parent: {run.info.run_id})")

    # 4. Call traced task - accesses parent's run (no @mlflow_run needed)
    result4 = await traced_child_task(15)
    print(f"Traced task: {result4} (should match parent: {run.info.run_id})")

    # 5. Call task without @mlflow_run - should return -1 (no run available)
    no_mlflow_result = await task_without_mlflow_run()
    print(f"Task without mlflow_run: {no_mlflow_result}")

    # 6. Call child task with run_mode="new" via context manager
    #    This task has no run_mode in decorator, so it inherits from context
    with mlflow_config(run_mode="new"):
        result5 = await child_task_inherits_mode(25)
    print(f"Child with run_mode=new via context: {result5} (should differ from parent)")

    # Verify parent's run context is unchanged after all child calls
    print("Parent task after child calls")
    print(f"Parent task - Run ID: {run.info.run_id}")

    return (
        f"Parent complete: new_run={result1}, inherited1={result2}, "
        f"inherited2={result3}, trace={result4}, no_mlflow={no_mlflow_result}, "
        f"new_via_context={result5}"
    )


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running Example: Parent/Child Task Logging with @mlflow_run")

    run = flyte.with_runcontext(
        custom_context=mlflow_config(
            experiment_name=f"/Users/{DATABRICKS_USERNAME}/flyte-mlflow-test",
            tags={"team": "ml"},
            link_host=DATABRICKS_HOST,
            link_template="{host}/ml/experiments/{experiment_id}/runs/{run_id}",
        ),
    ).run(parent_task)

    print(f"Run URL: {run.url}")
