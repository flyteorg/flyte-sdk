"""
## Key features:

- Automatic MLflow run management with `@mlflow_run` decorator
- Built-in autologging support via `autolog=True` parameter
- Auto-generated MLflow UI links via `link_host` config and the `Mlflow` link class
- Parent/child task support with run sharing
- Distributed training support (only rank 0 logs to MLflow)
- Configuration management with `mlflow_config()`

## Basic usage:

1. Manual logging with `@mlflow_run`:

   ```python
   from flyteplugins.mlflow import mlflow_run, get_mlflow_run

   @mlflow_run(
       tracking_uri="http://localhost:5000",
       experiment_name="my-experiment",
       tags={"team": "ml"},
   )
   @env.task
   async def train_model(learning_rate: float) -> str:
       import mlflow

       mlflow.log_param("lr", learning_rate)
       mlflow.log_metric("loss", 0.5)

       run = get_mlflow_run()
       return run.info.run_id
   ```

2. Automatic logging with `@mlflow_run(autolog=True)`:

   ```python
   from flyteplugins.mlflow import mlflow_run

   @mlflow_run(
       autolog=True,
       framework="sklearn",
       tracking_uri="http://localhost:5000",
       log_models=True,
       log_datasets=False,
       experiment_id="846992856162999",
   )
   @env.task
   async def train_sklearn_model():
       from sklearn.linear_model import LogisticRegression

       model = LogisticRegression()
       model.fit(X, y)  # Autolog captures parameters, metrics, and model
   ```

3. Workflow-level configuration with `mlflow_config()`:

   ```python
   from flyteplugins.mlflow import mlflow_config

   r = flyte.with_runcontext(
       custom_context=mlflow_config(
           tracking_uri="http://localhost:5000",
           experiment_id="846992856162999",
           tags={"team": "ml"},
       )
   ).run(train_model, learning_rate=0.001)
   ```

4. Per-task config overrides with context manager:

   ```python
   @mlflow_run
   @env.task
   async def parent_task():
       # Override config for a specific child task
       with mlflow_config(run_mode="new", tags={"role": "child"}):
           await child_task()
   ```

5. Run modes — control run creation vs sharing:

   ```python
   @mlflow_run                      # "auto": new run if no parent, else share parent's
   @mlflow_run(run_mode="new")      # Always create a new run
   ```

6. HPO — objective can be a Flyte task with `run_mode="new"`:

   ```python
   @mlflow_run(run_mode="new")
   @env.task
   def objective(params: dict) -> float:
       mlflow.log_params(params)
       loss = train(params)
       mlflow.log_metric("loss", loss)
       return loss
   ```

7. Distributed training (only rank 0 logs):

   ```python
   @mlflow_run  # Auto-detects rank from RANK env var
   @env.task
   async def distributed_train():
       ...
   ```

8. MLflow UI links — auto-generated via `link_host`:

   ```python
   from flyteplugins.mlflow import Mlflow, mlflow_config

   # Set link_host at workflow level — children with Mlflow() link
   # auto-get the URL after the parent creates the run.
   r = flyte.with_runcontext(
       custom_context=mlflow_config(
           tracking_uri="http://localhost:5000",
           link_host="http://localhost:5000",
       )
   ).run(parent_task)

   # Attach the link to child tasks:
   @mlflow_run
   @env.task(links=[Mlflow()])
   async def child_task(): ...

   # Custom URL template (e.g. Databricks):
   mlflow_config(
       link_host="https://dbc-xxx.cloud.databricks.com",
       link_template="{host}/ml/experiments/{experiment_id}/runs/{run_id}",
   )
   ```

Decorator order: `@mlflow_run` must be outermost (before `@env.task`):

```python
@mlflow_run
@env.task
async def my_task(): ...

@mlflow_run(autolog=True, framework="sklearn")
@env.task
async def my_task(): ...
```
"""

import flyte

from ._context import get_mlflow_context, mlflow_config
from ._decorator import mlflow_run
from ._link import Mlflow

__all__ = [
    "Mlflow",
    "get_mlflow_context",
    "get_mlflow_run",
    "mlflow_config",
    "mlflow_run",
]


__version__ = "0.1.0"


def get_mlflow_run():
    """
    Get the current MLflow run if within a `@mlflow_run` decorated task or trace.

    The run is started when the `@mlflow_run` decorator enters.
    Returns None if not within an `mlflow_run` context.

    Returns:
        `mlflow.ActiveRun` | `None`: The current MLflow active run or None.
    """
    ctx = flyte.ctx()
    if not ctx or not ctx.data:
        return None

    return ctx.data.get("_mlflow_run")
