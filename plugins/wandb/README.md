# Weights & Biases Plugin

- Parent and child tasks can log metrics to the same run by setting `new_run=False` in the `wandb_init` decorator.
- Traces can be used to initialize a W&B run independently of the parent task.
- `wandb_run` is added to the Flyte task context and can be accessed via `flyte.ctx().wandb_run`.
- `wandb_run` is set to `None` for tasks that are not initialized with `wandb_init`.
- If a child task needs to use the same run as its parent, `wandb_config` should not be set, as it will overwrite the run name and tags (this must be ensured by the user).
- `wandb_config` can be used to pass configuration to tasks enclosed within the context manager and can also be provided via `with_runcontext`.
- When the context manager exits, the configuration falls back to the parent task’s config.

```python
import flyte
from flyteplugins.wandb import (
    wandb_config,
    wandb_init,
)

env = flyte.TaskEnvironment(
    name="wandb-test",
    image=flyte.Image.from_debian_base()
    .with_apt_packages("git")
    .with_pip_packages(
        "git+https://github.com/flyteorg/flyte-sdk.git@991f788289cbfc16a9b49e61efd9f388a358e8d5",
        "git+https://github.com/flyteorg/flyte-sdk.git@8b1d5b85fc6a4082cb2c094c9c9a44b44e5fc08d#subdirectory=plugins/wandb",
    ),
    secrets=[flyte.Secret(key="wandb-api-key", as_env_var="WANDB_API_KEY")],
)


@flyte.trace
@wandb_init
async def traced_child_task(x: int) -> str:
    run = flyte.ctx().wandb_run

    print(f"Traced Child task - Run ID: {run.id}")
    print(f"Traced Child task - Project: {run.project}")
    print(f"Traced Child task - Entity: {run.entity}")
    print(f"Traced Child task - Name: {run.name}")
    print(f"Traced Child task - Tags: {run.tags}")

    # Log some metrics
    run.log({"traced_child_metric": x * 3, "input": x})

    return run.id


@env.task
@wandb_init(new_run=False)  # Use existing run from parent
async def grandchild_task(x: int) -> str:
    run = flyte.ctx().wandb_run

    print(f"Grandchild task - Run ID: {run.id}")
    print(f"Grandchild task - Project: {run.project}")
    print(f"Grandchild task - Entity: {run.entity}")
    print(f"Grandchild task - Name: {run.name}")
    print(f"Grandchild task - Tags: {run.tags}")

    # Log some metrics
    run.log({"child_metric": x * 4, "input": x})

    return run.id


@wandb_init
@env.task
async def child_task(x: int) -> str:
    run = flyte.ctx().wandb_run

    print(f"Child task - Run ID: {run.id}")
    print(f"Child task - Project: {run.project}")
    print(f"Child task - Entity: {run.entity}")
    print(f"Child task - Name: {run.name}")
    print(f"Child task - Tags: {run.tags}")

    # Log some metrics
    run.log({"child_metric": x * 2, "input": x})

    # Call grandchild task
    grandchild_result = await grandchild_task(x + 1)
    print(f"Grandchild result: {grandchild_result}")

    return run.id


@flyte.trace
async def traced_no_wandb_init() -> str | int:
    run = flyte.ctx().wandb_run
    print(f"Traced no wandb init task - Run ID: {run}")  # Should be None

    return run.id if run else -1


@env.task
@wandb_init
async def parent_task() -> str:
    run = flyte.ctx().wandb_run

    print(f"Parent task - Run ID: {run.id}")
    print(f"Parent task - Project: {run.project}")
    print(f"Parent task - Entity: {run.entity}")
    print(f"Parent task - Name: {run.name}")
    print(f"Parent task - Tags: {run.tags}")

    # Log parent metrics
    run.log({"parent_metric": 100})

    # Call child task
    # 1. Overwrite name and tags
    with wandb_config(name="child-run", tags=["child-task"]):
        result1 = await child_task(5)

    # 2. Use parent's config
    result2 = await child_task(10)

    # 3. Call traced child task with new config
    with wandb_config(name="traced-child-run", tags=["traced-child-task"]):
        result3 = await traced_child_task(15)

    # 4. Call traced task without wandb_init
    traced_no_wandb_result = await traced_no_wandb_init()

    # Verify parent's run context is unchanged
    print("Parent task after child calls")
    print(f"Parent task - Run ID: {run.id}")
    print(f"Parent task - Project: {run.project}")
    print(f"Parent task - Entity: {run.entity}")
    print(f"Parent task - Name: {run.name}")
    print(f"Parent task - Tags: {run.tags}")

    return f"Parent complete with children: {result1}, {result2}, {result3}, {traced_no_wandb_result}"


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.with_runcontext(
        custom_context=wandb_config(
            project="flyte-wandb-test",
            entity="samhita-alla",
            tags=["parent"],
        ),
    ).run(parent_task)

    print(run.url)
```
