"""
Example: Using @wandb_init decorator for parent/child task logging.

This example demonstrates:
- Basic wandb_init usage on tasks
- Parent/child task relationships with run reuse
- Traced tasks accessing parent's run
- Configuration with wandb_config context manager
"""

import flyte

from flyteplugins.wandb import (
    get_wandb_run,
    wandb_config,
    wandb_init,
)

env = flyte.TaskEnvironment(
    name="wandb-init-example",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-wandb"),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


# Traces can access parent task's wandb run (no @wandb_init decorator allowed on traces)
@flyte.trace
async def traced_child_task(x: int) -> str:
    run = get_wandb_run()

    print(f"Traced Child task - Run ID: {run.id}")
    print(f"Traced Child task - Project: {run.project}")
    print(f"Traced Child task - Entity: {run.entity}")

    # Log some metrics
    run.log({"traced_child_metric": x * 3, "input": x})

    return run.id


@wandb_init(run_mode="shared")  # Reuses parent's run
@env.task
def grandchild_task(x: int) -> str:
    run = get_wandb_run()

    print(f"Grandchild task - Run ID: {run.id}")
    print(f"Grandchild task - Project: {run.project}")

    # Log some metrics
    run.log({"grandchild_metric": x * 4, "input": x})

    return run.id


@wandb_init(run_mode="new")  # Always creates new run
@env.task
async def child_task(x: int) -> str:
    run = get_wandb_run()

    print(f"Child task - Run ID: {run.id}")
    print(f"Child task - Project: {run.project}")

    # Log some metrics
    run.log({"child_metric": x * 2, "input": x})

    # Call grandchild task - it will reuse this child's run
    grandchild_result = grandchild_task(x + 1)
    print(f"Grandchild result: {grandchild_result}")

    return run.id


@wandb_init  # run_mode="auto" by default - reuses parent if available
@env.task
async def child_task_with_config(x: int) -> str:
    run = get_wandb_run()

    print(f"Child task with config - Run ID: {run.id}")
    print(f"Child task with config - Tags: {run.tags}")
    print(f"Child task with config - Config: {run.config}")

    # Log some metrics
    run.log({"child_metric": x * 2, "input": x})

    return run.id


@env.task
async def task_without_wandb_init() -> str | int:
    # Task without @wandb_init - should return None
    run = get_wandb_run()
    print(f"Task without @wandb_init - Run: {run}")

    return run.id if run else -1


@wandb_init
@env.task
async def parent_task() -> str:
    run = get_wandb_run()

    print(f"Parent task - Run ID: {run.id}")
    print(f"Parent task - Project: {run.project}")
    print(f"Parent task - Entity: {run.entity}")
    print(f"Parent task - Tags: {run.tags}")

    # Log parent metrics
    run.log({"parent_metric": 100})

    # 1. Child task with run_mode="new" and custom config - creates new run
    with wandb_config(name="child-run", tags=["child-task"]):
        result1 = await child_task(5)

    # 2. Child task with run_mode="new" and parent's config - creates new run
    result2 = await child_task(10)

    # 3. Child task with run_mode="auto" and config override - reuses parent run
    with wandb_config(tags=["child-with-config"], config={"learning_rate": 0.01}):
        result3 = await child_task_with_config(20)

    # 4. Call traced task - accesses parent's run (no @wandb_init needed)
    result4 = await traced_child_task(15)

    # 5. Call task without @wandb_init - should return -1 (no run available)
    no_wandb_result = await task_without_wandb_init()

    # Verify parent's run context is unchanged
    print("Parent task after child calls")
    print(f"Parent task - Run ID: {run.id}")

    return (
        f"Parent complete: child1={result1}, child2={result2}, "
        f"child3={result3}, trace={result4}, no_wandb={no_wandb_result}"
    )


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running Example: Parent/Child Task Logging with @wandb_init")

    run = flyte.with_runcontext(
        custom_context=wandb_config(
            project="flyte-wandb-test",
            entity="samhita-alla",
            tags=["parent"],
        ),
    ).run(parent_task)

    print(f"Run URL: {run.url}")
