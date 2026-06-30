"""
Example: Using @wandb_init decorator for parent/child task logging.

This example demonstrates:
- Basic wandb_init usage on tasks
- Parent/child task relationships with run reuse
- Setting run_mode via wandb_config() for workflow-level defaults
- Overriding run_mode per-task with decorator argument
- Traced tasks accessing parent's run
- Configuration with wandb_config context manager
"""

import flyte

from flyteplugins.wandb import get_wandb_run, wandb_config, wandb_init

env = flyte.TaskEnvironment(
    name="wandb-init-example",
    image=flyte.Image.from_debian_base(name="wandb-init-example").with_pip_packages("flyteplugins-wandb"),
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


# This task inherits run_mode from context (no decorator arg specified)
@wandb_init
@env.task
def grandchild_task(x: int) -> str:
    run = get_wandb_run()

    print(f"Grandchild task - Run ID: {run.id}")
    print(f"Grandchild task - Project: {run.project}")

    # Log some metrics
    run.log({"grandchild_metric": x * 4, "input": x})

    return run.id


# This task overrides context's run_mode with decorator argument
# Always creates a new run regardless of context setting
@wandb_init(run_mode="new")
@env.task
async def child_task_new_run(x: int) -> str:
    run = get_wandb_run()

    print(f"Child task (new run) - Run ID: {run.id}")
    print(f"Child task (new run) - Project: {run.project}")

    # Log some metrics
    run.log({"child_metric": x * 2, "input": x})

    # Call grandchild task - it inherits run_mode="shared" from context
    # so it will share this child's run
    grandchild_result = grandchild_task(x + 1)
    print(f"Grandchild result: {grandchild_result}")

    return run.id


# This task inherits run_mode from context
# With run_mode="shared" in context, it will share the parent's run
@wandb_init
@env.task
async def child_task_inherits_mode(x: int) -> str:
    run = get_wandb_run()

    print(f"Child task (inherited mode) - Run ID: {run.id}")
    print(f"Child task (inherited mode) - Tags: {run.tags}")
    print(f"Child task (inherited mode) - Config: {run.config}")

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

    # 1. Child task with run_mode="new" (decorator override) - creates new run
    with wandb_config(name="child-run", tags=["child-task"]):
        result1 = await child_task_new_run(5)
    print(f"Child with new run: {result1} (should differ from parent)")

    # 2. Child task inheriting run_mode="shared" from context - shares parent's run
    result2 = await child_task_inherits_mode(10)
    print(f"Child inheriting mode: {result2} (should match parent: {run.id})")

    # 3. Child task with config override but still inheriting run_mode
    with wandb_config(tags=["child-with-config"], config={"learning_rate": 0.01}):
        result3 = await child_task_inherits_mode(20)
    print(f"Child with config override: {result3} (should match parent: {run.id})")

    # 4. Call traced task - accesses parent's run (no @wandb_init needed)
    result4 = await traced_child_task(15)
    print(f"Traced task: {result4} (should match parent: {run.id})")

    # 5. Call task without @wandb_init - should return -1 (no run available)
    no_wandb_result = await task_without_wandb_init()
    print(f"Task without wandb_init: {no_wandb_result}")

    # 6. Call child task with run_mode="new" via context manager
    #    This task has no run_mode in decorator, so it inherits from context
    with wandb_config(run_mode="new"):
        result5 = await child_task_inherits_mode(25)
    print(f"Child with run_mode=new via context: {result5} (should differ from parent)")

    # Verify parent's run context is unchanged
    print("Parent task after child calls")
    print(f"Parent task - Run ID: {run.id}")

    return (
        f"Parent complete: new_run={result1}, inherited1={result2}, "
        f"inherited2={result3}, trace={result4}, no_wandb={no_wandb_result}, new_via_context={result5}"
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
