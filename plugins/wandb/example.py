import asyncio
import time

import flyte
import wandb
from flyteplugins.wandb import (
    get_wandb_context,
    wandb_config,
    wandb_init,
    wandb_sweep,
    wandb_sweep_config,
)
from datetime import timedelta
from flyte._image import PythonWheels
from pathlib import Path

env = flyte.TaskEnvironment(
    name="wandb-test",
    image=flyte.Image.from_debian_base()
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent.parent / "dist",
            package_name="flyte",
            pre=True,
        )
    )
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent / "dist",
            package_name="flyteplugins-wandb",
            pre=True,
        ),
        name="wandb-test",
    ),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


@wandb_init
@flyte.trace
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


@wandb_init(new_run=False)
@env.task
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


@wandb_init(new_run=True)
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


@wandb_init
@env.task
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


@wandb_init
def objective():
    run = wandb.run
    config = run.config

    print(
        f"Training with lr={config.learning_rate}, batch_size={config.batch_size}, epochs={config.epochs}"
    )

    # Simulate training loop
    best_loss = float("inf")
    for epoch in range(config.epochs):
        # Simulate training metrics
        loss = 1.0 / (config.learning_rate * config.batch_size) + epoch * 0.1
        accuracy = min(
            0.95, config.learning_rate * config.batch_size * (epoch + 1) * 0.01
        )

        run.log(
            {
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
            }
        )

        best_loss = min(best_loss, loss)
        time.sleep(0.5)  # Simulate training time

    print(f"Training complete! Best loss: {best_loss}")


@env.task
async def sweep_agent(agent_id: int, sweep_id: str, count: int = 5) -> int:
    """
    Single sweep agent that pulls trials from W&B cloud controller.

    Multiple agents can run in parallel on different Flyte tasks, all working
    on the same sweep. W&B cloud controller coordinates work distribution.

    Flyte benefits:
    - Resource guarantees: 2-4 CPU, 4-8Gi memory per agent
    - Fault tolerance: Auto-retry up to 3 times on failure
    - Timeout protection: Prevents runaway jobs
    - Distributed execution: Each agent runs on separate nodes in the cluster
    - Dynamic scaling: Launch multiple agents based on workload
    """
    print(f"[Agent {agent_id}] Starting agent for sweep {sweep_id}")
    print(f"[Agent {agent_id}] Will run up to {count} trials")

    wandb.agent(
        sweep_id, function=objective, count=count, project=get_wandb_context().project
    )

    print(f"[Agent {agent_id}] Finished!")
    return agent_id


@wandb_sweep
@env.task
async def run_parallel_sweep(
    total_trials: int = 15, trials_per_agent: int = 5, max_agents: int = 10
) -> str:
    sweep_id = flyte.ctx().wandb_sweep_id

    # Dynamic scaling: Calculate optimal agent count based on workload
    num_agents = min(
        (total_trials + trials_per_agent - 1) // trials_per_agent,
        max_agents,
    )

    # Launch multiple agents in parallel
    agent_tasks = [
        sweep_agent.override(
            resources=flyte.Resources(cpu="2", memory="4Gi"),
            retries=3,
            timeout=timedelta(minutes=30),
        )(agent_id=i + 1, sweep_id=sweep_id, count=trials_per_agent)
        for i in range(num_agents)
    ]

    # Wait for all agents to complete (with fault tolerance per agent)
    results = await asyncio.gather(*agent_tasks)

    print(f"Sweep complete! All {len(results)} agents finished successfully")

    return sweep_id


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running Example 1: Parent/Child Task Logging")

    run_1 = flyte.with_runcontext(
        custom_context=wandb_config(project="flyte-wandb-test", tags=["parent"]),
    ).run(parent_task)

    print(run_1.url)

    print("Running Example 2: W&B Sweep with Local Controller")

    run_2 = flyte.with_runcontext(
        custom_context={
            **wandb_config(project="flyte-wandb-test"),
            **wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={
                    "learning_rate": {"min": 0.0001, "max": 0.1},
                    "batch_size": {"values": [16, 32, 64, 128]},
                    "epochs": {"values": [3, 5, 10]},
                },
                name="flyte-hyperparameter-optimization",
                description="Parallel agents",
            ),
        },
    ).run(run_parallel_sweep, total_trials=15, trials_per_agent=5, max_agents=10)

    print(run_2.url)
