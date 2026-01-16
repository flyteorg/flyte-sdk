"""
Example: Using @wandb_sweep decorator for hyperparameter optimization.

This example demonstrates:
- Creating W&B sweeps with @wandb_sweep decorator
- Running multiple sweep agents in parallel
- Objective function decorated with @wandb_init
- Resource allocation and fault tolerance for sweep agents
"""

import asyncio
import time
from datetime import timedelta

import flyte
import wandb

from flyteplugins.wandb import (
    get_wandb_context,
    get_wandb_sweep_id,
    wandb_config,
    wandb_init,
    wandb_sweep,
    wandb_sweep_config,
)

env = flyte.TaskEnvironment(
    name="wandb-sweep-example",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-wandb"),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


@wandb_init
def objective():
    """Objective function for W&B sweep - trains a model with hyperparameters."""
    run = wandb.run
    config = run.config

    print(f"Training with lr={config.learning_rate}, batch_size={config.batch_size}, epochs={config.epochs}")

    # Simulate training loop
    best_loss = float("inf")
    for epoch in range(config.epochs):
        # Simulate training metrics
        loss = 1.0 / (config.learning_rate * config.batch_size) + epoch * 0.1
        accuracy = min(0.95, config.learning_rate * config.batch_size * (epoch + 1) * 0.01)

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


@wandb_sweep
@env.task
async def sweep_agent(agent_id: int, sweep_id: str, count: int = 5) -> int:
    """
    Single sweep agent that pulls trials from W&B cloud controller.

    Multiple agents can run in parallel on different Flyte tasks, all working
    on the same sweep. W&B cloud controller coordinates work distribution.

    Flyte benefits:
    - Resource guarantees
    - Fault tolerance
    - Timeout protection
    - Distributed execution: Each agent runs on separate nodes in the cluster
    - Dynamic scaling: Launch multiple agents based on workload
    """
    print(f"[Agent {agent_id}] Starting agent for sweep {sweep_id}")
    print(f"[Agent {agent_id}] Will run up to {count} trials")

    wandb.agent(sweep_id, function=objective, count=count, project=get_wandb_context().project)

    print(f"[Agent {agent_id}] Finished!")
    return agent_id


@wandb_sweep
@env.task
async def run_parallel_sweep(total_trials: int = 15, trials_per_agent: int = 5, max_agents: int = 10) -> str:
    """
    Run a W&B sweep with multiple agents in parallel.

    This task creates a sweep and launches multiple sweep agents to run trials in parallel.
    Each agent is a separate Flyte task with its own resource allocation.
    """
    sweep_id = get_wandb_sweep_id()

    print(f"Starting sweep {sweep_id} with total {total_trials} trials")

    # Dynamic scaling: Calculate optimal agent count based on workload
    num_agents = min(
        (total_trials + trials_per_agent - 1) // trials_per_agent,
        max_agents,
    )

    print(f"Launching {num_agents} agents with {trials_per_agent} trials each")

    # Launch multiple agents in parallel with resource allocation
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

    print("Running Example: W&B Sweep with Parallel Agents")

    run = flyte.with_runcontext(
        custom_context={
            **wandb_config(project="flyte-wandb-test", entity="samhita-alla"),
            **wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={
                    "learning_rate": {"min": 0.0001, "max": 0.1},
                    "batch_size": {"values": [16, 32, 64, 128]},
                    "epochs": {"values": [3, 5, 10]},
                },
                description="Parallel sweep agents example",
            ),
        },
    ).run(run_parallel_sweep, total_trials=15, trials_per_agent=5, max_agents=10)

    print(f"Sweep URL: {run.url}")
