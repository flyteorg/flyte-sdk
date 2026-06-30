"""
Example: Accessing wandb sweep directories in both async and sync tasks.

This example demonstrates:
1. Using download_logs parameter in @wandb_sweep decorator
2. Async sweeps with parallel agents
3. Sync sweeps with sequential agents
4. Manual downloads using download_wandb_sweep_dirs()
"""

import asyncio
import time
from datetime import timedelta

import flyte
import wandb
from flyte.io import Dir

from flyteplugins.wandb import (
    download_wandb_sweep_dirs,
    get_wandb_context,
    get_wandb_run_dir,
    get_wandb_sweep_id,
    wandb_config,
    wandb_init,
    wandb_sweep,
    wandb_sweep_config,
)

env = flyte.TaskEnvironment(
    name="wandb-sweep-dir-example",
    image=flyte.Image.from_debian_base(name="wandb-sweep-dir-example").with_pip_packages("flyteplugins-wandb"),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


@wandb_init
def objective():
    """Objective function for W&B sweep - trains a model with hyperparameters."""
    run = wandb.run
    config = run.config

    print(f"Training with lr={config.learning_rate}, batch_size={config.batch_size}, epochs={config.epochs}")

    # Access local run directory for this trial
    local_dir = get_wandb_run_dir()
    print(f"Trial run dir: {local_dir}")

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
        time.sleep(0.5)

    print(f"Training complete! Best loss: {best_loss}")


@wandb_sweep
@env.task
async def sweep_agent_async(agent_id: int, sweep_id: str, count: int = 3) -> int:
    """Async sweep agent that runs trials."""
    print(f"[Async Agent {agent_id}] Starting agent for sweep {sweep_id}")
    print(f"[Async Agent {agent_id}] Will run up to {count} trials")

    wandb.agent(sweep_id, function=objective, count=count, project=get_wandb_context().project)

    print(f"[Async Agent {agent_id}] Finished!")
    return agent_id


@wandb_sweep(download_logs=True)
@env.task
async def run_async_sweep(total_trials: int = 6, trials_per_agent: int = 3, max_agents: int = 2) -> str:
    """
    Run an async sweep with parallel agents and automatic log downloads.

    Demonstrates:
    - Running multiple sweep agents in parallel (async)
    - Automatic download of all trial directories via download_logs=True
    """
    sweep_id = get_wandb_sweep_id()

    print(f"\n{'=' * 70}")
    print(f"ASYNC SWEEP: Starting sweep {sweep_id} with total {total_trials} trials")
    print(f"{'=' * 70}\n")

    # Calculate number of agents
    num_agents = min(
        (total_trials + trials_per_agent - 1) // trials_per_agent,
        max_agents,
    )

    print(f"Launching {num_agents} async agents in parallel with {trials_per_agent} trials each")

    # Launch agents in parallel
    agent_tasks = [
        sweep_agent_async.override(
            resources=flyte.Resources(cpu="2", memory="4Gi"),
            retries=3,
            timeout=timedelta(minutes=30),
        )(agent_id=i + 1, sweep_id=sweep_id, count=trials_per_agent)
        for i in range(num_agents)
    ]

    # Wait for all agents to complete
    await asyncio.gather(*agent_tasks)

    print(f"\nAsync sweep {sweep_id} completed!")

    return sweep_id


@wandb_sweep(download_logs=False)
@env.task
def sweep_agent_sync(agent_id: int, sweep_id: str, count: int = 3) -> int:
    """Sync sweep agent that runs trials sequentially."""
    print(f"[Sync Agent {agent_id}] Starting agent for sweep {sweep_id}")
    print(f"[Sync Agent {agent_id}] Will run up to {count} trials")

    wandb.agent(sweep_id, function=objective, count=count, project=get_wandb_context().project)

    print(f"[Sync Agent {agent_id}] Finished!")
    return agent_id


@wandb_sweep(download_logs=True)
@env.task
def run_sync_sweep(total_trials: int = 6, trials_per_agent: int = 3) -> dict[str, str | int | list[Dir]]:
    """
    Run a sync sweep with sequential agents and automatic log downloads.

    Demonstrates:
    - Running sweep agents sequentially (sync)
    - Automatic downloading of sweep logs after agents complete
    """
    sweep_id = get_wandb_sweep_id()

    print(f"\n{'=' * 70}")
    print(f"SYNC SWEEP: Starting sweep {sweep_id} with total {total_trials} trials")
    print(f"{'=' * 70}\n")

    # Run agents sequentially
    num_agents = (total_trials + trials_per_agent - 1) // trials_per_agent
    print(f"Running {num_agents} sync agents sequentially with {trials_per_agent} trials each")

    for i in range(num_agents):
        agent_id = sweep_agent_sync(agent_id=i + 1, sweep_id=sweep_id, count=trials_per_agent)
        print(f"  Agent {agent_id} completed")

    # Manually download sweep trial directories after completion (alternative to download_logs=True)
    print("\nManually downloading all trial directories from wandb cloud...")
    local_dirs = download_wandb_sweep_dirs(sweep_id=sweep_id)
    print(f"Downloaded {len(local_dirs)} trial directories")

    print(f"\nSync sweep {sweep_id} completed!")

    return {
        "sweep_id": sweep_id,
        "num_trials": len(local_dirs),
        "trial_dirs": [Dir.from_local_sync(d) for d in local_dirs],
    }


if __name__ == "__main__":
    flyte.init_from_config()

    print("W&B Sweep Example: Async (Parallel) vs Sync (Sequential)")

    # Run async sweep
    print("\n[1/2] Running ASYNC sweep with parallel agents...")
    async_run = flyte.with_runcontext(
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
                description="Async sweep with automatic download",
            ),
        },
    ).run(run_async_sweep, total_trials=6, trials_per_agent=3, max_agents=2)

    print(f"Async sweep URL: {async_run.url}")

    # Run sync sweep
    print("\n[2/2] Running SYNC sweep with sequential agents...")
    sync_run = flyte.with_runcontext(
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
                description="Sync sweep with manual download",
            ),
        },
    ).run(run_sync_sweep, total_trials=6, trials_per_agent=3)

    print(f"Sync sweep URL: {sync_run.url}")
