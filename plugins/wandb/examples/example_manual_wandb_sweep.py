"""
Example: Manual W&B sweep initialization without decorators.

This example demonstrates:
- Using the WandbSweep link class directly
- Manually creating sweeps with wandb.sweep()
- Manually running sweep agents with wandb.agent()
- Launching multiple sweep agents in parallel
- Overriding task links with sweep_id for child tasks
"""

import asyncio

import flyte
import wandb

from flyteplugins.wandb import WandbSweep, wandb_config

env = flyte.TaskEnvironment(
    name="wandb-manual-sweep-example",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-wandb"),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


def objective_function():
    """Objective function for manual sweep - uses wandb.run directly."""
    run = wandb.run
    config = run.config

    print(f"Training with lr={config.learning_rate}, batch_size={config.batch_size}, epochs={config.epochs}")

    # Simulate training
    best_loss = float("inf")
    for epoch in range(config.epochs):
        loss = 1.0 / (config.learning_rate * config.batch_size) + epoch * 0.1
        accuracy = min(0.95, config.learning_rate * config.batch_size * (epoch + 1) * 0.01)
        run.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})
        best_loss = min(best_loss, loss)

    print(f"Training complete! Best loss: {best_loss}")


# Single agent task
@env.task
async def sweep_agent_task(agent_id: int, sweep_id: str, count: int = 3) -> int:
    """
    Single sweep agent task.

    This task will have its link overridden at runtime with the sweep_id.
    """
    print(f"[Agent {agent_id}] Starting sweep {sweep_id}")
    print(f"[Agent {agent_id}] Running {count} trials")

    # Run sweep agent manually
    wandb.agent(
        sweep_id,
        function=objective_function,
        count=count,
        project="flyte-wandb-test",
    )

    print(f"[Agent {agent_id}] Finished!")
    return agent_id


# Parent task that creates sweep and launches agents
@env.task(
    links=(
        WandbSweep(
            project="flyte-wandb-test",
            entity="samhita-alla",
        ),
    )
)
async def manual_sweep_with_agents(num_agents: int = 2, trials_per_agent: int = 3) -> str:
    """
    Task that manually creates a sweep and launches multiple agents in parallel.

    Note: The WandbSweep link on this task points to the sweeps list page since
    sweep_id is not available at task definition time. Child agent tasks will have
    their links overridden with the specific sweep_id.
    """
    # Manually create sweep configuration
    sweep_config = {
        "method": "random",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.1},
            "batch_size": {"values": [16, 32, 64]},
            "epochs": {"values": [3, 5]},
        },
    }

    # Manually create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="flyte-wandb-test",
        entity="samhita-alla",
    )

    print(f"Created sweep: {sweep_id}")
    print(f"Launching {num_agents} agents in parallel")

    # Launch multiple sweep agents in parallel
    # Each agent's link is overridden with the sweep_id so it points to the specific sweep
    agent_tasks = [
        sweep_agent_task.override(
            links=(
                WandbSweep(
                    project="flyte-wandb-test",
                    entity="samhita-alla",
                    id=sweep_id,  # Provide sweep_id to link at runtime
                ),
            )
        )(agent_id=i + 1, sweep_id=sweep_id, count=trials_per_agent)
        for i in range(num_agents)
    ]

    # Wait for all agents to complete
    results = await asyncio.gather(*agent_tasks)

    print(f"Sweep {sweep_id} complete! All {len(results)} agents finished")
    return sweep_id


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.with_runcontext(custom_context=wandb_config(project="flyte-wandb-test", entity="samhita-alla")).run(
        manual_sweep_with_agents, num_agents=2, trials_per_agent=3
    )
    print(f"Sweep URL: {run.url}\n")
