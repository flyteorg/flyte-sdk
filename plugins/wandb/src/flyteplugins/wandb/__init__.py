"""
Weights & Biases Plugin

Key Features:
- Automatic W&B run initialization with @wandb_init decorator
- Automatic W&B links in Flyte UI pointing to runs and sweeps
- Parent/child task support with automatic run reuse
- W&B sweep creation and management with @wandb_sweep decorator
- Configuration management with wandb_config() and wandb_sweep_config()

Basic Usage:

1. Simple task with W&B logging:

   from flyteplugins.wandb import wandb_init, get_wandb_run

   @wandb_init(project="my-project", entity="my-team")
   @env.task
   async def train_model(learning_rate: float) -> str:
       run = get_wandb_run()
       run.log({"loss": 0.5, "learning_rate": learning_rate})
       return run.id

2. Parent/Child Tasks with Run Reuse:

   @wandb_init  # Automatically reuses parent's run ID
   @env.task
   async def child_task(x: int) -> str:
       run = get_wandb_run()
       run.log({"child_metric": x * 2})
       return run.id

   @wandb_init(project="my-project", entity="my-team")
   @env.task
   async def parent_task() -> str:
       run = get_wandb_run()
       run.log({"parent_metric": 100})

       # Child reuses parent's run by default (new_run="auto")
       await child_task(5)
       return run.id

3. Configuration with context manager:

   from flyteplugins.wandb import wandb_config

   run = flyte.with_runcontext(
       custom_context=wandb_config(
           project="my-project",
           entity="my-team",
           tags=["experiment-1"]
       )
   ).run(train_model, learning_rate=0.001)

4. Creating new runs for child tasks:

   @wandb_init(new_run=True)  # Always creates a new run
   @env.task
   async def independent_child() -> str:
       run = get_wandb_run()
       run.log({"independent_metric": 42})
       return run.id

5. Running sweep agents in parallel:

   import asyncio
   from flyteplugins.wandb import wandb_sweep, get_wandb_sweep_id, get_wandb_context

   @wandb_init
   async def objective():
       run = wandb.run
       config = run.config
       ...

       run.log({"loss": loss_value})

   @wandb_sweep
   @env.task
   async def sweep_agent(agent_id: int, sweep_id: str, count: int = 5) -> int:
       wandb.agent(sweep_id, function=objective, count=count, project=get_wandb_context().project)
       return agent_id

   @wandb_sweep
   @env.task
   async def run_parallel_sweep(num_agents: int = 2, trials_per_agent: int = 5) -> str:
       sweep_id = get_wandb_sweep_id()

       # Launch agents in parallel
       agent_tasks = [
           sweep_agent(agent_id=i + 1, sweep_id=sweep_id, count=trials_per_agent)
           for i in range(num_agents)
       ]

       # Wait for all agents to complete
       await asyncio.gather(*agent_tasks)
       return sweep_id

   # Run with 2 parallel agents
   run = flyte.with_runcontext(
       custom_context={
           **wandb_config(project="my-project", entity="my-team"),
           **wandb_sweep_config(
               method="random",
               metric={"name": "loss", "goal": "minimize"},
               parameters={
                   "learning_rate": {"min": 0.0001, "max": 0.1},
                   "batch_size": {"values": [16, 32, 64]},
               }
           )
       }
   ).run(run_parallel_sweep, num_agents=2, trials_per_agent=5)

Decorator Order:
    @wandb_init or @wandb_sweep must be the outermost decorator:

    @wandb_init
    @env.task
    async def my_task():
        ...

Helper Functions:
- get_wandb_run(): Access the current W&B run object (or None if not in a run)
- get_wandb_sweep_id(): Access the current sweep ID (or None if not in a sweep)
- get_wandb_context(): Access the current W&B context
- get_wandb_sweep_context(): Access the current W&B sweep context
"""

import flyte

from .context import (
    get_wandb_context,
    get_wandb_sweep_context,
    wandb_config,
    wandb_sweep_config,
)
from .decorator import wandb_init, wandb_sweep
from .link import Wandb, WandbSweep


__all__ = [
    "Wandb",
    "WandbSweep",
    "get_wandb_context",
    "get_wandb_run",
    "get_wandb_sweep_context",
    "get_wandb_sweep_id",
    "wandb_config",
    "wandb_init",
    "wandb_sweep",
    "wandb_sweep_config",
]

__version__ = "0.1.0"


def get_wandb_run():
    """
    Get the current wandb run if within a @wandb_init decorated task or trace.

    The run is initialized when the @wandb_init context manager is entered.
    Returns None if not within a wandb_init context.

    Returns:
        wandb.sdk.wandb_run.Run | None: The current wandb run object or None.
    """
    ctx = flyte.ctx()
    if not ctx or not ctx.data:
        return None

    return ctx.data.get("_wandb_run")


def get_wandb_sweep_id() -> str | None:
    """
    Get the current wandb sweep_id if within a @wandb_sweep decorated task.

    Returns None if not within a wandb_sweep context.

    Returns:
        str | None: The sweep ID or None.
    """
    ctx = flyte.ctx()
    if not ctx or not ctx.custom_context:
        return None

    return ctx.custom_context.get("_wandb_sweep_id")
