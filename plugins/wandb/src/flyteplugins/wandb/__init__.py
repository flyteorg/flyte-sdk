"""
## Key features:

- Automatic W&B run initialization with `@wandb_init` decorator
- Automatic W&B links in Flyte UI pointing to runs and sweeps
- Parent/child task support with automatic run reuse
- W&B sweep creation and management with `@wandb_sweep` decorator
- Configuration management with `wandb_config()` and `wandb_sweep_config()`
- Distributed training support (auto-detects PyTorch DDP/torchrun)

## Basic usage:

1. Simple task with W&B logging:

   ```python
   from flyteplugins.wandb import wandb_init, get_wandb_run

   @wandb_init(project="my-project", entity="my-team")
   @env.task
   async def train_model(learning_rate: float) -> str:
       wandb_run = get_wandb_run()
       wandb_run.log({"loss": 0.5, "learning_rate": learning_rate})
       return wandb_run.id
   ```

2. Parent/Child Tasks with Run Reuse:

   ```python
   @wandb_init  # Automatically reuses parent's run ID
   @env.task
   async def child_task(x: int) -> str:
       wandb_run = get_wandb_run()
       wandb_run.log({"child_metric": x * 2})
       return wandb_run.id

   @wandb_init(project="my-project", entity="my-team")
   @env.task
   async def parent_task() -> str:
       wandb_run = get_wandb_run()
       wandb_run.log({"parent_metric": 100})

       # Child reuses parent's run by default (run_mode="auto")
       await child_task(5)
       return wandb_run.id
   ```

3. Configuration with context manager:

   ```python
   from flyteplugins.wandb import wandb_config

   r = flyte.with_runcontext(
       custom_context=wandb_config(
           project="my-project",
           entity="my-team",
           tags=["experiment-1"]
       )
   ).run(train_model, learning_rate=0.001)
   ```

4. Creating new runs for child tasks:

   ```python
   @wandb_init(run_mode="new")  # Always creates a new run
   @env.task
   async def independent_child() -> str:
       wandb_run = get_wandb_run()
       wandb_run.log({"independent_metric": 42})
       return wandb_run.id
   ```

5. Running sweep agents in parallel:

   ```python
   import asyncio
   from flyteplugins.wandb import wandb_sweep, get_wandb_sweep_id, get_wandb_context

   @wandb_init
   async def objective():
       wandb_run = wandb.run
       config = wandb_run.config
       ...

       wandb_run.log({"loss": loss_value})

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
   r = flyte.with_runcontext(
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
   ```

6. Distributed Training Support:

   The plugin auto-detects distributed training from environment variables
   (RANK, WORLD_SIZE, LOCAL_RANK, etc.) set by torchrun/torch.distributed.elastic.

   The `rank_scope` parameter controls the scope of run creation:
   - `"global"` (default): Global scope - 1 run/group across all workers
   - `"worker"`: Worker scope - 1 run/group per worker

   By default (`run_mode="auto"`, `rank_scope="global"`):
   - Single-node: Only rank 0 logs (1 run)
   - Multi-node: Only global rank 0 logs (1 run)

   ```python
   from flyteplugins.pytorch.task import Elastic
   from flyteplugins.wandb import wandb_init, get_wandb_run

   torch_env = flyte.TaskEnvironment(
       name="torch_env",
       resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "5Gi"), gpu="V100:4"),
       plugin_config=Elastic(nnodes=2, nproc_per_node=2),
   )

   @wandb_init
   @torch_env.task
   async def train_distributed():
       torch.distributed.init_process_group("nccl")

       # Only global rank 0 gets a W&B run, other ranks get None
       run = get_wandb_run()
       if run:
           run.log({"loss": loss})

       return run.id if run else "non-primary-rank"
   ```

   Use `rank_scope="worker"` to get 1 run per worker:

   ```python
   @wandb_init(rank_scope="worker")
   @torch_env.task
   async def train_distributed_per_worker():
       # Multi-node: local rank 0 of each worker gets a W&B run (1 run per worker)
       run = get_wandb_run()
       if run:
           run.log({"loss": loss})
       return run.id if run else "non-primary-rank"
   ```

   Use `run_mode="shared"` for all ranks to log to shared run(s):

   ```python
   @wandb_init(run_mode="shared")  # rank_scope="global": 1 shared run across all ranks
   @torch_env.task
   async def train_distributed_shared():
       # All ranks log to the same W&B run (with x_label to identify each rank)
       run = get_wandb_run()
       run.log({"rank_metric": value})
       return run.id

   @wandb_init(run_mode="shared", rank_scope="worker")  # 1 shared run per worker
   @torch_env.task
   async def train_distributed_shared_per_worker():
       run = get_wandb_run()
       run.log({"rank_metric": value})
       return run.id
   ```

   Use `run_mode="new"` for each rank to have its own W&B run:

   ```python
   @wandb_init(run_mode="new")  # rank_scope="global": all runs in 1 group
   @torch_env.task
   async def train_distributed_separate_runs():
       # Each rank gets its own W&B run (grouped in W&B UI)
       # Run IDs: {base}-rank-{global_rank}
       run = get_wandb_run()
       run.log({"rank_metric": value})
       return run.id

   @wandb_init(run_mode="new", rank_scope="worker")  # runs grouped per worker
   @torch_env.task
   async def train_distributed_separate_runs_per_worker():
       run = get_wandb_run()
       run.log({"rank_metric": value})
       return run.id
   ```

Decorator order: `@wandb_init` or `@wandb_sweep` must be the outermost decorator:

```python
@wandb_init
@env.task
async def my_task():
    ...
```
"""

import json
import logging
import os
from typing import Optional

import flyte
from flyte.io import Dir

import wandb

from ._context import (
    get_wandb_context,
    get_wandb_sweep_context,
    wandb_config,
    wandb_sweep_config,
)
from ._decorator import (
    _get_distributed_info,
    wandb_init,
    wandb_sweep,
)
from ._link import Wandb, WandbSweep

logger = logging.getLogger(__name__)


__all__ = [
    "Wandb",
    "WandbSweep",
    "download_wandb_run_dir",
    "download_wandb_run_logs",
    "download_wandb_sweep_dirs",
    "download_wandb_sweep_logs",
    "get_distributed_info",
    "get_wandb_context",
    "get_wandb_run",
    "get_wandb_run_dir",
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
    Get the current wandb run if within a `@wandb_init` decorated task or trace.

    The run is initialized when the `@wandb_init` context manager is entered.
    Returns None if not within a `wandb_init` context.

    Returns:
        `wandb.sdk.wandb_run.Run` | `None`: The current wandb run object or None.
    """
    # First check Flyte context
    ctx = flyte.ctx()
    if ctx and ctx.data:
        run = ctx.data.get("_wandb_run")
        if run:
            return run

    # Fallback to wandb's global run
    return wandb.run


def get_wandb_sweep_id() -> str | None:
    """
    Get the current wandb `sweep_id` if within a `@wandb_sweep` decorated task.

    Returns `None` if not within a `wandb_sweep` context.

    Returns:
        `str` | `None`: The sweep ID or None.
    """
    ctx = flyte.ctx()
    if not ctx or not ctx.custom_context:
        return None

    return ctx.custom_context.get("_wandb_sweep_id")


def get_wandb_run_dir() -> Optional[str]:
    """
    Get the local directory path for the current wandb run.

    Use this for accessing files written by the current task without any
    network calls. For accessing files from other tasks (or after a task
    completes), use `download_wandb_run_dir()` instead.

    Returns:
        Local path to wandb run directory (`wandb.run.dir`) or `None` if no
        active run.
    """
    run = get_wandb_run()
    if run is None:
        return None
    return run.dir


def get_distributed_info() -> dict | None:
    """
    Get distributed training info if running in a distributed context.

    This function auto-detects distributed training from environment variables
    set by torchrun/torch.distributed.elastic.

    Returns:
        dict | None: Dictionary with distributed info or None if not distributed.
            - rank: Global rank (0 to world_size-1)
            - local_rank: Rank within the node (0 to local_world_size-1)
            - world_size: Total number of processes
            - local_world_size: Processes per node
            - worker_index: Node/worker index (0 to num_workers-1)
            - num_workers: Total number of nodes/workers
    """
    return _get_distributed_info()


def download_wandb_run_dir(
    run_id: Optional[str] = None,
    path: Optional[str] = None,
    include_history: bool = True,
) -> str:
    """
    Download wandb run data from wandb cloud.

    Downloads all run files and optionally exports metrics history to JSON.
    This enables access to wandb data from any task or after workflow completion.

    Downloaded contents:

        - summary.json - final summary metrics (always exported)
        - metrics_history.json - step-by-step metrics (if include_history=True)
        - Plus any files synced by wandb (requirements.txt, wandb_metadata.json, etc.)

    Args:
        run_id: The wandb run ID to download. If `None`, uses the current run's ID
            from context (useful for shared runs across tasks).
        path: Local directory to download files to. If `None`, downloads to
            `/tmp/wandb_runs/{run_id}`.
        include_history: If `True`, exports the step-by-step metrics history
            to `metrics_history.json`. Defaults to `True`.

    Returns:
        Local path where files were downloaded.

    Raises:
        `RuntimeError`: If no `run_id` provided and no active run in context.
        `wandb.errors.CommError`: If run not found in wandb cloud.

    Note:
        There may be a brief delay between when files are written locally and
        when they're available in wandb cloud. For immediate local access
        within the same task, use `get_wandb_run_dir()` instead.
    """
    # Determine run_id
    if run_id is None:
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            run_id = ctx.custom_context.get("_wandb_run_id")
        if run_id is None:
            run = get_wandb_run()
            if run:
                run_id = run.id
        if run_id is None:
            raise RuntimeError(
                "No run_id provided and no active wandb run found in context. "
                "Provide a run_id explicitly or call from within a @wandb_init task."
            )

    # Get entity/project from context
    wandb_ctx = get_wandb_context()
    entity = wandb_ctx.entity if wandb_ctx else None
    project = wandb_ctx.project if wandb_ctx else None

    # Build run path for API
    if entity and project:
        run_path = f"{entity}/{project}/{run_id}"
    elif project:
        run_path = f"{project}/{run_id}"
    else:
        # wandb API can sometimes work with just run_id if logged in
        run_path = run_id

    # Set download path
    if path is None:
        path = f"/tmp/wandb_runs/{run_id}"

    # Ensure directory exists
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create download directory {path}: {e}") from e

    # Download files from wandb cloud
    try:
        api = wandb.Api()
        api_run = api.run(run_path)
    except wandb.errors.AuthenticationError as e:
        # Must check AuthenticationError before CommError (it's a subclass)
        raise RuntimeError(
            f"Authentication failed when accessing wandb run '{run_path}'. "
            f"Please ensure WANDB_API_KEY is set correctly. Error: {e}"
        ) from e
    except wandb.errors.CommError as e:
        raise RuntimeError(
            f"Failed to fetch wandb run '{run_path}' from wandb cloud. "
            f"The run may not exist, or you may not have access to it. "
            f"Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching wandb run '{run_path}': {e}") from e

    try:
        for file in api_run.files():
            file.download(root=path, replace=True)
    except Exception as e:
        raise RuntimeError(f"Failed to download files for run '{run_id}': {e}") from e

    # Export summary to JSON
    try:
        summary_data = dict(api_run.summary)
        if summary_data:
            with open(os.path.join(path, "summary.json"), "w") as f:
                json.dump(summary_data, f, indent=2, default=str)
    except (IOError, OSError) as e:
        raise RuntimeError(f"Failed to write summary.json for run '{run_id}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to export summary data for run '{run_id}': {e}") from e

    # Export metrics history to JSON
    if include_history:
        try:
            history = api_run.history()
            if history:
                with open(os.path.join(path, "metrics_history.json"), "w") as f:
                    json.dump(history, f, indent=2, default=str)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Failed to write metrics_history.json for run '{run_id}': {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to export history data for run '{run_id}': {e}") from e

    return path


def download_wandb_sweep_dirs(
    sweep_id: Optional[str] = None,
    base_path: Optional[str] = None,
    include_history: bool = True,
) -> list[str]:
    """
    Download all run data for a wandb sweep.

    Queries the wandb API for all runs in the sweep and downloads their files
    and metrics history. This is useful for collecting results from all sweep
    trials after completion.

    Args:
        sweep_id: The wandb sweep ID. If `None`, uses the current sweep's ID
            from context (set by `@wandb_sweep` decorator).
        base_path: Base directory to download files to. Each run's files will be
            in a subdirectory named by run_id. If `None`, uses `/tmp/wandb_runs/`.
        include_history: If `True`, exports the step-by-step metrics history
            to metrics_history.json for each run. Defaults to `True`.

    Returns:
        List of local paths where run data was downloaded.

    Raises:
        RuntimeError: If no sweep_id provided and no active sweep in context.
        wandb.errors.CommError: If sweep not found in wandb cloud.
    """
    # Determine sweep_id
    if sweep_id is None:
        sweep_id = get_wandb_sweep_id()
        if sweep_id is None:
            raise RuntimeError(
                "No sweep_id provided and no active wandb sweep found in context. "
                "Provide a sweep_id explicitly or call from within a @wandb_sweep task."
            )

    # Get entity/project from context
    wandb_ctx = get_wandb_context()
    entity = wandb_ctx.entity if wandb_ctx else None
    project = wandb_ctx.project if wandb_ctx else None

    if not entity or not project:
        raise RuntimeError("Cannot query sweep without entity and project. Set them via wandb_config().")

    # Query sweep runs via wandb API
    try:
        api = wandb.Api()
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        run_ids = [run.id for run in sweep.runs]
    except wandb.errors.AuthenticationError as e:
        # Must check AuthenticationError before CommError (it's a subclass)
        raise RuntimeError(
            f"Authentication failed when accessing wandb sweep '{entity}/{project}/{sweep_id}'. "
            f"Please ensure WANDB_API_KEY is set correctly. Error: {e}"
        ) from e
    except wandb.errors.CommError as e:
        raise RuntimeError(
            f"Failed to fetch wandb sweep '{entity}/{project}/{sweep_id}' from wandb cloud. "
            f"The sweep may not exist, or you may not have access to it. "
            f"Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching wandb sweep '{entity}/{project}/{sweep_id}': {e}") from e

    # Download each run's data
    downloaded_paths = []
    failed_runs = []

    for run_id in run_ids:
        path = f"{base_path or '/tmp/wandb_runs'}/{run_id}"
        try:
            download_wandb_run_dir(run_id=run_id, path=path, include_history=include_history)
            downloaded_paths.append(path)
        except Exception as e:
            # Log failure but continue with other runs
            failed_runs.append((run_id, str(e)))

    # If some runs failed, include that information
    if failed_runs:
        failed_info = ", ".join([f"{rid} ({err})" for rid, err in failed_runs])
        if not downloaded_paths:
            # All runs failed
            raise RuntimeError(
                f"Failed to download all {len(run_ids)} runs for sweep '{sweep_id}'. Failed runs: {failed_info}"
            )
        else:
            # Some runs succeeded, log warning but continue
            logger.warning(
                f"Failed to download {len(failed_runs)}/{len(run_ids)} runs for sweep '{sweep_id}'. "
                f"Failed runs: {failed_info}"
            )

    return downloaded_paths


@flyte.trace
async def download_wandb_run_logs(run_id: str) -> Dir:
    """
    Traced function to download wandb run logs after task completion.

    This function is called automatically when `download_logs=True` is set
    in `@wandb_init` or `wandb_config()`. The downloaded files appear as a
    trace output in the Flyte UI.

    Args:
        run_id: The wandb run ID to download.

    Returns:
        Dir containing the downloaded wandb run files.

    Raises:
        RuntimeError: If download fails (network error, run not found, auth failure, etc.)
    """
    path = download_wandb_run_dir(run_id=run_id)
    return await Dir.from_local(path)


@flyte.trace
async def download_wandb_sweep_logs(sweep_id: str) -> Dir:
    """
    Traced function to download wandb sweep logs after task completion.

    This function is called automatically when `download_logs=True` is set
    in `@wandb_sweep` or `wandb_sweep_config()`. The downloaded files appear as a
    trace output in the Flyte UI.

    Args:
        sweep_id: The wandb sweep ID to download.

    Returns:
        Dir containing the downloaded wandb sweep run files.

    Raises:
        RuntimeError: If download fails (network error, sweep not found, auth failure, etc.)
    """
    paths = download_wandb_sweep_dirs(sweep_id=sweep_id)

    # Return the base directory containing all run subdirectories
    base_path = os.path.dirname(paths[0]) if paths else "/tmp/wandb_runs"
    return await Dir.from_local(base_path)
