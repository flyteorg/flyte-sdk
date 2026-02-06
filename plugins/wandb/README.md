# Weights & Biases Plugin

This plugin provides integration between Flyte and Weights & Biases (W&B) for experiment tracking, including support for distributed training with PyTorch Elastic.

## Quickstart

```python
from flyteplugins.wandb import wandb_init, wandb_config, get_wandb_run

@wandb_init(project="my-project", entity="my-team")
@env.task
def train():
    run = get_wandb_run()
    run.log({"loss": 0.5, "accuracy": 0.9})
```

## Core concepts

### Decorator order

`@wandb_init` and `@wandb_sweep` must be the **outermost decorators** (applied after `@env.task`):

```python
@wandb_init  # Outermost
@env.task   # Task decorator
def my_task():
    ...
```

### Run modes

The `run_mode` parameter controls how W&B runs are created:

- **`"auto"`** (default): Creates a new run if no parent exists, otherwise shares the parent's run
- **`"new"`**: Always creates a new W&B run with a unique ID
- **`"shared"`**: Always shares the parent's run ID (useful for child tasks)

### Accessing the run

Use `get_wandb_run()` to access the current W&B run:

```python
from flyteplugins.wandb import get_wandb_run

run = get_wandb_run()
if run:
    run.log({"metric": value})
```

Returns `None` if not within a `@wandb_init` decorated task or if the current rank should not log (in distributed training).

## Distributed training

The plugin automatically detects distributed training environments (PyTorch Elastic) and configures W&B appropriately.

### Environment variables

Distributed training is detected via these environment variables (set by `torchrun`/`torch.distributed.elastic`):

| Variable | Description |
|----------|-------------|
| `RANK` | Global rank of the process |
| `WORLD_SIZE` | Total number of processes |
| `LOCAL_RANK` | Rank within the current node |
| `LOCAL_WORLD_SIZE` | Number of processes per node |
| `GROUP_RANK` | Worker/node index (0, 1, 2, ...) |

### Rank scope

The `rank_scope` parameter controls the granularity of W&B runs in multi-node distributed training:

- **`"global"`** (default): Treat all workers as one unit → **1 run** (or 1 group for `run_mode="new"`)
- **`"worker"`**: Treat each worker/node independently → **N runs** (or N groups for `run_mode="new"`)

The effect of `rank_scope` depends on `run_mode`:

#### run_mode="auto" + rank_scope

```python
# Global scope (default): Only global rank 0 logs → 1 run total
@wandb_init
@multi_node_env.task
def train():
    run = get_wandb_run()  # Non-None only for global rank 0
    ...

# Worker scope: Local rank 0 of each worker logs → N runs (1 per worker)
@wandb_init(rank_scope="worker")
@multi_node_env.task
def train():
    run = get_wandb_run()  # Non-None for local_rank 0 on each worker
    ...
```

#### run_mode="shared" + rank_scope

```python
# Global scope: All ranks log to 1 shared run
@wandb_init(run_mode="shared")
@multi_node_env.task
def train():
    run = get_wandb_run()  # All ranks get a run object, all log to same run
    ...

# Worker scope: All ranks on each worker share a run → N runs total
@wandb_init(run_mode="shared", rank_scope="worker")
@multi_node_env.task
def train():
    run = get_wandb_run()  # All ranks get a run, grouped by worker
    ...
```

#### run_mode="new" + rank_scope

```python
# Global scope: Each rank gets own run, all grouped together → N×M runs, 1 group
@wandb_init(run_mode="new")
@multi_node_env.task
def train():
    run = get_wandb_run()  # Each rank has its own run
    # Run IDs: {base}-rank-{global_rank}
    ...

# Worker scope: Each rank gets own run, grouped per worker → N×M runs, N groups
@wandb_init(run_mode="new", rank_scope="worker")
@multi_node_env.task
def train():
    run = get_wandb_run()  # Each rank has its own run
    # Run IDs: {base}-worker-{idx}-rank-{local_rank}
    ...
```

### Run modes in distributed context

| run_mode | rank_scope | Who initializes W&B? | W&B Runs | Grouping |
|----------|------------|----------------------|----------|----------|
| `"auto"` | `"global"` | global rank 0 only | 1 | - |
| `"auto"` | `"worker"` | local_rank 0 per worker | N | - |
| `"shared"` | `"global"` | all ranks (shared mode) | 1 | - |
| `"shared"` | `"worker"` | all ranks (shared mode) | N | - |
| `"new"` | `"global"` | all ranks | N×M | 1 group |
| `"new"` | `"worker"` | all ranks | N×M | N groups |

Where N = number of workers/nodes, M = processes per worker.

### Run ID patterns

| Scenario | Run ID Pattern | Group |
|----------|----------------|-------|
| Single-node auto/shared | `{base}` | - |
| Single-node new | `{base}-rank-{rank}` | `{base}` |
| Multi-node auto (global) | `{base}` | - |
| Multi-node auto (worker) | `{base}-worker-{idx}` | - |
| Multi-node shared (global) | `{base}` | - |
| Multi-node shared (worker) | `{base}-worker-{idx}` | - |
| Multi-node new (global) | `{base}-rank-{global_rank}` | `{base}` |
| Multi-node new (worker) | `{base}-worker-{idx}-rank-{local_rank}` | `{base}-worker-{idx}` |

Where `{base}` = `{run_name}-{action_name}`

### Example: Distributed training task

```python
from flyteplugins.wandb import wandb_init, wandb_config, get_wandb_run, get_distributed_info
from flyteplugins.pytorch.task import Elastic

# Multi-node environment (2 nodes, 4 GPUs each)
multi_node_env = flyte.TaskEnvironment(
    name="multi_node_env",
    resources=flyte.Resources(gpu="V100:4", shm="auto"),
    plugin_config=Elastic(nproc_per_node=4, nnodes=2),
    secrets=flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY"),
)

@wandb_init  # run_mode="auto", rank_scope="global" by default → 1 run total
@multi_node_env.task
def train_multi_node():
    import torch.distributed as dist
    dist.init_process_group("nccl")

    run = get_wandb_run()  # Returns run for global rank 0 only, None for others
    dist_info = get_distributed_info()

    # Training loop...
    if run:
        run.log({"loss": loss.item()})

    dist.destroy_process_group()
```

### Worker scope for per-worker logging

Use `rank_scope="worker"` when you want each worker/node to have its own W&B run:

```python
@wandb_init(rank_scope="worker")  # 1 run per worker
@multi_node_env.task
def train_per_worker():
    run = get_wandb_run()  # Returns run for local_rank 0 of each worker

    if run:
        # Each worker logs to its own run
        run.log({"loss": loss.item(), "worker": dist_info["worker_index"]})
```

### Shared mode for all-Rank logging

Use `run_mode="shared"` when you want all ranks to log to the same W&B run:

```python
@wandb_init(run_mode="shared")
@multi_node_env.task
def train_all_ranks_log():
    run = get_wandb_run()  # All ranks get a run object

    # All ranks can log - W&B handles deduplication
    run.log({"loss": loss.item(), "rank": dist.get_rank()})
```

### New mode for per-rank runs

Use `run_mode="new"` when you want each rank to have its own W&B run:

```python
@wandb_init(run_mode="new")
@multi_node_env.task
def train_per_rank():
    run = get_wandb_run()  # Each rank gets its own run

    # Runs are grouped in W&B UI for easy comparison
    run.log({"loss": loss.item()})
```

## Configuration

### wandb_config

Use `wandb_config()` to pass configuration that propagates to child tasks:

```python
from flyteplugins.wandb import wandb_config

# With flyte.with_runcontext
run = flyte.with_runcontext(
    custom_context=wandb_config(
        project="my-project",
        entity="my-team",
        tags=["experiment-1"],
    )
).run(my_task)

# As a context manager
with wandb_config(project="override-project"):
    await child_task()
```

### Configuring run_mode and rank_scope

Both `run_mode` and `rank_scope` can be set via decorator or context:

```python
# Via decorator (takes precedence)
@wandb_init(run_mode="shared", rank_scope="worker")
@multi_node_env.task
def train():
    ...

# Via context (useful for dynamic configuration)
run = flyte.with_runcontext(
    custom_context=wandb_config(
        project="my-project",
        run_mode="shared",
        rank_scope="worker",
    )
).run(train)
```

When both are specified, **decorator arguments take precedence** over context config.

### Decorator vs context config

| Source | Scope | Use case |
|--------|-------|----------|
| Decorator (`@wandb_init(...)`) | Current task and traces only | Static per-task config |
| Context (`wandb_config(...)`) | Propagates to child tasks | Dynamic/shared config |

Priority order (highest to lowest):
1. Decorator arguments
2. Context config (`wandb_config`)
3. Defaults (`run_mode="auto"`, `rank_scope="global"`)

## W&B links

Tasks decorated with `@wandb_init` or `@wandb_sweep` automatically get W&B links in the Flyte UI:

- With `rank_scope="global"` (default): A single link to the one W&B run
- With `rank_scope="worker"`: Each worker gets its own link
- Links point directly to the corresponding W&B runs or sweeps
- Project/entity are retrieved from decorator parameters or context configuration

## Sweeps

Use `@wandb_sweep` to create W&B sweeps:

```python
from flyteplugins.wandb import wandb_sweep, wandb_sweep_config, get_wandb_sweep_id

@wandb_init
def objective():
    # Training logic - this runs for each sweep trial
    run = get_wandb_run()
    config = run.config  # Sweep parameters are passed via run.config

    # Train with sweep-suggested hyperparameters
    model = train(lr=config.lr, batch_size=config.batch_size)
    wandb.log({"loss": loss, "accuracy": accuracy})

@wandb_sweep
@env.task
def run_sweep():
    sweep_id = get_wandb_sweep_id()

    # Launch sweep agents to run trials
    # count=10 means run 10 trials total
    wandb.agent(sweep_id, function=objective, count=10)
```

**Note:** A maximum of **20 sweep agents** can be launched at a time.

Configure sweeps with `wandb_sweep_config()`:

```python
run = flyte.with_runcontext(
    custom_context=wandb_sweep_config(
        method="bayes",
        metric={"name": "loss", "goal": "minimize"},
        parameters={"lr": {"min": 1e-5, "max": 1e-2}},
        project="my-project",
    )
).run(run_sweep)
```

## Downloading logs

Set `download_logs=True` to download W&B run/sweep logs after task completion. The download I/O is traced by Flyte's `@flyte.trace`, making the logs visible in the Flyte UI:

```python
@wandb_init(download_logs=True)
@env.task
def train():
    ...

# Or via context
wandb_config(download_logs=True)
wandb_sweep_config(download_logs=True)
```

The downloaded logs include all files uploaded to W&B during the run (metrics, artifacts, etc.).

## API reference

### Functions

- `get_wandb_run()` - Get the current W&B run object (or `None`)
- `get_wandb_sweep_id()` - Get the current sweep ID (or `None`)
- `get_distributed_info()` - Get distributed training info dict (or `None`)
- `wandb_config(...)` - Create W&B configuration for context
- `wandb_sweep_config(...)` - Create sweep configuration for context

### Decorators

- `@wandb_init` - Initialize W&B for a task or function
  - `run_mode`: `"auto"` (default), `"new"`, or `"shared"`
  - `rank_scope`: `"global"` (default) or `"worker"` - controls which ranks log in distributed training
  - `download_logs`: If `True`, download W&B logs after task completion
  - `project`, `entity`: W&B project and entity names
- `@wandb_sweep` - Create a W&B sweep for a task

### Links

- `Wandb` - Link class for W&B runs
- `WandbSweep` - Link class for W&B sweeps

### Types

- `RankScope` - Literal type: `"global"` | `"worker"`
- `RunMode` - Literal type: `"auto"` | `"new"` | `"shared"`
