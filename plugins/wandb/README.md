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

### Run modes in distributed context

| Mode | Single-Node | Multi-Node |
|------|-------------|------------|
| `"auto"` | Only rank 0 logs → 1 run | Local rank 0 of each worker logs → N runs (1 per worker) |
| `"shared"` | All ranks log to 1 shared run | All ranks per worker log to shared run → N runs (1 per worker) |
| `"new"` | Each rank gets its own run (grouped) → N runs | Each rank gets its own run (grouped per worker) → N×GPUs runs |

### Run ID patterns

| Scenario | Run ID Pattern |
|----------|----------------|
| Single-node auto/shared | `{run_name}-{action_name}` |
| Single-node new | `{run_name}-{action_name}-rank-{rank}` |
| Multi-node auto/shared | `{run_name}-{action_name}-worker-{worker_index}` |
| Multi-node new | `{run_name}-{action_name}-worker-{worker_index}-rank-{local_rank}` |

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

@wandb_init  # run_mode="auto" by default
@multi_node_env.task
def train_multi_node():
    import torch.distributed as dist
    dist.init_process_group("nccl")

    run = get_wandb_run()  # Returns run for local_rank 0, None for others
    dist_info = get_distributed_info()

    # Training loop...
    if run:
        run.log({"loss": loss.item()})

    dist.destroy_process_group()
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

### Decorator vs context config

- **Decorator arguments** (`@wandb_init(project=...)`) are available only within the current task and its traces
- **Context config** (`wandb_config(...)`) propagates to child tasks

## W&B links

Tasks decorated with `@wandb_init` or `@wandb_sweep` automatically get W&B links in the Flyte UI:

- For distributed training with multiple workers, each worker gets its own link
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
- `@wandb_sweep` - Create a W&B sweep for a task

### Links

- `Wandb` - Link class for W&B runs
- `WandbSweep` - Link class for W&B sweeps
