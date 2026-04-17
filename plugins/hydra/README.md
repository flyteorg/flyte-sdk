# flyteplugins-hydra

Hydra support for Flyte tasks.

This package lets you compose Hydra configs, run Flyte tasks, and submit Hydra
sweeps as Flyte executions. It provides three entry points:

- `hydra/launcher=flyte` for scripts that already use `@hydra.main`.
- `hydra_run` and `hydra_sweep` for Python SDK callers.
- `flyte hydra run` for CLI users who want Hydra config composition without a
  `@hydra.main` wrapper.

Remote execution is the default for Flyte Hydra runs. Use `mode="local"`,
`--local`, or `hydra.launcher.mode=local` when you want local execution.

Remote runs print the Flyte run URL immediately after submission, before waiting.
Remote sweeps submit every job first, then wait for submitted runs concurrently.
Waiting is capped at 32 worker threads by default; use `wait_max_workers`,
`--wait-max-workers`, or `hydra.launcher.wait_max_workers` to tune the cap. Use
`wait=False`, `--no-wait`, or `hydra.launcher.wait=false` for fire-and-forget
submission.

## Requirements

Tasks should accept an OmegaConf `DictConfig` input:

```python
from omegaconf import DictConfig


@env.task
async def pipeline(cfg: DictConfig, dataset: str) -> float:
    ...
```

`flyteplugins-omegaconf` must be installed in the same environment as this plugin.
It registers Flyte type transformers so `DictConfig` and `ListConfig` values
serialize correctly through Flyte. The dependency is declared in this plugin's
`pyproject.toml`.

Install this plugin locally into the same environment as `flyte`:

```bash
pip install flyteplugins-hydra
```

You may need `flyteplugins-hydra` in the image if you're using `apply_task_env`
to compose task configs for child tasks.

## Which entry point should I use?

Use `hydra/launcher=flyte` when your script already has a `@hydra.main` entry
point and you want standard Hydra CLI behavior, including `--multirun` and custom
sweepers.

Use `flyte hydra run` when you want a Flyte-style CLI command that imports a task
from a Python file, composes a Hydra config, and runs that task. The target script
does not need to execute its `@hydra.main` function for this path.

Use `hydra_run` and `hydra_sweep` when Python code should submit runs directly,
for example from examples, notebooks, tests, or another orchestration script.

All three paths ultimately use `FlyteLauncher` for Flyte submission.

## YAML Config Example

Given this config:

```yaml
# conf/training.yaml
defaults:
  - optimizer: adam
  - model: resnet
  - _self_

data:
  path: s3://my-bucket/imagenet
  dataset: imagenet

training:
  epochs: 30
  batch_size: 64
```

and this task:

```python
@env.task
async def pipeline(cfg: DictConfig, dataset: str) -> float:
    ...
```

you can run it with any of the entry points below.

## Hydra launcher

Use this path from a `@hydra.main` script.

Remote single run:

```bash
python train.py hydra/launcher=flyte hydra.launcher.mode=remote
```

Local single run:

```bash
python train.py hydra/launcher=flyte hydra.launcher.mode=local
```

Remote grid sweep:

```bash
python train.py --multirun \
  hydra/launcher=flyte hydra.launcher.mode=remote \
  hydra.launcher.wait_max_workers=64 \
  optimizer.lr=0.001,0.01,0.1 training.epochs=10,20
```

Fire-and-forget sweep submission:

```bash
python train.py --multirun \
  hydra/launcher=flyte hydra.launcher.wait=false \
  optimizer.lr=0.001,0.01,0.1
```

Optuna sweep:

```bash
python train.py --multirun \
  hydra/launcher=flyte hydra.launcher.mode=remote \
  hydra/sweeper=optuna hydra.sweeper.n_trials=20 \
  hydra.sweeper.n_jobs=4 \
  "optimizer.lr=interval(1e-4,1e-1)"
```

## Python SDK

Use `hydra_run` for one composed config:

```python
from flyteplugins.hydra import hydra_run

run = hydra_run(
    pipeline,
    config_path="conf",
    config_name="training",
    overrides=["optimizer.lr=0.01"],
    dataset="s3://my-bucket/imagenet",
    mode="remote",
    wait=True,
    wait_max_workers=64,
)
```

Use `hydra_sweep` for grid sweeps:

```python
from flyteplugins.hydra import hydra_sweep

runs = hydra_sweep(
    pipeline,
    config_path="conf",
    config_name="training",
    overrides=["optimizer.lr=0.001,0.01,0.1", "training.epochs=10,20"],
    dataset="s3://my-bucket/imagenet",
    mode="remote",
)
```

Custom Hydra sweepers are supported by passing Hydra sweeper overrides:

```python
runs = hydra_sweep(
    pipeline,
    config_path="conf",
    config_name="training",
    overrides=[
        "hydra/sweeper=optuna",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=4",
        "optimizer.lr=interval(1e-4,1e-1)",
    ],
    dataset="s3://my-bucket/imagenet",
    mode="remote",
)
```

`hydra_run` returns the Flyte run for no-wait remote runs. Waited remote runs
return a wrapper with both `url` and the resolved task output value. The wrapper
is float-castable so Hydra sweepers such as Optuna can consume scalar objectives.

## Flyte CLI

The CLI command is registered through the `flyte.plugins.cli.commands` entry
point.

Remote single run:

```bash
flyte hydra run --config-path conf --config-name training \
  train.py pipeline --dataset s3://my-bucket/imagenet
```

Local single run:

```bash
flyte hydra run --local --config-path conf --config-name training \
  train.py pipeline --dataset s3://my-bucket/imagenet
```

Remote grid sweep:

```bash
flyte hydra run --multirun --config-path conf --config-name training \
  --wait-max-workers 64 \
  train.py pipeline --dataset s3://my-bucket/imagenet \
  --cfg "optimizer.lr=0.001,0.01,0.1" --cfg "training.epochs=10,20"
```

Use the task's `DictConfig` parameter name for app-level overrides. For
`cfg: DictConfig`, use `--cfg`. For `config: DictConfig`, use `--config`:

```bash
flyte hydra run --config-path conf --config-name training \
  train.py pipeline_with_config --dataset s3://my-bucket/imagenet \
  --config optimizer.lr=0.01 --config training.epochs=20
```

Use `--hydra-override` for Hydra namespace overrides, including `hydra.*` values
and `hydra/...` config group selections:

```bash
flyte hydra run --multirun --config-path conf --config-name training \
  train.py pipeline --dataset s3://my-bucket/imagenet \
  --hydra-override "hydra/sweeper=optuna" \
  --hydra-override "hydra.sweeper.n_trials=20" \
  --hydra-override "hydra.sweeper.n_jobs=4" \
  --cfg "optimizer.lr=interval(1e-4,1e-1)"
```

### Override support

The plugin keeps app config overrides separate from Hydra runtime overrides.
This mirrors Hydra's own command-line grammar while avoiding conflicts with
Flyte task arguments.

App-level overrides target your composed config and are passed through the
task's `DictConfig` parameter flag:

```bash
# pipeline(cfg: DictConfig, ...)
flyte hydra run --config-path conf --config-name training \
  train.py pipeline \
  --cfg optimizer.lr=0.01 \
  --cfg training.epochs=20

# pipeline_with_config(config: DictConfig, ...)
flyte hydra run --config-path conf --config-name training \
  train.py pipeline_with_config \
  --config optimizer.lr=0.01
```

Hydra runtime overrides use `--hydra-override`:

```bash
flyte hydra run --config-path conf --config-name training \
  train.py pipeline \
  --hydra-override hydra.run.dir=./outputs/exp1 \
  --hydra-override hydra/launcher=flyte
```

Sweeps use the same split. Basic sweeps accept comma-separated values, while
custom sweepers can use Hydra sweep functions such as `choice(...)` and
`interval(...)`:

```bash
flyte hydra run --multirun --config-path conf --config-name training \
  train.py pipeline --dataset s3://my-bucket/imagenet \
  --hydra-override hydra/sweeper=optuna \
  --hydra-override hydra.sweeper.n_trials=20 \
  --hydra-override hydra.sweeper.n_jobs=4 \
  --cfg "optimizer.lr=interval(1e-4,1e-1)" \
  --cfg "training.epochs=choice(10,20,50)"
```

Supported app override forms are the standard Hydra forms:

```bash
--cfg optimizer.lr=0.01          # set a value
--cfg optimizer=sgd              # select a config group
--cfg +task_env=a100             # append a config group or key
--cfg ++optimizer.lr=0.05        # force set or create
--cfg ~training.warmup_steps     # delete a key
```

`flyte hydra run` also inherits relevant `flyte run` flags such as `--project`,
`--domain`, `--image`, `--name`, `--service-account`, `--raw-data-path`,
`--copy-style`, `--debug`, `--local`, and `--follow`. `--follow` is handled after
launch and cannot be combined with `--no-wait`.

## Shell Completion

Install shell completion for the `flyte` executable using Click's completion
hook. For zsh:

```zsh
eval "$(_FLYTE_COMPLETE=zsh_source flyte)"
```

For bash:

```bash
eval "$(_FLYTE_COMPLETE=bash_source flyte)"
```

If you run Flyte through a wrapper, use the exact command name you type. For
example:

```zsh
eval "$(_FLYTE_COMPLETE=zsh_source uv run flyte)"
```

`flyte hydra run` adds Hydra-aware completions after `SCRIPT TASK_NAME`. It
loads the script, inspects the task signature, and suggests the app override
flag that matches the task's `DictConfig` parameter:

```bash
# pipeline(cfg: DictConfig, ...)
flyte hydra run --config-path conf --config-name training \
  train.py pipeline --c<TAB>
# suggests --cfg

# pipeline_with_config(config: DictConfig, ...)
flyte hydra run --config-path conf --config-name training \
  train.py pipeline_with_config --co<TAB>
# suggests --config
```

Override values are completed with Hydra's own completion engine:

```bash
flyte hydra run --config-path conf --config-name training \
  train.py pipeline --cfg optimizer.<TAB>
# suggests optimizer.lr=, optimizer.weight_decay=, ...

flyte hydra run --config-path conf --config-name training \
  train.py pipeline --cfg +task_env=<TAB>
# suggests task_env config group options

flyte hydra run --config-path conf --config-name training \
  train.py pipeline --hydra-override hydra/launcher=<TAB>
# suggests hydra launcher choices
```

Completion imports the target script to inspect the task. Keep task definitions
and `ConfigStore` registration import-safe, and avoid expensive top-level work
in scripts used with `flyte hydra run`.

## Structured configs

Structured configs work as long as they are registered before the plugin composes
the config. `flyte hydra run` imports the script first, so top-level
`ConfigStore.instance().store(...)` calls are available.

```python
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class TrainingConf:
    epochs: int = 30
    batch_size: int = 64


@dataclass
class RootConf:
    training: TrainingConf = field(default_factory=TrainingConf)


ConfigStore.instance().store(name="structured_training", node=RootConf)
```

Run a structured config without YAML:

```bash
flyte hydra run --config-name structured_training \
  train.py pipeline --dataset s3://my-bucket/imagenet
```

Run the same structured config through `@hydra.main`:

```bash
python train.py --config-name structured_training
```

If the structured config also uses YAML config groups, keep `--config-path conf`.
If it is fully registered with `ConfigStore`, omit `--config-path`.

Do not register structured configs only inside `if __name__ == "__main__":` or
inside the `@hydra.main` function body; `flyte hydra run` needs them at import
time.

## Task environment overrides

By default, `task_env` is the config key used for entry-task `task.override`
kwargs. Values are nested under the task name:

```yaml
task_env:
  pipeline:
    resources:
      cpu: "2"
      memory: 8Gi
  train_model:
    resources:
      cpu: "16"
      memory: 64Gi
      gpu: "A100:1"
```

To run the launched task with a prebuilt container image, set `image`. The
plugin lowers this to a `flyte.PodTemplate`, so the image is expected to
already exist. `primary_container_name` is optional and defaults to `primary`:

```yaml
task_env:
  pipeline:
    image: ghcr.io/acme/flyte-training:latest
    primary_container_name: main
    resources:
      cpu: "4"
      memory: 16Gi
```

If the task already has an inline `flyte.PodTemplate`, the plugin deep-copies
it and only sets the image on the primary container. If the task references a
cluster pod template by name, keep the image in that referenced pod template
instead; there is no inline template object for the plugin to patch safely.

The YAML task environment intentionally does not model the full Kubernetes
`V1PodSpec`. Keep advanced pod configuration in Python task/environment code,
and use Hydra task-env presets for the common image and resources knobs.

Example presets live in `examples/conf/task_env/a100.yaml` and
`examples/conf/task_env/prebuilt_image.yaml`.

The launcher applies overrides only to the task it launches. Child tasks should
be overridden in user code where those child tasks are called. Use
`apply_task_env` to apply the same `resources` and `image` handling to child
tasks:

```python
from flyteplugins.hydra import apply_task_env


@env.task
async def pipeline(cfg: DictConfig, dataset: str) -> float:
    data = await preprocess(cfg)
    train_task = apply_task_env(train_model, cfg)
    _, val_loss = await train_task(cfg, data)
    return val_loss
```

Use `task_env_key` in the Python SDK or `--task-env-key` in the CLI if your
config uses a different key:

```python
hydra_run(..., task_env_key="task_environment")
```

```bash
flyte hydra run --task-env-key task_environment ...
```

## Override cheatsheet

- App-level value override: `optimizer.lr=0.01`
- Config group selection: `optimizer=sgd`
- Append a new key: `+task_env=a100` or `+training.grad_clip=1.0`
- Force an override: `++optimizer.lr=0.05`
- Delete a key: `~training.warmup_steps`
- Hydra runtime override: `hydra.run.dir=./outputs/exp1`
- Hydra plugin selection: `hydra/sweeper=optuna`

For `flyte hydra run`, pass app-level overrides with the task config parameter
flag, such as `--cfg` or `--config`. Pass Hydra runtime overrides with
`--hydra-override`.
