"""Comprehensive examples for flyteplugins-hydra.

Covers every combination of:
  - Entry point: hydra_run / hydra_sweep / @hydra.main / flyte hydra run CLI
  - Mode:        local / remote
  - Config:      YAML dir / structured ConfigStore / merged
  - Overrides:   value (key=val), config group (optimizer=sgd),
                 append (+key=val), force (++key=val), delete (~key),
                 hydra-namespace (hydra.run.dir=...)
  - Sweepers:    BasicSweeper (grid), Optuna (Bayesian), Nevergrad
  - Hardware:    task_env config group sweep
  - run_options: service_account, name, copy_style, raw_data_path

Run individual sections by passing --section=<name>.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from train import pipeline, pipeline_with_config, pipeline_with_podtemplate

from flyteplugins.hydra import hydra_run as _hydra_run
from flyteplugins.hydra import hydra_sweep as _hydra_sweep

CONFIG_PATH = "conf"
_FLYTE_STATE = {"initialized": False}


def _ensure_flyte_config() -> None:
    """Initialize Flyte for SDK examples without import-time side effects.

    ``train.py`` initializes Flyte only inside its ``@hydra.main`` entry point.
    This module imports ``pipeline`` directly, so that entry point never runs.
    """
    if _FLYTE_STATE["initialized"]:
        return

    import flyte

    flyte.init_from_config()
    _FLYTE_STATE["initialized"] = True


def hydra_run(*args, **kwargs):
    _ensure_flyte_config()
    return _hydra_run(*args, **kwargs)


def hydra_sweep(*args, **kwargs):
    _ensure_flyte_config()
    return _hydra_sweep(*args, **kwargs)


def _completed_result_value(run):
    """Return a completed value without echoing the run URL after waits."""
    if hasattr(run, "value"):
        return run.value
    if getattr(run, "url", None) is None:
        return run
    return None


def _print_remote_result(run) -> None:
    value = _completed_result_value(run)
    if value is not None:
        print("remote result:", value)


def _print_remote_results(runs) -> None:
    for i, run in enumerate(runs):
        value = _completed_result_value(run)
        if value is not None:
            print(f"[{i}] result={value}")


# 1. Single run: hydra_run
def ex_single_local():
    """Basic local run — config composed from YAML, no overrides."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        dataset="s3://my-bucket/imagenet",
        mode="local",
    )
    print("local run result:", run.outputs())


def ex_single_remote():
    """Basic remote run — submits to Flyte, waits for completion."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        dataset="s3://my-bucket/imagenet",
        mode="remote",
        wait=False,
    )
    _print_remote_result(run)


def ex_value_overrides():
    """Value overrides — change lr and epochs without editing YAML."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["optimizer.lr=0.01", "training.epochs=20"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_non_cfg_config_param():
    """Override a DictConfig task param named ``config`` instead of ``cfg``."""
    run = hydra_run(
        pipeline_with_config,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["optimizer.lr=0.01", "training.epochs=20"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_config_group_selection():
    """Config group selection — swap optimizer from adam to sgd."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["optimizer=sgd", "optimizer.lr=0.005"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_model_and_optimizer_groups():
    """Select both model and optimizer config groups simultaneously."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["model=vit", "optimizer=sgd"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_append_key():
    """Append a key that doesn't exist in the YAML (+key=value)."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["training.warmup_steps=500", "+training.grad_clip=1.0"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_force_override():
    """Force-set a key regardless of whether it exists (++key=value)."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["++optimizer.lr=0.05", "++training.batch_size=128"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_delete_key():
    """Remove a key from the composed config (~key)."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["~training.warmup_steps"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_hydra_namespace_override():
    """Hydra-namespace value override — redirect .hydra/ output dir."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["hydra.run.dir=./outputs/exp1", "hydra.verbose=true"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_task_env_preset():
    """Select hardware preset via task_env config group."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["+task_env=a100"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_prebuilt_image():
    """Use task_env prebuilt image overrides for pipeline and train_model."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["+task_env=prebuilt_image"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
        wait=False,
    )
    _print_remote_result(run)


def ex_prebuilt_image_with_podtemplate():
    """Merge task_env image into an entry task that already has a pod template."""
    run = hydra_run(
        pipeline_with_podtemplate,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["+task_env=prebuilt_image"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
        wait=False,
    )
    _print_remote_result(run)


def ex_run_options():
    """Pass flyte.with_runcontext options — name, service_account, etc."""
    run = hydra_run(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["optimizer.lr=0.01"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
        run_options={
            "name": "my-training-run",
            "copy_style": "all",
            "debug": True,
        },
    )
    _print_remote_result(run)


@dataclass
class OptimizerConf:
    lr: float = 0.001
    weight_decay: float = 1e-4


@dataclass
class ModelConf:
    name: str = "resnet50"
    hidden_dim: int = 512


@dataclass
class TrainConf:
    epochs: int = 30
    batch_size: int = 64


@dataclass
class DataConf:
    path: str = MISSING
    dataset: str = "imagenet"


@dataclass
class RootConf:
    optimizer: OptimizerConf = field(default_factory=OptimizerConf)
    model: ModelConf = field(default_factory=ModelConf)
    training: TrainConf = field(default_factory=TrainConf)
    data: DataConf = field(default_factory=lambda: DataConf(path="s3://my-bucket/imagenet"))


def ex_structured_config():
    """Structured config — no YAML files, schema enforced at assignment."""
    cs = ConfigStore.instance()
    cs.store(name="structured_config", node=RootConf)

    run = hydra_run(
        pipeline,
        config_path=None,  # no YAML dir
        config_name="structured_config",
        overrides=["optimizer.lr=0.01", "training.epochs=20"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


def ex_structured_config_missing_field():
    """MISSING field must be provided at call time or Hydra raises."""
    cs = ConfigStore.instance()
    cs.store(name="structured_config", node=RootConf)

    run = hydra_run(
        pipeline,
        config_path=None,
        config_name="structured_config",
        overrides=["data.path=s3://other-bucket/data"],  # fills MISSING
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_result(run)


# 2. Sweeps: hydra_sweep
def ex_grid_sweep_local():
    """Grid sweep, 6 jobs, run locally (sequential)."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["optimizer.lr=0.001,0.01,0.1", "training.epochs=10,20"],
        dataset="s3://my-bucket/imagenet",
        mode="local",
    )
    for i, r in enumerate(runs):
        print(f"[{i}] result={r}")


def ex_grid_sweep_remote():
    """Grid sweep, 6 parallel remote executions."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["optimizer.lr=0.001,0.01,0.1", "training.epochs=10,20"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_results(runs)


def ex_config_group_sweep():
    """Sweep over config groups — train once per optimizer."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["optimizer=adam,sgd"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_results(runs)


def ex_model_x_optimizer_sweep():
    """Cartesian product: 2 models x 3 lr values = 6 jobs."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["model=resnet,vit", "optimizer.lr=0.001,0.01,0.1"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_results(runs)


def ex_hardware_sweep():
    """Sweep over hardware presets — same job on A10G and A100."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["+task_env=a10g,a100"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_results(runs)


def ex_hardware_x_hparam_sweep():
    """2 hardware presets x 3 lr values = 6 jobs."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["+task_env=a10g,a100", "optimizer.lr=0.001,0.01,0.1"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_results(runs)


def ex_sweep_with_run_options():
    """Grid sweep with per-job run options."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=["optimizer.lr=0.001,0.01,0.1"],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
        run_options={"copy_style": "all", "debug": True},
    )
    _print_remote_results(runs)


def ex_sweep_hydra_output_dir():
    """Override sweep output dir (hydra.* value override, no full runtime needed)."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=[
            "optimizer.lr=0.001,0.01,0.1",
            "hydra.sweep.dir=./outputs/sweep1",
            "hydra.sweep.subdir=${hydra.job.num}",
        ],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_results(runs)


# Custom sweepers (trigger full Hydra runtime via hydra/ prefix)
def ex_optuna_sweep():
    """TPE/Bayesian optimization via Optuna sweeper (requires hydra-optuna-sweeper).

    Install: pip install hydra-optuna-sweeper
    Optuna manages a study and calls FlyteLauncher.launch() per trial batch.
    """
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=[
            "hydra/sweeper=optuna",
            "hydra.sweeper.n_trials=30",
            "hydra.sweeper.n_jobs=5",
            "optimizer.lr=interval(1e-4,1e-1)",
            "optimizer.weight_decay=interval(1e-6,1e-2)",
            "model=choice(resnet,vit)",
        ],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_results(runs)


def ex_optuna_with_config_group():
    """Optuna sweep over continuous params and config group choices."""
    runs = hydra_sweep(
        pipeline,
        config_path=CONFIG_PATH,
        config_name="training",
        overrides=[
            "hydra/sweeper=optuna",
            "hydra.sweeper.n_trials=15",
            "hydra.sweeper.n_jobs=4",
            "optimizer=choice(adam,sgd)",
            "optimizer.lr=interval(1e-4,1e-1)",
        ],
        dataset="s3://my-bucket/imagenet",
        mode="remote",
    )
    _print_remote_results(runs)


# 3. CLI reference (flyte hydra run): Run these from the shell inside examples/
CLI_EXAMPLES = """
# --- single runs -----------------------------------------------------------------------------

# remote
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet

# force local
flyte hydra run --local --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet

# DictConfig parameter named 'config' instead of 'cfg'
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline_with_config --dataset s3://my-bucket/imagenet \\
    --config optimizer.lr=0.01 --config training.epochs=20

# value overrides
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg optimizer.lr=0.01 --cfg training.epochs=20

# config group selection
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg optimizer=sgd --cfg model=vit

# append new key
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg "+training.val_steps=500"

# force-override key
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg "++optimizer.lr=0.05"

# delete key
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg "~training.warmup_steps"

# hydra-namespace value override
flyte hydra run --config-path conf --config-name training \\
    --hydra-override "hydra.run.dir=./outputs/exp1" \\
    train.py pipeline --dataset s3://my-bucket/imagenet

# task_env hardware preset
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg "+task_env=a100"

# task_env with a prebuilt primary-container image
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg "+task_env=prebuilt_image"

# task_env with a prebuilt primary-container image and pod template
flyte hydra run --config-path conf --config-name training \\
    train.py pipeline_with_podtemplate --dataset s3://my-bucket/imagenet \\
    --cfg "+task_env=prebuilt_image"

# with standard flyte run options
flyte hydra run --config-path conf --config-name training \\
    --project my-project --domain development \\
    --name my-run \\
    --follow \\
    train.py pipeline --dataset s3://my-bucket/imagenet

# structured config (no YAML, schema enforced at assignment)
flyte hydra run --config-name structured_training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg optimizer.lr=0.01 --cfg training.epochs=20

# --- grid sweeps (--multirun) ------------------------------------------------

# 6 remote jobs — lr x epochs cartesian product
flyte hydra run --multirun --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg "optimizer.lr=0.001,0.01,0.1" --cfg "training.epochs=10,20"

# sweep config groups
flyte hydra run --multirun --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg "optimizer=adam,sgd" --cfg "model=resnet,vit"

# hardware sweep
flyte hydra run --multirun --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --cfg "+task_env=a10g,a100"

# redirect sweep output dirs
flyte hydra run --multirun --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --hydra-override "hydra.sweep.dir=./outputs/sweep1" \\
    --cfg "optimizer.lr=0.001,0.01,0.1"

# --- custom sweepers (hydra/ prefix triggers full runtime) -------------------

# Optuna TPE/Bayesian sweep
flyte hydra run --multirun --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --hydra-override "hydra/sweeper=optuna" \\
    --hydra-override "hydra.sweeper.n_trials=20" \\
    --hydra-override "hydra.sweeper.n_jobs=4" \\
    --cfg "optimizer.lr=interval(1e-4,1e-1)" \\
    --cfg "training.epochs=choice(10,20,50)"

# Optuna TPE sweep
flyte hydra run --multirun --config-path conf --config-name training \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --hydra-override "hydra/sweeper=optuna" \\
    --hydra-override "hydra.sweeper.n_trials=30" \\
    --cfg "optimizer.lr=interval(1e-4,1e-1)" \\
    --cfg "model=choice(resnet,vit)"

# Optuna sweep with project/domain targeting
flyte hydra run --multirun --config-path conf --config-name training \\
    --project flytetester --domain production \\
    --name my-sweep \\
    train.py pipeline --dataset s3://my-bucket/imagenet \\
    --hydra-override "hydra/sweeper=optuna" \\
    --hydra-override "hydra.sweeper.n_trials=20" \\
    --hydra-override "hydra.sweeper.n_jobs=4" \\
    --cfg "optimizer.lr=interval(1e-4,1e-1)"
"""


# Runner
EXAMPLES = {
    # single runs
    "single_local": ex_single_local,
    "single_remote": ex_single_remote,
    "value_overrides": ex_value_overrides,
    "non_cfg_config_param": ex_non_cfg_config_param,
    "config_group": ex_config_group_selection,
    "multi_group": ex_model_and_optimizer_groups,
    "append_key": ex_append_key,
    "force_override": ex_force_override,
    "delete_key": ex_delete_key,
    "hydra_namespace": ex_hydra_namespace_override,
    "task_env_preset": ex_task_env_preset,
    "prebuilt_image": ex_prebuilt_image,
    "prebuilt_image_podtemplate": ex_prebuilt_image_with_podtemplate,
    "run_options": ex_run_options,
    "structured_config": ex_structured_config,
    "structured_missing": ex_structured_config_missing_field,
    # grid sweeps
    "grid_local": ex_grid_sweep_local,
    "grid_remote": ex_grid_sweep_remote,
    "group_sweep": ex_config_group_sweep,
    "model_x_optimizer": ex_model_x_optimizer_sweep,
    "hardware_sweep": ex_hardware_sweep,
    "hardware_x_hparam": ex_hardware_x_hparam_sweep,
    "sweep_run_options": ex_sweep_with_run_options,
    "sweep_output_dir": ex_sweep_hydra_output_dir,
    # custom sweepers
    "optuna": ex_optuna_sweep,
    "optuna_group": ex_optuna_with_config_group,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hydra plugin examples")
    parser.add_argument(
        "example",
        nargs="?",
        choices=[*list(EXAMPLES), "cli", "all"],
        default="cli",
        help="Which example to run (default: print CLI examples)",
    )
    args = parser.parse_args()

    if args.example == "cli":
        print(CLI_EXAMPLES)
    elif args.example == "all":
        for name, fn in EXAMPLES.items():
            print(f"\n{'─' * 60}")
            print(f"  {name}")
            print(f"{'─' * 60}")
            fn()
    else:
        EXAMPLES[args.example]()
