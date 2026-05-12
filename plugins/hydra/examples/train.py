"""Flyte tasks used by all example entry points.

Can also be run directly as a @hydra.main script:

    # YAML config (default)
    python train.py

    # local single run
    python train.py hydra/launcher=flyte hydra.launcher.mode=local

    # add task-environment overrides from a YAML config group
    python train.py +task_env=a100

    # run with a prebuilt image override for pipeline and train_model
    python train.py +task_env=prebuilt_image

    # Python dataclass config
    python train.py --config-name structured_training

    # select structured config groups; they populate the same optimizer/model keys
    python train.py --config-name structured_training structured_optimizer=sgd \\
        optimizer.lr=0.005

    # add task-environment overrides from a structured config group
    python train.py --config-name structured_training +structured_task_env=a100

    # remote grid sweep — 6 parallel executions
    # hydra.launcher.mode defaults to remote; shown here for clarity
    python train.py --multirun \\
        hydra/launcher=flyte hydra.launcher.mode=remote \\
        optimizer.lr=0.001,0.01,0.1 training.epochs=10,20

    # fire-and-forget sweep submission
    python train.py --multirun \\
        hydra/launcher=flyte hydra.launcher.wait=false \\
        optimizer.lr=0.001,0.01,0.1

    # local grid sweep — 3 sequential local runs
    python train.py --multirun \\
        hydra/launcher=flyte hydra.launcher.mode=local \\
        optimizer.lr=0.001,0.01,0.1

    # Optuna sweep (requires: pip install hydra-optuna-sweeper)
    python train.py --multirun \\
        hydra/launcher=flyte hydra.launcher.mode=remote \\
        hydra/sweeper=optuna hydra.sweeper.n_trials=20 \\
        hydra.sweeper.n_jobs=4 \\
        "optimizer.lr=interval(1e-4,1e-1)"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import flyte
import hydra
from flyte._image import PythonWheels
from flyte.io import Dir
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from kubernetes.client import V1Container, V1PodSpec
from omegaconf import MISSING, DictConfig, OmegaConf

from flyteplugins.hydra import apply_task_env


@dataclass
class AdamConf:
    lr: float = 0.001
    weight_decay: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SgdConf:
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4


@dataclass
class ResNetConf:
    name: str = "resnet50"
    hidden_dim: int = 512
    num_layers: int = 50
    dropout: float = 0.1


@dataclass
class VitConf:
    name: str = "vit_base"
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.0


@dataclass
class TrainingConf:
    epochs: int = 30
    batch_size: int = 64
    warmup_steps: int = 0


@dataclass
class DataConf:
    path: str = "s3://my-bucket/imagenet"
    dataset: str = "imagenet"


@dataclass
class EmptyTaskEnvConf:
    pass


@dataclass
class A10GTaskEnvConf:
    train_model: dict[str, Any] = field(
        default_factory=lambda: {
            "resources": {
                "cpu": "2",
                "memory": "4Gi",
                "gpu": "A10G:1",
            },
        },
    )


@dataclass
class A100TaskEnvConf:
    pipeline: dict[str, Any] = field(
        default_factory=lambda: {
            "resources": {
                "cpu": "2",
                "memory": "6Gi",
            },
        },
    )
    train_model: dict[str, Any] = field(
        default_factory=lambda: {
            "resources": {
                "cpu": "16",
                "memory": "64Gi",
                "gpu": "A100:1",
            },
        },
    )


@dataclass
class StructuredTrainingConf:
    # These groups are registered under structured_* names and packaged into
    # optimizer/model/task_env so they do not shadow the YAML groups.
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"structured_optimizer": "adam"},
            {"structured_model": "resnet"},
            "_self_",
        ],
    )
    optimizer: Any = MISSING
    model: Any = MISSING
    data: DataConf = field(default_factory=DataConf)
    training: TrainingConf = field(default_factory=TrainingConf)
    task_env: Any = field(default_factory=dict)


def _register_structured_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="structured_training", node=StructuredTrainingConf)
    cs.store(
        group="structured_optimizer",
        name="adam",
        node=AdamConf,
        package="optimizer",
    )
    cs.store(
        group="structured_optimizer",
        name="sgd",
        node=SgdConf,
        package="optimizer",
    )
    cs.store(
        group="structured_model",
        name="resnet",
        node=ResNetConf,
        package="model",
    )
    cs.store(
        group="structured_model",
        name="vit",
        node=VitConf,
        package="model",
    )
    cs.store(
        group="structured_task_env",
        name="none",
        node=EmptyTaskEnvConf,
        package="task_env",
    )
    cs.store(
        group="structured_task_env",
        name="a10g",
        node=A10GTaskEnvConf,
        package="task_env",
    )
    cs.store(
        group="structured_task_env",
        name="a100",
        node=A100TaskEnvConf,
        package="task_env",
    )


_register_structured_configs()


def _flyte_launcher_mode() -> str | None:
    hydra_cfg = HydraConfig.get()
    launcher_target = OmegaConf.select(hydra_cfg, "launcher._target_")
    if launcher_target != "hydra_plugins.hydra_flyte_launcher.FlyteLauncher":
        return None
    return OmegaConf.select(hydra_cfg, "launcher.mode")


env = flyte.TaskEnvironment(
    name="training",
    image=flyte.Image.from_debian_base(name="training")
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-hydra",
            pre=True,
        ),
    )
    .with_pip_packages("flyteplugins-omegaconf"),
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


@env.task
async def preprocess(cfg: DictConfig) -> flyte.io.Dir:
    """Download and preprocess dataset."""
    import pathlib
    import tempfile

    print(f"Preprocessing dataset at {cfg.data.path}")

    # simulate: return a local dir
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "data.txt").write_text("preprocessed")
    return await Dir.from_local(d)


@env.task
async def train_model(cfg: DictConfig, data: flyte.io.Dir) -> tuple[flyte.io.Dir, float]:
    """Train the model; returns (checkpoint_dir, val_loss)."""
    import pathlib
    import random
    import tempfile

    print(
        f"Training {cfg.model.name} | "
        f"optimizer={cfg.optimizer} | "
        f"epochs={cfg.training.epochs} | "
        f"batch_size={cfg.training.batch_size}"
    )
    val_loss = random.uniform(0.1, 2.0)

    d = pathlib.Path(tempfile.mkdtemp())
    (d / "model.pt").write_text(f"loss={val_loss}")
    return await Dir.from_local(d), val_loss


@env.task
async def pipeline(cfg: DictConfig, dataset: str) -> float:
    """End-to-end pipeline: preprocess, train, return val_loss."""
    data = await preprocess(cfg)
    train_task = apply_task_env(train_model, cfg)
    _, val_loss = await train_task(cfg, data)
    print(f"pipeline done — val_loss={val_loss:.4f}  dataset={dataset}")
    return val_loss


@env.task
async def pipeline_with_config(config: DictConfig, dataset: str) -> float:
    """Same pipeline, using ``config`` instead of ``cfg`` as the parameter."""
    data = await preprocess(config)
    train_task = apply_task_env(train_model, config)
    _, val_loss = await train_task(config, data)
    print(f"pipeline done — val_loss={val_loss:.4f}  dataset={dataset}")
    return val_loss


@env.task(
    pod_template=flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(containers=[V1Container(name="primary", image="python:3.12.13")]),
    )
)
async def pipeline_with_podtemplate(cfg: DictConfig, dataset: str) -> float:
    """End-to-end pipeline: preprocess, train, return val_loss."""
    data = await preprocess(cfg)
    train_task = apply_task_env(train_model, cfg)
    _, val_loss = await train_task(cfg, data)
    print(f"pipeline done — val_loss={val_loss:.4f}  dataset={dataset}")
    return val_loss


@hydra.main(version_base=None, config_path="conf", config_name="training")
def main(cfg: DictConfig):
    flyte.init_from_config()
    entry_task = apply_task_env(pipeline, cfg)
    mode = _flyte_launcher_mode()
    runner = flyte.with_runcontext(mode=mode) if mode else flyte
    run = runner.run(
        entry_task,
        cfg=cfg,
        dataset=cfg.data.dataset,
    )
    if mode is None:
        print(run.url)
    return run


if __name__ == "__main__":
    main()
