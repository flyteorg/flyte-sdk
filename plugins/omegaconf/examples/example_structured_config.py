"""
Example: Structured configs (dataclass-backed DictConfig) as task inputs and outputs.

Covers:
- Flat structured config
- Nested structured config (dataclass inside dataclass)
- Structured config with List[T] fields (layer sizes, augmentation names)
- Dataclass-backed values inside Dict[str, Dataclass] and List[Dataclass] fields
- Enum, Path, Optional, bytes, and MISSING fields surviving task hops
- OmegaConf.merge() to apply overrides on top of a structured config
- Type validation: assigning a wrong type raises ValidationError in the receiving task
- MISSING fields: serialize successfully and still raise MissingMandatoryValue on access
- Dataclass/DictConfig resolution: structured config comes back typed; plain dict comes back as dict
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import flyte
from flyte._image import PythonWheels
from flyte.errors import RuntimeUserError
from omegaconf import (
    MISSING,
    DictConfig,
    OmegaConf,
    ValidationError,
)

from flyteplugins.omegaconf import log_yaml

env = flyte.TaskEnvironment(
    name="omegaconf-structured-example",
    image=flyte.Image.from_debian_base(name="omegaconf-structured-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-omegaconf",
        ),
    ),
)


@dataclass
class OptimizerConf:
    lr: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass
class ModelConf:
    hidden_dim: int = 512
    num_layers: int = 6
    dropout: float = 0.1


@dataclass
class TrainingConf:
    epochs: int = 10
    batch_size: int = 32
    grad_clip: float = 1.0


@dataclass
class TrainConf:
    optimizer: OptimizerConf = field(default_factory=OptimizerConf)
    model: ModelConf = field(default_factory=ModelConf)
    training: TrainingConf = field(default_factory=TrainingConf)
    experiment_name: str = "default"


@dataclass
class NetworkConf:
    """Structured config with List[T] fields."""

    layer_sizes: list[int] = field(default_factory=lambda: [512, 256, 128])
    activations: list[str] = field(default_factory=lambda: ["relu", "relu", "sigmoid"])
    dropout_rates: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.0])


@dataclass
class TrainConfWithMissing:
    """Config with required fields that remain MISSING until accessed."""

    data_path: str = MISSING  # required — no default
    epochs: int = 10
    lr: float = 0.001


class RunMode(Enum):
    TRAIN = "train"
    EVAL = "eval"


@dataclass
class CallbackConf:
    name: str = "early_stop"
    patience: int = 3
    monitor: str = MISSING


@dataclass
class AdvancedTrainConf:
    mode: RunMode = RunMode.TRAIN
    checkpoint_dir: Path = Path("/tmp/checkpoints")
    maybe_seed: Optional[int] = None
    payload: bytes = b"default-token"
    callbacks_by_name: dict[str, CallbackConf] = field(
        default_factory=lambda: {
            "early_stop": CallbackConf(name="early_stop", patience=3),
            "checkpoint": CallbackConf(name="checkpoint", patience=1, monitor="val_loss"),
        }
    )
    callbacks: list[CallbackConf] = field(
        default_factory=lambda: [
            CallbackConf(name="lr_monitor", patience=2, monitor="lr"),
            CallbackConf(name="nan_guard", patience=1, monitor="loss"),
        ]
    )


@env.task
async def scale_learning_rate(cfg: DictConfig, factor: float) -> DictConfig:
    """Receives a structured config, scales lr, returns updated config.
    The receiving task gets back a TrainConf-backed DictConfig (type validated).
    """
    updated = OmegaConf.merge(cfg, {"optimizer": {"lr": cfg.optimizer.lr * factor}})
    return updated


@env.task
async def validate_config(cfg: DictConfig) -> str:
    """Verifies the config is properly typed and returns a summary."""
    assert OmegaConf.get_type(cfg) == TrainConf, f"Expected TrainConf, got {OmegaConf.get_type(cfg)}"
    return (
        f"experiment={cfg.experiment_name} "
        f"lr={cfg.optimizer.lr} "
        f"epochs={cfg.training.epochs} "
        f"hidden_dim={cfg.model.hidden_dim}"
    )


@env.task
async def type_validation_check(cfg: DictConfig) -> bool:
    """Verifies that assigning a wrong type raises ValidationError."""
    try:
        cfg.optimizer.lr = "not-a-float"  # should raise
        return False
    except (ValidationError, Exception):
        return True


@env.task
async def summarize_network(cfg: DictConfig) -> str:
    """Receives a NetworkConf-backed DictConfig, reads its list fields."""
    assert OmegaConf.get_type(cfg) == NetworkConf
    return f"layers={list(cfg.layer_sizes)} activations={list(cfg.activations)} dropouts={list(cfg.dropout_rates)}"


@env.task
async def append_output_layer(cfg: DictConfig, output_dim: int) -> DictConfig:
    """Appends a new layer to the network config's list fields."""
    return OmegaConf.merge(
        cfg,
        {
            "layer_sizes": [*list(cfg.layer_sizes), output_dim],
            "activations": [*list(cfg.activations), "softmax"],
            "dropout_rates": [*list(cfg.dropout_rates), 0.0],
        },
    )


@env.task
async def use_filled_config(cfg: DictConfig) -> str:
    """Receives a config with all MISSING fields filled. Simply reads from it."""
    return f"data_path={cfg.data_path} epochs={cfg.epochs} lr={cfg.lr}"


@env.task(report=True)
async def inspect_advanced_config(cfg: DictConfig) -> str:
    """Verifies nested dataclass containers and rich scalar types survive roundtrip."""
    await log_yaml.aio(cfg, title="Inspecting advanced structured config")

    assert OmegaConf.get_type(cfg) == AdvancedTrainConf
    assert OmegaConf.get_type(cfg.callbacks_by_name["early_stop"]) == CallbackConf
    assert OmegaConf.get_type(cfg.callbacks[0]) == CallbackConf
    assert isinstance(cfg.mode, RunMode)
    assert isinstance(cfg.checkpoint_dir, Path)
    assert cfg.maybe_seed is None
    assert isinstance(cfg.payload, bytes)

    try:
        _ = cfg.callbacks_by_name["early_stop"].monitor
        missing_preserved = False
    except Exception:
        missing_preserved = True

    return (
        f"mode={cfg.mode.value} "
        f"checkpoint={cfg.checkpoint_dir} "
        f"dict_callback={cfg.callbacks_by_name['checkpoint'].monitor} "
        f"list_callback={cfg.callbacks[0].name} "
        f"missing_preserved={missing_preserved}"
    )


@env.task
async def check_config_resolution(structured_cfg: DictConfig, plain_cfg: DictConfig) -> tuple[str, str]:
    """Demonstrates deserialization resolution.

    structured_cfg was created from a dataclass → comes back as dataclass-backed DictConfig.
    plain_cfg was created from a plain dict → comes back as dict-backed DictConfig.
    """
    structured_type = OmegaConf.get_type(structured_cfg).__name__
    plain_type = OmegaConf.get_type(plain_cfg).__name__
    return structured_type, plain_type


@env.task
async def structured_config_pipeline() -> str:
    """End-to-end: create structured config → pass between tasks → validate."""
    cfg = OmegaConf.structured(
        TrainConf(
            optimizer=OptimizerConf(lr=0.01, weight_decay=1e-5),
            model=ModelConf(hidden_dim=768, num_layers=12),
            training=TrainingConf(epochs=50, batch_size=64),
            experiment_name="vit-large",
        )
    )
    scaled = await scale_learning_rate(cfg, factor=0.1)
    summary = await validate_config(scaled)
    return summary


@env.task
async def structured_merge_pipeline() -> str:
    """Merge CLI-style overrides onto a structured config base."""
    base = OmegaConf.structured(TrainConf())
    overrides = OmegaConf.create(
        {
            "optimizer": {"lr": 0.05},
            "training": {"epochs": 100},
            "experiment_name": "sweep-run-1",
        }
    )
    cfg = OmegaConf.merge(base, overrides)
    scaled = await scale_learning_rate(cfg, factor=2.0)
    summary = await validate_config(scaled)
    return summary


@env.task
async def type_safety_pipeline() -> bool:
    """Verify type validation is enforced in the receiving task."""
    cfg = OmegaConf.structured(TrainConf())
    result = await type_validation_check(cfg)
    return result


@env.task
async def list_fields_pipeline() -> str:
    """Structured config with List[T] fields survives task hops with types intact."""
    cfg = OmegaConf.structured(
        NetworkConf(
            layer_sizes=[256, 128, 64],
            activations=["relu", "relu", "relu"],
            dropout_rates=[0.2, 0.2, 0.1],
        )
    )
    with_output = await append_output_layer(cfg, output_dim=10)
    summary = await summarize_network(with_output)
    return summary


@env.task
async def missing_field_pipeline() -> tuple[str, bool]:
    """Demonstrates MISSING field behaviour.

    - Passing a config with MISSING fields serializes successfully.
    - Accessing the unfilled field in the receiving task still raises.
    - Passing a fully-filled config works fine.
    """
    # Case 1: filled — works
    filled_cfg = OmegaConf.structured(TrainConfWithMissing(data_path="/data/imagenet"))
    summary = await use_filled_config(filled_cfg)

    # Case 2: unfilled — roundtrips successfully, but accessing the field still raises
    unfilled_cfg = OmegaConf.structured(TrainConfWithMissing())
    raised_on_access = False
    try:
        await use_filled_config(unfilled_cfg)
    except RuntimeUserError:
        raised_on_access = True

    return summary, raised_on_access


@env.task
async def advanced_container_pipeline() -> str:
    """Structured config with dataclasses nested inside dict and list fields."""
    cfg = OmegaConf.structured(AdvancedTrainConf())
    return await inspect_advanced_config(cfg)


@env.task
async def config_resolution_pipeline() -> tuple[str, str]:
    """Shows that structured configs come back typed; plain dicts come back as dict.

    structured_type -> "TrainConf"  (dataclass-backed, type-validated)
    plain_type      -> "dict"       (plain DictConfig, no schema)
    """
    structured_cfg = OmegaConf.structured(TrainConf())
    plain_cfg = OmegaConf.create({"optimizer": {"lr": 0.001}, "epochs": 10})
    structured_type, plain_type = await check_config_resolution(structured_cfg, plain_cfg)
    return structured_type, plain_type


if __name__ == "__main__":
    flyte.init_from_config()

    print("=== Structured config pipeline ===")
    run = flyte.run(structured_config_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Structured merge pipeline ===")
    run = flyte.run(structured_merge_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Type safety pipeline ===")
    run = flyte.run(type_safety_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== List fields pipeline ===")
    run = flyte.run(list_fields_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== MISSING field pipeline ===")
    run = flyte.run(missing_field_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Advanced container pipeline ===")
    run = flyte.run(advanced_container_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Config resolution pipeline ===")
    run = flyte.run(config_resolution_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
