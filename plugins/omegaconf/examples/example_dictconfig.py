"""
Example: DictConfig as task inputs and outputs.

Covers:
- Plain dict-based DictConfig
- YAML-loaded DictConfig (configs/training.yaml)
- Merged config (base + overrides)
- OmegaConf interpolation (resolved at serialization)
- DictConfig as both input and output
- Modifying and returning a DictConfig from a task
- Plain DictConfig values that are dataclass-backed configs
- Plain DictConfig values that are pathlib.Path references
- User config keys that overlap with the plugin payload shape ("kind", "values")
- DictConfig with list values (e.g. layer sizes, augmentation pipeline)
- Deeply nested DictConfig (3+ levels)
"""

from dataclasses import dataclass
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from omegaconf import DictConfig, OmegaConf

from flyteplugins.omegaconf import log_yaml

env = flyte.TaskEnvironment(
    name="omegaconf-dictconfig-example",
    image=flyte.Image.from_debian_base(name="omegaconf-dictconfig-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-omegaconf",
        ),
    ),
)


@dataclass
class TransformConf:
    name: str
    probability: float = 1.0


@env.task(report=True)
async def normalize_config(cfg: DictConfig) -> DictConfig:
    """Receives a plain DictConfig, adds a field, returns it."""
    await log_yaml.aio(cfg, title="Original config in normalize_config")
    return OmegaConf.merge(cfg, {"normalized": True})


@env.task(report=True)
async def print_config(cfg: DictConfig) -> str:
    """Receives a DictConfig, returns a string summary."""
    await log_yaml.aio(cfg, title="Config in print_config")
    return OmegaConf.to_yaml(cfg)


@env.task(report=True)
async def use_interpolated_config(cfg: DictConfig) -> float:
    """Receives a config where interpolations are already resolved at serialization time."""
    # By the time this runs, cfg.optimizer.lr is already the resolved float (not "${base_lr}")
    await log_yaml.aio(cfg, title="Config in use_interpolated_config")
    return float(cfg.optimizer.lr)


@env.task(report=True)
async def merge_with_overrides(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """Merges two DictConfigs inside a task and returns the result."""
    await log_yaml.aio(base_cfg, title="Base config in merge_with_overrides")
    await log_yaml.aio(override_cfg, title="Override config in merge_with_overrides")

    return OmegaConf.merge(base_cfg, override_cfg)


@env.task
async def plain_dictconfig_pipeline() -> str:
    cfg = OmegaConf.create(
        {
            "optimizer": {"lr": 0.001, "weight_decay": 1e-4},
            "training": {"epochs": 10, "batch_size": 32},
        }
    )
    normalized = await normalize_config(cfg)
    summary = await print_config(normalized)
    return summary


@env.task
async def interpolation_pipeline() -> float:
    cfg = OmegaConf.create(
        {
            "base_lr": 0.01,
            "optimizer": {"lr": "${base_lr}", "momentum": 0.9},
        }
    )
    # Interpolation is resolved by to_container(resolve=True) before serialization
    result = await use_interpolated_config(cfg)
    return result


@env.task
async def merge_pipeline() -> DictConfig:
    base = OmegaConf.create(
        {
            "optimizer": {"lr": 0.001},
            "training": {"epochs": 10},
            "model": {"hidden_dim": 512},
        }
    )
    override = OmegaConf.create(
        {
            "optimizer": {"lr": 0.01},
            "training": {"epochs": 20},
        }
    )
    merged = await merge_with_overrides(base, override)
    return merged


@env.task(report=True)
async def double_layer_sizes(cfg: DictConfig) -> DictConfig:
    """Receives a config whose values include lists; doubles each layer size."""
    await log_yaml.aio(cfg, title="Config in double_layer_sizes")

    doubled = [size * 2 for size in cfg.model.layer_sizes]
    return OmegaConf.merge(cfg, {"model": {"layer_sizes": doubled}})


@env.task(report=True)
async def summarize_architecture(cfg: DictConfig) -> str:
    await log_yaml.aio(cfg, title="Config in summarize_architecture")
    layers = list(cfg.model.layer_sizes)
    augmentations = list(cfg.data.augmentations)
    return f"layers={layers} depth={len(layers)} augmentations={augmentations}"


@env.task(report=True)
async def summarize_transforms(cfg: DictConfig) -> str:
    """Receives a plain DictConfig whose values are dataclass-backed configs."""
    await log_yaml.aio(cfg, title="Config in summarize_transforms")
    assert OmegaConf.get_type(cfg.train) == TransformConf
    assert OmegaConf.get_type(cfg.eval) == TransformConf
    return f"train={cfg.train.name}:{cfg.train.probability:.2f} eval={cfg.eval.name}:{cfg.eval.probability:.2f}"


@env.task(report=True)
async def summarize_path_refs(cfg: DictConfig) -> str:
    """Receives a plain DictConfig containing Path values."""
    await log_yaml.aio(cfg, title="Config in summarize_path_refs")
    assert isinstance(cfg.model_path, Path)

    return f"model_path={cfg.model_path}"


@env.task(report=True)
async def summarize_payload_shaped_user_config(cfg: DictConfig) -> str:
    """Receives user data with keys that overlap with the plugin's internal payload fields."""
    await log_yaml.aio(cfg, title='User config with "kind" and "values" keys')

    assert cfg.kind == "training-job"

    # cfg.values resolves to the DictConfig.values() method,
    # hence the need for bracket notation to access the user config key named "values"
    assert cfg["values"].lr == 0.001
    assert cfg["values"].epochs == 10

    return f"kind={cfg.kind} lr={cfg['values'].lr} epochs={cfg['values'].epochs}"


@env.task
async def list_values_pipeline() -> str:
    """DictConfig whose values include ListConfigs (layer sizes, augmentation pipeline)."""
    cfg = OmegaConf.create(
        {
            "model": {
                "layer_sizes": [64, 128, 256, 512],
                "activations": ["relu", "relu", "relu", "sigmoid"],
            },
            "data": {
                "augmentations": ["random_flip", "random_crop", "color_jitter"],
                "input_size": [224, 224],
            },
        }
    )
    doubled = await double_layer_sizes(cfg)
    summary = await summarize_architecture(doubled)
    return summary


@env.task
async def dataclass_values_pipeline() -> str:
    """Plain DictConfig with dataclass-backed values under each key."""
    cfg = OmegaConf.create(
        {
            "train": TransformConf(name="random_crop", probability=0.8),
            "eval": TransformConf(name="center_crop", probability=1.0),
        }
    )
    return await summarize_transforms(cfg)


@env.task
async def path_refs_pipeline() -> str:
    """Plain DictConfig with a configured Path value."""
    cfg = OmegaConf.create(
        {"model_path": Path("/opt/models/model.bin")},
    )
    return await summarize_path_refs(cfg)


@env.task
async def payload_marker_collision_pipeline() -> str:
    """User keys named "kind" and "values" are preserved as normal config data."""
    cfg = OmegaConf.create(
        {
            "kind": "training-job",
            "values": {
                "lr": 0.001,
                "epochs": 10,
            },
        }
    )
    return await summarize_payload_shaped_user_config(cfg)


@env.task
async def extract_leaf(cfg: DictConfig) -> float:
    """Reads a deeply nested value."""
    return float(cfg.experiment.model.encoder.attention.num_heads)


@env.task(report=True)
async def scale_lr(cfg: DictConfig, factor: float) -> DictConfig:
    """Scales the learning rate in a YAML-loaded config."""
    await log_yaml.aio(cfg, title="Original config in scale_lr")

    new_lr = float(cfg.optimizer.lr) * factor
    return OmegaConf.merge(cfg, {"optimizer": {"lr": new_lr}})


@env.task
async def yaml_pipeline() -> DictConfig:
    """Load a config from a YAML file, pass it through tasks, return the result."""
    cfg = OmegaConf.load(Path("configs") / "training.yaml")

    # Scale up the learning rate by 10x for a warm-up experiment
    scaled = await scale_lr(cfg, factor=10.0)
    return scaled


@env.task
async def deep_nesting_pipeline() -> float:
    """DictConfig nested 4 levels deep."""
    cfg = OmegaConf.create(
        {
            "experiment": {
                "name": "deep-nest",
                "model": {
                    "encoder": {
                        "attention": {
                            "num_heads": 8,
                            "head_dim": 64,
                            "dropout": 0.1,
                        },
                        "ffn": {
                            "hidden_dim": 2048,
                            "activation": "gelu",
                        },
                    },
                    "decoder": {
                        "num_layers": 6,
                    },
                },
            }
        }
    )
    num_heads = await extract_leaf(cfg)
    return num_heads


if __name__ == "__main__":
    flyte.init_from_config()

    print("=== Plain DictConfig pipeline ===")
    run = flyte.run(plain_dictconfig_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Interpolation pipeline ===")
    run = flyte.run(interpolation_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Merge pipeline ===")
    run = flyte.run(merge_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== List values pipeline ===")
    run = flyte.run(list_values_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Dataclass values pipeline ===")
    run = flyte.run(dataclass_values_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Path refs pipeline ===")
    run = flyte.run(path_refs_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Payload marker collision pipeline ===")
    run = flyte.run(payload_marker_collision_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== YAML pipeline ===")
    run = flyte.with_runcontext(copy_style="all").run(yaml_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Deep nesting pipeline ===")
    run = flyte.run(deep_nesting_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
