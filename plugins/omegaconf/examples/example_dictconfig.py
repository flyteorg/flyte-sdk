"""
Example: DictConfig as task inputs and outputs.

Covers:
- Plain dict-based DictConfig
- YAML-loaded DictConfig (configs/training.yaml)
- Merged config (base + overrides)
- OmegaConf interpolation (resolved at serialization)
- DictConfig as both input and output
- Modifying and returning a DictConfig from a task
- DictConfig with list values (e.g. layer sizes, augmentation pipeline)
- Deeply nested DictConfig (3+ levels)
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import flyte
from flyte._image import PythonWheels

env = flyte.TaskEnvironment(
    name="omegaconf-dictconfig-example",
    image=flyte.Image.from_debian_base(name="omegaconf-dictconfig-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-omegaconf",
        ),
    ),
)


@env.task
async def normalize_config(cfg: DictConfig) -> DictConfig:
    """Receives a plain DictConfig, adds a field, returns it."""
    return OmegaConf.merge(cfg, {"normalized": True})


@env.task
async def print_config(cfg: DictConfig) -> str:
    """Receives a DictConfig, returns a string summary."""
    return OmegaConf.to_yaml(cfg)


@env.task
async def use_interpolated_config(cfg: DictConfig) -> float:
    """Receives a config where interpolations are already resolved at serialization time."""
    # By the time this runs, cfg.optimizer.lr is already the resolved float (not "${base_lr}")
    return float(cfg.optimizer.lr)


@env.task
async def merge_with_overrides(
    base_cfg: DictConfig, override_cfg: DictConfig
) -> DictConfig:
    """Merges two DictConfigs inside a task and returns the result."""
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


@env.task
async def double_layer_sizes(cfg: DictConfig) -> DictConfig:
    """Receives a config whose values include lists; doubles each layer size."""
    doubled = [size * 2 for size in cfg.model.layer_sizes]
    return OmegaConf.merge(cfg, {"model": {"layer_sizes": doubled}})


@env.task
async def summarize_architecture(cfg: DictConfig) -> str:
    layers = list(cfg.model.layer_sizes)
    augmentations = list(cfg.data.augmentations)
    return f"layers={layers} depth={len(layers)} augmentations={augmentations}"


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
async def extract_leaf(cfg: DictConfig) -> float:
    """Reads a deeply nested value."""
    return float(cfg.experiment.model.encoder.attention.num_heads)


@env.task
async def scale_lr(cfg: DictConfig, factor: float) -> DictConfig:
    """Scales the learning rate in a YAML-loaded config."""
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

    print("\n=== YAML pipeline ===")
    run = flyte.with_runcontext(copy_style="all").run(yaml_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Deep nesting pipeline ===")
    run = flyte.run(deep_nesting_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
