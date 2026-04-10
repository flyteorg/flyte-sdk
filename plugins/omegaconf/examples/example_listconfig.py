"""
Example: ListConfig as task inputs and outputs.

Covers:
- Simple list of primitives
- List of strings (e.g. layer names, optimizer choices)
- Nested list (list of lists)
- List of dicts (round-trips as ListConfig of DictConfigs)
- List of dataclass instances (round-trips as ListConfig of typed DictConfigs)
- ListConfig as output of one task, input to another
- Building a ListConfig inside a task and returning it
"""

from dataclasses import dataclass
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from omegaconf import DictConfig, ListConfig, OmegaConf

env = flyte.TaskEnvironment(
    name="omegaconf-listconfig-example",
    image=flyte.Image.from_debian_base(name="omegaconf-listconfig-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-omegaconf",
        ),
    ),
)


@dataclass
class LayerConf:
    name: str
    width: int
    activation: str


@env.task
async def scale_values(values: ListConfig, factor: float) -> ListConfig:
    """Multiplies each numeric value in the list by factor."""
    return OmegaConf.create([v * factor for v in values])


@env.task
async def filter_above_threshold(values: ListConfig, threshold: float) -> ListConfig:
    """Returns only values above the threshold."""
    return OmegaConf.create([v for v in values if v > threshold])


@env.task
async def build_lr_schedule(base_lr: float, num_stages: int) -> ListConfig:
    """Builds a learning rate schedule as a ListConfig."""
    return OmegaConf.create([base_lr * (0.1**i) for i in range(num_stages)])


@env.task
async def select_best_config(configs: ListConfig) -> DictConfig:
    """Receives a list of DictConfigs, returns the one with the highest lr."""
    best = max(OmegaConf.to_container(configs), key=lambda c: c["optimizer"]["lr"])
    return OmegaConf.create(best)


@env.task
async def flatten_grid(grid: ListConfig) -> ListConfig:
    """Flattens a list of lists into a single list."""
    flat = [item for sublist in OmegaConf.to_container(grid) for item in sublist]
    return OmegaConf.create(flat)


@env.task
async def summarize_list(values: ListConfig) -> str:
    return f"count={len(values)} min={min(values):.4f} max={max(values):.4f}"


@env.task
async def summarize_layer_configs(layers: ListConfig) -> str:
    """Receives a list of dataclass-backed DictConfigs and inspects their types."""
    assert OmegaConf.get_type(layers[0]) == LayerConf
    names = [layer.name for layer in layers]
    widths = [layer.width for layer in layers]
    return f"layers={names} widths={widths} final={layers[-1].activation}"


@env.task
async def lr_schedule_pipeline() -> str:
    """Build an LR schedule, filter it, and summarize."""
    schedule = await build_lr_schedule(base_lr=0.1, num_stages=6)
    filtered = await filter_above_threshold(schedule, threshold=1e-4)
    summary = await summarize_list(filtered)
    return summary


@env.task
async def list_of_configs_pipeline() -> DictConfig:
    """Pass a list of DictConfigs between tasks, pick the best one."""
    configs = OmegaConf.create(
        [
            {"optimizer": {"lr": 0.001}, "training": {"epochs": 10}},
            {"optimizer": {"lr": 0.01}, "training": {"epochs": 20}},
            {"optimizer": {"lr": 0.1}, "training": {"epochs": 5}},
        ]
    )
    best = await select_best_config(configs)
    return best


@env.task
async def nested_list_pipeline() -> ListConfig:
    """Flatten a grid of hyperparameter combinations."""
    grid = OmegaConf.create(
        [
            [0.001, 0.01, 0.1],
            [10, 20, 50],
        ]
    )
    flat = await flatten_grid(grid)
    return flat


@env.task
async def list_of_strings_pipeline() -> ListConfig:
    """Pass a list of strings through a task."""
    scaled = await scale_values(OmegaConf.create([1.0, 2.0, 3.0, 4.0]), factor=0.5)
    return scaled


@env.task
async def total_params(layer_configs: ListConfig) -> int:
    """Receives a list of dicts each containing their own list of layer sizes."""
    total = 0
    for block in layer_configs:
        for size in block.layer_sizes:
            total += size
    return total


@env.task
async def nested_complex_pipeline() -> int:
    """ListConfig where each element is a DictConfig that itself has list values."""
    layer_configs = OmegaConf.create(
        [
            {
                "name": "encoder",
                "layer_sizes": [768, 512, 256],
                "dropout": [0.1, 0.1, 0.0],
            },
            {"name": "bottleneck", "layer_sizes": [128], "dropout": [0.2]},
            {
                "name": "decoder",
                "layer_sizes": [256, 512, 768],
                "dropout": [0.0, 0.1, 0.1],
            },
        ]
    )
    total = await total_params(layer_configs)
    return total


@env.task
async def dataclass_list_pipeline() -> str:
    """ListConfig whose elements are dataclass-backed configs."""
    layers = OmegaConf.create(
        [
            LayerConf(name="encoder", width=768, activation="gelu"),
            LayerConf(name="bottleneck", width=128, activation="relu"),
            LayerConf(name="decoder", width=768, activation="linear"),
        ]
    )
    return await summarize_layer_configs(layers)


if __name__ == "__main__":
    flyte.init_from_config()

    print("=== LR schedule pipeline ===")
    run = flyte.run(lr_schedule_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== List of configs pipeline ===")
    run = flyte.run(list_of_configs_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Nested list pipeline ===")
    run = flyte.run(nested_list_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== List of strings pipeline ===")
    run = flyte.run(list_of_strings_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Nested complex pipeline ===")
    run = flyte.run(nested_complex_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")

    print("\n=== Dataclass list pipeline ===")
    run = flyte.run(dataclass_list_pipeline)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
