"""
Example: DictConfig and ListConfig flowing through a realistic multi-task pipeline.

Covers:
- cfg passed from parent to multiple child tasks
- Child task modifying cfg and returning it to parent
- ListConfig produced by one task consumed by another
- Mixed: task takes both DictConfig and ListConfig inputs
- Structured config surviving multiple task hops
"""

from dataclasses import dataclass, field
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from omegaconf import DictConfig, ListConfig, OmegaConf

env = flyte.TaskEnvironment(
    name="omegaconf-pipeline-example",
    image=flyte.Image.from_debian_base(name="omegaconf-pipeline-example").clone(
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


@dataclass
class DataConf:
    path: str = ""
    preprocessed: bool = False


@dataclass
class ResultsConf:
    val_loss: float = 0.0
    final_lr: float = 0.0
    num_lr_steps: int = 0


@dataclass
class TrainConf:
    optimizer: OptimizerConf = field(default_factory=OptimizerConf)
    data: DataConf = field(default_factory=DataConf)
    results: ResultsConf = field(default_factory=ResultsConf)
    epochs: int = 10
    batch_size: int = 32
    experiment: str = "baseline"


@env.task
async def preprocess(cfg: DictConfig, dataset: str) -> DictConfig:
    """First stage: fills in the data section of cfg."""
    return OmegaConf.merge(cfg, {"data": {"path": dataset, "preprocessed": True}})


@env.task
async def build_schedule(cfg: DictConfig) -> ListConfig:
    """Produces an LR schedule from cfg."""
    lrs = [cfg.optimizer.lr * (0.5**i) for i in range(cfg.epochs)]
    return OmegaConf.create(lrs)


@env.task
async def train(cfg: DictConfig, lr_schedule: ListConfig) -> tuple[DictConfig, float]:
    """Simulates training. Returns final cfg (with results filled in) and val loss."""
    final_lr = float(lr_schedule[-1])
    val_loss = final_lr * 10  # placeholder
    result_cfg = OmegaConf.merge(
        cfg,
        {
            "results": {
                "val_loss": val_loss,
                "final_lr": final_lr,
                "num_lr_steps": len(lr_schedule),
            }
        },
    )
    return result_cfg, val_loss


@env.task
async def evaluate(result_cfg: DictConfig, val_loss: float) -> str:
    """Final stage: formats a report from the result config."""
    return (
        f"experiment={result_cfg.experiment} "
        f"data={result_cfg.data.path} "
        f"val_loss={val_loss:.6f} "
        f"final_lr={result_cfg.results.final_lr:.6f} "
        f"lr_steps={result_cfg.results.num_lr_steps}"
    )


@env.task
async def training_pipeline(dataset: str) -> str:
    """Full pipeline: cfg flows through preprocess → build_schedule → train → evaluate."""
    cfg = OmegaConf.structured(
        TrainConf(
            optimizer=OptimizerConf(lr=0.01, weight_decay=1e-5),
            epochs=5,
            batch_size=64,
            experiment="structured-cfg-pipeline",
        )
    )

    # Each hop serializes/deserializes cfg — structured config survives intact
    preprocessed_cfg = await preprocess(cfg, dataset=dataset)
    lr_schedule = await build_schedule(preprocessed_cfg)
    result_cfg, val_loss = await train(preprocessed_cfg, lr_schedule=lr_schedule)
    report = await evaluate(result_cfg, val_loss=val_loss)
    return report


if __name__ == "__main__":
    flyte.init_from_config()

    print("=== Multi-task training pipeline ===")
    run = flyte.run(training_pipeline, dataset="s3://my-bucket/imagenet")
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
