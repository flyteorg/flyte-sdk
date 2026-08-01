"""Artifacts end to end: publish, produce, and consume.

Artifacts are addressable assets — files, directories, dataframes, or
structured models. Primitive values (str, int, ...) are NOT allowed as
artifacts, and an artifact must be a top-level task output (nesting a wrapped
value inside another model fails at serialization time).

Two ways an artifact is born:
- Explicit publish: `flyte.remote.Artifact.create(...)` from anywhere (no task).
- Produced by a task: mark the task `produces_artifacts=True` and wrap the
  output with `flyte.artifacts.new(value, Metadata)`. After the task succeeds
  the platform extracts the artifact and records the producing action as its
  source.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame, Dir, File

env = flyte.TaskEnvironment(
    name="artifact_example",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
    # pandas + pyarrow need more than the default task memory.
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)


# 1. A File artifact: the file uploads to blob storage; the artifact stores the
#    typed reference.
@env.task(produces_artifacts=True)
async def produce_file() -> File:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("model weights bytes")
    file = await File.from_local(f.name)
    return artifacts.new(file, artifacts.Metadata(name="weights-file", description="Raw weights file"))


# 2. A Dir artifact: a whole directory as one asset (e.g. a sharded checkpoint).
@env.task(produces_artifacts=True)
async def produce_dir() -> Dir:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "shard-0.bin").write_text("shard 0")
    (tmp / "shard-1.bin").write_text("shard 1")
    d = await Dir.from_local(str(tmp))
    return artifacts.new(d, artifacts.Metadata(name="checkpoint-dir", description="Sharded checkpoint"))


# 3. A DataFrame artifact: stored as parquet, consumable as a typed dataframe.
@env.task(produces_artifacts=True)
async def produce_dataframe() -> DataFrame:
    df = pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]})
    metadata = artifacts.Metadata(name="training-set", description="Toy training data", data={"rows": "3"})
    return artifacts.new(DataFrame.from_df(df), metadata)


# 4. A structured model artifact (dataclass), with a model card.
@dataclass
class TrainedModel:
    architecture: str
    accuracy: float
    labels: list[str]


@env.task(produces_artifacts=True)
async def produce_dataclass() -> TrainedModel:
    model = TrainedModel(architecture="resnet50", accuracy=0.92, labels=["cat", "dog"])
    card = artifacts.Card.create_from(
        content="<h1>Model Card</h1><p>ResNet50 toy classifier.</p>", format="html", card_type="model"
    )
    metadata = artifacts.Metadata.create_model_metadata(
        name="trained-model",
        description="A toy classifier",
        framework="PyTorch",
        model_type="Neural Network",
        architecture="ResNet50",
        task="Image Classification",
        modality=("image",),
        serial_format="pt",
        short_description="ResNet50 toy classifier.",
        card=card,
    )
    return artifacts.new(model, metadata)


# 5. A pydantic model artifact.
class EvalReport(BaseModel):
    dataset: str
    f1: float


@env.task(produces_artifacts=True)
async def produce_pydantic() -> EvalReport:
    report = EvalReport(dataset="toy-eval", f1=0.87)
    return artifacts.new(report, artifacts.Metadata(name="eval-report", description="Eval metrics"))


# Consuming artifacts: they bind like normal typed inputs.
@env.task
async def consume(model: TrainedModel, data: DataFrame) -> str:
    df = await data.open(pd.DataFrame).all()
    return f"{model.architecture} trained on {len(df)} rows"


# Driver: fans out to the producers. Child outputs come back unwrapped, so the
# driver itself produces no artifacts — each child records its own.
@env.task
async def produce_all() -> tuple[File, Dir, DataFrame, TrainedModel, EvalReport]:
    return (
        await produce_file(),
        await produce_dir(),
        await produce_dataframe(),
        await produce_dataclass(),
        await produce_pydantic(),
    )


if __name__ == "__main__":
    flyte.init_from_config()

    from flyte.remote import Artifact

    # Explicit publish from the local machine (no task): a File artifact.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("published from local")
    published = Artifact.create(
        File.from_local_sync(f.name),
        name="local-file-artifact",
        description="Published from local",
        python_type=File,
    )
    print(f"published {published.name}@{published.version}")

    # Primitives are rejected.
    try:
        artifacts.new("a plain string", artifacts.Metadata(name="nope"))
    except TypeError as e:
        print(f"as expected: {e}")

    # Produce artifacts from tasks; the producing action is recorded as source.
    run = flyte.run(produce_all)
    print(run.url)
    run.wait()

    # Consume: fetch by name and bind as typed inputs.
    model = Artifact.get("trained-model")
    data = Artifact.get("training-set")
    result = flyte.run(consume, model=model, data=data)
    print(result.url)
