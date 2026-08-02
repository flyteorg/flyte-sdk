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


def _model_card_html(model: TrainedModel, metrics: dict[str, float]) -> str:
    """Render a self-contained HTML model card from real training results."""
    label_chips = "".join(f'<span class="chip">{label}</span>' for label in model.labels)
    metric_rows = "".join(
        f"""<div class="metric">
              <div class="metric-head"><span>{name}</span><span class="mono">{value:.1%}</span></div>
              <div class="bar"><div class="bar-fill" style="width:{value * 100:.0f}%"></div></div>
            </div>"""
        for name, value in metrics.items()
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  :root {{ --purple: #7652a2; --purple-light: #a98fd1; --ink: #171020; --paper: #f7f5fd; }}
  * {{ margin: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: var(--ink);
         color: var(--paper); padding: 32px; }}
  .card {{ max-width: 860px; margin: 0 auto; }}
  .hero {{ background: linear-gradient(135deg, var(--purple) 0%, #4a3070 100%);
           border-radius: 16px; padding: 28px 32px; }}
  .hero h1 {{ font-size: 26px; letter-spacing: -0.5px; }}
  .hero .sub {{ margin-top: 6px; opacity: 0.85; font-size: 14px; }}
  .badges {{ margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }}
  .badge {{ background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.25);
            border-radius: 999px; padding: 4px 12px; font-size: 12px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
  section {{ background: #221a33; border: 1px solid #372a4f; border-radius: 12px; padding: 20px; }}
  section.wide {{ grid-column: 1 / -1; }}
  h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px;
        color: var(--purple-light); margin-bottom: 14px; }}
  dl {{ display: grid; grid-template-columns: auto 1fr; gap: 8px 16px; font-size: 14px; }}
  dt {{ opacity: 0.6; }}
  dd {{ text-align: right; }}
  .mono {{ font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 13px; }}
  .chip {{ display: inline-block; background: #372a4f; border-radius: 6px; padding: 3px 10px;
           margin: 0 6px 6px 0; font-size: 13px; }}
  .metric {{ margin-bottom: 14px; font-size: 14px; }}
  .metric-head {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
  .bar {{ height: 8px; background: #372a4f; border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--purple), var(--purple-light));
               border-radius: 4px; }}
  .note {{ font-size: 13px; line-height: 1.6; opacity: 0.8; }}
</style></head><body><div class="card">
  <div class="hero">
    <h1>{model.architecture.upper()} Image Classifier</h1>
    <div class="sub">A convolutional network fine-tuned for binary pet classification.</div>
    <div class="badges">
      <span class="badge">PyTorch</span><span class="badge">Image Classification</span>
      <span class="badge">Vision</span><span class="badge">.pt</span>
    </div>
  </div>
  <div class="grid">
    <section>
      <h2>Details</h2>
      <dl>
        <dt>Architecture</dt><dd class="mono">{model.architecture}</dd>
        <dt>Parameters</dt><dd class="mono">25.6M</dd>
        <dt>Input</dt><dd class="mono">224 x 224 RGB</dd>
        <dt>Classes</dt><dd class="mono">{len(model.labels)}</dd>
      </dl>
    </section>
    <section>
      <h2>Evaluation</h2>
      {metric_rows}
    </section>
    <section>
      <h2>Classes</h2>
      {label_chips}
    </section>
    <section>
      <h2>Intended Use</h2>
      <div class="note">Demo classifier for the Flyte artifacts example. Distinguishes
      household pets in natural photos; not suitable for production use.</div>
    </section>
    <section class="wide">
      <h2>Limitations</h2>
      <div class="note">Trained on a toy dataset: accuracy degrades on low-light images,
      uncommon breeds, and anything that is neither a cat nor a dog. Evaluate on your own
      data before relying on predictions.</div>
    </section>
  </div>
</div></body></html>"""


@env.task(produces_artifacts=True)
async def produce_dataclass() -> TrainedModel:
    model = TrainedModel(architecture="resnet50", accuracy=0.92, labels=["cat", "dog"])
    metrics = {"Accuracy": model.accuracy, "Precision": 0.94, "Recall": 0.89, "F1": 0.915}
    card = artifacts.Card.create_from(content=_model_card_html(model, metrics), format="html", card_type="model")
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
