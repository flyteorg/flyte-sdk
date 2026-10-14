"""
Image Classification - Training Script

Fine-tunes a small vision transformer model on HuggingFace datasets and
**publishes the result as a model artifact**.

The task is marked `produces_artifacts=True` and wraps its `flyte.io.Dir` output
with `flyte.artifacts.new(...)`. On success the platform registers a new version
of the `beans-classifier` artifact, records this action as its source, and
attaches the rendered HTML model card. Two things downstream hang off that
publish, without either of them naming this run:

- `batch_inference.py` deploys an `OnArtifact` trigger, so a new model version
  automatically kicks off a batch-inference run against it;
- `serving.py` mounts the artifact with `flyte.app.ArtifactValue`, so a plain
  `flyte deploy` picks up the model with no run name to thread through.

Usage:
    flyte run training.py finetune_image_model

    Or with custom parameters:
    flyte run training.py finetune_image_model \\
        --dataset_name="ethz/food101" \\
        --num_epochs=5 \\
        --batch_size=32

Inspect what was published:
    flyte get artifact beans-classifier
"""

import json
import logging
from pathlib import Path

import numpy as np
from datasets import load_dataset
from model_artifact import MODEL_ARTIFACT_NAME
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

import flyte
import flyte.artifacts as artifacts
import flyte.io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create image from local dependencies (will use pyproject.toml)
training_image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=Path("pyproject.toml"), extra_args="--extra training"
)

training_env = flyte.TaskEnvironment(
    name="image_finetune_training",
    image=training_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu=1, disk="10Gi"),
    cache=flyte.Cache("auto", "1.0"),
    env_vars={"HF_XET_HIGH_PERFORMANCE": "1"},
)


def _model_card_html(
    *,
    base_model: str,
    dataset_name: str,
    labels: list[str],
    hyperparams: dict[str, str],
    metrics: dict[str, float],
) -> str:
    """
    Render a self-contained HTML model card from the real training results.

    HTML rather than markdown on purpose: the UI renders HTML cards in an iframe,
    which works against any object store, while a markdown card needs a browser
    fetch of the presigned URL and therefore CORS configured on the bucket.
    """
    label_chips = "".join(f'<span class="chip">{label}</span>' for label in labels)
    param_rows = "".join(f"<dt>{name}</dt><dd class='mono'>{value}</dd>" for name, value in hyperparams.items())
    metric_rows = (
        "".join(
            f"""<div class="metric">
              <div class="metric-head"><span>{name}</span><span class="mono">{value:.1%}</span></div>
              <div class="bar"><div class="bar-fill" style="width:{min(value, 1.0) * 100:.0f}%"></div></div>
            </div>"""
            for name, value in metrics.items()
        )
        or "<div class='note'>No validation split in this dataset — nothing was evaluated.</div>"
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
  .mono {{ font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 13px;
           overflow-wrap: anywhere; }}
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
    <h1>{MODEL_ARTIFACT_NAME}</h1>
    <div class="sub">{base_model} fine-tuned on {dataset_name} for image classification.</div>
    <div class="badges">
      <span class="badge">PyTorch</span><span class="badge">Transformers</span>
      <span class="badge">Image Classification</span><span class="badge">safetensors</span>
    </div>
  </div>
  <div class="grid">
    <section>
      <h2>Training</h2>
      <dl>{param_rows}</dl>
    </section>
    <section>
      <h2>Evaluation</h2>
      {metric_rows}
    </section>
    <section>
      <h2>Classes ({len(labels)})</h2>
      {label_chips}
    </section>
    <section>
      <h2>Intended Use</h2>
      <div class="note">Demo classifier for the Flyte artifacts + triggers example. Serve it with
      <span class="mono">serving.py</span>, or score a batch of images with
      <span class="mono">batch_inference.py</span> — both bind to this artifact by name.</div>
    </section>
    <section class="wide">
      <h2>Limitations</h2>
      <div class="note">Trained for a handful of epochs on a small public dataset to keep the example
      cheap to run. Accuracy degrades on out-of-distribution photos, and the label set is closed —
      every input is forced into one of the classes above. Evaluate on your own data before relying
      on predictions.</div>
    </section>
  </div>
</div></body></html>"""


# `produces_artifacts=True` tells the platform to look for wrapped values among this
# task's outputs. Without it the `artifacts.new(...)` wrapper is silently unwrapped
# and you get a plain Dir back — no artifact, no trigger, nothing for the app to bind.
@training_env.task(produces_artifacts=True)
async def finetune_image_model(
    dataset_name: str = "AI-Lab-Makerere/beans",
    model_name: str = "WinKawaks/vit-tiny-patch16-224",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
) -> flyte.io.Dir:
    """
    Fine-tune a small vision transformer model and publish it as a model artifact.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "AI-Lab-Makerere/beans", "uoft-cs/cifar10", "ethz/food101")
        model_name: HuggingFace model name (small ViT models work best)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training

    Returns:
        Directory containing the fine-tuned model and processor, published as a new
        version of the `beans-classifier` model artifact.
    """
    logger.info(f"Starting fine-tuning: model={model_name}, dataset={dataset_name}")

    # Load dataset with XET acceleration
    logger.info(f"Loading dataset {dataset_name} with XET acceleration...")
    dataset = load_dataset(dataset_name)

    # Get label information - try different common column names
    train_data = dataset["train"]
    if hasattr(train_data.features.get("labels", None), "names"):
        labels = train_data.features["labels"].names
    elif hasattr(train_data.features.get("label", None), "names"):
        labels = train_data.features["label"].names
    else:
        # Fallback: extract unique labels from the data
        label_col = "labels" if "labels" in train_data.column_names else "label"
        labels = sorted(set(train_data[label_col]))

    num_labels = len(labels)
    id2label = dict(enumerate(labels))
    label2id = {label: i for i, label in enumerate(labels)}

    logger.info(f"Dataset loaded: {num_labels} classes - {labels}")

    # Load model and processor
    logger.info(f"Loading model and processor: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Preprocessing function
    def preprocess_images(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        inputs = processor(images, return_tensors="pt")
        if "labels" in examples:
            inputs["labels"] = examples["labels"]
        elif "label" in examples:
            inputs["labels"] = examples["label"]
        return inputs

    # Prepare datasets
    logger.info("Preprocessing datasets...")
    train_dataset = dataset["train"].with_transform(preprocess_images)
    val_dataset = dataset["validation"].with_transform(preprocess_images) if "validation" in dataset else None

    # Training arguments
    output_dir = Path("/tmp/finetuned_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="accuracy" if val_dataset else None,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Metric function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics if val_dataset else None,
    )

    # Train
    logger.info(f"Starting training for {num_epochs} epochs...")
    trainer.train()

    # Evaluate so the model card and the artifact attrs carry real numbers rather
    # than claims. Skipped when the dataset ships no validation split.
    metrics: dict[str, float] = {}
    if val_dataset:
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {eval_metrics}")
        if "eval_accuracy" in eval_metrics:
            metrics["Accuracy"] = float(eval_metrics["eval_accuracy"])

    # Save final model and processor
    final_model_dir = Path("/tmp/final_model")
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)

    # Save label mapping
    with open(final_model_dir / "label_mapping.json", "w") as f:  # noqa: ASYNC230
        json.dump({"id2label": id2label, "label2id": label2id}, f)

    model_dir = await flyte.io.Dir.from_local(final_model_dir)

    # Render and upload the model card. The card is stored alongside the artifact
    # and rendered by the UI; it is metadata about the model, not part of the Dir.
    card = await artifacts.Card.create_from.aio(
        content=_model_card_html(
            base_model=model_name,
            dataset_name=dataset_name,
            labels=[str(label) for label in labels],
            hyperparams={
                "Base model": model_name,
                "Dataset": dataset_name,
                "Epochs": str(num_epochs),
                "Batch size": str(batch_size),
                "Learning rate": f"{learning_rate:g}",
                "Classes": str(num_labels),
            },
            metrics=metrics,
        ),
        format="html",
        card_type="model",
    )

    # `attrs` are free-form key/values stored on the artifact version and shown in the
    # UI — keep them to things worth filtering a model list by. `create_model_metadata`
    # additionally stamps the reserved kind key, so this registers as kind="model".
    #
    # No `version=` is passed: the platform derives one from the producing action, which
    # is what you want here. Pass one explicitly only when an external identifier (a git
    # sha, an upstream commit) is the real identity of the weights.
    metadata = artifacts.Metadata.create_model_metadata(
        name=MODEL_ARTIFACT_NAME,
        description=f"{model_name} fine-tuned on {dataset_name} ({num_labels} classes)",
        framework="PyTorch",
        model_type="Vision Transformer",
        architecture=model_name,
        task="Image Classification",
        modality=("image",),
        serial_format="safetensors",
        card=card,
        attrs={
            "base_model": model_name,
            "dataset": dataset_name,
            "num_classes": str(num_labels),
            "epochs": str(num_epochs),
            **({"accuracy": f"{metrics['Accuracy']:.4f}"} if "Accuracy" in metrics else {}),
        },
    )

    logger.info(f"Fine-tuning complete! Publishing artifact {MODEL_ARTIFACT_NAME}")

    # `artifacts.new` is a zero-copy wrapper: downstream tasks still receive a plain
    # flyte.io.Dir. It must be returned as a *top-level* output — nesting it inside a
    # dataclass or model drops the metadata at serialization time.
    return artifacts.new(model_dir, metadata)


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=Path(__file__).parent,
    )

    # Run training pipeline. Note the env sets `cache=flyte.Cache("auto", "1.0")`:
    # re-running with identical inputs is a cache hit, which reuses the previous
    # outputs and therefore publishes no new artifact version (and fires no
    # trigger). Vary an input — or use
    # `flyte.with_runcontext(overwrite_cache=True).run(...)` — to force a
    # genuinely new model version.
    run = flyte.run(
        finetune_image_model,
        dataset_name="AI-Lab-Makerere/beans",
        model_name="WinKawaks/vit-tiny-patch16-224",
        num_epochs=3,
        batch_size=32,
    )
    print(f"Training Run URL: {run.url}")
    run.wait()

    # The artifact is registered once the action succeeds. Anything bound to the
    # name — the batch-inference trigger, the serving app — picks it up from here.
    from flyte.remote import Artifact

    published = Artifact.get(MODEL_ARTIFACT_NAME)
    print(f"Published {published.name}@{published.version} (kind={published.kind}): {published.url}")
