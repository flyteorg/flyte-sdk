"""
Image Classification - Training Script

Fine-tunes a small vision transformer model on HuggingFace datasets.
This script handles the model training and saves the fine-tuned model.

Usage:
    flyte run training.py finetune_image_model

    Or with custom parameters:
    flyte run training.py finetune_image_model \\
        --dataset_name="food101" \\
        --num_epochs=5 \\
        --batch_size=32
"""

import json
import logging
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

import flyte
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
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu=1),
    cache=flyte.Cache("auto", "1.0"),
    env_vars={"HF_XET_HIGH_PERFORMANCE": "1"},
)


@training_env.task
async def finetune_image_model(
    dataset_name: str = "beans",
    model_name: str = "WinKawaks/vit-tiny-patch16-224",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
) -> flyte.io.Dir:
    """
    Fine-tune a small vision transformer model on an image classification dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "beans", "cifar10", "food101")
        model_name: HuggingFace model name (small ViT models work best)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training

    Returns:
        Directory containing the fine-tuned model and processor
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
        inputs["labels"] = examples["labels"]
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

    # Save final model and processor
    final_model_dir = Path("/tmp/final_model")
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)

    # Save label mapping
    with open(final_model_dir / "label_mapping.json", "w") as f:  # noqa: ASYNC230
        json.dump({"id2label": id2label, "label2id": label2id}, f)

    logger.info("Fine-tuning complete!")
    return await flyte.io.Dir.from_local(final_model_dir)


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=Path(__file__).parent,
    )

    # Run training pipeline
    run = flyte.run(
        finetune_image_model,
        dataset_name="beans",
        model_name="WinKawaks/vit-tiny-patch16-224",
        num_epochs=3,
        batch_size=32,
    )
    print(f"Training Run URL: {run.url}")
