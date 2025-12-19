# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flyte>=2.0.0b39",
#     "transformers[torch]",
#     "torch",
#     "torchvision",
#     "datasets",
#     "pillow",
#     "numpy",
#     "fastapi",
#     "uvicorn",
#     "python-multipart",
# ]
# ///
"""
Image Classification - Fine-tuning and Serving Pipeline

This script demonstrates:
1. Fine-tuning a small vision transformer model (ViT-tiny)
2. Training on HuggingFace datasets with XET for fast transfers
3. Saving the fine-tuned model
4. Serving the model via FastAPI with image upload endpoint

The model is fine-tuned on the beans dataset (plant disease classification)
or a custom dataset passed by the user.

USAGE:

1. Enable XET for fast HuggingFace transfers (optional but recommended):
   export HF_XET_HIGH_PERFORMANCE=1

2. Train the model:
   flyte run fine_tune_serve.py finetune_image_model

   Or with custom parameters:
   flyte run fine_tune_serve.py finetune_image_model \\
       --dataset_name="food101" \\
       --num_epochs=5 \\
       --batch_size=32

3. Serve the trained model:
   flyte serve fine_tune_serve.py

4. Test the API:
   curl -X POST http://localhost:8000/predict \\
       -F "file=@/path/to/image.jpg"

Available datasets: beans, cifar10, food101, imagenet-1k, etc.
The model used is WinKawaks/vit-tiny-patch16-224 (very small and efficient).
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

import flyte
import flyte.io
from flyte.app import Input
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# TRAINING TASKS
# =============================================================================

# Create image from script dependencies
training_image = flyte.Image.from_uv_script(__file__, name="image_finetune_train", pre=True)

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
        labels = sorted(list(set(train_data[label_col])))

    num_labels = len(labels)
    id2label = {i: label for i, label in enumerate(labels)}
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
    import json
    with open(final_model_dir / "label_mapping.json", "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f)

    logger.info("Fine-tuning complete!")
    return await flyte.io.Dir.from_local(final_model_dir)


# =============================================================================
# SERVING APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    """
    logger.info("Starting up: Loading model and processor...")

    # Load model from mounted directory
    model_path = Path("/tmp/finetuned_model")

    if model_path.exists():
        await load_model_artifacts(model_path)
        logger.info("Startup complete: Model loaded successfully")
    else:
        logger.warning(f"Model not found at {model_path}. App will start but won't be ready.")

    yield

    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="Fine-tuned vision model serving API",
    version="1.0.0",
    lifespan=lifespan,
)

# Create Flyte FastAPI App Environment
serving_image = flyte.Image.from_uv_script(__file__, name="image_finetune_serve", pre=True)

env = FastAPIAppEnvironment(
    name="image-classification-api",
    app=app,
    description="Serving fine-tuned image classification model",
    image=serving_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    requires_auth=False,
    inputs=[
        Input(
            name="model",
            value=flyte.app.RunOutput(
                task_name="fine_tune_serve.finetune_image_model",
                type="directory"
            ),
            mount="/tmp/finetuned_model",
        )
    ],
)

# Application state
app.state.model: Optional[AutoModelForImageClassification] = None
app.state.processor: Optional[AutoImageProcessor] = None
app.state.id2label: dict = {}
app.state.label2id: dict = {}


# Response models
class PredictionResult(BaseModel):
    label: str
    confidence: float


class ClassificationResponse(BaseModel):
    predictions: list[PredictionResult]
    top_prediction: PredictionResult


async def load_model_artifacts(model_path: Path):
    """
    Load the fine-tuned model, processor, and label mappings.
    """
    logger.info(f"Loading model from {model_path}")

    # Load label mapping
    import json
    with open(model_path / "label_mapping.json", "r") as f:
        label_data = json.load(f)
        app.state.id2label = {int(k): v for k, v in label_data["id2label"].items()}
        app.state.label2id = label_data["label2id"]

    # Load processor and model
    app.state.processor = AutoImageProcessor.from_pretrained(str(model_path))
    app.state.model = AutoModelForImageClassification.from_pretrained(str(model_path))
    app.state.model.eval()

    logger.info(f"Model loaded with {len(app.state.id2label)} classes")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy" if app.state.model is not None else "not_ready",
        "model_loaded": app.state.model is not None,
        "num_classes": len(app.state.id2label),
        "classes": list(app.state.label2id.keys()) if app.state.label2id else [],
    }


@app.post("/predict", response_model=ClassificationResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image.

    Upload an image file and get classification predictions.
    """
    if app.state.model is None or app.state.processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and validate image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    logger.info(f"Classifying image: {file.filename}")

    # Preprocess and predict
    inputs = app.state.processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = app.state.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    # Get all predictions sorted by confidence
    predictions = []
    for idx, prob in enumerate(probs.tolist()):
        label = app.state.id2label[idx]
        predictions.append(PredictionResult(label=label, confidence=prob))

    # Sort by confidence descending
    predictions.sort(key=lambda x: x.confidence, reverse=True)

    return ClassificationResponse(
        predictions=predictions,
        top_prediction=predictions[0],
    )


@app.get("/classes")
async def get_classes():
    """
    Get all available classification classes.
    """
    if not app.state.id2label:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "num_classes": len(app.state.id2label),
        "classes": list(app.state.label2id.keys()),
    }


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=Path(__file__).parent,
        log_level=logging.DEBUG,
    )

    # Option 1: Run training pipeline
    # Uncomment to train the model first
    run = flyte.run(
        finetune_image_model,
        dataset_name="beans",
        model_name="WinKawaks/vit-tiny-patch16-224",
        num_epochs=3,
        batch_size=32,
    )
    print(f"Training Run URL: {run.url}")
    run.wait()
    print("Training completed!")

    # Option 2: Serve the model (requires training to be completed first)
    print("To serve the model, use: flyte serve fine_tune_serve.py")
