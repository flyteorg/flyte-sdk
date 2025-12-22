"""
Image Classification - Serving Script

Serves a fine-tuned vision model via FastAPI with image upload endpoint.
This script handles model loading and inference.

Usage:
    flyte serve serving.py env

Test the API:
    curl -X POST http://localhost:8000/predict -F "file=@/path/to/image.jpg"
"""

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiofiles
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

import flyte
from flyte.app import Parameter
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
serving_image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=Path("pyproject.toml"), extra_args="--extra serving"
)

env = FastAPIAppEnvironment(
    name="image-classification-api",
    app=app,
    description="Serving fine-tuned image classification model",
    image=serving_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    include=[
        "index.html",
    ],
    requires_auth=False,
    parameters=[
        Parameter(
            name="model",
            value=flyte.app.RunOutput(task_name="image_finetune_training.finetune_image_model", type="directory"),
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
    with open(model_path / "label_mapping.json", "r") as f:  # noqa: ASYNC230
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


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Serve the web interface for image classification.
    """
    html_path = Path(__file__).parent / "index.html"
    with aiofiles.open(html_path, "r") as f:
        return await f.read()


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
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e!s}")

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
    )

    # Serve the model (requires training to be completed first)
    deployed_app = flyte.serve(env)
    print(f"Model served at {deployed_app.url}, {deployed_app.endpoint}")
    print(
        "You can test it on the FastAPI /docs page with an image from "
        "https://www.vegetables.bayer.com/ca/en-ca/resources/agronomic-spotlights/foliar-fungal-diseases-bean-2.html"
    )
