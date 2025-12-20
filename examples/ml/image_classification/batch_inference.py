"""
Image Classification - Batch Inference Script

Runs batch inference on a large number of images efficiently using:
- Reusable containers (model loaded once, amortized across many images)
- Chunking (multiple images per task for better GPU utilization)
- Parallel processing (multiple replicas running concurrently)

Usage:
    flyte run batch_inference.py batch_inference_pipeline \\
        --model_dir=<model_directory> \\
        --images_dir=<images_directory> \\
        --chunk_size=100

The pipeline will:
1. Discover all images in the directory
2. Partition them into chunks
3. Process each chunk in parallel with reusable containers
4. Generate a comprehensive report with all predictions
"""

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

import flyte
import flyte.io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image from local dependencies
batch_image = (
    flyte.Image.from_debian_base()
    .with_uv_project(pyproject_file=Path("pyproject.toml"))
    .with_pip_packages("unionai-reuse>=0.1.9", "pandas", "pyarrow")
)

# Worker environment with reusable containers
# Using higher concurrency and more replicas for better GPU utilization
worker_env = flyte.TaskEnvironment(
    name="batch_inference_worker",
    image=batch_image,
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
    reusable=flyte.ReusePolicy(
        replicas=8,  # 8 replicas running in parallel
        concurrency=2,  # Each replica can handle 2 tasks concurrently
        idle_ttl=300,  # Keep alive for 5 minutes after idle
        scaledown_ttl=300,
    ),
)

# Driver environment for orchestration
driver_env = flyte.TaskEnvironment(
    name="batch_inference_driver",
    image=batch_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[worker_env],
)


@lru_cache(maxsize=1)
def load_model(model_path: str) -> tuple[AutoModelForImageClassification, AutoImageProcessor, Dict]:
    """
    Lazily load and cache the model, processor, and label mappings.
    This ensures the model is loaded only once per worker container.
    """
    logger.info(f"Loading model from {model_path}")

    # Load label mapping
    with open(Path(model_path) / "label_mapping.json", "r") as f:
        label_data = json.load(f)
        id2label = {int(k): v for k, v in label_data["id2label"].items()}

    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model loaded on GPU")
    else:
        logger.info("Model loaded on CPU")

    return model, processor, id2label


@worker_env.task(cache="auto", retries=3)
async def process_image_batch(
    model_dir: flyte.io.Dir, image_paths: List[str], batch_size: int = 32
) -> flyte.io.File:
    """
    Process a batch of images and return predictions.

    This task is designed to be run in reusable containers, so the model
    is loaded once (via lru_cache) and reused across all invocations.

    Args:
        model_dir: Directory containing the fine-tuned model
        image_paths: List of image file paths to process
        batch_size: Batch size for processing (for GPU efficiency)

    Returns:
        File containing predictions in JSON format
    """
    logger.info(f"Processing batch of {len(image_paths)} images")

    # Download model directory (cached across invocations)
    model_path = await model_dir.download()

    # Load model (cached with lru_cache)
    model, processor, id2label = load_model(str(model_path))

    predictions = []

    # Process images in mini-batches for GPU efficiency
    for i in range(0, len(image_paths), batch_size):
        mini_batch_paths = image_paths[i : i + batch_size]

        # Load images
        images = []
        valid_paths = []
        for img_path in mini_batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                predictions.append(
                    {
                        "image_path": img_path,
                        "error": str(e),
                        "predictions": [],
                    }
                )

        if not images:
            continue

        # Process batch
        inputs = processor(images=images, return_tensors="pt", padding=True)

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Convert to predictions
        for idx, (img_path, prob_vector) in enumerate(zip(valid_paths, probs)):
            # Get top 3 predictions
            top_k = min(3, len(id2label))
            top_probs, top_indices = torch.topk(prob_vector, top_k)

            image_predictions = []
            for prob, label_idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
                image_predictions.append(
                    {
                        "label": id2label[label_idx],
                        "confidence": float(prob),
                    }
                )

            predictions.append(
                {
                    "image_path": img_path,
                    "predictions": image_predictions,
                    "top_prediction": image_predictions[0] if image_predictions else None,
                }
            )

        logger.info(f"Processed {len(images)} images in mini-batch")

    # Save predictions to file
    output_file = Path("/tmp") / f"predictions_{hash(tuple(image_paths))}.json"
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Saved {len(predictions)} predictions to {output_file}")

    return await flyte.io.File.from_local(str(output_file))


@driver_env.task(cache="auto")
async def batch_inference_pipeline(
    model_dir: flyte.io.Dir,
    images_dir: flyte.io.Dir,
    chunk_size: int = 100,
    batch_size: int = 32,
) -> flyte.io.File:
    """
    Run batch inference on all images in a directory.

    This pipeline:
    1. Discovers all images in the directory
    2. Partitions them into chunks
    3. Processes each chunk in parallel using reusable containers
    4. Aggregates results into a comprehensive report

    Args:
        model_dir: Directory containing the fine-tuned model
        images_dir: Directory containing images to process
        chunk_size: Number of images per chunk (affects parallelism)
        batch_size: Mini-batch size for GPU processing

    Returns:
        File containing the complete inference report
    """
    logger.info("Starting batch inference pipeline")

    # Download images directory
    images_path = await images_dir.download()
    images_dir_path = Path(images_path)

    # Discover all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    all_images = []
    for ext in image_extensions:
        all_images.extend(images_dir_path.rglob(f"*{ext}"))
        all_images.extend(images_dir_path.rglob(f"*{ext.upper()}"))

    all_image_paths = [str(img) for img in all_images]
    logger.info(f"Found {len(all_image_paths)} images to process")

    if not all_image_paths:
        raise ValueError(f"No images found in {images_path}")

    # Partition into chunks
    chunks = []
    for i in range(0, len(all_image_paths), chunk_size):
        chunk = all_image_paths[i : i + chunk_size]
        chunks.append(chunk)

    logger.info(f"Partitioned into {len(chunks)} chunks of ~{chunk_size} images each")

    # Process all chunks in parallel
    tasks = []
    for idx, chunk in enumerate(chunks):
        with flyte.group(f"chunk-{idx}"):
            task = asyncio.create_task(process_image_batch(model_dir, chunk, batch_size))
            tasks.append(task)

    logger.info(f"Processing {len(tasks)} chunks in parallel...")
    result_files = await asyncio.gather(*tasks)

    # Aggregate results
    logger.info("Aggregating results...")
    all_predictions = []

    for result_file in result_files:
        file_path = await result_file.download()
        with open(file_path, "r") as f:
            predictions = json.load(f)
            all_predictions.extend(predictions)

    # Generate comprehensive report
    report = generate_report(all_predictions)

    # Save report
    report_path = Path("/tmp") / "batch_inference_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Generated report with {len(all_predictions)} predictions")

    return await flyte.io.File.from_local(str(report_path))


def generate_report(predictions: List[Dict]) -> Dict:
    """
    Generate a comprehensive report from predictions.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Report dictionary with statistics and results
    """
    # Create DataFrame for easier analysis
    df_data = []
    for pred in predictions:
        if pred.get("error"):
            df_data.append(
                {
                    "image_path": pred["image_path"],
                    "top_label": None,
                    "top_confidence": None,
                    "error": pred["error"],
                }
            )
        elif pred.get("top_prediction"):
            df_data.append(
                {
                    "image_path": pred["image_path"],
                    "top_label": pred["top_prediction"]["label"],
                    "top_confidence": pred["top_prediction"]["confidence"],
                    "error": None,
                }
            )

    df = pd.DataFrame(df_data)

    # Calculate statistics
    total_images = len(predictions)
    successful = len(df[df["error"].isna()])
    failed = len(df[df["error"].notna()])

    # Label distribution
    label_counts = df[df["error"].isna()]["top_label"].value_counts().to_dict()

    # Confidence statistics
    if successful > 0:
        avg_confidence = float(df[df["error"].isna()]["top_confidence"].mean())
        min_confidence = float(df[df["error"].isna()]["top_confidence"].min())
        max_confidence = float(df[df["error"].isna()]["top_confidence"].max())
    else:
        avg_confidence = min_confidence = max_confidence = 0.0

    # Build report
    report = {
        "summary": {
            "total_images": total_images,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_images if total_images > 0 else 0.0,
        },
        "confidence_stats": {
            "average": avg_confidence,
            "min": min_confidence,
            "max": max_confidence,
        },
        "label_distribution": label_counts,
        "predictions": predictions,
    }

    return report


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=Path(__file__).parent,
    )

    # Example: Run batch inference
    # You need to provide:
    # 1. model_dir: Output from training.py (flyte.io.Dir)
    # 2. images_dir: Directory containing images to classify

    # For testing, you could run:
    # model_dir = flyte.io.Dir.from_existing_remote("s3://bucket/path/to/model")
    # images_dir = flyte.io.Dir.from_existing_remote("s3://bucket/path/to/images")

    # run = flyte.run(
    #     batch_inference_pipeline,
    #     model_dir=model_dir,
    #     images_dir=images_dir,
    #     chunk_size=100,
    #     batch_size=32,
    # )
    # print(f"Batch Inference Run URL: {run.url}")

    print(
        """
To run batch inference:

1. First, ensure you have a trained model from training.py

2. Prepare a directory with images to classify

3. Run the pipeline:
   flyte run batch_inference.py batch_inference_pipeline \\
       --model_dir=<path_or_remote_ref> \\
       --images_dir=<path_or_remote_ref> \\
       --chunk_size=100

The pipeline will process all images in parallel and generate a comprehensive report.
"""
    )
