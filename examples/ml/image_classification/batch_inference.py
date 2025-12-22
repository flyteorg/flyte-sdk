"""
Image Classification - Batch Inference with DataFrames & Reports

Runs batch inference on a large number of images efficiently using:
- DataFrame-based processing (efficient serialization and analysis)
- Streaming aggregation (low memory footprint using asyncio.as_completed)
- Reusable containers (model loaded once, amortized across many images)
- Chunking (multiple images per task for better GPU utilization)
- Parallel processing (multiple replicas running concurrently)
- Interactive HTML reports (charts and visualizations with Flyte reports)

Usage:
    # Run with interactive report (recommended):
    flyte run batch_inference.py batch_inference_with_report \\
        --model_dir=<model_directory> \\
        --images_dir=<images_directory> \\
        --chunk_size=100

    # Run inference only (no report):
    flyte run batch_inference.py batch_inference_pipeline \\
        --model_dir=<model_directory> \\
        --images_dir=<images_directory> \\
        --chunk_size=100

The pipeline will:
1. Discover all images in the directory
2. Partition them into chunks
3. Process each chunk in parallel with reusable containers
4. Aggregate results as they complete (streaming) to minimize memory usage
5. Return a DataFrame with all predictions
6. Generate an interactive HTML report (if using batch_inference_with_report)
"""

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import torch
from batch_inference_report import batch_image, generate_batch_inference_report, report_env
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

import flyte
import flyte.io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
) -> flyte.io.DataFrame:
    """
    Process a batch of images and return predictions as a DataFrame.

    This task is designed to be run in reusable containers, so the model
    is loaded once (via lru_cache) and reused across all invocations.

    Args:
        model_dir: Directory containing the fine-tuned model
        image_paths: List of image file paths to process
        batch_size: Batch size for processing (for GPU efficiency)

    Returns:
        DataFrame with columns: image_path, top_label, top_confidence,
        second_label, second_confidence, third_label, third_confidence, error
    """
    logger.info(f"Processing batch of {len(image_paths)} images")

    # Download model directory (cached across invocations)
    model_path = await model_dir.download()

    # Load model (cached with lru_cache)
    model, processor, id2label = load_model(str(model_path))

    # Lists to build DataFrame columns
    df_image_paths = []
    df_top_labels = []
    df_top_confidences = []
    df_second_labels = []
    df_second_confidences = []
    df_third_labels = []
    df_third_confidences = []
    df_errors = []

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
                # Add error row
                df_image_paths.append(img_path)
                df_top_labels.append(None)
                df_top_confidences.append(None)
                df_second_labels.append(None)
                df_second_confidences.append(None)
                df_third_labels.append(None)
                df_third_confidences.append(None)
                df_errors.append(str(e))

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

        # Convert to DataFrame rows
        for idx, (img_path, prob_vector) in enumerate(zip(valid_paths, probs)):
            # Get top 3 predictions
            top_k = min(3, len(id2label))
            top_probs, top_indices = torch.topk(prob_vector, top_k)

            # Convert to lists
            probs_list = top_probs.cpu().tolist()
            labels_list = [id2label[idx.item()] for idx in top_indices.cpu()]

            # Add to DataFrame columns
            df_image_paths.append(img_path)
            df_top_labels.append(labels_list[0] if len(labels_list) > 0 else None)
            df_top_confidences.append(probs_list[0] if len(probs_list) > 0 else None)
            df_second_labels.append(labels_list[1] if len(labels_list) > 1 else None)
            df_second_confidences.append(probs_list[1] if len(probs_list) > 1 else None)
            df_third_labels.append(labels_list[2] if len(labels_list) > 2 else None)
            df_third_confidences.append(probs_list[2] if len(probs_list) > 2 else None)
            df_errors.append(None)

        logger.info(f"Processed {len(images)} images in mini-batch")

    # Create PyArrow table
    table = pa.table(
        {
            "image_path": df_image_paths,
            "top_label": df_top_labels,
            "top_confidence": df_top_confidences,
            "second_label": df_second_labels,
            "second_confidence": df_second_confidences,
            "third_label": df_third_labels,
            "third_confidence": df_third_confidences,
            "error": df_errors,
        }
    )

    logger.info(f"Created DataFrame with {len(df_image_paths)} predictions")

    return flyte.io.DataFrame.from_df(table)


@driver_env.task(cache="auto")
async def batch_inference_pipeline(
    model_dir: flyte.io.Dir,
    images_dir: flyte.io.Dir,
    chunk_size: int = 100,
    batch_size: int = 32,
    aggregation_batch_size: int = 10,
) -> flyte.io.DataFrame:
    """
    Run batch inference on all images in a directory using streaming aggregation.

    This pipeline:
    1. Discovers all images in the directory
    2. Partitions them into chunks
    3. Processes each chunk in parallel using reusable containers
    4. Aggregates results as they complete (streaming) to minimize memory usage
    5. Returns a combined DataFrame with all predictions

    Args:
        model_dir: Directory containing the fine-tuned model
        images_dir: Directory containing images to process
        chunk_size: Number of images per chunk (affects parallelism)
        batch_size: Mini-batch size for GPU processing
        aggregation_batch_size: Number of DataFrames to combine at once

    Returns:
        DataFrame with all inference results
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

    # Process all chunks in parallel with streaming aggregation
    tasks = []
    for idx, chunk in enumerate(chunks):
        with flyte.group(f"chunk-{idx}"):
            task = asyncio.create_task(process_image_batch(model_dir, chunk, batch_size))
            tasks.append(task)

    logger.info(f"Processing {len(tasks)} chunks in parallel with streaming aggregation...")

    # Use streaming aggregation to keep memory footprint low
    accumulated_dfs = []
    final_batches = []
    completed_count = 0

    # Process results as they complete
    for task in asyncio.as_completed(tasks):
        result_df = await task
        accumulated_dfs.append(result_df)
        completed_count += 1

        logger.info(f"Completed {completed_count}/{len(tasks)} chunks")

        # When batch is full, combine and clear to reduce memory
        if len(accumulated_dfs) >= aggregation_batch_size:
            logger.info(f"Combining {len(accumulated_dfs)} DataFrames...")
            combined = await combine_dataframes(accumulated_dfs)
            final_batches.append(combined)
            accumulated_dfs.clear()

    # Handle remaining DataFrames
    if accumulated_dfs:
        logger.info(f"Combining final {len(accumulated_dfs)} DataFrames...")
        combined = await combine_dataframes(accumulated_dfs)
        final_batches.append(combined)

    # Final aggregation of all batches
    logger.info(f"Final aggregation of {len(final_batches)} batches...")
    final_df = await combine_dataframes(final_batches)

    logger.info("Batch inference pipeline completed successfully")
    return final_df


async def combine_dataframes(dfs: List[flyte.io.DataFrame]) -> flyte.io.DataFrame:
    """
    Efficiently combine multiple Flyte DataFrames into one using PyArrow.

    Args:
        dfs: List of Flyte DataFrames to combine

    Returns:
        Combined Flyte DataFrame
    """
    if not dfs:
        # Return empty DataFrame with correct schema
        empty_table = pa.table(
            {
                "image_path": pa.array([], type=pa.string()),
                "top_label": pa.array([], type=pa.string()),
                "top_confidence": pa.array([], type=pa.float64()),
                "second_label": pa.array([], type=pa.string()),
                "second_confidence": pa.array([], type=pa.float64()),
                "third_label": pa.array([], type=pa.string()),
                "third_confidence": pa.array([], type=pa.float64()),
                "error": pa.array([], type=pa.string()),
            }
        )
        return flyte.io.DataFrame.from_df(empty_table)

    if len(dfs) == 1:
        return dfs[0]

    # Load all tables and concatenate using PyArrow
    tables = [await df.open(pa.Table).all() for df in dfs]
    combined_table = pa.concat_tables(tables)

    logger.info(f"Combined {len(dfs)} DataFrames into table with {combined_table.num_rows} rows")

    return flyte.io.DataFrame.from_df(combined_table)


@driver_env.task(cache="auto")
async def create_sample_images(num_images: int = 50) -> flyte.io.Dir:
    """
    Create sample images for testing the batch inference pipeline.

    Generates random colored images with different patterns to simulate a real dataset.

    Args:
        num_images: Number of sample images to create

    Returns:
        Directory containing the generated sample images
    """
    import random
    import tempfile

    from PIL import Image, ImageDraw

    logger.info(f"Creating {num_images} sample images...")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    sample_dir = Path(temp_dir) / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Generate random images
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    patterns = ["solid", "gradient", "circles", "stripes"]

    for i in range(num_images):
        # Create a 224x224 RGB image (standard for image classification)
        img = Image.new("RGB", (224, 224), color="white")
        draw = ImageDraw.Draw(img)

        # Choose random color and pattern
        color = random.choice(colors)
        pattern = random.choice(patterns)

        # Draw pattern
        if pattern == "solid":
            draw.rectangle([(0, 0), (224, 224)], fill=color)
        elif pattern == "gradient":
            for y in range(224):
                intensity = int(255 * (y / 224))
                draw.line([(0, y), (224, y)], fill=(intensity, 0, 255 - intensity))
        elif pattern == "circles":
            for _ in range(10):
                x, y = random.randint(0, 224), random.randint(0, 224)
                r = random.randint(10, 50)
                draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)
        elif pattern == "stripes":
            for x in range(0, 224, 20):
                draw.rectangle([(x, 0), (x + 10, 224)], fill=color)

        # Save image
        img_path = sample_dir / f"sample_{i:04d}.jpg"
        img.save(img_path)

    logger.info(f"Created {num_images} sample images in {sample_dir}")

    # Upload directory
    return await flyte.io.Dir.from_local(str(sample_dir))


driver_with_report_env = driver_env.clone_with("image-classification-report", depends_on=[driver_env, report_env])


@driver_with_report_env.task(cache="auto")
async def batch_inference_with_report(
    model_dir: flyte.io.Dir,
    images_dir: flyte.io.Dir,
    chunk_size: int = 100,
    batch_size: int = 32,
    aggregation_batch_size: int = 10,
) -> flyte.io.DataFrame:
    """
    Main workflow: Run batch inference and generate an interactive report.

    This workflow:
    1. Runs batch inference on all images (returns DataFrame)
    2. Generates an interactive HTML report with visualizations
    3. Returns the full DataFrame for downstream processing

    Args:
        model_dir: Directory containing the fine-tuned model
        images_dir: Directory containing images to process
        chunk_size: Number of images per chunk (affects parallelism)
        batch_size: Mini-batch size for GPU processing
        aggregation_batch_size: Number of DataFrames to combine at once

    Returns:
        DataFrame with all inference results
    """
    logger.info("Starting batch inference workflow with report generation")

    # Run batch inference
    results_df = await batch_inference_pipeline(
        model_dir=model_dir,
        images_dir=images_dir,
        chunk_size=chunk_size,
        batch_size=batch_size,
        aggregation_batch_size=aggregation_batch_size,
    )

    # Generate interactive report
    await generate_batch_inference_report(results_df)

    logger.info("Batch inference workflow completed successfully")
    return results_df


@driver_with_report_env.task(cache="auto")
async def batch_inference_demo(
    model_dir: flyte.io.Dir,
    num_sample_images: int = 50,
    chunk_size: int = 20,
) -> flyte.io.DataFrame:
    """
    Demo workflow: Create sample images and run batch inference with report.

    This is a complete end-to-end demonstration that:
    1. Creates sample images for testing
    2. Runs batch inference on those images
    3. Generates an interactive report
    4. Returns the full DataFrame with results

    Args:
        model_dir: Directory containing the fine-tuned model
        num_sample_images: Number of sample images to create
        chunk_size: Number of images per chunk (affects parallelism)

    Returns:
        DataFrame with all inference results
    """
    logger.info(f"Starting batch inference demo with {num_sample_images} sample images")

    # Create sample images
    images_dir = await create_sample_images(num_images=num_sample_images)

    # Run batch inference with report
    results_df = await batch_inference_with_report(
        model_dir=model_dir,
        images_dir=images_dir,
        chunk_size=chunk_size,
        batch_size=10,
        aggregation_batch_size=5,
    )

    logger.info("Batch inference demo completed successfully")
    return results_df


if __name__ == "__main__":
    import flyte.models
    import flyte.remote

    flyte.init_from_config(
        root_dir=Path(__file__).parent,
    )

    training_runs = flyte.remote.Run.listall(
        in_phase=(flyte.models.ActionPhase.SUCCEEDED,),
        task_name="image_finetune_training.finetune_image_model",
        sort_by=("created_at", "desc"),
        limit=1,
    )

    training_run = next(training_runs)
    outputs = training_run.outputs()

    r = flyte.run(batch_inference_demo, outputs[0])
    print(r.url)

    print(
        """
Image Classification - Batch Inference

This module provides efficient batch inference with interactive reporting:

WORKFLOWS:
----------

1. batch_inference_pipeline - Returns DataFrame only (no report)
   Returns: flyte.io.DataFrame with predictions

2. batch_inference_with_report - Runs inference + generates interactive report
   Returns: flyte.io.DataFrame with predictions + HTML report

3. batch_inference_demo - End-to-end demo with sample images
   Creates sample images, runs inference, and generates report
   Returns: flyte.io.DataFrame with predictions + HTML report

USAGE:
------

# Run end-to-end demo with sample images (for testing):
flyte run batch_inference.py batch_inference_demo \\
    --model_dir=<path_or_remote_ref> \\
    --num_sample_images=50 \\
    --chunk_size=20

# Run with automatic report generation (recommended):
flyte run batch_inference.py batch_inference_with_report \\
    --model_dir=<path_or_remote_ref> \\
    --images_dir=<path_or_remote_ref> \\
    --chunk_size=100

# Run inference only (no report):
flyte run batch_inference.py batch_inference_pipeline \\
    --model_dir=<path_or_remote_ref> \\
    --images_dir=<path_or_remote_ref> \\
    --chunk_size=100

FEATURES:
---------
✓ Efficient DataFrame-based processing
✓ Streaming aggregation for low memory footprint
✓ Parallel processing with reusable containers
✓ Interactive HTML reports with charts
✓ Label distribution and confidence analysis
✓ Detailed results table
✓ End-to-end demo with sample data generation

OUTPUT:
-------
- DataFrame with columns: image_path, top_label, top_confidence,
  second_label, second_confidence, third_label, third_confidence, error
- Interactive HTML report (when using batch_inference_with_report or demo)

FILES:
------
- batch_inference.py: Main inference pipeline
- batch_inference_report.py: Interactive report generation
"""
    )
