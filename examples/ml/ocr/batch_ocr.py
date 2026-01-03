"""
Batch Document OCR with Qwen Vision-Language Model

Simple pipeline for extracting text from document images using Qwen2.5-VL-3B.
Demonstrates reusable GPU workers, parallel processing, and caching.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

import flyte
import flyte.io
import pyarrow as pa

from ocr_processor import get_ocr_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model ID
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Base image with OCR dependencies
ocr_image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=Path("pyproject.toml"),
    extra_args="--extra ocr"
)

# GPU worker environment - runs OCR tasks with model caching
gpu_worker = flyte.TaskEnvironment(
    name="qwen_ocr_worker",
    description="GPU worker for Qwen VL OCR processing with model caching",
    image=ocr_image,
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
        gpu=1,
    ),
    reusable=flyte.ReusePolicy(
        replicas=4,
        concurrency=1,
        idle_ttl=600,
    ),
    secrets="HF_HUB_TOKEN",
    cache=flyte.Cache("auto", "1.0"),
)

# Driver environment - orchestrates the workflow
driver = flyte.TaskEnvironment(
    name="ocr_driver",
    description="Driver for orchestrating batch OCR workflows",
    image=ocr_image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    depends_on=[gpu_worker],
    cache=flyte.Cache("auto", "1.0"),
)


@gpu_worker.task(retries=3)
async def process_batch(image_files: list[flyte.io.File]) -> list[dict]:
    """
    Process a batch of images using Qwen OCR.

    Args:
        image_files: List of image files to process

    Returns:
        List of results with document_id, extracted_text, success, token_count
    """
    logger.info(f"Processing batch of {len(image_files)} images")

    # Get cached OCR processor (loads model once per worker)
    processor = await get_ocr_processor(MODEL_ID)

    # Process all images in parallel
    tasks = [processor.process_document(image_file) for image_file in image_files]
    results = await asyncio.gather(*tasks)

    logger.info(f"Batch complete: {sum(r['success'] for r in results)}/{len(results)} successful")
    return results


@driver.task
async def load_dataset(sample_size: int = 100) -> flyte.io.Dir:
    """
    Load DocumentVQA dataset from HuggingFace.

    Args:
        sample_size: Number of samples to load (0 for all)

    Returns:
        Directory containing document images
    """
    from datasets import load_dataset

    logger.info(f"Loading DocumentVQA dataset (sample_size={sample_size})")

    dataset = load_dataset("HuggingFaceM4/DocumentVQA", split="train", streaming=True)

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Download images
    count = 0
    for idx, sample in enumerate(dataset):
        if sample_size > 0 and count >= sample_size:
            break

        image = sample.get("image")
        if image is None:
            continue

        image_path = images_dir / f"doc_{idx:05d}.png"
        image.save(image_path)
        count += 1

        if count % 10 == 0:
            logger.info(f"Downloaded {count} images")

    logger.info(f"Downloaded {count} images to {images_dir}")
    return await flyte.io.Dir.from_local(images_dir)


@driver.task
async def batch_ocr(images_dir: flyte.io.Dir, chunk_size: int = 20) -> flyte.io.DataFrame:
    """
    Run batch OCR on all images in a directory.

    Args:
        images_dir: Directory containing images
        chunk_size: Number of images per chunk for parallel processing

    Returns:
        DataFrame with columns: document_id, extracted_text, success, token_count
    """
    logger.info("Starting batch OCR")

    # List all image files
    all_files = await images_dir.list_files()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"}
    image_files = [f for f in all_files if any(f.path.lower().endswith(ext) for ext in image_extensions)]

    logger.info(f"Found {len(image_files)} images")

    if not image_files:
        raise ValueError(f"No images found in {images_dir.path}")

    # Split into chunks
    chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]
    logger.info(f"Split into {len(chunks)} chunks")

    # Process chunks in parallel
    tasks = [process_batch(chunk) for chunk in chunks]
    all_results = await asyncio.gather(*tasks)

    # Flatten results
    results = [item for sublist in all_results for item in sublist]

    # Create DataFrame
    table = pa.table({
        "document_id": [r["document_id"] for r in results],
        "extracted_text": [r["extracted_text"] for r in results],
        "success": [r["success"] for r in results],
        "token_count": [r["token_count"] for r in results],
    })

    logger.info(f"OCR complete: {table.num_rows} documents processed")
    return flyte.io.DataFrame.from_df(table)


@driver.task
async def run_ocr_pipeline(sample_size: int = 10, chunk_size: int = 20) -> flyte.io.DataFrame:
    """
    Complete OCR pipeline: load dataset and extract text.

    Args:
        sample_size: Number of documents to process (0 for all)
        chunk_size: Number of images per processing chunk

    Returns:
        DataFrame with OCR results
    """
    logger.info(f"Starting OCR pipeline (sample_size={sample_size})")

    # Load dataset
    images_dir = await load_dataset(sample_size=sample_size)

    # Run OCR
    results = await batch_ocr(images_dir=images_dir, chunk_size=chunk_size)

    logger.info("Pipeline complete")
    return results


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)

    print("""
Batch Document OCR with Qwen Vision-Language Model
===================================================

USAGE:
------

# Process 10 sample documents:
flyte run batch_ocr.py run_ocr_pipeline --sample_size=10

# Process 100 documents:
flyte run batch_ocr.py run_ocr_pipeline --sample_size=100

# Process on existing image directory:
flyte run batch_ocr.py batch_ocr --images_dir=/path/to/images

FEATURES:
---------
✓ Qwen2.5-VL-3B for OCR
✓ Reusable GPU workers with model caching
✓ Parallel batch processing
✓ DocumentVQA dataset integration
✓ Content-based caching

OUTPUT:
-------
DataFrame with:
- document_id: Image identifier
- extracted_text: OCR output
- success: Boolean success flag
- token_count: Number of tokens generated
""")
