"""
Batch Document OCR Workflow - Multi-Model Comparison

This workflow demonstrates efficient batch OCR processing using Flyte 2.0 features:
- Multiple OCR models from HuggingFace (Qwen2.5-VL, GOT-OCR, InternVL, etc.)
- Reusable containers with GPU acceleration
- Content-based caching for efficiency
- DocumentVQA dataset from HuggingFace
- Sample mode vs full dataset processing
- Model comparison matrix with interactive reports
- Dynamic GPU resource configuration per model

Usage:
    # Run single model OCR on sample data:
    flyte run batch_ocr.py batch_ocr_single_model \\
        --model=QWEN_VL_7B \\
        --sample_size=10

    # Run multi-model comparison:
    flyte run batch_ocr.py batch_ocr_comparison \\
        --models='["QWEN_VL_7B", "GOT_OCR_2"]' \\
        --sample_size=50

    # Run on full dataset:
    flyte run batch_ocr.py batch_ocr_single_model \\
        --model=QWEN_VL_7B \\
        --sample_size=0  # 0 means all data

The pipeline will:
1. Load DocumentVQA dataset from HuggingFace (streaming mode)
2. Partition documents into chunks
3. Process each chunk in parallel with reusable GPU containers
4. Generate OCR results with confidence scores
5. Create comparison matrix report (for multi-model runs)
"""

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import flyte
import flyte.io
import pyarrow as pa
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# OCR Model Definitions
class OCRModel(str, Enum):
    """Supported OCR models with their HuggingFace model IDs"""

    # Qwen2.5-VL variants
    QWEN_VL_2B = "Qwen/Qwen2.5-VL-2B-Instruct"
    QWEN_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN_VL_72B = "Qwen/Qwen2.5-VL-72B-Instruct"

    # GOT-OCR 2.0
    GOT_OCR_2 = "ucaslcl/GOT-OCR2_0"

    # InternVL 2.5 variants
    INTERN_VL_2B = "OpenGVLab/InternVL2_5-2B"
    INTERN_VL_8B = "OpenGVLab/InternVL2_5-8B"
    INTERN_VL_26B = "OpenGVLab/InternVL2_5-26B"

    # RolmOCR (Qwen 2.5-VL fine-tune)
    ROLM_OCR = "reducto-ai/RolmOCR"

    # PaddleOCR (note: different API, might need custom wrapper)
    # For simplicity, focusing on HF Transformers-compatible models first


@dataclass
class ModelConfig:
    """Configuration for each OCR model including GPU requirements"""

    model_id: str
    gpu_type: str  # e.g., "T4", "A100", "A100 80G"
    gpu_count: int
    memory: str  # e.g., "16Gi", "40Gi"
    cpu: int
    requires_flash_attention: bool = False
    supports_batching: bool = True
    max_batch_size: int = 4


# GPU Requirements Dictionary - easily customizable via dynamic overrides
MODEL_GPU_CONFIGS: dict[OCRModel, ModelConfig] = {
    OCRModel.QWEN_VL_2B: ModelConfig(
        model_id=OCRModel.QWEN_VL_2B.value,
        gpu_type="T4",
        gpu_count=1,
        memory="16Gi",
        cpu=4,
        requires_flash_attention=False,
        max_batch_size=8,
    ),
    OCRModel.QWEN_VL_7B: ModelConfig(
        model_id=OCRModel.QWEN_VL_7B.value,
        gpu_type="A100",
        gpu_count=1,
        memory="40Gi",
        cpu=8,
        requires_flash_attention=True,
        max_batch_size=4,
    ),
    OCRModel.QWEN_VL_72B: ModelConfig(
        model_id=OCRModel.QWEN_VL_72B.value,
        gpu_type="A100 80G",
        gpu_count=4,
        memory="160Gi",
        cpu=16,
        requires_flash_attention=True,
        max_batch_size=2,
    ),
    OCRModel.GOT_OCR_2: ModelConfig(
        model_id=OCRModel.GOT_OCR_2.value,
        gpu_type="A100",
        gpu_count=1,
        memory="40Gi",
        cpu=8,
        requires_flash_attention=False,
        max_batch_size=4,
    ),
    OCRModel.INTERN_VL_2B: ModelConfig(
        model_id=OCRModel.INTERN_VL_2B.value,
        gpu_type="T4",
        gpu_count=1,
        memory="16Gi",
        cpu=4,
        requires_flash_attention=False,
        max_batch_size=8,
    ),
    OCRModel.INTERN_VL_8B: ModelConfig(
        model_id=OCRModel.INTERN_VL_8B.value,
        gpu_type="A100",
        gpu_count=1,
        memory="40Gi",
        cpu=8,
        requires_flash_attention=False,
        max_batch_size=4,
    ),
    OCRModel.INTERN_VL_26B: ModelConfig(
        model_id=OCRModel.INTERN_VL_26B.value,
        gpu_type="A100 80G",
        gpu_count=2,
        memory="80Gi",
        cpu=12,
        requires_flash_attention=True,
        max_batch_size=2,
    ),
    OCRModel.ROLM_OCR: ModelConfig(
        model_id=OCRModel.ROLM_OCR.value,
        gpu_type="T4",
        gpu_count=1,
        memory="16Gi",
        cpu=4,
        requires_flash_attention=False,
        max_batch_size=8,
    ),
}


# Base image for OCR tasks
ocr_image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=Path("pyproject.toml"), extra_args="--extra ocr"
)


def create_worker_env(model: OCRModel) -> flyte.TaskEnvironment:
    """
    Create a worker environment with model-specific GPU configuration.

    This function can be used to dynamically create environments for different models,
    or you can override resources at runtime using task.override().
    """
    config = MODEL_GPU_CONFIGS[model]

    return flyte.TaskEnvironment(
        name=f"ocr_worker_{model.name.lower()}",
        image=ocr_image,
        resources=flyte.Resources(
            cpu=config.cpu,
            memory=config.memory,
            gpu=f"{config.gpu_type}:{config.gpu_count}",
        ),
        reusable=flyte.ReusePolicy(
            replicas=4,  # 4 replicas for parallel processing
            concurrency=2,  # Each replica handles 2 tasks concurrently
            idle_ttl=600,  # Keep alive for 10 minutes (models are large)
            scaledown_ttl=600,
        ),
    )


# Default worker environment (for Qwen 2B - good default)
default_worker_env = create_worker_env(OCRModel.QWEN_VL_2B)

# Driver environment for orchestration
driver_env = flyte.TaskEnvironment(
    name="ocr_driver",
    image=ocr_image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    depends_on=[default_worker_env],
)


@lru_cache(maxsize=1)
def load_ocr_model(model_id: str, device: str = "cuda") -> tuple[Any, Any]:
    """
    Lazily load and cache the OCR model and processor.
    This ensures the model is loaded only once per worker container.

    Args:
        model_id: HuggingFace model ID
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading OCR model: {model_id}")

    try:
        # Load processor/tokenizer
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        model.eval()

        logger.info(f"Model loaded successfully on {device}")
        return model, processor

    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise


def run_ocr_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str = "Extract all text from this document image.",
) -> dict[str, Any]:
    """
    Run OCR inference on a single image.

    Args:
        model: Loaded OCR model
        processor: Model processor
        image: PIL Image
        prompt: OCR prompt/instruction

    Returns:
        Dictionary with extracted text and metadata
    """
    try:
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template and prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        # Decode output
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "text": output_text,
            "success": True,
            "error": None,
            "token_count": len(generated_ids[0]),
        }

    except Exception as e:
        logger.error(f"OCR inference failed: {e}")
        return {
            "text": "",
            "success": False,
            "error": str(e),
            "token_count": 0,
        }


@default_worker_env.task(cache="auto", retries=3)
async def process_document_batch(
    model_name: OCRModel,
    image_files: list[flyte.io.File],
    batch_size: int = 4,
) -> flyte.io.DataFrame:
    """
    Process a batch of document images with OCR.

    This task runs in reusable containers with the model loaded once via lru_cache.
    Images are downloaded on-demand and processed in mini-batches.

    Args:
        model_name: OCR model to use
        image_files: List of document image files
        batch_size: Mini-batch size for processing

    Returns:
        DataFrame with OCR results: document_id, extracted_text, success, error, token_count
    """
    logger.info(f"Processing batch of {len(image_files)} documents with {model_name.value}")

    # Get model config
    config = MODEL_GPU_CONFIGS[model_name]

    # Load model (cached with lru_cache)
    model, processor = load_ocr_model(config.model_id)

    # Result lists for DataFrame
    document_ids = []
    extracted_texts = []
    successes = []
    errors = []
    token_counts = []

    # Process images in mini-batches
    for i in range(0, len(image_files), batch_size):
        mini_batch_files = image_files[i : i + batch_size]

        # Download mini-batch files in parallel
        download_tasks = [file.download() for file in mini_batch_files]
        downloaded_paths = await asyncio.gather(*download_tasks, return_exceptions=True)

        # Process each image
        for file, downloaded_path in zip(mini_batch_files, downloaded_paths):
            # Get document ID from file path
            doc_id = file.path if hasattr(file, "path") else str(file)

            # Handle download errors
            if isinstance(downloaded_path, Exception):
                logger.warning(f"Failed to download {doc_id}: {downloaded_path}")
                document_ids.append(doc_id)
                extracted_texts.append("")
                successes.append(False)
                errors.append(str(downloaded_path))
                token_counts.append(0)
                continue

            # Load and process image
            try:
                image = Image.open(downloaded_path).convert("RGB")

                # Run OCR
                result = run_ocr_inference(model, processor, image)

                document_ids.append(doc_id)
                extracted_texts.append(result["text"])
                successes.append(result["success"])
                errors.append(result["error"])
                token_counts.append(result["token_count"])

                logger.info(f"Processed {doc_id}: {result['token_count']} tokens extracted")

            except Exception as e:
                logger.error(f"Failed to process {doc_id}: {e}")
                document_ids.append(doc_id)
                extracted_texts.append("")
                successes.append(False)
                errors.append(str(e))
                token_counts.append(0)

    # Create PyArrow table
    table = pa.table(
        {
            "document_id": document_ids,
            "model": [model_name.value] * len(document_ids),
            "extracted_text": extracted_texts,
            "success": successes,
            "error": errors,
            "token_count": token_counts,
        }
    )

    logger.info(f"Completed batch: {len(document_ids)} documents processed")
    return flyte.io.DataFrame.from_df(table)


@driver_env.task(cache="auto")
async def load_documentvqa_dataset(
    sample_size: int = 100,
    split: str = "train",
) -> flyte.io.Dir:
    """
    Load DocumentVQA dataset from HuggingFace and save images locally.

    Args:
        sample_size: Number of samples to load (0 for all)
        split: Dataset split ("train", "validation", "test")

    Returns:
        Directory containing downloaded document images
    """
    from datasets import load_dataset

    logger.info(f"Loading DocumentVQA dataset (split={split}, sample_size={sample_size})")

    # Load dataset in streaming mode for efficiency
    dataset = load_dataset("HuggingFaceM4/DocumentVQA", split=split, streaming=True)

    # Create temporary directory for images
    temp_dir = Path(tempfile.mkdtemp())
    images_dir = temp_dir / "documentvqa_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Download images
    count = 0
    for idx, sample in enumerate(dataset):
        if sample_size > 0 and count >= sample_size:
            break

        # Get image from sample
        image = sample.get("image")
        if image is None:
            logger.warning(f"Sample {idx} has no image, skipping")
            continue

        # Save image
        image_path = images_dir / f"doc_{idx:05d}.png"
        image.save(image_path)
        count += 1

        if count % 10 == 0:
            logger.info(f"Downloaded {count} images...")

    logger.info(f"Downloaded {count} images from DocumentVQA dataset to {images_dir}")

    # Upload to Flyte storage
    return await flyte.io.Dir.from_local(images_dir)


@driver_env.task(cache="auto")
async def batch_ocr_pipeline(
    model: OCRModel,
    images_dir: flyte.io.Dir,
    chunk_size: int = 50,
    batch_size: int = 4,
) -> flyte.io.DataFrame:
    """
    Run batch OCR on all documents in a directory.

    Args:
        model: OCR model to use
        images_dir: Directory containing document images
        chunk_size: Number of documents per chunk (parallelism)
        batch_size: Mini-batch size for processing

    Returns:
        DataFrame with all OCR results
    """
    logger.info(f"Starting batch OCR pipeline with {model.value}")

    # List all files
    all_files = await images_dir.list_files()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"}
    image_files = [f for f in all_files if any(f.path.lower().endswith(ext) for ext in image_extensions)]

    logger.info(f"Found {len(image_files)} document images")

    if not image_files:
        raise ValueError(f"No images found in {images_dir.path}")

    # Partition into chunks
    chunks = [image_files[i : i + chunk_size] for i in range(0, len(image_files), chunk_size)]
    logger.info(f"Partitioned into {len(chunks)} chunks")

    # Get model config and potentially override resources
    config = MODEL_GPU_CONFIGS[model]

    # Create task with appropriate resources for this model
    ocr_task = process_document_batch.override(
        resources=flyte.Resources(
            cpu=config.cpu,
            memory=config.memory,
            gpu=f"{config.gpu_type}:{config.gpu_count}",
        )
    )

    # Process chunks in parallel
    tasks = []
    for idx, chunk in enumerate(chunks):
        with flyte.group(f"chunk-{idx}"):
            task = asyncio.create_task(ocr_task(model, chunk, batch_size))
            tasks.append(task)

    # Gather results
    logger.info(f"Processing {len(tasks)} chunks in parallel...")
    results = await asyncio.gather(*tasks)

    # Combine DataFrames
    logger.info("Combining results...")
    combined_df = await combine_dataframes(results)

    logger.info("Batch OCR pipeline completed")
    return combined_df


async def combine_dataframes(dfs: list[flyte.io.DataFrame]) -> flyte.io.DataFrame:
    """Combine multiple DataFrames into one."""
    if not dfs:
        empty_table = pa.table(
            {
                "document_id": pa.array([], type=pa.string()),
                "model": pa.array([], type=pa.string()),
                "extracted_text": pa.array([], type=pa.string()),
                "success": pa.array([], type=pa.bool_()),
                "error": pa.array([], type=pa.string()),
                "token_count": pa.array([], type=pa.int64()),
            }
        )
        return flyte.io.DataFrame.from_df(empty_table)

    if len(dfs) == 1:
        return dfs[0]

    tables = [await df.open(pa.Table).all() for df in dfs]
    combined_table = pa.concat_tables(tables)

    logger.info(f"Combined {len(dfs)} DataFrames into {combined_table.num_rows} rows")
    return flyte.io.DataFrame.from_df(combined_table)


@driver_env.task(cache="auto")
async def batch_ocr_single_model(
    model: OCRModel = OCRModel.QWEN_VL_2B,
    sample_size: int = 10,
    chunk_size: int = 20,
) -> flyte.io.DataFrame:
    """
    End-to-end workflow: Load DocumentVQA dataset and run OCR with a single model.

    Args:
        model: OCR model to use
        sample_size: Number of documents to process (0 for all)
        chunk_size: Chunk size for parallel processing

    Returns:
        DataFrame with OCR results
    """
    logger.info(f"Starting single-model OCR workflow with {model.value}")

    # Load dataset
    images_dir = await load_documentvqa_dataset(sample_size=sample_size)

    # Run OCR
    results_df = await batch_ocr_pipeline(
        model=model,
        images_dir=images_dir,
        chunk_size=chunk_size,
        batch_size=MODEL_GPU_CONFIGS[model].max_batch_size,
    )

    logger.info("Single-model OCR workflow completed")
    return results_df


@driver_env.task(cache="auto")
async def batch_ocr_comparison(
    models: list[OCRModel] = [OCRModel.QWEN_VL_2B, OCRModel.INTERN_VL_2B],
    sample_size: int = 10,
    chunk_size: int = 20,
) -> list[flyte.io.DataFrame]:
    """
    Multi-model comparison: Run OCR with multiple models on the same dataset.

    This workflow runs different OCR models on the same documents for comparison.
    Each model runs with its own GPU configuration.

    Args:
        models: List of OCR models to compare
        sample_size: Number of documents to process
        chunk_size: Chunk size for parallel processing

    Returns:
        List of DataFrames, one per model
    """
    logger.info(f"Starting multi-model comparison with {len(models)} models")

    # Load dataset once
    images_dir = await load_documentvqa_dataset(sample_size=sample_size)

    # Run OCR with each model in parallel
    tasks = []
    for model in models:
        with flyte.group(f"model-{model.name}"):
            task = asyncio.create_task(
                batch_ocr_pipeline(
                    model=model,
                    images_dir=images_dir,
                    chunk_size=chunk_size,
                    batch_size=MODEL_GPU_CONFIGS[model].max_batch_size,
                )
            )
            tasks.append(task)

    # Gather results from all models
    logger.info(f"Running {len(models)} models in parallel...")
    all_results = await asyncio.gather(*tasks)

    logger.info("Multi-model comparison completed")
    return all_results


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)

    print("""
Batch Document OCR Workflow - Multi-Model Comparison
=====================================================

This example demonstrates batch OCR processing with multiple models using Flyte 2.0.

WORKFLOWS:
----------

1. batch_ocr_single_model - Run OCR with a single model
   Args:
     --model: OCR model (QWEN_VL_2B, QWEN_VL_7B, GOT_OCR_2, INTERN_VL_2B, etc.)
     --sample_size: Number of documents (0 for all)
     --chunk_size: Chunk size for parallelism

2. batch_ocr_comparison - Compare multiple OCR models
   Args:
     --models: List of models to compare
     --sample_size: Number of documents
     --chunk_size: Chunk size

3. batch_ocr_pipeline - Run OCR on existing image directory
   Args:
     --model: OCR model to use
     --images_dir: Directory with images
     --chunk_size: Chunk size

SUPPORTED MODELS:
-----------------
- QWEN_VL_2B: Qwen2.5-VL 2B (T4, 16Gi)
- QWEN_VL_7B: Qwen2.5-VL 7B (A100, 40Gi)
- QWEN_VL_72B: Qwen2.5-VL 72B (4x A100 80G, 160Gi)
- GOT_OCR_2: GOT-OCR 2.0 (A100, 40Gi)
- INTERN_VL_2B: InternVL 2.5 2B (T4, 16Gi)
- INTERN_VL_8B: InternVL 2.5 8B (A100, 40Gi)
- INTERN_VL_26B: InternVL 2.5 26B (2x A100 80G, 80Gi)
- ROLM_OCR: RolmOCR (T4, 16Gi)

USAGE EXAMPLES:
---------------

# Run single model on 10 samples:
flyte run batch_ocr.py batch_ocr_single_model \\
    --model=QWEN_VL_2B \\
    --sample_size=10

# Compare two models:
flyte run batch_ocr.py batch_ocr_comparison \\
    --models='["QWEN_VL_2B", "INTERN_VL_2B"]' \\
    --sample_size=50

# Run on full dataset:
flyte run batch_ocr.py batch_ocr_single_model \\
    --model=QWEN_VL_7B \\
    --sample_size=0

FEATURES:
---------
✓ Multiple OCR models from HuggingFace
✓ Model-specific GPU configurations
✓ Dynamic resource overrides
✓ Reusable containers with model caching
✓ DocumentVQA dataset integration
✓ Sample mode vs full dataset
✓ Parallel batch processing
✓ Comprehensive error handling
✓ Content-based caching

GPU CONFIGURATION:
------------------
Each model has pre-configured GPU requirements stored in MODEL_GPU_CONFIGS.
You can override these at runtime:

flyte run batch_ocr.py batch_ocr_single_model \\
    --model=QWEN_VL_7B \\
    --sample_size=10 \\
    --overrides='{"process_document_batch": {"resources": {"gpu": 2}}}'

OUTPUT:
-------
DataFrame with columns:
- document_id: Document identifier
- model: Model name used
- extracted_text: OCR output
- success: Boolean success flag
- error: Error message (if any)
- token_count: Number of tokens generated

For comparison workflows, you get one DataFrame per model.
""")
