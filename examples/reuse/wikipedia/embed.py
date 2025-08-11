# Reimplementation of https://www.union.ai/docs/v1/selfmanaged/tutorials/parallel-processing-and-job-scheduling/wikipedia-embeddings/
# Wikipedia Embeddings Example, using flyte 2.0 API
# Based on: https://github.com/unionai/unionai-examples/blob/main/tutorials/wikipedia_embeddings_on_actor/wikipedia_embeddings_on_actor.py

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import torch

import flyte
import flyte.io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an image with required dependencies for embeddings
embedding_image = flyte.Image.from_debian_base().with_pip_packages(
    "datasets", "sentence-transformers", "pandas", "torch", "requests>=2.29.0", "transformers", "huggingface-hub"
)

# Create a reusable environment for embedding tasks
embedding_env = flyte.TaskEnvironment(
    name="wikipedia-embedder-env",
    resources=flyte.Resources(memory="12Gi", cpu=5, gpu=1),
    reusable=flyte.ReusePolicy(
        replicas=20,
        idle_ttl=900,
    ),
    image=embedding_image,
)

# Environment for model downloading
download_env = flyte.TaskEnvironment(
    name="model-downloader",
    resources=flyte.Resources(memory="5Gi", cpu=2),
    image=embedding_image,
)


@download_env.task
async def download_model(embedding_model: str) -> flyte.io.Dir:
    """Download and cache the embedding model from HuggingFace."""
    from huggingface_hub import snapshot_download

    ctx = flyte.current_context()
    working_dir = Path(ctx.working_directory)
    cached_model_dir = working_dir / "cached_model"

    logger.info(f"Downloading model: {embedding_model}")
    snapshot_download(embedding_model, local_dir=cached_model_dir)
    logger.info("Model download complete")

    return str(cached_model_dir)


async def _load_model(model_path: str) -> str:
    """Helper function to load the model path."""
    # This is a placeholder for any additional logic needed to load the model
    return model_path


@embedding_env.task
async def encode(df: flyte.io.DataFrame, batch_size: int, model_path: str) -> torch.Tensor:
    """Encode text data using the cached SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    local_path = Path("/tmp/embedding-model")

    # Only copy model if not already cached locally
    if not local_path.exists():
        logger.info("Copying model to local cache...")
        shutil.copytree(src=model_path, dst=str(local_path))
        logger.info("Model copied to local cache")
    else:
        logger.info("Using locally cached model")

    # Load the model (fast after first load in reusable container)
    encoder = SentenceTransformer(str(local_path))
    encoder.max_seq_length = 256

    logger.info(f"Encoding {len(df)} texts on device: {encoder.device}")

    embeddings = encoder.encode(
        df["text"],
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


# Main orchestration environment (non-reusable)
main_env = embedding_env.clone_with(
    name="wikipedia-main",
    reusable=None,
    depends_on=[embedding_env, download_env],
    resources=flyte.Resources(memory="4Gi", cpu=3, gpu=0),
)


@main_env.task
async def main(
    dataset_name: str = "wikipedia",
    dataset_version: str = "20220301.en",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 128,
    num_proc: int = 4,
) -> Dict[str, Any]:
    """
    Main workflow that processes Wikipedia dataset and creates embeddings.
    Downloads model in parallel while streaming through dataset partitions.
    """
    import pathlib

    from datasets import DownloadConfig, load_dataset_builder

    logger.info("Starting Wikipedia embedding workflow")
    logger.info(f"Dataset: {dataset_name} {dataset_version}")
    logger.info(f"Model: {embedding_model}")

    # Start model download in parallel
    model_download_task = asyncio.create_task(download_model(embedding_model))
    logger.info("Started model download in background...")

    # Stream through dataset partitions
    async def partition_generator() -> AsyncGenerator[flyte.io.DataFrame, None]:
        """Smart partition generator - only downloads metadata, yields partitions as needed."""
        logger.info(f"Initializing dataset builder for {dataset_name} {dataset_version}")
        dsb = load_dataset_builder(dataset_name, dataset_version, cache_dir="/tmp/hfds", trust_remote_code=True)

        logger.info("Downloading and preparing dataset (smart partitioning)...")
        dsb.download_and_prepare(
            file_format="parquet",
            download_config=DownloadConfig(disable_tqdm=True, num_proc=num_proc),
        )

        logger.info("Dataset prepared, streaming partitions...")
        p = pathlib.Path(dsb.cache_dir)

        partition_count = 0
        for f in p.iterdir():
            if "parquet" in f.name:
                logger.info(f"Yielding partition {partition_count}: {f.name}")
                yield flyte.io.DataFrame(uri=str(f))
                partition_count += 1

        logger.info(f"Finished streaming {partition_count} partitions")

    # Process partitions as they become available
    logger.info("Starting partition processing...")
    embedding_tasks = []
    partition_count = 0

    # Wait for model download to complete
    model_path = await model_download_task
    logger.info("Model download completed, starting encoding...")

    # Process each partition
    async for partition in partition_generator():
        # Load partition data
        df = partition.to_pandas()

        if len(df) > 0:
            logger.info(f"Processing partition {partition_count} with {len(df)} rows")

            # Create encoding task for this partition
            encoding_task = encode(df, batch_size, model_path)
            embedding_tasks.append(encoding_task)
            partition_count += 1
        else:
            logger.warning(f"Skipping empty partition {partition_count}")

    logger.info(f"Created {len(embedding_tasks)} encoding tasks, waiting for completion...")

    # Wait for all encoding tasks to complete
    embeddings = await asyncio.gather(*embedding_tasks)

    # Compute final statistics
    total_embeddings = sum(tensor.shape[0] for tensor in embeddings if tensor.numel() > 0)
    embedding_dim = embeddings[0].shape[1] if embeddings and embeddings[0].numel() > 0 else 0

    logger.info(f"Workflow complete: {total_embeddings} embeddings generated")

    return {
        "partitions_processed": len(embeddings),
        "total_embeddings_generated": total_embeddings,
        "embedding_dimension": embedding_dim,
        "model_used": embedding_model,
        "batch_size": batch_size,
        "dataset": f"{dataset_name}:{dataset_version}",
    }


if __name__ == "__main__":
    # Example usage
    flyte.init_from_config("../../../config.yaml")

    print("Starting Wikipedia embeddings workflow...")
    print("Using smart partition streaming for efficient processing")

    run = flyte.run(
        main,
        dataset_name="wikipedia",
        dataset_version="20220301.en",
        embedding_model="BAAI/bge-small-en-v1.5",
        batch_size=128,
        num_proc=4,
    )
    print(f"Workflow URL: {run.url}")

    # Wait for completion and get results
    result = run.wait()

    if result:
        print("\n🎉 Results:")
        print(f"📊 Processed {result['partitions_processed']} partitions")
        print(f"🔢 Generated {result['total_embeddings_generated']} embeddings")
        print(f"📐 Embedding dimension: {result['embedding_dimension']}")
        print(f"🤖 Model: {result['model_used']}")
        print(f"📚 Dataset: {result['dataset']}")
        print(f"⚡ Batch size: {result['batch_size']}")
    else:
        print("❌ Workflow failed or returned no results")
