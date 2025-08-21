# /// script
# requires-python = "==3.12"
# dependencies = [
#    "sentence-transformers",
#    "pandas",
#    "torch",
#    "requests>=2.29.0",
#    "transformers",
#    "huggingface-hub",
#    "flyte>=2.0.0b6",
#    "pyarrow",
# ]
# ///
import asyncio
import logging

# Import our reusable data processing tracker
from pathlib import Path
from typing import AsyncGenerator, Dict

import pandas as pd
import torch
from async_lru import alru_cache
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from sentence_transformers import SentenceTransformer

import flyte
import flyte.io
from flyte.report import DataProcessingTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an image with required dependencies for embeddings
embedding_image = flyte.Image.from_uv_script(__file__, name="flyte", registry="ghcr.io/flyteorg")

# Create a reusable environment for embedding tasks
embedding_env = flyte.TaskEnvironment(
    name="wikipedia_embedder",
    resources=flyte.Resources(memory="12Gi", cpu=5, gpu=1),
    reusable=flyte.ReusePolicy(
        replicas=20,
        idle_ttl=900,
    ),
    image=embedding_image,
)

# Main orchestration environment (non-reusable)
main_env = embedding_env.clone_with(
    name="wikipedia_main",
    reusable=None,
    depends_on=[embedding_env],
    resources=flyte.Resources(memory="16Gi", cpu=3),
)


@alru_cache(1)
async def _load_model(embedding_model: str) -> SentenceTransformer:
    """Helper function to load the model path."""

    local_path = Path("/tmp/embedding-model")

    # Only copy model if not already cached locally
    if not local_path.exists():
        logger.info(f"Downloading model: {embedding_model}")
        snapshot_download(embedding_model, local_dir=str(local_path))
        logger.info("Model download complete")
    else:
        logger.info("Using locally cached model")

    # Load the model (fast after first load in reusable container)
    encoder = SentenceTransformer(str(local_path))
    encoder.max_seq_length = 256
    return encoder


@embedding_env.task
async def encode(df: pd.DataFrame, batch_size: int, model_name: str) -> torch.Tensor:
    """Encode text data using the cached SentenceTransformer model."""

    encoder = await _load_model(model_name)

    logger.info(f"Encoding {len(df)} texts on device: {encoder.device}")

    embeddings = encoder.encode(
        df["text"],
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


@main_env.task(report=True)
async def main(
    dataset_name: str = "wikipedia",
    dataset_version: str = "20220301.en",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 128,
    num_proc: int = 4,
) -> Dict[str, str]:
    """
    Main workflow that processes Wikipedia dataset and creates embeddings.
    Downloads model in parallel while streaming through dataset partitions.
    """

    logger.info("Starting Wikipedia embedding workflow")
    logger.info(f"Dataset: {dataset_name} {dataset_version}")
    logger.info(f"Model: {embedding_model}")

    # Get dataset info first to estimate total records
    repo_id = f"{dataset_name}"
    revision = dataset_version if dataset_version != "20220301.en" else "main"

    try:
        files = list_repo_files(repo_id, revision=revision, repo_type="dataset")
        parquet_files = [f for f in files if f.endswith(".parquet") and "train" in f]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {repo_id}/{revision}")

        # Estimate total records (rough estimate: 10k records per parquet file)
        estimated_records = len(parquet_files) * 10000
    except Exception as e:
        logger.error(f"Error accessing dataset {repo_id}: {e}")
        raise e

    # Initialize the data processing tracker
    tracker = DataProcessingTracker(
        total_records=estimated_records,
        title="📄 Wikipedia Embeddings Pipeline",
        description=f"Processing {dataset_name} dataset with {embedding_model} model...",
    )

    await tracker.initialize()
    await tracker.log_activity("Starting dataset download and preparation", "info")

    # Stream through dataset partitions using HF Hub
    async def partition_generator() -> AsyncGenerator[flyte.io.DataFrame, None]:
        """Download Wikipedia parquet files directly from HuggingFace Hub."""
        logger.info(f"Loading {dataset_name} {dataset_version} from HuggingFace Hub")

        try:
            logger.info(f"Found {len(parquet_files)} parquet files to process")
            await tracker.log_activity(f"Found {len(parquet_files)} data partitions to process", "success")

            # Download and yield each parquet file
            for i, parquet_file in enumerate(parquet_files):
                logger.info(f"Processing partition {i + 1}/{len(parquet_files)}: {parquet_file}")

                await tracker.log_activity(
                    f"Downloading partition {i + 1}/{len(parquet_files)}: {parquet_file}", "info"
                )

                # Download the parquet file
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=parquet_file,
                    revision=revision,
                    repo_type="dataset",
                )

                await tracker.log_activity(f"Successfully downloaded partition {i + 1}", "success")
                yield flyte.io.DataFrame(uri=file_path)

        except Exception as e:
            await tracker.log_activity(f"Error accessing dataset: {e}", "error")
            raise

    # Process partitions as they become available
    logger.info("Starting partition processing...")
    await tracker.log_activity("Starting embedding generation for all partitions", "info")

    embedding_tasks = []
    partition_count = 0
    processed_records = 0

    with flyte.group("wikipedia"):
        # Process each partition
        async for partition_df in partition_generator():
            if partition_df is None:
                break

            # Estimate records in this partition (for demo, assume 10k per partition)
            partition_records = 10000

            logger.info("Processing partition with texts")
            await tracker.log_activity(f"Creating embedding task for partition {partition_count + 1}", "info")

            # Create encoding task for this partition
            encoding_task = encode(partition_df, batch_size, embedding_model)
            embedding_tasks.append(encoding_task)
            partition_count += 1

            # Update progress tracker
            processed_records += partition_records
            await tracker.update_progress(processed_records)

        logger.info(f"Created {len(embedding_tasks)} encoding tasks, waiting for completion...")
        await tracker.log_activity(
            f"All {len(embedding_tasks)} embedding tasks created, waiting for completion", "info"
        )

        # Wait for all encoding tasks to complete
        embeddings = await asyncio.gather(*embedding_tasks)

        await tracker.log_activity("All embedding tasks completed successfully", "success")

    # Compute final statistics
    total_embeddings = sum(tensor.shape[0] for tensor in embeddings if tensor.numel() > 0)
    embedding_dim = embeddings[0].shape[1] if embeddings and embeddings[0].numel() > 0 else 0

    logger.info(f"Workflow complete: {total_embeddings} embeddings generated")

    # Complete the tracking
    final_stats = await tracker.complete(
        f"Wikipedia embeddings pipeline completed! Generated {total_embeddings:,} embeddings using {embedding_model}"
    )

    return {
        "partitions_processed": str(len(embeddings)),
        "total_embeddings_generated": str(total_embeddings),
        "embedding_dimension": str(embedding_dim),
        "model_used": embedding_model,
        "batch_size": str(batch_size),
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
