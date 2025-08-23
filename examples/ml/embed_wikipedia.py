# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b17",
#    "sentence-transformers",
#    "datasets",
#    "huggingface-hub",
#    "hf-transfer",
# ]
# ///


"""
Wikipedia Article Embedding with Modern BERT using Hugging Face Datasets

This Flyte 2 script demonstrates how to:
1. Load Wikipedia articles using the Hugging Face datasets library
2. Preprocess and clean the text content
3. Generate BERT embeddings using modern sentence-transformers
4. Store the results for further processing

Requirements:
- datasets (Hugging Face)
- sentence-transformers
- pandas
- numpy
"""

import asyncio
import json
import logging
import tempfile

import numpy as np
import pandas as pd
from async_lru import alru_cache
from sentence_transformers import SentenceTransformer

import flyte
import flyte.io

# Configure logging
logger = logging.getLogger(__name__)


image = flyte.Image.from_uv_script(__file__, name="embed_wikipedia_image").with_pip_packages("unionai-reuse")

driver = flyte.TaskEnvironment(
    name="embed_wikipedia_driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="4Gi", disk="16Gi"),
    # reusable=flyte.ReusePolicy(replicas=4, concurrency=2, idle_ttl=60),
)

N_GPUS = 1
worker = flyte.TaskEnvironment(
    name="embed_wikipedia_worker",
    image=image,
    resources=flyte.Resources(cpu=4, memory="16Gi", disk="16Gi", gpu=1),
    env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    # reusable=flyte.ReusePolicy(replicas=12, concurrency=1, idle_ttl=60),
)


@driver.task(cache="auto")
async def load_partitions(num_proc: int = 4) -> list[flyte.io.DataFrame]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info("wikimedia/wikipedia")
    # Each file is stored in info.siblings
    parquet_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")]
    print(parquet_files)
    partitions = []
    for i, f in enumerate(parquet_files):
        print(f"Adding partition {i}: {f} to encoding tasks")
        partitions.append(flyte.io.DataFrame(uri=str(f)))
    return partitions


@alru_cache(maxsize=32)
async def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@worker.task(retries=10)
async def embed_batch(batch: pd.DataFrame, encode_batch_size: int = 256) -> flyte.io.File:
    model = await load_embedding_model("BAAI/bge-small-en-v1.5")
    print(f"model loaded {model}")

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    embeddings: list[np.ndarray] = model.encode(
        batch["text"].tolist(),
        show_progress_bar=True,
        batch_size=encode_batch_size,
    )
    output = []
    for i, embedding in enumerate(embeddings):
        output.append(
            dict(
                title=batch["title"].iloc[i],
                id=batch["id"].iloc[i],
                embedding=embedding.tolist(),
            )
        )

    with open(temp_file.name, "w") as f:
        json.dump(output, f)

    file = await flyte.io.File.from_local(temp_file.name)
    return file


@driver.task(retries=10)
async def embed_articles(df: pd.DataFrame, batch_size: int) -> list[flyte.io.File]:
    print(f"encoding {len(df)} articles")
    embedded_batches = []

    # Create batch indexes for dataframe
    n = len(df)
    indexes = []
    if n > batch_size:
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            indexes.append((i, end_idx))
    else:
        indexes.append((0, n))

    for i, idx in enumerate(indexes):
        embedded_batches.append(embed_batch(df.iloc[idx[0] : idx[1]]))
    return await asyncio.gather(*embedded_batches)


@driver.task
async def embed_wikipedia(
    limit: int = 8,
    batch_size: int = 8,
    num_proc: int = 4,
) -> list[flyte.io.File]:
    partitions = await load_partitions(num_proc)

    embedding_tasks = []
    for i, partition in enumerate(partitions):
        if limit and limit != -1 and i > limit:
            break
        embedding_tasks.append(embed_articles(partition, batch_size))

    embeddings = await asyncio.gather(*embedding_tasks)
    # Flatten list of lists into single list
    return [file for batch in embeddings for file in batch]


if __name__ == "__main__":
    # Usage:
    # Run this with limit=-1 to embed all articles in the dataset (~61MM rows)
    # flyte.init()
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(embed_wikipedia, limit=1, batch_size=50_000)
    print(run.url)
