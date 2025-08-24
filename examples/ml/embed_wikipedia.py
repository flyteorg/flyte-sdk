# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b17",
#    "sentence-transformers",
#    "huggingface-hub",
#    "hf-transfer",
#    "datasets",
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
import logging
import os
import tempfile
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_url
from sentence_transformers import SentenceTransformer

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


@lru_cache(maxsize=1)
def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Lazily load and cache the SentenceTransformer model."""
    return SentenceTransformer(model_name)


@worker.task(cache="auto", retries=2)
async def embed_shard_to_file(repo_id: str, filename: str, model_name: str, batch_size: int = 32) -> flyte.io.File:
    """
    Stream one parquet shard, embed in batches, write embeddings to a file.

    Args:
        repo_id: Hugging Face dataset repo id (e.g. "wikimedia/wikipedia").
        filename: Path of the shard inside the dataset repo.
        model_name: SentenceTransformer model name.
        batch_size: Number of texts per embedding batch.

    Returns:
        str: Path to the saved `.pt` file containing embeddings (torch.Tensor).
    """
    model: SentenceTransformer = get_model(model_name)

    # Get shard URL
    file_url: str = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")

    # Stream dataset shard
    ds = load_dataset("parquet", data_files=file_url, split="train", streaming=True)

    # Prepare output file
    shard_name: str = filename.replace("/", "_")
    out_path: str = os.path.join(tempfile.gettempdir(), f"{shard_name}.pt")

    all_embeddings: list[torch.Tensor] = []
    batch: list[str] = []

    async for row in _aiter(ds):
        text: str = row.get("text", "")
        if not text:
            continue
        batch.append(text)

        if len(batch) >= batch_size:
            embeddings: torch.Tensor = await asyncio.to_thread(
                model.encode, batch, convert_to_tensor=True, show_progress_bar=False
            )
            all_embeddings.append(embeddings.cpu())
            batch = []

    if batch:
        embeddings: torch.Tensor = await asyncio.to_thread(
            model.encode, batch, convert_to_tensor=True, show_progress_bar=False
        )
        all_embeddings.append(embeddings.cpu())

    if all_embeddings:
        tensor: torch.Tensor = torch.cat(all_embeddings, dim=0)
        torch.save(tensor, out_path)

    return await flyte.io.File.from_local(out_path)


async def _aiter(sync_iterable) -> AsyncGenerator[Dict[str, Any], None]:
    """Wrap a synchronous iterable into an async generator."""
    loop = asyncio.get_running_loop()
    for row in sync_iterable:
        yield await loop.run_in_executor(None, lambda r=row: r)


@driver.task(cache="auto")
async def main():
    from huggingface_hub import HfApi

    repo_id = "wikimedia/wikipedia"
    model_name = "all-MiniLM-L6-v2"

    api = HfApi()
    info = api.dataset_info("wikimedia/wikipedia")
    # Each file is stored in info.siblings
    parquet_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")]
    print(parquet_files)
    filename = parquet_files[0]  # For demo, just process the first shard
    out_file = await embed_shard_to_file(repo_id, filename, model_name=model_name, batch_size=64)
    print("✅ Embeddings written to:", out_file)


if __name__ == "__main__":
    # Usage:
    # Run this with limit=-1 to embed all articles in the dataset (~61MM rows)
    # flyte.init()
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(main)
    print(run.url)
