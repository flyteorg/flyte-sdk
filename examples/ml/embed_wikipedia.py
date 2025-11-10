# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b17",
#    "sentence-transformers",
#    "tokenizers==0.22.1",
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
from collections import defaultdict
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_url
from sentence_transformers import SentenceTransformer

import flyte.io

# Configure logging
logger = logging.getLogger(__name__)

image = flyte.Image.from_uv_script(__file__, name="embed_wikipedia_image").with_pip_packages(
    "unionai-reuse>=0.1.5",
)

N_GPUS = 1
worker = flyte.TaskEnvironment(
    name="embed_wikipedia_worker",
    image=image,
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
    env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    reusable=flyte.ReusePolicy(replicas=16, concurrency=1, idle_ttl=120, scaledown_ttl=120),
    secrets="HF_HUB_TOKEN",
)


driver = flyte.TaskEnvironment(
    name="embed_wikipedia_driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="4Gi", disk="16Gi"),
    secrets="HF_HUB_TOKEN",
    depends_on=[worker],
)


@lru_cache(maxsize=1)
def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Lazily load and cache the SentenceTransformer model."""
    return SentenceTransformer(model_name)


@worker.task(cache="auto", retries=4)
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
    logger.warning(f"Embedding shard {filename} from repo {repo_id} using model {model_name}")
    model: SentenceTransformer = get_model(model_name)
    logger.warning(f"Model loaded on device: {model.device}")

    # Get shard URL
    file_url: str = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")

    # Stream dataset shard
    ds = load_dataset("parquet", data_files=file_url, split="train", streaming=True, token=os.getenv("HF_HUB_TOKEN"))

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
            embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=True)
            all_embeddings.append(embeddings.cpu())
            batch = []
            print(f"Collected {len(all_embeddings)} articles so far...")

    if batch:
        embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=True)
        all_embeddings.append(embeddings.cpu())

    print(f"Saving {len(all_embeddings)} batches of embeddings to {out_path}")

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
async def main(batch_size: int = 64, shard: str = "all") -> list[flyte.io.File]:
    from huggingface_hub import HfApi

    repo_id = "wikimedia/wikipedia"
    model_name = "all-MiniLM-L6-v2"

    api = HfApi(token=os.getenv("HF_HUB_TOKEN"))
    info = api.dataset_info("wikimedia/wikipedia")
    # Each file is stored in info.siblings
    parquet_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")]
    print(f"Found {len(parquet_files)} parquet files in the dataset", flush=True)
    grouped_coros = defaultdict(list)
    for filename in parquet_files:
        shard_id = filename.split("/")[0]
        grouped_coros[shard_id].append(
            embed_shard_to_file(repo_id, filename, model_name=model_name, batch_size=batch_size)
        )

    if shard == "all":
        tasks = []
        for shard_id, coros in grouped_coros.items():
            print(f"Processing shard {shard_id} with {len(coros)} files", flush=True)
            with flyte.group(f"shard-{shard_id}"):
                shard_tasks = [asyncio.create_task(coro) for coro in coros]
                tasks.extend(shard_tasks)
        return await asyncio.gather(*tasks)
    else:
        if shard not in grouped_coros:
            raise ValueError(f"Shard {shard} not found. Available shards: {list(grouped_coros.keys())}")
        print(f"Processing only shard {shard} with {len(grouped_coros[shard])} files", flush=True)
        with flyte.group(f"shard-{shard}"):
            tasks = [asyncio.create_task(coro) for coro in grouped_coros[shard]]
            return await asyncio.gather(*tasks)


async def high_mem_examples():
    files = ["20231101.sv/train-00000-of-00005.parquet", "20231101.war/train-00000-of-00001.parquet"]
    large_task = embed_shard_to_file.override(resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1), reusable="off")

    coros = []
    for file in files:
        coros.append(flyte.run.aio(large_task, "wikimedia/wikipedia", file, "all-MiniLM-L6-v2", batch_size=256))

    results = await asyncio.gather(*coros)
    for r in results:
        print(r.url)
        print(r.url)


if __name__ == "__main__":
    # Usage:
    # Run this with limit=-1 to embed all articles in the dataset (~61MM rows)
    # flyte.init()
    flyte.init_from_config()
    run = flyte.run(main, 256, shard="20231101.en")
    print(run.url)
    # asyncio.run(high_mem_examples())
