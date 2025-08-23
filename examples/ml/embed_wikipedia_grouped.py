# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte",
#    "async-lru",
#    "sentence-transformers",
#    "datasets",
#    "pandas",
#    "unionai-reuse",
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
from dataclasses import dataclass

import datasets
import pandas as pd
from async_lru import alru_cache
from sentence_transformers import SentenceTransformer

import flyte
import flyte.io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


image = flyte.Image.from_uv_script(__file__, name="embed_wikipedia_image")

driver = flyte.TaskEnvironment(
    name="embed_wiki_grouped_driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="10Gi", disk="64Gi"),
)

worker = flyte.TaskEnvironment(
    name="embed_wiki_grouped_worker",
    image=image,
    resources=flyte.Resources(cpu=8, memory="12Gi", gpu=1),
    reusable=flyte.ReusePolicy(replicas=8, concurrency=1),
)


@dataclass
class Article:
    title: str
    text: str
    wiki_id: str


@dataclass
class ArticleEmbedding(Article):
    """Data structure for article embeddings"""

    embedding: list[float]
    text_length: int
    language: str


@alru_cache(maxsize=32)
async def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@worker.task(retries=5)
async def embed_articles(batch: pd.DataFrame, encode_batch_size: int = 64) -> flyte.io.File:
    model = await load_embedding_model("BAAI/bge-small-en-v1.5")
    model.max_seq_length = 256
    print(f"model loaded {model}")

    print(f"encoding {len(batch)} articles")
    embeddings = model.encode(
        batch["text"].tolist(),
        show_progress_bar=True,
        batch_size=encode_batch_size,
    )

    file_contents = []
    for i, embedding in enumerate(embeddings):
        article_embedding = dict(
            title=batch["title"].iloc[i],
            wiki_id=batch["wiki_id"].iloc[i],
            embedding=embedding.tolist(),
        )
        file_contents.append(article_embedding)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, "w") as f:
        json.dump(file_contents, f)
    dir = await flyte.io.File.from_local(temp_file.name)
    return dir


@driver.task(cache="auto")
async def prepare_batches(
    limit: int = 8,
    batch_size: int = 8,
) -> list[flyte.io.DataFrame]:
    """
    Task to prepare batches of articles from Wikipedia dataset
    Returns a list of files, where each file contains a batch of articles
    """
    dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    print(f"dataset loaded {dataset}")

    output = []
    batch = []

    def create_batch(batch: list[dict]) -> flyte.io.DataFrame:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        pd.DataFrame(batch).to_parquet(temp_file.name)
        return flyte.io.DataFrame(uri=temp_file.name)

    for i, article in enumerate(dataset["train"]):
        if limit and limit != -1 and i > limit:
            break

        batch.append(dict(title=article["title"], text=article["text"], wiki_id=article["id"]))

        if len(batch) == batch_size:
            output.append(create_batch(batch))
            batch = []

    # Handle remaining articles in last batch
    if batch:
        output.append(create_batch(batch))

    return output


@driver.task
async def embed_wikipedia(
    limit: int = 8,
    batch_size: int = 8,
    embedding_group_size: int = 4,
) -> list[flyte.io.File]:
    batches = await prepare_batches(limit=limit, batch_size=batch_size)
    embeddings = []
    group_number = 1
    for i, batch in enumerate(batches):
        with flyte.group(f"embedding_tasks_{group_number}"):
            embeddings.append(asyncio.create_task(embed_articles(batch)))
        if i % embedding_group_size == 0:
            group_number += 1

    return await asyncio.gather(*embeddings)


if __name__ == "__main__":
    # Usage:
    # Run this with limit=-1 to embed all articles in the dataset (~61MM rows)
    # flyte.init()
    flyte.init_from_config("/Users/nielsbantilan/.flyte/config.yaml")
    run = flyte.run(embed_wikipedia, limit=1_000_000, batch_size=256, embedding_group_size=100)
    print(run.url)
