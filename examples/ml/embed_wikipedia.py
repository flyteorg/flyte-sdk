# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte",
#    "async-lru",
#    "sentence-transformers",
#    "datasets",
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
from dataclasses import asdict, dataclass
from pathlib import Path

import datasets
from async_lru import alru_cache
from sentence_transformers import SentenceTransformer

import flyte
import flyte.io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


image = flyte.Image.from_uv_script(__file__, name="embed_wikipedia_image")

driver = flyte.TaskEnvironment(
    name="driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="4Gi", disk="64Gi"),
)

N_GPUS = 1
worker = flyte.TaskEnvironment(
    name="worker",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi", gpu=f"T4:{N_GPUS}"),
    reusable=flyte.ReusePolicy(replicas=4, concurrency=16),
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
async def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("nomic-ai/modernbert-embed-base")


@worker.task
async def embed_articles(batch: list[Article]) -> list[ArticleEmbedding]:
    model = await load_embedding_model()
    print(f"model loaded {model}")

    print(f"encoding {len(batch)} articles")
    embeddings = model.encode(
        [f"{article.title}\n{article.text}" for article in batch],
    )

    article_embeddings = []
    for i, embedding in enumerate(embeddings):
        article_embeddings.append(
            ArticleEmbedding(
                title=batch[i].title,
                wiki_id=batch[i].wiki_id,
                text=batch[i].text,
                embedding=embedding.tolist(),
                text_length=len(batch[i].text),
                language="en",
            )
        )
    print(f"encoded {len(article_embeddings)} articles")
    return article_embeddings


async def embedding_to_sink(dir: Path, batch: list[ArticleEmbedding]):
    """Writes embeddings to file"""
    article_embeddings = [asdict(article_embedding) for article_embedding in await embed_articles(batch)]

    for article_embedding in article_embeddings:
        fname = f"{article_embedding['title']}_{article_embedding['wiki_id']}.json"
        with (dir / fname).open("w") as f:
            json.dump(article_embedding, f)


@driver.task
async def embed_wikipedia(
    limit: int = 8,
    batch_size: int = 8,
    embedding_group_size: int = 4,
) -> flyte.io.Dir:
    dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    print(f"dataset loaded {dataset}")

    embedding_tasks = []
    temp_dir = tempfile.mkdtemp()

    batch = []
    group_number = 1
    for i, article in enumerate(dataset["train"]):
        if limit and limit != -1 and i > limit:
            break
        batch.append(Article(title=article["title"], text=article["text"], wiki_id=article["id"]))

        if len(batch) == batch_size:
            embedding_tasks.append(embedding_to_sink(Path(temp_dir), batch))
            batch = []

        if len(embedding_tasks) == embedding_group_size:
            with flyte.group(f"embedding_tasks_{group_number}"):
                group_number += 1
                await asyncio.gather(*embedding_tasks)
                embedding_tasks = []

    # embed any remaining articles
    if batch:
        embedding_tasks.append(embedding_to_sink(Path(temp_dir), batch))
    if embedding_tasks:
        with flyte.group(f"embedding_tasks_{group_number}"):
            await asyncio.gather(*embedding_tasks)

    print(f"Files written to {temp_dir}")
    for file in Path(temp_dir).glob("*.json"):
        with file.open("r") as f:
            print(json.load(f))

    dir = await flyte.io.Dir.from_local(temp_dir)
    return dir


if __name__ == "__main__":
    # Usage:
    # Run this with limit=-1 to embed all articles in the dataset (~61MM rows)
    # flyte.init()
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(embed_wikipedia, limit=1000, batch_size=1, embedding_group_size=100)
    print(run.url)
