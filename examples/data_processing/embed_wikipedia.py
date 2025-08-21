# ///script
# requires-python = "==3.13"
# dependencies = [
#    "flyte",
#    "async-lru",
#    "sentence-transformers",
#    "wikipedia-api",
#    "pandas",
#    "numpy",
#    "requests",
#    "beautifulsoup4",
#    "lxml"
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
from dataclasses import dataclass, asdict
from pathlib import Path

import flyte
import flyte.io
import datasets
from async_lru import alru_cache
from sentence_transformers import SentenceTransformer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


image = flyte.Image.from_uv_script(__file__, name="embed_wikipedia_image")

driver = flyte.TaskEnvironment(
    name="driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi", disk="16Gi"),
    reusable=flyte.ReusePolicy(replicas=2)
)

worker = flyte.TaskEnvironment(
    name="worker",
    image=image,
    resources=flyte.Resources(cpu=4, memory="4Gi"),
)


@dataclass
class ArticleEmbedding:
    """Data structure for article embeddings"""
    title: str
    wiki_id: str
    embedding: list[float]
    text_length: int
    language: str


@alru_cache(maxsize=32)
async def load_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer("nomic-ai/modernbert-embed-base")
    return model


@worker.task
async def embed_article(title: str, text: str, wiki_id: str) -> ArticleEmbedding:
    model = await load_embedding_model()
    embedding = model.encode(f"{title}\n{text}")
    return ArticleEmbedding(
        title=title,
        wiki_id=wiki_id,
        embedding=embedding.tolist(),
        text_length=len(text),
        language="en"
    )


async def embedding_to_sink(file: Path, title: str, text: str, wiki_id: str):
    """Writes embeddings to file"""
    article_embedding = await embed_article(title, text, wiki_id)
    with file.open("w") as f:
        json.dump(asdict(article_embedding), f)
    del article_embedding


@driver.task
async def embed_wikipedia() -> flyte.io.Dir:
    dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

    embeddings = []
    temp_dir = tempfile.mkdtemp()
    with flyte.group("embed_articles"):
        for i, article in enumerate(dataset["train"]):
            new_file = Path(temp_dir) / f"{article['title']}_{article['id']}.json"
            embeddings.append(embedding_to_sink(new_file, article["title"], article["text"], article["id"]))
            if i > 10:
                break
        await asyncio.gather(*embeddings)

    print(f"Files written to {temp_dir}")
    for file in Path(temp_dir).glob("*.json"):
        with file.open("r") as f:
            print(json.load(f))

    dir = await flyte.io.Dir.from_local(temp_dir)
    return dir


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(embed_wikipedia)
    print(run.url)
