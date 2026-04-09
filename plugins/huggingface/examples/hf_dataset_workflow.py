"""
Example: HuggingFace Datasets with Flyte.

This example demonstrates:
- Prefetching a dataset from HuggingFace Hub to remote storage
- Passing datasets.Dataset between tasks via the type transformer
- Creating and returning new datasets from tasks
"""

import datasets
import flyte

from flyteplugins.huggingface import hf_dataset

env = flyte.TaskEnvironment(
    name="hf-dataset-example",
    image=flyte.Image.from_debian_base(name="hf-dataset-example").with_pip_packages(
        "flyteplugins-huggingface",
    ),
)


@env.task
async def tokenize(ds: datasets.Dataset) -> datasets.Dataset:
    """Simple tokenization: split text into word count."""
    word_counts = [len(text.split()) for text in ds["text"]]
    ds = ds.add_column("word_count", word_counts)
    return ds


@env.task
async def filter_long(ds: datasets.Dataset) -> datasets.Dataset:
    """Keep only rows with more than 100 words."""
    return ds.filter(lambda row: row["word_count"] > 100)


@env.task
async def summary(ds: datasets.Dataset) -> str:
    return f"{len(ds)} rows, columns: {ds.column_names}"


if __name__ == "__main__":
    flyte.init()

    # Prefetch dataset from HuggingFace Hub
    run = hf_dataset(repo="stanfordnlp/imdb", split="train")
    run.wait()
    data_dir = run.outputs()[0]

    # Use datasets between tasks via the type transformer
    ds = datasets.Dataset.from_parquet(str(data_dir.path) + "/*.parquet")

    result = flyte.with_runcontext("local").run(tokenize, ds)
    tokenized = result.outputs()[0]

    result = flyte.with_runcontext("local").run(filter_long, tokenized)
    filtered = result.outputs()[0]

    result = flyte.with_runcontext("local").run(summary, filtered)
    print(result.outputs()[0])
