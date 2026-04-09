"""
Example: HuggingFace Datasets with Flyte.

This example demonstrates:
- Prefetching a dataset from HuggingFace Hub to remote storage
- Loading a prefetched Dir into a datasets.Dataset inside a task
- Passing datasets.Dataset between tasks via the type transformer
- Creating and returning new datasets from tasks
"""

import datasets
import flyte
import pyarrow.parquet as pq

from flyteplugins.huggingface import hf_dataset

env = flyte.TaskEnvironment(
    name="hf-dataset-example",
    image=flyte.Image.from_debian_base(name="hf-dataset-example").with_pip_packages(
        "flyteplugins-huggingface",
    ),
)


@env.task
async def load_from_dir(data_dir: flyte.io.Dir) -> datasets.Dataset:
    """Load parquet files from a prefetched Dir into a datasets.Dataset."""
    tables = []
    async for file in data_dir.walk():
        if file.path.endswith(".parquet"):
            local = await file.download()
            tables.append(pq.read_table(local))
    import pyarrow as pa

    return datasets.Dataset(pa.concat_tables(tables))


@env.task
async def tokenize(ds: datasets.Dataset) -> datasets.Dataset:
    """Simple tokenization: add word count column."""
    word_counts = [len(text.split()) for text in ds["text"]]
    return ds.add_column("word_count", word_counts)


@env.task
async def filter_long(ds: datasets.Dataset) -> datasets.Dataset:
    """Keep only rows with more than 100 words."""
    return ds.filter(lambda row: row["word_count"] > 100)


@env.task
async def summary(ds: datasets.Dataset) -> str:
    return f"{len(ds)} rows, columns: {ds.column_names}"


if __name__ == "__main__":
    flyte.init()

    # 1. Prefetch dataset from HuggingFace Hub to remote storage
    run = hf_dataset(repo="stanfordnlp/imdb", split="train")
    run.wait()
    data_dir = run.outputs()[0]

    # 2. Load into datasets.Dataset inside a task
    run = flyte.with_runcontext("local").run(load_from_dir, data_dir)
    ds = run.outputs()[0]

    # 3. Pass datasets.Dataset between tasks via the type transformer
    run = flyte.with_runcontext("local").run(tokenize, ds)
    tokenized = run.outputs()[0]

    run = flyte.with_runcontext("local").run(filter_long, tokenized)
    filtered = run.outputs()[0]

    run = flyte.with_runcontext("local").run(summary, filtered)
    print(run.outputs()[0])
