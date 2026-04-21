"""
Example: HuggingFace Datasets with Flyte.

This example demonstrates:
- Uncached hf:// source materialization
- Shared dataset materialization with cache_root
- Dataset configs, splits, and all-split loading
- Loading selected columns with Annotated
- Using datasets.IterableDataset for lazy streaming from remote storage
- Serializing task-produced datasets.Dataset objects to parquet, locally
"""

from collections import OrderedDict
import logging
import os
from pathlib import Path
import tempfile
from typing import Annotated, Any, Callable

import datasets
from flyteplugins.huggingface import from_hf

import flyte
from flyte._image import PythonWheels
from flyte.io import DataFrame

env = flyte.TaskEnvironment(
    name="hf-dataset-example",
    image=flyte.Image.from_debian_base(name="hf-dataset-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-huggingface",
        ),
    ),
)


RUN_MODE = os.environ.get("HF_EXAMPLE_RUN_MODE", "local")
IS_LOCAL_RUN = RUN_MODE == "local"

LOG_LEVEL_NAME = os.environ.get("HF_EXAMPLE_LOG_LEVEL", "INFO")
LOG_LEVEL = (
    int(LOG_LEVEL_NAME)
    if LOG_LEVEL_NAME.isdigit()
    else getattr(logging, LOG_LEVEL_NAME.upper(), logging.INFO)
)

LOCAL_HF_CACHE_ROOT = tempfile.mkdtemp(prefix="flyte-hf-cache-")
REMOTE_HF_CACHE_ROOT = "s3://my-bucket/flyte-hf-cache"  # TODO: replace with your own remote cache path for remote runs
HF_CACHE_ROOT = REMOTE_HF_CACHE_ROOT if RUN_MODE == "remote" else LOCAL_HF_CACHE_ROOT

HF_PARQUET_REVISION = "refs/convert/parquet"
IMDB_CONFIG = "plain_text"

IMDB_TRAIN = from_hf(
    "stanfordnlp/imdb",
    name=IMDB_CONFIG,
    split="train",
    cache_root=HF_CACHE_ROOT,
)
GLUE_MRPC_TRAIN = from_hf(
    "glue",
    name="mrpc",
    split="train",
    cache_root=HF_CACHE_ROOT,
)
IMDB_ALL_SPLITS = from_hf(
    "stanfordnlp/imdb",
    name=IMDB_CONFIG,
    cache_root=HF_CACHE_ROOT,
)
IMDB_TRAIN_EXPLICIT_REVISION = from_hf(
    "stanfordnlp/imdb",
    name=IMDB_CONFIG,
    split="train",
    revision=HF_PARQUET_REVISION,
    cache_root=HF_CACHE_ROOT,
)


def dataset_summary(ds: datasets.Dataset) -> str:
    return f"{len(ds)} rows, columns: {ds.column_names}"


def iterable_dataset_summary(_ds: datasets.IterableDataset) -> str:
    return "IterableDataset"


def dataframe_reference_summary(df: DataFrame) -> str:
    details = [f"uri={df.uri!r}", f"format={df.format!r}"]
    if df.hash:
        details.append(f"hash={df.hash!r}")
    return f"DataFrame({', '.join(details)})"


def output_summary(output: Any) -> str:
    if isinstance(output, DataFrame):
        return dataframe_reference_summary(output)
    if isinstance(output, datasets.Dataset):
        return dataset_summary(output)
    if isinstance(output, datasets.IterableDataset):
        return iterable_dataset_summary(output)
    return str(output)


def print_run(
    label: str,
    run,
    formatter: Callable[[Any], str] | None = None,
) -> Any | None:
    print(f"{label} URL: {run.url}")
    try:
        run.wait()
        output = run.outputs()[0]
        output_text = (formatter or output_summary)(output)
    except Exception as e:
        print(f"{label} output: unavailable ({type(e).__name__}: {e})")
        return None

    print(f"{label} output: {output_text}")
    return output


def run_task(task, *args, **kwargs):
    return flyte.with_runcontext(
        RUN_MODE,
        log_level=LOG_LEVEL,
        env_vars={"LOG_LEVEL": str(LOG_LEVEL)},
    ).run(task, *args, **kwargs)


@env.task
async def count_imdb_uncached_source(
    ds: datasets.Dataset = from_hf(
        "stanfordnlp/imdb",
        name=IMDB_CONFIG,
        split="train",
    ),
) -> int:
    """Materialize to a generated Flyte raw-data path for this execution."""
    return len(ds)


@env.task
async def tokenize_imdb_cached_source(
    ds: datasets.Dataset = IMDB_TRAIN,
) -> datasets.Dataset:
    """Add a word-count column. On first run the plugin fetches the IMDB
    dataset from HuggingFace Hub and caches it in remote storage; subsequent
    runs skip the download entirely."""
    word_counts = [len(text.split()) for text in ds["text"]]
    return ds.add_column("word_count", word_counts)


@env.task
async def count_glue_mrpc_cached_source(ds: datasets.Dataset = GLUE_MRPC_TRAIN) -> int:
    """Load a dataset repo that requires a config/subset name."""
    return len(ds)


@env.task
async def list_imdb_cached_all_splits(
    ds: datasets.Dataset = IMDB_ALL_SPLITS,
) -> list[str]:
    """Omitting split fetches all converted parquet splits for the config."""
    return ds.column_names


@env.task
async def count_imdb_cached_explicit_revision(
    ds: datasets.Dataset = IMDB_TRAIN_EXPLICIT_REVISION,
) -> int:
    """Use a specific Hugging Face revision for converted parquet files."""
    return len(ds)


@env.task
async def sample_imdb_cached_text_column(
    ds: Annotated[
        datasets.Dataset,
        OrderedDict(text=str),
    ] = IMDB_TRAIN,
) -> list[str]:
    """Only the requested columns are read from parquet."""
    return ds["text"][:10]


@env.task
async def filter_tokenized_reviews(ds: datasets.Dataset) -> datasets.Dataset:
    """Return a new in-memory datasets.Dataset produced by task code."""
    return ds.filter(lambda row: row["word_count"] > 100)


@env.task
async def summarize_dataset(ds: datasets.Dataset) -> str:
    return f"{len(ds)} rows, columns: {ds.column_names}"


@env.task
async def stream_tokenize_imdb_cached_source(
    ds: datasets.IterableDataset = IMDB_TRAIN,
) -> datasets.IterableDataset:
    """IterableDataset variant: stream from cached parquet without a full load."""

    def add_word_count(batch):
        batch["word_count"] = [len(t.split()) for t in batch["text"]]
        return batch

    return ds.map(add_word_count, batched=True)


@env.task
async def count_stream_sample(ds: datasets.IterableDataset) -> int:
    """Consume an IterableDataset produced by another task."""
    count = 0
    for _row in ds.take(1_000):
        count += 1
    return count


if __name__ == "__main__":
    flyte.init_from_config(log_level=LOG_LEVEL)

    # Scenario 1: uncached hf:// source.
    uncached_run = run_task(count_imdb_uncached_source)
    print_run("uncached imdb count", uncached_run)

    # Scenario 2: cached hf:// source. The second run should reuse cache_root.
    cached_run = run_task(tokenize_imdb_cached_source)
    tokenized = print_run("cached imdb tokenize", cached_run)

    second_cached_run = run_task(tokenize_imdb_cached_source)
    print_run("cached imdb tokenize again", second_cached_run)

    # Scenario 3: config, split, revision, and column projection.
    glue_run = run_task(count_glue_mrpc_cached_source)
    print_run("cached glue/mrpc count", glue_run)

    all_splits_run = run_task(
        list_imdb_cached_all_splits.override(resources=flyte.Resources(memory="2Gi"))
    )
    print_run("cached imdb all-splits columns", all_splits_run)

    revision_run = run_task(count_imdb_cached_explicit_revision)
    print_run("cached imdb explicit revision count", revision_run)

    text_column_run = run_task(sample_imdb_cached_text_column)
    print_run(
        "cached imdb text column sample",
        text_column_run,
        lambda rows: f"{len(rows)} rows",
    )

    # Scenario 4: IterableDataset source.
    stream_run = run_task(stream_tokenize_imdb_cached_source)
    stream = print_run(
        "cached imdb stream tokenize",
        stream_run,
        iterable_dataset_summary,
    )

    # Scenario 5: local-only in-memory Dataset output round trip.
    if IS_LOCAL_RUN and tokenized is not None:
        filtered_run = run_task(filter_tokenized_reviews, tokenized)
        filtered = print_run("filter tokenized reviews", filtered_run, dataset_summary)

        if filtered is not None:
            summary_run = run_task(summarize_dataset, filtered)
            print_run("summarize filtered dataset", summary_run)

    if IS_LOCAL_RUN and stream is not None:
        stream_count_run = run_task(count_stream_sample, stream)
        print_run("count stream sample", stream_count_run)
