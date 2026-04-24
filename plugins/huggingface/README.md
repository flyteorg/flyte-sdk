# Hugging Face Plugin

Native Flyte support for Hugging Face integrations in Flyte.

This plugin provides dataset support for Hugging Face `datasets.Dataset`
and `datasets.IterableDataset` objects. It gives you two related capabilities:

1. Use `from_hf(...)` to reference a dataset on the Hugging Face Hub as a task
   input default.
2. Pass Hugging Face dataset objects between Flyte tasks with automatic Parquet
   serialization.

The plugin works by treating Hub datasets as Parquet-backed structured data. For
Hub sources, it first resolves the dataset's converted Parquet shards, then
materializes them either into a generated path for the current run or into a
shared artifact registry rooted at `cache_root`.

## Installation

```bash
pip install flyteplugins-huggingface
```

## Quick start

```python
import datasets
import flyte
from flyteplugins.huggingface.datasets import from_hf

env = flyte.TaskEnvironment(name="hf-example")

@env.task
async def count_reviews(
    ds: datasets.Dataset = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
    ),
) -> int:
    return len(ds)
```

At the Flyte literal level this source is represented as an `hf://` URI, for
example:

```text
hf://stanfordnlp/imdb?name=plain_text&split=train
```

The task receives a hydrated `datasets.Dataset`. The `hf://` URI is only the
reference used between Flyte and the plugin.

## `from_hf(...)`

`from_hf(...)` is the entry point for Hub-backed task defaults:

```python
from flyteplugins.huggingface.datasets import from_hf

from_hf(
    repo: str,
    *,
    name: str | None = None,
    split: str | None = None,
    revision: str | None = None,
    cache_root: str | None = None,
)
```

Arguments:

- `repo`: Hugging Face dataset repo, such as `"stanfordnlp/imdb"` or `"glue"`.
- `name`: Optional dataset config/subset.
- `split`: Optional split such as `"train"` or `"validation"`.
- `revision`: Optional Hub revision. Defaults to `refs/convert/parquet`.
- `cache_root`: Optional shared remote cache root for cross-run reuse.

`from_hf(...)` returns a Flyte `DataFrame` reference, not an eagerly loaded
dataset object. When the task input is typed as `datasets.Dataset` or
`datasets.IterableDataset`, the plugin decoder materializes that reference into
the requested Hugging Face type.

## Config resolution

If you specify `name`, the plugin uses that config directly.

If you omit `name`, the plugin resolves the config as follows:

1. Try actual converted-parquet config `default`.
2. If `default` does not exist and there is exactly one config, use that one.
3. If there are multiple configs, raise an error and ask for `name=...`.

Examples:

```python
# Works: imdb has a single converted-parquet config, plain_text.
from_hf("stanfordnlp/imdb", split="train")

# Required: glue has multiple configs such as mrpc, sst2, qnli, ...
from_hf("glue", name="mrpc", split="train")
```

Using `name=` explicitly is recommended in examples and production code because
it makes the UI literal and task signature more obvious.

## Split behavior

If you specify `split`, only that split is materialized:

```python
@env.task
async def train_split(
    ds: datasets.Dataset = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
    ),
) -> int:
    return len(ds)
```

If you omit `split`, the plugin reads every converted Parquet split under the
resolved config and presents them as one dataset stream/table:

```python
@env.task
async def all_splits(
    ds: datasets.Dataset = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
    ),
) -> list[str]:
    return ds.column_names
```

That means the result is a combined dataset, not a mapping of split name to
dataset.

## Cross-run reuse with `cache_root`

Without `cache_root`, a Hub source is materialized into a generated path for the
current execution only.

With `cache_root`, the plugin uses a shared cache registry so later runs can
skip the Hub download entirely:

```python
@env.task
async def train_cached(
    ds: datasets.Dataset = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
        cache_root="s3://my-bucket/flyte-hf-cache",
    ),
) -> int:
    return len(ds)
```

The shared cache layout is:

```text
{cache_root}/huggingface/datasets/
  by-key/{source-cache-key}.json
  blobs/{source-cache-key}/...
```

The cache key is derived from:

- repo
- config name
- split
- revision
- resolved Parquet shard metadata

This means the cache is stable across runs as long as the underlying converted
Parquet source does not change.

The canonical artifact location is always
`{cache_root}/huggingface/datasets/blobs/{source-cache-key}/...`. The registry
record under `by-key/` is metadata for that cache key.

## What the plugin logs

When `LOG_LEVEL` is `INFO` or lower, the plugin logs whether it is:

- checking the shared dataset cache
- materializing from the Hugging Face Hub
- using a cached artifact
- reading Parquet from a local or remote directory

This is the easiest way to confirm whether a run is reading from the Hub or
from your shared cache artifact.

## `datasets.Dataset` between tasks

You can return and pass real `datasets.Dataset` objects between tasks:

```python
import datasets
import flyte

env = flyte.TaskEnvironment(name="hf-transform")


@env.task
async def create_dataset() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "text": ["hello", "world", "flyte"],
            "label": [0, 1, 0],
        }
    )


@env.task
async def filter_positive(ds: datasets.Dataset) -> datasets.Dataset:
    return ds.filter(lambda row: row["label"] == 1)
```

Task-produced in-memory datasets are serialized to Parquet automatically. This
is separate from `from_hf(...)`, which is a source reference rather than a
materialized dataset object.

## `datasets.IterableDataset`

Use `datasets.IterableDataset` when you want row streaming behavior instead of a
fully materialized table:

```python
@env.task
async def stream_reviews(
    ds: datasets.IterableDataset = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
        cache_root="s3://my-bucket/flyte-hf-cache",
    ),
) -> datasets.IterableDataset:
    def add_length(batch):
        batch["length"] = [len(text) for text in batch["text"]]
        return batch

    return ds.map(add_length, batched=True)
```

Notes:

- The returned Hugging Face `IterableDataset` is consumed with normal synchronous
  iteration.
- Internally the plugin streams row batches from Parquet files.
- Iterable outputs are written back as sharded Parquet directories.

## Column projection

Use a Flyte structured-dataset column annotation when you only want selected
columns:

```python
from collections import OrderedDict
from typing import Annotated


@env.task
async def load_text_only(
    ds: Annotated[datasets.Dataset, OrderedDict(text=str)] = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
    ),
) -> list[str]:
    return ds["text"][:10]
```

The plugin uses the annotation to request only those columns when reading
Parquet.

## Revision selection

Use `revision=` if you want to pin a specific converted-Parquet revision:

```python
@env.task
async def pinned_revision(
    ds: datasets.Dataset = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
        revision="refs/convert/parquet",
        cache_root="s3://my-bucket/flyte-hf-cache",
    ),
) -> int:
    return len(ds)
```

If you do not specify a revision, the plugin uses `refs/convert/parquet`.

## Local vs remote behavior

There are two distinct layers to keep in mind:

1. Task inputs and outputs inside Flyte tasks.
2. What your launcher process sees when a run completes.

Inside a task, a parameter typed as `datasets.Dataset` or
`datasets.IterableDataset` is hydrated by the plugin into a Hugging Face object.

Outside the task, especially for remote runs, outputs are often represented to
the launcher as Flyte `DataFrame` references rather than already-opened Hugging
Face dataset objects. That is expected: the structured dataset literal remains
the transport format.

## Private datasets

Set `HF_TOKEN` in the task environment to access private Hugging Face datasets.
Without it, the plugin uses anonymous Hub access.

## Failure modes

Common issues:

- Missing `name` for a dataset with multiple configs:
  the plugin raises and asks for `name=...`.
- No converted Parquet shards available:
  the dataset may not have an auto-converted Parquet representation yet.
- Remote cache path credentials:
  your Flyte runtime must be able to read and write the chosen `cache_root`.

## Example

See the example workflow in `plugins/huggingface/examples/hf_dataset_workflow.py`
for end-to-end local and remote scenarios.
