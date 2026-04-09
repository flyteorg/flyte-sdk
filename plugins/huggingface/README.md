# Hugging Face Datasets Plugin

Native support for HuggingFace Datasets in Flyte: prefetch datasets from the Hub to remote storage and pass `datasets.Dataset` between tasks with automatic Parquet serialization.

## Installation

```bash
pip install flyteplugins-huggingface
```

## Prefetch from HuggingFace Hub

Stream a dataset from the Hub directly to Flyte's remote storage:

```python
import flyte
from flyteplugins.huggingface import hf_dataset

flyte.init(endpoint="my-flyte-endpoint")

run = hf_dataset(repo="stanfordnlp/imdb", split="train")
run.wait()
data_dir = run.outputs()[0]  # flyte.io.Dir with parquet files
```

## Type transformer

Pass `datasets.Dataset` between tasks with automatic serialization:

```python
import flyte
import datasets

env = flyte.TaskEnvironment(
    name="hf-example",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "flyteplugins-huggingface",
    ),
)


@env.task
async def create_dataset() -> datasets.Dataset:
    return datasets.Dataset.from_dict({
        "text": ["hello", "world", "foo"],
        "label": [0, 1, 0],
    })


@env.task
async def filter_positive(ds: datasets.Dataset) -> datasets.Dataset:
    return ds.filter(lambda x: x["label"] == 1)
```

## Column filtering

Use type annotations to load only specific columns:

```python
from typing import Annotated
from collections import OrderedDict

@env.task
async def load_text_only(
    ds: Annotated[datasets.Dataset, OrderedDict(text=str)],
) -> list:
    return ds["text"]
```
