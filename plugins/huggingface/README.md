# Hugging Face Datasets Plugin

Native support for `datasets.Dataset` as a Flyte DataFrame type, with Parquet serialization and column filtering.

## Installation

```bash
pip install flyteplugins-huggingface
```

## Usage

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


@env.task
async def main() -> int:
    ds = await create_dataset()
    filtered = await filter_positive(ds)
    return len(filtered)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
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
