# Flyte Pandera Plugin

This plugin integrates [Pandera](https://pandera.readthedocs.io/) with [Flyte](https://flyte.org/), enabling automatic runtime validation of pandas DataFrames against Pandera schemas as data flows between tasks.

## Installation

```bash
pip install flyteplugins-pandera
```

## Usage

Define a Pandera schema using `DataFrameModel` and use `pandera.typing.DataFrame` as your task's type annotation:

```python
import flyte
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

env = flyte.TaskEnvironment(name="my-env")


class UserSchema(pa.DataFrameModel):
    name: Series[str]
    age: Series[int] = pa.Field(ge=0, le=120)
    email: Series[str]


@env.task
async def generate_users() -> DataFrame[UserSchema]:
    return pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
        "email": ["alice@example.com", "bob@example.com"],
    })


@env.task
async def process_users(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
    df["age"] = df["age"] + 1
    return df
```

DataFrames are automatically validated against the schema on both input and output. If validation fails, a detailed error report is generated.

## Configuration

Control validation behavior using `ValidationConfig` with `typing.Annotated`:

```python
from typing import Annotated
from flyteplugins.pandera import ValidationConfig

@env.task
async def lenient_task(
    df: Annotated[DataFrame[UserSchema], ValidationConfig(on_error="warn")]
) -> DataFrame[UserSchema]:
    return df
```

- `on_error="raise"` (default): Raises an exception on validation failure
- `on_error="warn"`: Logs a warning and continues with the original data
