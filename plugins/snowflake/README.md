# Snowflake Plugin for Flyte

Run Snowflake SQL queries as Flyte tasks with parameterized inputs, key-pair authentication, batch inserts, and DataFrame support.

## Installation

```bash
pip install flyteplugins-snowflake

# With DataFrame support (read query results as pandas DataFrames):
pip install flyteplugins-snowflake[dataframe]
```

## Quick start

```python
from flyteplugins.snowflake import Snowflake, SnowflakeConfig

import flyte

config = SnowflakeConfig(
    account="myorg-myaccount",
    user="flyte_user",
    database="ANALYTICS",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
)

query = Snowflake(
    name="count_users",
    query_template="SELECT COUNT(*) FROM users",
    plugin_config=config,
    snowflake_private_key="snowflake-pk",
)
```

## Authentication

The plugin supports Snowflake [key-pair authentication](https://docs.snowflake.com/en/user-guide/key-pair-auth). Pass secret keys via `snowflake_private_key` (and optionally `snowflake_private_key_passphrase`).

```python
task = Snowflake(
    name="my_task",
    query_template="SELECT 1",
    plugin_config=config,
    snowflake_private_key="private-key",
    snowflake_private_key_passphrase="passphrase",
    # Generates env vars: PRIVATE_KEY, PASSPHRASE
)
```

For other auth methods (password, OAuth, etc.), pass them via `connection_kwargs`:

```python
config = SnowflakeConfig(
    account="myorg-myaccount",
    user="flyte_user",
    database="ANALYTICS",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
    connection_kwargs={"password": "...", "role": "ADMIN"},
)
```

## Parameterized queries

Use `%(name)s` placeholders and typed `inputs`:

```python
lookup = Snowflake(
    name="lookup_user",
    query_template="SELECT * FROM users WHERE id = %(user_id)s",
    plugin_config=config,
    inputs={"user_id": int},
    output_dataframe_type=pd.DataFrame,
    snowflake_private_key="snowflake-pk",
)
```

## Batch inserts

Set `batch=True` to expand list inputs into multi-row `VALUES` clauses:

```python
insert_rows = Snowflake(
    name="insert_users",
    query_template="INSERT INTO users (id, name, age) VALUES (%(id)s, %(name)s, %(age)s)",
    plugin_config=config,
    inputs={"id": list[int], "name": list[str], "age": list[int]},
    snowflake_private_key="snowflake-pk",
    batch=True,
)

# Calling with id=[1,2], name=["Alice","Bob"], age=[30,25] expands to:
# INSERT INTO users (id, name, age) VALUES (%(id_0)s, %(name_0)s, %(age_0)s), (%(id_1)s, %(name_1)s, %(age_1)s)
```

## Reading results as DataFrames

Install with `pip install flyteplugins-snowflake[dataframe]`, then set `output_dataframe_type`:

```python
import pandas as pd

select_task = Snowflake(
    name="get_users",
    query_template="SELECT * FROM users",
    plugin_config=config,
    output_dataframe_type=pd.DataFrame,
    snowflake_private_key="snowflake-pk",
)
```

## Full example

```python
import pandas as pd
from flyteplugins.snowflake import Snowflake, SnowflakeConfig

import flyte

config = SnowflakeConfig(
    user="KEVIN",
    account="PWGJLTH-XKB21544",
    database="FLYTE",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
)

insert_task = Snowflake(
    name="insert_rows",
    inputs={"id": list[int], "name": list[str], "age": list[int]},
    plugin_config=config,
    query_template="INSERT INTO FLYTE.PUBLIC.TEST (ID, NAME, AGE) VALUES (%(id)s, %(name)s, %(age)s)",
    snowflake_private_key="snowflake",
    batch=True,
)

select_task = Snowflake(
    name="select_all",
    output_dataframe_type=pd.DataFrame,
    plugin_config=config,
    query_template="SELECT * FROM FLYTE.PUBLIC.TEST",
    snowflake_private_key="snowflake",
)

snowflake_env = flyte.TaskEnvironment.from_task("snowflake_env", insert_task, select_task)

env = flyte.TaskEnvironment(
    name="example_env",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-snowflake[dataframe]"),
    secrets=[flyte.Secret(key="snowflake", as_env_var="SNOWFLAKE_PRIVATE_KEY")],
    depends_on=[snowflake_env],
)


@env.task
async def main(ids: list[int], names: list[str], ages: list[int]) -> float:
    await insert_task(id=ids, name=names, age=ages)
    df = await select_task()
    return df["AGE"].mean().item()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(
        main, ids=[123, 456], names=["Kevin", "Alice"], ages=[30, 25],
    )
    print(run.url)
```
