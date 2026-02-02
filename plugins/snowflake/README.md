# Snowflake Plugin for Flyte

This plugin provides Snowflake integration for Flyte, enabling you to run Snowflake queries as Flyte tasks.

## Installation

```bash
pip install flyteplugins-snowflake
```

## Usage

```python
from flyteplugins.snowflake import Snowflake, SnowflakeConfig

config = SnowflakeConfig(
    account="myaccount.us-east-1",
    user="myuser",
    database="mydb",
    schema="PUBLIC",
    warehouse="mywarehouse",
)

task = Snowflake(
    name="my_query",
    query_template="INSERT INTO FLYTE.PUBLIC.TEST (ID, NAME, AGE) VALUES (%(id)s, %(name)s, %(age)s);",
    plugin_config=config,
    inputs={"id": int, "name": str, "age": int},
    snowflake_private_key="snowflake-private-key-secret",
)
```
