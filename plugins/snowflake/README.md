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
    query_template="SELECT * FROM mytable WHERE id = {{ .user_id }}",
    plugin_config=config,
    inputs={"user_id": int},
    snowflake_private_key="snowflake-private-key-secret",
)
```
