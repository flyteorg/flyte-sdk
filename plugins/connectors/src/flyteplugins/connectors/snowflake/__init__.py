"""
Key features:

- Parameterized SQL queries with typed inputs
- Key-pair and password-based authentication
- Returns query results as DataFrames
- Automatic links to the Snowflake query dashboard in the Flyte UI
- Query cancellation on task abort

Basic usage example:
```python
import flyte
from flyte.io import DataFrame
from flyteplugins.connectors.snowflake import Snowflake, SnowflakeConfig

config = SnowflakeConfig(
    account="myorg-myaccount",
    user="flyte_user",
    database="ANALYTICS",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
)

count_users = Snowflake(
    name="count_users",
    query_template="SELECT COUNT(*) FROM users",
    plugin_config=config,
    output_dataframe_type=DataFrame,
)

flyte.TaskEnvironment.from_task("snowflake_env", count_users)

if __name__ == "__main__":
    flyte.init_from_config()

    # Run locally (connector runs in-process, requires credentials and packages locally)
    run = flyte.with_runcontext(mode="local").run(count_users)

    # Run remotely (connector runs on the control plane)
    run = flyte.with_runcontext(mode="remote").run(count_users)

    print(run.url)
```
"""

from flyteplugins.connectors.snowflake.connector import SnowflakeConnector
from flyteplugins.connectors.snowflake.task import Snowflake, SnowflakeConfig

__all__ = ["Snowflake", "SnowflakeConfig", "SnowflakeConnector"]
