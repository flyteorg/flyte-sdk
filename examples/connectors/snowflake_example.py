from flyteplugins.snowflake import Snowflake, SnowflakeConfig

import flyte
from flyte.io import DataFrame

"""
Example of using Snowflake connector.

Create a snowflake database and table before running this example:
CREATE DATABASE FLYTE;
CREATE SCHEMA FLYTE.PUBLIC;
CREATE TABLE FLYTE.PUBLIC.TEST (ID INT, NAME VARCHAR, AGE INT);

NOTE: You can get the SnowflakeConfig's metadata from the Snowflake console by executing the following query:

SELECT
    CURRENT_USER() AS "User",
    CONCAT(CURRENT_ORGANIZATION_NAME(), '-', CURRENT_ACCOUNT_NAME()) AS "Account",
    CURRENT_DATABASE() AS "Database",
    CURRENT_SCHEMA() AS "Schema",
    CURRENT_WAREHOUSE() AS "Warehouse";

"""

snowflake_task = Snowflake(
    name="snowflake",
    inputs={"id": int, "name": str, "age": int},
    output_dataframe_type=DataFrame,
    plugin_config=SnowflakeConfig(
        user="KEVIN",
        account="PWGJLTH-XKB21544",
        database="FLYTE",
        schema="PUBLIC",
        warehouse="COMPUTE_WH",
    ),
    query_template="INSERT INTO FLYTE.PUBLIC.TEST (ID, NAME, AGE) VALUES (%(id)s, %(name)s, %(age)s);",
    snowflake_private_key="SNOWFLAKE",
)

flyte.TaskEnvironment.from_task("snowflake_env", snowflake_task)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(snowflake_task, id=123, name="Kevin", age=30)
    print(run.url)
