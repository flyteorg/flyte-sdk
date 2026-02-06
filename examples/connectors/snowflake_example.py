import pandas as pd
from flyteplugins.snowflake import Snowflake, SnowflakeConfig

import flyte

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

sf_config = SnowflakeConfig(
    user="KEVIN",
    account="PWGJLTH-XKB21544",
    database="FLYTE",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
)

# Batch insert: accepts list inputs and inserts multiple rows
snowflake_insert_task = Snowflake(
    name="snowflake",
    inputs={"id": list[int], "name": list[str], "age": list[int]},
    plugin_config=sf_config,
    query_template="INSERT INTO FLYTE.PUBLIC.TEST (ID, NAME, AGE) VALUES (%(id)s, %(name)s, %(age)s);",
    snowflake_private_key="snowflake",
    batch=True,
)

snowflake_select_task = Snowflake(
    name="snowflake_select",
    output_dataframe_type=pd.DataFrame,
    plugin_config=sf_config,
    query_template="SELECT * FROM FLYTE.PUBLIC.TEST;",
    snowflake_private_key="snowflake",
)

snowflake_env = flyte.TaskEnvironment.from_task("snowflake_env", snowflake_insert_task, snowflake_select_task)

env = flyte.TaskEnvironment(
    name="snowflake_example_env",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-snowflake"),
    secrets=[flyte.Secret(key="snowflake", as_env_var="SNOWFLAKE_PRIVATE_KEY")],
    depends_on=[snowflake_env],
)


@env.task
async def downstream_task(df: pd.DataFrame) -> float:
    mean_age = df["AGE"].mean()
    return mean_age.item()


@env.task
async def main(ids: list[int], names: list[str], ages: list[int]) -> float:
    await snowflake_insert_task(id=ids, name=names, age=ages)
    df = await snowflake_select_task()
    return await downstream_task(df)

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local").run(main, ids=[123, 456], names=["Kevin", "Alice"], ages=[30, 25])
    print(run.url)
