"""
Pandera + Flyte: PySpark SQL (`pandera.typing.pyspark_sql.DataFrame`).

Requires ``flyteplugins-spark`` and a Spark session in the task context (same
pattern as ``examples/plugins/spark_dataframe_example.py``).
"""

from __future__ import annotations

import logging

import pandera.typing.pyspark_sql as pt
import pyspark.sql.types as T
from flyteplugins.spark.task import Spark
from pandera.pyspark import DataFrameModel, Field
from pyspark.sql import SparkSession
from typing_extensions import cast

import flyte

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(name="pandera-pyspark-sql", python_version=(3, 10), registry="ghcr.io/flyteorg", extendable=True)
    .with_pip_packages("flyteplugins-spark", "flyteplugins-pandera", "pandera[pyspark]", pre=True)
    .with_pip_packages("pandas", "pyarrow")
)

spark_conf = Spark(
    spark_conf={
        "spark.driver.memory": "1000M",
        "spark.executor.memory": "1000M",
        "spark.executor.cores": "1",
        "spark.executor.instances": "1",
        "spark.driver.cores": "1",
    },
)

env = flyte.TaskEnvironment(
    name="pandera_pyspark_sql",
    plugin_config=spark_conf,
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class PersonSchema(DataFrameModel):
    """PySpark SQL rows."""

    name: str = Field()
    age: int = Field(ge=0)


@env.task(report=True)
async def people_from_spark() -> pt.DataFrame[PersonSchema]:
    spark = cast(SparkSession, flyte.ctx().data["spark_session"])
    data = [("Ada", 36), ("Grace", 85)]
    schema = T.StructType(
        [
            T.StructField("name", T.StringType(), False),
            T.StructField("age", T.IntegerType(), False),
        ]
    )
    return spark.createDataFrame(data, schema=schema)


if __name__ == "__main__":
    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.run(people_from_spark)
    print(run.url)
    run.wait()
    print("pyspark_sql pandera example finished:", run.outputs())
