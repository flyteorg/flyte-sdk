"""
Pandera + Flyte: pandas-on-Spark (`pandera.typing.pyspark.DataFrame`).

Uses ``pyspark.pandas`` (Koalas API) with ``pandera.pyspark.DataFrameModel``.

.. note::
    ``flyteplugins-spark`` registers parquet I/O for ``pyspark.sql.DataFrame``,
    not ``pyspark.pandas.DataFrame``. This example matches the pandera typing
    surface supported by ``flyteplugins-pandera``; add a dataframe encoder for
    pandas-on-Spark if you need remote round-trips.
"""

from __future__ import annotations

import argparse
import logging

import pandera.typing.pyspark as pt
import pyspark.pandas as ps
from flyteplugins.spark.task import Spark
from pandera.pyspark import DataFrameModel, Field
from pyspark.sql import SparkSession
from typing_extensions import cast

import flyte

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(name="pandera-pyspark-pandas", python_version=(3, 10), extendable=True)
    .with_pip_packages("flyte>=2.0.9", "flyteplugins-spark", "flyteplugins-pandera", "pandera[pyspark]", pre=True)
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
    name="pandera_pyspark_pandas",
    plugin_config=spark_conf,
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class LabelSchema(DataFrameModel):
    """Pandas-on-Spark rows."""

    id: int = Field(ge=0)
    label: str = Field()


@env.task(report=True)
async def labels_ps() -> pt.DataFrame[LabelSchema]:
    # Spark session is injected by flyteplugins-spark; pandas-on-Spark uses it.
    _ = cast(SparkSession, flyte.ctx().data["spark_session"])
    return ps.DataFrame({"id": [1, 2], "label": ["a", "b"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandera + pandas-on-Spark Flyte example.")
    parser.add_argument(
        "--mode",
        choices=("local", "remote"),
        default="remote",
        help="Run tasks locally or submit to a remote Flyte cluster.",
    )
    args = parser.parse_args()

    flyte.init_from_config(log_level=logging.DEBUG)
    try:
        run = flyte.with_runcontext(args.mode).run(labels_ps)
        print(run.url)
        run.wait()
        print("pyspark.pandas pandera example finished:", run.outputs())
    except Exception as exc:
        print("Run may fail without a Flyte encoder for pyspark.pandas.DataFrame:", exc)
