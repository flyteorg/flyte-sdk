from pathlib import Path
from typing import cast, Annotated

import pandas
import pyspark
from flytekit import kwtypes

from flyteplugins.spark.task import Spark
from pyspark.sql import SparkSession

import flyte
from flyte.io import File

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(name="spark", python_version=(3, 10), registry="ghcr.io/flyteorg")
    # .with_pip_packages("flyteplugins-spark", pre=True)
    .with_pip_packages("pandas", "pyarrow", "fastparquet", "Jinja2")
    .with_source_folder(Path(__file__).parent.parent.parent / "plugins/spark", "./spark")
    .with_env_vars({"PYTHONPATH": "./spark/src:${PYTHONPATH}"})
    .with_local_v2()
)

task_env = flyte.TaskEnvironment(
    name="sum_of_all_ages", resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)

spark_conf = Spark(
    spark_conf={
        "spark.driver.memory": "1000M",
        "spark.executor.memory": "1000M",
        "spark.executor.cores": "1",
        "spark.executor.instances": "2",
        "spark.driver.cores": "1",
        "spark.kubernetes.file.upload.path": "/opt/spark/work-dir",
        "spark.jars": "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar,https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.2/hadoop-aws-3.2.2.jar,https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar",
    },
)

spark_env = flyte.TaskEnvironment(
    name="spark_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("800Mi", "1600Mi")),
    plugin_config=spark_conf,
    image=image,
    depends_on=[task_env],
)

columns = kwtypes(name=str, age=int)


@spark_env.task
def spark_df() -> Annotated[flyte.io.DataFrame, columns]:
    """
    This task returns a Spark dataset that conforms to the defined schema.
    """
    sess = flyte.ctx().data["spark_session"]
    return flyte.io.DataFrame.from_df(
        val=sess.createDataFrame(
            [
                ("Alice", 5),
                ("Bob", 10),
                ("Charlie", 15),
            ],
            ["name", "age"],
        ),
    )


@spark_env.task
def sum_of_all_ages(sd: Annotated[flyte.io.DataFrame, columns]) -> int:
    df: pandas.DataFrame = sd.open(pandas.DataFrame).all()
    return int(df["age"].sum())


@spark_env.task
async def dataframe_transformer() -> pyspark.sql.DataFrame:
    spark = flyte.ctx().data["spark_session"]

    csv_data = "age,name\n10,alice\n20,bob\n30,charlie\n40,david\n50,edward\n60,frank"
    file = await create_new_remote_file(csv_data)
    print("Remote file path:", file.path)

    return spark.read.csv(
        path=file.path,
        header=True,
        inferSchema=False,
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local").run(dataframe_transformer)
    print("run name:", run.name)
    print("run url:", run.url)

    run.wait()
