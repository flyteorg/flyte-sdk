import collections
from typing import Annotated, OrderedDict, Type, cast

import pyspark
from flyteplugins.spark.task import Spark
from pyspark.sql import SparkSession

import flyte
from flyte.io import File

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(name="spark", python_version=(3, 10), registry="ghcr.io/flyteorg")
    .with_pip_packages("flyteplugins-spark", pre=True)
    .with_pip_packages("pandas", "pyarrow")
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


def kwtypes(**kwargs) -> OrderedDict[str, Type]:
    d = collections.OrderedDict()
    for k, v in kwargs.items():
        d[k] = v
    return d


columns = kwtypes(name=str, age=int)


@spark_env.task
async def sum_of_all_ages(sd: Annotated[pyspark.sql.DataFrame, columns]) -> int:
    """
    This task computes the sum of all ages in the provided Spark DataFrame.
    """
    total_age = sd.groupBy().sum("age").collect()[0][0]
    print("Total age sum:", total_age)
    return total_age


@spark_env.task
async def create_new_remote_file(content: str) -> File:
    """
    Demonstrates File.new_remote() - creating a new remote file asynchronously.
    """
    f = File.new_remote()
    async with f.open("wb") as fh:
        await fh.write(content.encode("utf-8"))
    print(f"Created new remote file at: {f.path}")
    return f


@spark_env.task
async def dataframe_transformer() -> int:
    spark = flyte.ctx().data["spark_session"]
    spark = cast(SparkSession, spark)

    csv_data = "age,name\n10,alice\n20,bob\n30,charlie\n40,david\n50,edward\n60,frank"
    file = await create_new_remote_file(csv_data)

    file = cast(File, file)
    print("Remote file path:", file.path)

    spark_df = spark.read.csv(
        path=file.path,
        header=True,
        inferSchema=True,
    )

    return await sum_of_all_ages(spark_df)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(dataframe_transformer)
    print("run name:", run.name)
    print("run url:", run.url)

    run.wait()
