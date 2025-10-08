# # Spark Example

import pandas
from flyteidl.core import types_pb2
from flyteplugins.spark.task import Spark

import flyte
from flyte._context import internal_ctx
from flyte.io import DataFrame
from plugins.spark.src.flyteplugins.spark.sd_transformers import SparkToParquetEncodingHandler

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(name="spark", python_version=(3, 10), registry="ghcr.io/flyteorg")
    .with_pip_packages("flyteplugins-spark", pre=True)
)

task_env = flyte.TaskEnvironment(
    name="sum_of_all_ages", resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)

spark_conf = Spark(
    spark_conf={
        "spark.driver.memory": "3000M",
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
    resources=flyte.Resources(cpu=(1, 2), memory=("3000Mi", "5000Mi")),
    plugin_config=spark_conf,
    image=image,
    depends_on=[task_env],
)


@task_env.task
async def sum_of_all_ages(sd: DataFrame) -> int:
    df: pandas.DataFrame = await sd.open(pandas.DataFrame).all()
    return int(df["age"].sum())


@spark_env.task
async def dataframe_transformer() -> int:
    """
    This task returns a Spark dataset that conforms to the defined schema.
    """
    ctx = internal_ctx()
    spark = ctx.data.task_context.data["spark_session"]
    h = SparkToParquetEncodingHandler()
    sd = await h.encode(
        dataframe=spark.createDataFrame(
            [
                ("Alice", 5),
                ("Bob", 10),
                ("Charlie", 15),
            ],
            ["name", "age"],
        ),
        structured_dataset_type=types_pb2.StructuredDatasetType(),
    )

    return await sum_of_all_ages(sd)


# ## Execute locally
# You can execute the code locally as if it was a normal Python script.
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(dataframe_transformer)
    print("run name:", run.name)
    print("run url:", run.url)

    run.wait()
