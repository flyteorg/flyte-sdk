# # Spark Example
import random
from operator import add

from flyteplugins.spark.task import Spark

import flyte.remote
from flyte._context import internal_ctx

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(name="spark", python_version="3.10")
    .with_pip_packages("flyteplugins-spark")
)

task_env = flyte.TaskEnvironment(
    name="get_pi", resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)


spark_env = flyte.TaskEnvironment(
    name="spark_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")),
    plugin_config=Spark(
        spark_conf={
            "spark.driver.memory": "1000M",
            "spark.executor.memory": "1000M",
            "spark.executor.cores": "1",
            "spark.executor.instances": "2",
            "spark.driver.cores": "1",
            "spark.kubernetes.file.upload.path": "/opt/spark/work-dir",
            "spark.jars": "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar",
            # For AWS environments, you can use the following jars:
            # "spark.jars": "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.2/hadoop-aws-3.2.2.jar,https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar"
        },
        executor_path="/opt/venv/bin/python",
        applications_path="local:///opt/venv/bin/runtime.py",
    ),
    image=image,
)


def f(_):
    x = random.random() * 2 - 1
    y = random.random() * 2 - 1
    return 1 if x**2 + y**2 <= 1 else 0


@task_env.task
async def get_pi(count: int, partitions: int) -> float:
    return 4.0 * count / partitions


@spark_env.task
async def hello_spark_nested(partitions: int = 3) -> float:
    n = 1 * partitions
    ctx = internal_ctx()
    spark = ctx.data.task_context.data["spark_session"]
    count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)

    return await get_pi(count, partitions)


# ## Execute locally
# You can execute the code locally as if it was a normal Python script.

if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(hello_spark_nested)
    print("run name:", run.name)
    print("run url:", run.url)
    run.wait(run)

    action_details = flyte.remote.ActionDetails.get(run_name=run.name, name="a0")
    for log in action_details.pb2.attempts[-1].log_info:
        print(f"{log.name}: {log.uri}")
