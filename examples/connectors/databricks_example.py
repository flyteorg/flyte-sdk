import asyncio
import os
import random
from operator import add
from pathlib import Path

from flyteplugins.connectors.databricks.task import Databricks

import flyte.remote
from flyte.storage import S3

image = (
    # https://hub.docker.com/r/databricksruntime/python/tags
    flyte.Image.from_base("databricksruntime/standard:16.4-LTS")
    .clone(name="spark", registry="ghcr.io/flyteorg", registry_secret="docker-g")
    .with_apt_packages("git", "vim")
    .with_env_vars({"UV_PYTHON": "/databricks/python3/bin/python"})
    .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@0227af26f82353fb828d099921b15b0dffee676f#subdirectory=plugins/connectors", pre=True)
    .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@0227af26f82353fb828d099921b15b0dffee676f#subdirectory=plugins/spark", pre=True)
    # .with_pip_packages("flyteplugins-connectors", pre=True)
    # .with_source_folder(Path(__file__).parent.parent.parent / "plugins/connectors", "/opt/connectors")
    .with_local_v2()
    .with_pip_packages("nest-asyncio", "aiohttp", "click==8.2.0")
    .with_env_vars({"Hello": "World2"})
)

# image = "pingsutw/spark:e0d6d6210ccdff13475ab65483ddb9b3"

task_env = flyte.TaskEnvironment(
    name="get_pi", resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)

databricks_conf = Databricks(
    spark_conf={
        "spark.driver.memory": "2000M",
        "spark.executor.memory": "1000M",
        "spark.executor.cores": "1",
        "spark.executor.instances": "2",
        "spark.driver.cores": "1",
        "spark.kubernetes.file.upload.path": "/opt/spark/work-dir",
        "spark.jars": "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar,https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.2/hadoop-aws-3.2.2.jar,https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar",
    },
    executor_path="/databricks/python3/bin/python",
    databricks_conf={
        "run_name": "flytekit databricks plugin example",
        "new_cluster": {
            "spark_version": "13.3.x-scala2.12",
            "autoscale": {
                "min_workers": 1,
                "max_workers": 3,
            },
            "node_type_id": "m6i.large",
            "num_workers": 3,
            "aws_attributes": {
                "availability": "SPOT_WITH_FALLBACK",
                "instance_profile_arn": "arn:aws:iam::339713193121:instance-profile/databricks-demo",
                "ebs_volume_type": "GENERAL_PURPOSE_SSD",
                "ebs_volume_count": 1,
                "ebs_volume_size": 100,
                "first_on_demand": 1,
            },
        },
        # "existing_cluster_id": "1113-204018-tb9vr2fm",
        "timeout_seconds": 3600,
        "max_retries": 1,
    },
    databricks_instance="dbc-6f73e2e6-19e4.cloud.databricks.com",
    databricks_token="DATABRICKS_TOKEN",
)

databricks_env = flyte.TaskEnvironment(
    name="databricks_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("3000Mi", "5000Mi")),
    plugin_config=databricks_conf,
    image=image,
    depends_on=[task_env],
    env_vars={
        "AWS_REGION": "us-west-1",
        "AWS_DEFAULT_REGION": "us-west-1",
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_SESSION_TOKEN": os.getenv("AWS_SESSION_TOKEN"),
    }
)


def f(_):
    x = random.random() * 2 - 1
    y = random.random() * 2 - 1
    return 1 if x**2 + y**2 <= 1 else 0


@task_env.task
async def get_pi(count: int, partitions: int) -> float:
    return 4.0 * count / partitions


@databricks_env.task
async def hello_databricks_nested(partitions: int = 3) -> float:
    n = 1 * partitions
    spark = flyte.ctx().data["spark_session"]
    count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
    print("countttt:", count, flush=True)
    return await get_pi(count, partitions)


async def test_main():
    import flyte.storage as storage
    proto_str = b"".join([c async for c in storage.get_stream(path="s3://my-v2-connector/test/inputs.pb")])
    print(proto_str)

if __name__ == "__main__":
    flyte.init_from_config(storage=S3().auto())
    run = flyte.with_runcontext(
        mode="local",
        raw_data_path="s3://my-v2-connector/").run(hello_databricks_nested)
    print("run name:", run.name)
    print("run urlllll:", run.url)

    # asyncio.run(test_main())
