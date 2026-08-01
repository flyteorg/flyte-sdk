# Spark operator migration checks. Run against a cluster, then compare:
#   t1_smoke            — job submits and completes (any operator/client combo)
#   t2_conf_mapping     — plugin-derived limit-cores confs survive the client's key-constant renames
#   t3_security_context — pod security context reaches the driver; fails (uid 185) on old-CRD clusters
#   t4_disk             — task disk request; check executor pod ephemeral-storage via kubectl (template path only)
#   t5_executor_pod_disk — ephemeral-storage via explicit executor_pod template; expected to NO-OP today:
#                          overlay containers merge by name against a generated container name and get dropped
import asyncio
import os
from typing import cast

from flyteplugins.spark.task import Spark
from kubernetes.client import V1Container, V1PodSecurityContext, V1PodSpec, V1ResourceRequirements
from pyspark.sql import SparkSession

import flyte

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(
        name="spark",
        python_version=(3, 10),
        registry="ghcr.io/flyteorg",
        extendable=True,
        platform=("linux/amd64", "linux/arm64"),
    )
    .with_pip_packages("flyteplugins-spark", "kubernetes")
)

# keeps driver/executor pods alive long enough to inspect them with kubectl
HOLD_SECONDS = 120

base_conf = {
    "spark.driver.memory": "1000M",
    "spark.executor.memory": "1000M",
    "spark.executor.cores": "1",
    "spark.executor.instances": "2",
    "spark.driver.cores": "1",
    "spark.kubernetes.file.upload.path": "/opt/spark/work-dir",
}

spark_env = flyte.TaskEnvironment(
    name="spark_migrate",
    resources=flyte.Resources(cpu=(1, 2), memory=("800Mi", "1600Mi")),
    plugin_config=Spark(spark_conf=base_conf),
    image=image,
)

# runAsUser only reaches the pods where the SparkApplication CRD accepts pod templates
sc_env = flyte.TaskEnvironment(
    name="spark_migrate_sc",
    resources=flyte.Resources(cpu=(1, 2), memory=("800Mi", "1600Mi")),
    plugin_config=Spark(
        spark_conf=base_conf,
        driver_pod=flyte.PodTemplate(pod_spec=V1PodSpec(containers=[], security_context=V1PodSecurityContext(run_as_user=1000))),
    ),
    image=image,
)

disk_env = flyte.TaskEnvironment(
    name="spark_migrate_disk",
    resources=flyte.Resources(cpu=(1, 2), memory=("800Mi", "1600Mi"), disk="10Gi"),
    plugin_config=Spark(spark_conf=base_conf),
    image=image,
)


@spark_env.task
async def t1_smoke() -> int:
    spark = cast(SparkSession, flyte.ctx().data["spark_session"])
    total = spark.sparkContext.parallelize(range(100)).sum()
    await asyncio.sleep(HOLD_SECONDS)
    return total


@spark_env.task
async def t2_conf_mapping() -> str:
    spark = cast(SparkSession, flyte.ctx().data["spark_session"])
    driver_limit = spark.sparkContext.getConf().get("spark.kubernetes.driver.limit.cores")
    executor_limit = spark.sparkContext.getConf().get("spark.kubernetes.executor.limit.cores")
    await asyncio.sleep(HOLD_SECONDS)
    assert driver_limit == "1", f"driver limit-cores not mapped: {driver_limit}"
    assert executor_limit == "1", f"executor limit-cores not mapped: {executor_limit}"
    return f"driver={driver_limit} executor={executor_limit}"


@sc_env.task
async def t3_security_context() -> int:
    uid = os.getuid()
    await asyncio.sleep(HOLD_SECONDS)
    assert uid == 1000, f"pod template securityContext not applied, uid={uid}"
    return uid


executor_pod_disk_env = flyte.TaskEnvironment(
    name="spark_migrate_execpod_disk",
    resources=flyte.Resources(cpu=(1, 2), memory=("800Mi", "1600Mi")),
    plugin_config=Spark(
        spark_conf=base_conf,
        executor_pod=flyte.PodTemplate(
            pod_spec=V1PodSpec(
                containers=[
                    V1Container(
                        name="primary",
                        resources=V1ResourceRequirements(requests={"ephemeral-storage": "9Gi"}),
                    )
                ],
            ),
        ),
    ),
    image=image,
)


@disk_env.task
async def t4_disk() -> int:
    # kubectl: executor pods should carry a 10Gi ephemeral-storage request on template-capable clusters
    spark = cast(SparkSession, flyte.ctx().data["spark_session"])
    count = spark.range(1_000_000).repartition(4).count()
    await asyncio.sleep(HOLD_SECONDS)
    return count


@executor_pod_disk_env.task
async def t5_executor_pod_disk() -> int:
    # kubectl: check executor pods for a 9Gi ephemeral-storage request; absence = overlay container dropped
    spark = cast(SparkSession, flyte.ctx().data["spark_session"])
    count = spark.range(1_000_000).repartition(4).count()
    await asyncio.sleep(HOLD_SECONDS)
    return count


if __name__ == "__main__":
    flyte.init_from_config()
    for task in (t1_smoke, t2_conf_mapping, t3_security_context, t4_disk, t5_executor_pod_disk):
        run = flyte.with_runcontext(mode="remote").run(task)
        print(task.name, run.url)
