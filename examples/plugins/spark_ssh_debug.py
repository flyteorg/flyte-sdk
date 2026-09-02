"""Reproduce: SSH-into-task debug against a Spark task (Slack thread, ticket #6694).

Mirrors the customer's script: a Spark task launched via ``with_debugcontext`` then
``SSHDebug.connect`` to render the ssh config. Run with::

    python examples/plugins/spark_ssh_debug.py
"""

from flyteplugins.spark.task import Spark
from flyteplugins.union import with_debugcontext
from flyteplugins.union.remote import SSHDebug

import flyte
import flyte.remote

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(
        name="spark-ssh-debug",
        python_version=(3, 10),
        registry="ghcr.io/flyteorg",
        extendable=True,
        platform=("linux/amd64",),
    )
    .with_apt_packages("curl")
    .with_pip_packages("flyteplugins-spark>=2.6.9")
)

spark_conf = Spark(
    spark_conf={
        "spark.driver.cores": "1",
        "spark.driver.memory": "2000M",
        "spark.executor.cores": "1",
        "spark.executor.memory": "1000M",
        "spark.executor.instances": "1",
        "spark.kubernetes.file.upload.path": "/opt/spark/work-dir",
        "spark.eventLog.enabled": "false",
    },
)

spark_env = flyte.TaskEnvironment(
    name="spark_ssh_debug",
    resources=flyte.Resources(cpu=(1, 2), memory=("2000Mi", "3000Mi")),
    plugin_config=spark_conf,
    image=image,
    env_vars={"HOME": "/opt/spark/work-dir"},
)


@spark_env.task
async def spark_hello() -> int:
    spark = flyte.ctx().data["spark_session"]
    return spark.sparkContext.parallelize(range(1, 11), 2).sum()


if __name__ == "__main__":
    import sys

    flyte.init_from_config(sys.argv[1] if len(sys.argv) > 1 else None)
    run = with_debugcontext().run(spark_hello)
    print("run name:", run.name)
    print("run url:", run.url)
    info = SSHDebug.connect(run.name)
    print("wss url:", info.wss_url)
    print(info.ssh_config)
