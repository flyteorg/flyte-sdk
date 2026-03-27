"""NotebookTask with Spark — runs a notebook inside a Spark driver pod.

The Spark plugin configures the SparkContext via the K8s operator before
execution. Inside the notebook, create the session directly:

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("FlyteSpark").getOrCreate()

The notebook runs in its own kernel subprocess, so the SparkSession cannot
be passed through the Flyte context as it can with regular @task functions.
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyteplugins.spark import Spark

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="spark_example",
    image=flyte.Image.from_base("apache/spark:3.5.8-python3")
    .clone(
        name="spark-example",
        extendable=True,
    )
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        )
    )
    .with_pip_packages("flyteplugins-spark")
    .with_pip_packages("pandas", "pyarrow"),
)

spark_notebook = NotebookTask(
    name="spark_analysis",
    notebook_path="notebooks/spark_analysis.ipynb",
    task_environment=env,
    plugin_config=Spark(
        spark_conf={
            "spark.driver.memory": "1000M",
            "spark.executor.memory": "1000M",
            "spark.executor.cores": "1",
            "spark.executor.instances": "2",
            "spark.driver.cores": "1",
            "spark.kubernetes.file.upload.path": "/opt/spark/work-dir",
            "spark.jars": "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar,https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.2/hadoop-aws-3.2.2.jar,https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar",
        },
    ),
    inputs={"data": list},
    outputs={"total": int, "count": int},
)


@env.task
def spark_workflow(data: list = [1, 2, 3, 4, 5]) -> tuple[int, int]:
    total, count = spark_notebook(data=data)
    return total, count


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(spark_workflow)

    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
