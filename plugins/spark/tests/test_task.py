import sys
from unittest.mock import MagicMock, patch

import pytest
from flyte import PodTemplate
from flyte.models import SerializationContext
from kubernetes.client import V1Container, V1EnvVar, V1LocalObjectReference, V1PodSpec

from flyteplugins.spark.task import PysparkFunctionTask, Spark


@pytest.mark.parametrize("spark_conf,hadoop_conf", [(None, None), ({"a": "b"}, {"c": "d"})])
def test_spark_post_init(spark_conf, hadoop_conf):
    s = Spark(spark_conf=spark_conf, hadoop_conf=hadoop_conf)
    assert isinstance(s.spark_conf, dict)
    assert isinstance(s.hadoop_conf, dict)


@patch("flyteplugins.spark.task.flyte.ctx")
@patch("flyteplugins.spark.task.shutil.make_archive")
@pytest.mark.asyncio
async def test_pre_cluster(mock_make_archive, mock_ctx):
    # Patch pyspark in sys.modules
    mock_spark_session = MagicMock()
    mock_spark_context = MagicMock()
    mock_spark_session.builder.appName.return_value.getOrCreate.return_value = mock_spark_session
    mock_spark_session.sparkContext = mock_spark_context
    sys.modules["pyspark"] = MagicMock()
    sys.modules["pyspark"].sql = MagicMock()
    sys.modules["pyspark"].sql.SparkSession = mock_spark_session

    mock_ctx.return_value.is_in_cluster.return_value = True
    task = PysparkFunctionTask(plugin_config=Spark(), name="n", interface=None, func=lambda: None)
    result = await task.pre()
    assert "spark_session" in result
    mock_make_archive.assert_called()
    mock_spark_context.addPyFile.assert_called()


def test_custom_config():
    pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="foo", env=[V1EnvVar(name="hello", value="world")])],
            image_pull_secrets=[V1LocalObjectReference(name="regcred-test")],
        )
    )
    sctx = SerializationContext(
        version="123",
    )

    spark = Spark(
        spark_conf={"a": "b"},
        hadoop_conf={"c": "d"},
        applications_path="/main.py",
        executor_path="/usr/bin/python3",
        driver_pod=pod_template,
        executor_pod=pod_template,
    )
    task = PysparkFunctionTask(plugin_config=spark, name="n", interface=None, func=lambda: None)
    result = task.custom_config(sctx)
    assert result["sparkConf"] == {"a": "b"}
    assert result["hadoopConf"] == {"c": "d"}
    assert result["mainApplicationFile"] == "/main.py"
    assert result["executorPath"] == "/usr/bin/python3"
    assert "driverPod" in result
    assert "executorPod" in result

    spark = Spark()
    task1 = PysparkFunctionTask(plugin_config=spark, name="n", interface=None, func=lambda: None)
    result = task1.custom_config(sctx)
    assert result["mainApplicationFile"] == "local:///opt/venv/bin/runtime.py"
    assert result["executorPath"] == "/opt/venv/bin/python"
