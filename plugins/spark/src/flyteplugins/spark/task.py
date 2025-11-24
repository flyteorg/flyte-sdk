import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import flyte
from flyte import PodTemplate
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext
from flyteidl2.plugins.spark_pb2 import SparkApplication, SparkJob
from google.protobuf.json_format import MessageToDict

DEFAULT_SPARK_CONTEXT_NAME = "FlyteSpark"


@dataclass
class Spark(object):
    """
    Use this to configure a SparkContext for a your task. Task's marked with this will automatically execute
    natively onto K8s as a distributed execution of spark

    Attributes:
        spark_conf (Optional[Dict[str, str]]): Spark configuration dictionary.
        hadoop_conf (Optional[Dict[str, str]]): Hadoop configuration dictionary.
        executor_path (Optional[str]): Path to the Python binary for PySpark execution.
        applications_path (Optional[str]): Path to the main application file.
        driver_pod (Optional[PodTemplate]): Pod template for the driver pod.
        executor_pod (Optional[PodTemplate]): Pod template for the executor pods.
    """

    spark_conf: Optional[Dict[str, str]] = None
    hadoop_conf: Optional[Dict[str, str]] = None
    executor_path: Optional[str] = None
    applications_path: Optional[str] = None
    driver_pod: Optional[PodTemplate] = None
    executor_pod: Optional[PodTemplate] = None

    def __post_init__(self):
        if self.spark_conf is None:
            self.spark_conf = {}

        if self.hadoop_conf is None:
            self.hadoop_conf = {}


@dataclass(kw_only=True)
class PysparkFunctionTask(AsyncFunctionTaskTemplate):
    """
    Actual Plugin that transforms the local python code for execution within a spark context
    """

    plugin_config: Spark
    task_type: str = "spark"
    debuggable: bool = True

    async def pre(self, *args, **kwargs) -> Dict[str, Any]:
        import pyspark as _pyspark

        sess = _pyspark.sql.SparkSession.builder.appName(DEFAULT_SPARK_CONTEXT_NAME).getOrCreate()

        if flyte.ctx().is_in_cluster():
            base_dir = tempfile.mkdtemp()
            file_name = "flyte_wf"
            file_format = "zip"
            shutil.make_archive(f"{base_dir}/{file_name}", file_format, os.getcwd())
            sess.sparkContext.addPyFile(f"{base_dir}/{file_name}.{file_format}")

        return {"spark_session": sess}

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        driver_pod = self.plugin_config.driver_pod.to_k8s_pod() if self.plugin_config.driver_pod else None
        executor_pod = self.plugin_config.executor_pod.to_k8s_pod() if self.plugin_config.executor_pod else None

        job = SparkJob(
            sparkConf=self.plugin_config.spark_conf,
            hadoopConf=self.plugin_config.hadoop_conf,
            mainApplicationFile=self.plugin_config.applications_path or "local://" + sctx.get_entrypoint_path(),
            executorPath=self.plugin_config.executor_path or sctx.interpreter_path,
            mainClass="",
            applicationType=SparkApplication.PYTHON,
            driverPod=driver_pod,
            executorPod=executor_pod,
        )

        return MessageToDict(job)

    async def post(self, return_vals: Any) -> Any:
        ctx = flyte.ctx()
        if ctx and ctx.action.name == "a0":
            # Only stop the SparkSession if it was created by the parent task in the debug mode.
            # This is to make sure that the SparkSession is stopped by
            # parent action only when debugging in the interactive mode.
            # Note: The action name is always "a0" in the debug mode.
            import pyspark as _pyspark

            sess = _pyspark.sql.SparkSession.builder.appName(DEFAULT_SPARK_CONTEXT_NAME).getOrCreate()
            sess.stop()


TaskPluginRegistry.register(Spark, PysparkFunctionTask)
