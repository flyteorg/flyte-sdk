from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from flyte._task_plugins import TaskPluginRegistry
from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.models import SerializationContext
from flyteidl2.plugins.spark_pb2 import SparkApplication, SparkJob
from flyteplugins.spark import Spark
from flyteplugins.spark.task import PysparkFunctionTask
from google.protobuf.json_format import MessageToDict


@dataclass
class Databricks(Spark):
    """
    Use this to configure a Databricks task. Task's marked with this will automatically execute
    natively onto databricks platform as a distributed execution of spark

    Args:
        databricks_conf: Databricks job configuration compliant with API version 2.1, supporting 2.0 use cases.
        For the configuration structure, visit here.https://docs.databricks.com/dev-tools/api/2.0/jobs.html#request-structure
        For updates in API 2.1, refer to: https://docs.databricks.com/en/workflows/jobs/jobs-api-updates.html
        databricks_instance: Domain name of your deployment. Use the form <account>.cloud.databricks.com.
        databricks_token: the name of the secret containing the Databricks token for authentication.
    """

    databricks_conf: Optional[Dict[str, Union[str, dict]]] = None
    databricks_instance: Optional[str] = None
    databricks_token: Optional[str] = None


class DatabricksFunctionTask(AsyncConnectorExecutorMixin, PysparkFunctionTask):
    """
    Actual Plugin that transforms the local python code for execution within a spark context
    """

    plugin_config: Databricks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "databricks"

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
            databricksConf=self.plugin_config.databricks_conf,
            databricksInstance=self.plugin_config.databricks_instance,
        )

        cfg = MessageToDict(job)
        cfg["secrets"] = {"databricks_token": self.plugin_config.databricks_token}

        return cfg


TaskPluginRegistry.register(Databricks, DatabricksFunctionTask)
