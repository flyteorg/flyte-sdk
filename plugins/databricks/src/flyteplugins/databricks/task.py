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
    """Configuration for a Databricks task.

    Tasks configured with this will execute natively on Databricks as a
    distributed PySpark job. Extends `Spark` with Databricks-specific
    cluster and authentication settings.

    Attributes:
        spark_conf: Spark configuration key-value pairs, e.g.
            `{"spark.executor.memory": "4g"}`.
        hadoop_conf: Hadoop configuration key-value pairs.
        executor_path: Path to the Python binary used for PySpark execution.
            Defaults to the interpreter path from the serialization context.
        applications_path: Path to the main application file. Defaults to
            the task entrypoint path.
        driver_pod: Pod template applied to the Spark driver pod.
        executor_pod: Pod template applied to the Spark executor pods.
        databricks_conf: Databricks job configuration dict compliant with
            the Databricks Jobs API v2.1 (also supports v2.0 use cases).
            Typically includes `new_cluster` or `existing_cluster_id`,
            `run_name`, and other job settings.
        databricks_instance: Domain name of your Databricks deployment,
            e.g. `"myorg.cloud.databricks.com"`.
        databricks_token: Name of the Flyte secret containing the Databricks
            API token used for authentication.
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
