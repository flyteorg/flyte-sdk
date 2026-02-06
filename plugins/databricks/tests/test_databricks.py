import pathlib

import pytest
from flyte import PodTemplate
from flyte.models import SerializationContext
from kubernetes.client import V1Container, V1EnvVar, V1LocalObjectReference, V1PodSpec

from flyteplugins.databricks.task import Databricks, DatabricksFunctionTask


@pytest.fixture
def serialization_context() -> SerializationContext:
    return SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )


class TestDatabricksConfig:
    def test_databricks_config_creation_minimal(self):
        """Test creating Databricks config with no parameters"""
        config = Databricks()
        assert config.databricks_conf is None
        assert config.databricks_instance is None
        assert config.spark_conf == {}
        assert config.hadoop_conf == {}

    def test_databricks_config_creation_with_databricks_conf(self):
        """Test creating Databricks config with databricks_conf"""
        databricks_conf = {
            "new_cluster": {
                "spark_version": "11.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 2,
            }
        }
        config = Databricks(databricks_conf=databricks_conf)
        assert config.databricks_conf == databricks_conf
        assert config.databricks_instance is None

    def test_databricks_config_creation_with_databricks_instance(self):
        """Test creating Databricks config with databricks_instance"""
        config = Databricks(databricks_instance="mycompany.cloud.databricks.com")
        assert config.databricks_conf is None
        assert config.databricks_instance == "mycompany.cloud.databricks.com"

    def test_databricks_config_creation_with_all_parameters(self):
        """Test creating Databricks config with all parameters"""
        databricks_conf = {
            "existing_cluster_id": "1234-567890-abcdef",
            "timeout_seconds": 3600,
        }
        spark_conf = {"spark.executor.memory": "4g"}
        hadoop_conf = {"fs.s3a.endpoint": "s3.amazonaws.com"}

        config = Databricks(
            databricks_conf=databricks_conf,
            databricks_instance="mycompany.cloud.databricks.com",
            spark_conf=spark_conf,
            hadoop_conf=hadoop_conf,
        )

        assert config.databricks_conf == databricks_conf
        assert config.databricks_instance == "mycompany.cloud.databricks.com"
        assert config.spark_conf == spark_conf
        assert config.hadoop_conf == hadoop_conf

    def test_databricks_config_inherits_from_spark(self):
        """Test that Databricks inherits from Spark"""
        from flyteplugins.spark import Spark

        config = Databricks()
        assert isinstance(config, Spark)


class TestDatabricksFunctionTask:
    def test_databricks_task_type_constant(self):
        """Test that the task type constant is set correctly"""
        config = Databricks()
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        assert task._DATABRICKS_TASK_TYPE == "databricks"

    def test_databricks_task_creation_minimal(self):
        """Test creating DatabricksFunctionTask with minimal config"""
        config = Databricks()
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        assert task.name == "test_task"
        assert task.plugin_config == config

    def test_custom_config_minimal(self, serialization_context):
        """Test custom_config with minimal Databricks configuration"""
        config = Databricks()
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        result = task.custom_config(serialization_context)

        # MessageToDict only includes non-None/non-empty fields
        assert result["mainApplicationFile"] == "local:///opt/venv/bin/runtime.py"
        assert result["executorPath"] == "/opt/venv/bin/python"
        # databricksConf and databricksInstance should not be in result when None
        assert "databricksConf" not in result
        assert "databricksInstance" not in result

    def test_custom_config_with_databricks_conf(self, serialization_context):
        """Test custom_config with databricks_conf specified"""
        databricks_conf = {
            "new_cluster": {
                "spark_version": "11.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 2,
            }
        }
        config = Databricks(databricks_conf=databricks_conf)
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        result = task.custom_config(serialization_context)

        assert result["databricksConf"] == databricks_conf
        # databricksInstance should not be in result when None
        assert "databricksInstance" not in result

    def test_custom_config_with_databricks_instance(self, serialization_context):
        """Test custom_config with databricks_instance specified"""
        config = Databricks(databricks_instance="mycompany.cloud.databricks.com")
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        result = task.custom_config(serialization_context)

        # databricksConf should not be in result when None
        assert "databricksConf" not in result
        assert result["databricksInstance"] == "mycompany.cloud.databricks.com"

    def test_custom_config_with_spark_and_hadoop_conf(self, serialization_context):
        """Test custom_config with spark_conf and hadoop_conf"""
        spark_conf = {"spark.executor.memory": "4g", "spark.executor.cores": "2"}
        hadoop_conf = {"fs.s3a.endpoint": "s3.amazonaws.com", "fs.s3a.access.key": "test-key"}

        config = Databricks(spark_conf=spark_conf, hadoop_conf=hadoop_conf)
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        result = task.custom_config(serialization_context)

        assert result["sparkConf"] == spark_conf
        assert result["hadoopConf"] == hadoop_conf

    def test_custom_config_with_custom_paths(self, serialization_context):
        """Test custom_config with custom applications_path and executor_path"""
        config = Databricks(
            applications_path="/custom/path/app.py",
            executor_path="/custom/python/bin/python3",
        )
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        result = task.custom_config(serialization_context)

        assert result["mainApplicationFile"] == "/custom/path/app.py"
        assert result["executorPath"] == "/custom/python/bin/python3"

    def test_custom_config_with_pod_templates(self, serialization_context):
        """Test custom_config with driver and executor pod templates"""
        pod_template = PodTemplate(
            pod_spec=V1PodSpec(
                containers=[V1Container(name="spark", env=[V1EnvVar(name="TEST_ENV", value="test_value")])],
                image_pull_secrets=[V1LocalObjectReference(name="regcred-test")],
            )
        )

        config = Databricks(driver_pod=pod_template, executor_pod=pod_template)
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        result = task.custom_config(serialization_context)

        assert "driverPod" in result
        assert "executorPod" in result
        assert result["driverPod"] is not None
        assert result["executorPod"] is not None

    def test_custom_config_with_all_parameters(self, serialization_context):
        """Test custom_config with all Databricks parameters"""
        databricks_conf = {
            "new_cluster": {
                "spark_version": "11.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 2,
            },
            "timeout_seconds": 3600,
        }
        spark_conf = {"spark.executor.memory": "8g"}
        hadoop_conf = {"fs.s3a.endpoint": "s3.amazonaws.com"}
        pod_template = PodTemplate(
            pod_spec=V1PodSpec(
                containers=[V1Container(name="spark", env=[V1EnvVar(name="ENV_VAR", value="value")])],
            )
        )

        config = Databricks(
            databricks_conf=databricks_conf,
            databricks_instance="mycompany.cloud.databricks.com",
            spark_conf=spark_conf,
            hadoop_conf=hadoop_conf,
            applications_path="/app/main.py",
            executor_path="/usr/bin/python3",
            driver_pod=pod_template,
            executor_pod=pod_template,
        )
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        result = task.custom_config(serialization_context)

        assert result["sparkConf"] == spark_conf
        assert result["hadoopConf"] == hadoop_conf
        assert result["mainApplicationFile"] == "/app/main.py"
        assert result["executorPath"] == "/usr/bin/python3"
        assert result["databricksConf"] == databricks_conf
        assert result["databricksInstance"] == "mycompany.cloud.databricks.com"
        assert "driverPod" in result
        assert "executorPod" in result

    def test_custom_config_without_pod_templates(self, serialization_context):
        """Test custom_config when driver_pod and executor_pod are None"""
        config = Databricks()
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        result = task.custom_config(serialization_context)

        # Should handle None pod templates gracefully - they won't be in result
        assert "driverPod" not in result
        assert "executorPod" not in result

    def test_databricks_task_inherits_from_pyspark(self):
        """Test that DatabricksFunctionTask inherits from PysparkFunctionTask"""
        from flyteplugins.spark.task import PysparkFunctionTask

        config = Databricks()
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        assert isinstance(task, PysparkFunctionTask)

    def test_databricks_task_uses_async_connector_mixin(self):
        """Test that DatabricksFunctionTask uses AsyncConnectorExecutorMixin"""
        from flyte.connectors import AsyncConnectorExecutorMixin

        config = Databricks()
        task = DatabricksFunctionTask(plugin_config=config, name="test_task", interface=None, func=lambda: None)

        # Check that it's in the MRO (Method Resolution Order)
        assert AsyncConnectorExecutorMixin in type(task).__mro__
