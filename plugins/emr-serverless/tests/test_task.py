from pathlib import Path

import pytest
from flyte._task_plugins import TaskPluginRegistry
from flyte.models import NativeInterface, SerializationContext

from flyteplugins.emr_serverless import (
    EMRServerless,
    EMRServerlessFunctionTask,
    EMRServerlessHiveJobDriver,
    EMRServerlessSparkJobDriver,
    EMRServerlessTask,
)


@pytest.fixture
def serialization_context() -> SerializationContext:
    return SerializationContext(
        project="project",
        domain="development",
        version="version",
        org="organization",
        input_path="s3://bucket/inputs.pb",
        output_path="s3://bucket/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=Path.cwd(),
    )


class TestDrivers:
    def test_spark_driver(self) -> None:
        driver = EMRServerlessSparkJobDriver(
            entry_point="s3://bucket/main.py",
            entry_point_arguments=["--count", "10"],
            spark_submit_parameters="--conf spark.executor.memory=4g",
        )
        assert driver.to_dict() == {
            "entryPoint": "s3://bucket/main.py",
            "entryPointArguments": ["--count", "10"],
            "sparkSubmitParameters": "--conf spark.executor.memory=4g",
        }

    def test_hive_driver(self) -> None:
        driver = EMRServerlessHiveJobDriver(
            query="s3://bucket/query.sql",
            init_query_file="s3://bucket/init.sql",
            parameters="--hiveconf date=2026-01-01",
        )
        assert driver.to_dict() == {
            "query": "s3://bucket/query.sql",
            "initQueryFile": "s3://bucket/init.sql",
            "parameters": "--hiveconf date=2026-01-01",
        }


class TestConfig:
    def test_pythonic_mode(self) -> None:
        config = EMRServerless(execution_role_arn="arn:aws:iam::123456789012:role/job")
        assert not config.is_script_mode

    def test_script_modes(self) -> None:
        spark = EMRServerless(
            execution_role_arn="arn:aws:iam::123456789012:role/job",
            spark_job_driver=EMRServerlessSparkJobDriver("s3://bucket/main.py"),
        )
        hive = EMRServerless(
            execution_role_arn="arn:aws:iam::123456789012:role/job",
            application_type="HIVE",
            hive_job_driver=EMRServerlessHiveJobDriver("SELECT 1"),
        )
        assert spark.get_job_driver() == {"sparkSubmit": {"entryPoint": "s3://bucket/main.py"}}
        assert hive.get_job_driver() == {"hive": {"query": "SELECT 1"}}

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"execution_role_arn": ""}, "execution_role_arn"),
            ({"execution_role_arn": "role", "application_type": "INVALID"}, "application_type"),
            ({"execution_role_arn": "role", "application_type": "HIVE"}, "hive_job_driver"),
            (
                {
                    "execution_role_arn": "role",
                    "spark_job_driver": EMRServerlessSparkJobDriver("s3://bucket/main.py"),
                    "hive_job_driver": EMRServerlessHiveJobDriver("SELECT 1"),
                },
                "Only one",
            ),
            ({"execution_role_arn": "role", "execution_timeout_minutes": 0}, "execution_timeout"),
            ({"execution_role_arn": "role", "architecture": "INVALID"}, "architecture"),
        ],
    )
    def test_validation(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            EMRServerless(**kwargs)

    def test_all_fields_round_trip(self) -> None:
        config = EMRServerless(
            execution_role_arn="role",
            application_id="app123",
            release_label="emr-7.5.0",
            application_name="analytics",
            sync_image=False,
            inject_flyte_env=False,
            spark_job_driver=EMRServerlessSparkJobDriver(
                "s3://bucket/main.py",
                ["one"],
                "--conf spark.executor.cores=2",
            ),
            spark_submit_parameters="--conf spark.driver.cores=1",
            application_configuration=[{"classification": "spark-defaults"}],
            runtime_configuration=[{"classification": "spark"}],
            scheduler_configuration={"maxConcurrentRuns": 5},
            auto_stop_config={"enabled": True, "idleTimeoutMinutes": 15},
            architecture="ARM64",
            retry_policy={"maxAttempts": 3},
            configuration_overrides={"monitoringConfiguration": {}},
            tags={"team": "data"},
            execution_timeout_minutes=120,
            initial_capacity={"DRIVER": {"workerCount": 1}},
            maximum_capacity={"cpu": "100vCPU"},
            network_configuration={"subnetIds": ["subnet-example"]},
            image_configuration={"imageUri": "example.invalid/spark:v1"},
            region="us-west-2",
            client_token="token",
        )
        assert EMRServerless.from_dict(config.to_dict()) == config

    def test_effective_configuration_merge(self) -> None:
        config = EMRServerless(
            execution_role_arn="role",
            application_configuration=[{"classification": "spark-defaults"}],
            configuration_overrides={"monitoringConfiguration": {"s3MonitoringConfiguration": {}}},
        )
        merged = config.get_effective_configuration_overrides()
        assert merged is not None
        assert merged["applicationConfiguration"] == [{"classification": "spark-defaults"}]
        assert "monitoringConfiguration" in merged


class TestTask:
    def test_public_task_alias_and_registration(self) -> None:
        assert EMRServerlessTask is EMRServerlessFunctionTask
        assert TaskPluginRegistry.find(EMRServerless) is EMRServerlessFunctionTask

    def test_pythonic_outputs_are_supported(self, serialization_context: SerializationContext) -> None:
        task = EMRServerlessTask(
            plugin_config=EMRServerless(execution_role_arn="role", application_id="app123"),
            name="python-task",
            interface=None,
            func=lambda: 1,
        )
        assert task.task_type == "emr_serverless"
        assert task.custom_config(serialization_context)["application_id"] == "app123"

    def test_script_outputs_are_rejected(self, serialization_context: SerializationContext) -> None:
        task = EMRServerlessTask(
            plugin_config=EMRServerless(
                execution_role_arn="role",
                application_id="app123",
                spark_job_driver=EMRServerlessSparkJobDriver("s3://bucket/main.py"),
            ),
            name="script-task",
            interface=NativeInterface(inputs={}, outputs={"value": int}),
            func=lambda: 1,
        )
        with pytest.raises(ValueError, match="cannot declare Flyte outputs"):
            task.custom_config(serialization_context)
