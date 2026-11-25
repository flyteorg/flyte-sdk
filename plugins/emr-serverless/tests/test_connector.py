import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError
from flyteidl2.connector.connector_pb2 import TaskExecutionMetadata
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.identifier_pb2 import (
    NodeExecutionIdentifier,
    TaskExecutionIdentifier,
    WorkflowExecutionIdentifier,
)
from flyteidl2.core.literals_pb2 import KeyValuePair
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct

from flyteplugins.emr_serverless import (
    EMRServerless,
    EMRServerlessConnector,
    EMRServerlessHiveJobDriver,
    EMRServerlessJobMetadata,
    EMRServerlessSparkJobDriver,
)
from flyteplugins.emr_serverless.connector import (
    _ENV_ALLOW_CREATE_APPLICATION,
    _ENV_ALLOWED_APPLICATION_IDS,
    _ENV_ALLOWED_APPLICATION_NAME_PREFIXES,
    _ENV_ALLOWED_EXECUTION_ROLE_ARNS,
    _ENV_ALLOWED_REGIONS,
    _ENV_APPLICATION_NAME_PREFIX,
    _ENV_ENTRYPOINT_BUCKET,
    FLYTE_TAGS,
    _execution_token,
)


def template(
    config: EMRServerless, *, args: list[str] | None = None, image: str = "example.invalid/flyte:v1"
) -> TaskTemplate:
    custom = Struct()
    ParseDict(config.to_dict(), custom)
    result = TaskTemplate(type="emr_serverless", custom=custom)
    result.id.name = "package.module.task"
    result.container.image = image
    result.container.args.extend(
        args
        or [
            "a0",
            "--inputs",
            "s3://bucket/inputs.pb",
            "--outputs-path",
            "s3://bucket/run/a0",
            "--tgz",
            "s3://bucket/code.tgz",
            "--dest",
            ".",
        ]
    )
    result.container.env.append(KeyValuePair(key="TASK_ENV", value="value"))
    return result


def handler() -> MagicMock:
    value = MagicMock()
    value.region = "us-west-2"
    value.client.meta.region_name = "us-west-2"
    value.find_application_by_name = AsyncMock(return_value=None)
    value.create_application = AsyncMock(return_value="newapp123")
    value.get_application = AsyncMock(
        return_value={"applicationId": "app123", "state": "STARTED", "imageConfiguration": {}}
    )
    value.update_application = AsyncMock()
    value.stop_application_if_needed = AsyncMock(return_value=True)
    value.start_application_if_needed = AsyncMock(return_value=True)
    value.start_job_run = AsyncMock(return_value="job123")
    value.get_job_run = AsyncMock(return_value={"state": "RUNNING"})
    value.cancel_job_run = AsyncMock()
    return value


def execution_metadata() -> TaskExecutionMetadata:
    return TaskExecutionMetadata(
        task_execution_id=TaskExecutionIdentifier(
            task_id={"project": "project", "domain": "development", "name": "task", "version": "v1"},
            node_execution_id=NodeExecutionIdentifier(
                node_id="n0",
                execution_id=WorkflowExecutionIdentifier(
                    project="project",
                    domain="development",
                    name="run123",
                    org="organization",
                ),
            ),
            retry_attempt=2,
        ),
        environment_variables={
            "_U_RUN_BASE": "s3://bucket/run123",
            "RUN_NAME": "run123",
            "ACTION_NAME": "a0",
        },
    )


@pytest.fixture(autouse=True)
def connector_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_ALLOWED_REGIONS, "us-west-2")
    monkeypatch.setenv(_ENV_ALLOWED_EXECUTION_ROLE_ARNS, "role")
    monkeypatch.setenv(_ENV_ALLOWED_APPLICATION_IDS, "app123,resolved123")
    monkeypatch.setenv(_ENV_ALLOWED_APPLICATION_NAME_PREFIXES, "managed-,flyte-v2-e2e-emr-")


@pytest.mark.asyncio
class TestCreate:
    async def test_operator_policy_is_required_before_aws_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        monkeypatch.delenv(_ENV_ALLOWED_EXECUTION_ROLE_ARNS)

        with pytest.raises(ValueError, match=_ENV_ALLOWED_EXECUTION_ROLE_ARNS):
            await connector.create(
                template(
                    EMRServerless(
                        execution_role_arn="role",
                        application_id="app123",
                        sync_image=False,
                    )
                ),
                "s3://bucket/run/a0",
            )

        aws.get_application.assert_not_awaited()
        aws.start_job_run.assert_not_awaited()

    async def test_operator_policy_rejects_region_and_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)

        with pytest.raises(ValueError, match="execution role"):
            await connector.create(
                template(
                    EMRServerless(
                        execution_role_arn="other-role",
                        application_id="app123",
                        sync_image=False,
                    )
                ),
                "s3://bucket/run/a0",
            )

        with pytest.raises(ValueError, match="AWS region"):
            await connector.create(
                template(
                    EMRServerless(
                        execution_role_arn="role",
                        application_id="app123",
                        region="us-east-1",
                        sync_image=False,
                    )
                ),
                "s3://bucket/run/a0",
            )

        aws.start_job_run.assert_not_awaited()

    async def test_existing_application_requires_allowlist_or_owner_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        monkeypatch.delenv(_ENV_ALLOWED_APPLICATION_IDS)
        aws.get_application.return_value = {
            "applicationId": "app123",
            "state": "STARTED",
            "tags": {},
        }
        config = EMRServerless(
            execution_role_arn="role",
            application_id="app123",
            spark_job_driver=EMRServerlessSparkJobDriver("s3://bucket/main.py"),
            sync_image=False,
        )

        with pytest.raises(ValueError, match="connector-owned"):
            await connector.create(template(config), "s3://bucket/run/a0")

        aws.start_application_if_needed.assert_not_awaited()
        aws.start_job_run.assert_not_awaited()

        aws.get_application.return_value.update(
            {
                "name": "unapproved-app",
                "tags": dict(FLYTE_TAGS),
            }
        )
        with pytest.raises(ValueError, match=_ENV_ALLOWED_APPLICATION_NAME_PREFIXES):
            await connector.create(template(config), "s3://bucket/run/a0")

        aws.get_application.return_value["name"] = "managed-app"
        result = await connector.create(template(config), "s3://bucket/run/a0")
        assert result.application_id == "app123"
        aws.start_job_run.assert_awaited_once()

    async def test_managed_tags_cannot_be_overridden(self) -> None:
        assert EMRServerlessConnector._merge_tags({"Application": "other", "ManagedBy": "other", "team": "data"}) == {
            "Application": "flyte",
            "ManagedBy": "flyte-connector",
            "team": "data",
        }

    async def test_script_spark_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = EMRServerless(
            execution_role_arn="role",
            application_id="app123",
            spark_job_driver=EMRServerlessSparkJobDriver(
                "s3://bucket/main.py",
                ["10"],
                "--conf spark.executor.memory=4g",
            ),
            spark_submit_parameters="--conf spark.driver.memory=2g",
            application_configuration=[{"classification": "spark-defaults"}],
            retry_policy={"maxAttempts": 3},
            tags={"team": "data"},
            sync_image=False,
        )
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)

        meta = await connector.create(
            template(config), "s3://bucket/run/a0", task_execution_metadata=execution_metadata()
        )

        assert meta.is_script_mode
        request = aws.start_job_run.await_args.kwargs
        assert request["application_id"] == "app123"
        assert request["retry_policy"] == {"maxAttempts": 3}
        assert request["configuration_overrides"]["applicationConfiguration"]
        params = request["job_driver"]["sparkSubmit"]["sparkSubmitParameters"]
        assert "spark.executor.memory=4g" in params
        assert "spark.driver.memory=2g" in params
        assert "spark.emr-serverless.driverEnv._U_RUN_BASE=s3://bucket/run123" in params
        assert "spark.executorEnv.TASK_ENV=value" in params
        assert len(request["client_token"]) == 64

    async def test_hive_request_has_no_spark_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        config = EMRServerless(
            execution_role_arn="role",
            application_id="app123",
            application_type="HIVE",
            hive_job_driver=EMRServerlessHiveJobDriver("SELECT 1"),
            sync_image=False,
        )
        await connector.create(template(config), "s3://bucket/run/a0", task_execution_metadata=execution_metadata())
        assert aws.start_job_run.await_args.kwargs["job_driver"] == {"hive": {"query": "SELECT 1"}}

    async def test_environment_injection_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        config = EMRServerless(
            execution_role_arn="role",
            application_id="app123",
            spark_job_driver=EMRServerlessSparkJobDriver(
                "s3://bucket/main.py",
                spark_submit_parameters="--conf spark.executor.memory=4g",
            ),
            inject_flyte_env=False,
            sync_image=False,
        )
        await connector.create(template(config), "s3://bucket/run/a0", task_execution_metadata=execution_metadata())
        params = aws.start_job_run.await_args.kwargs["job_driver"]["sparkSubmit"]["sparkSubmitParameters"]
        assert params == "--conf spark.executor.memory=4g"

    async def test_pythonic_uses_a0_code_bundle_and_image_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = EMRServerless(execution_role_arn="role", application_id="app123")
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        monkeypatch.setattr(connector, "_ensure_entrypoint_on_s3", lambda handler, config: "s3://bucket/entrypoint.py")
        task = template(config)

        meta = await connector.create(task, "s3://bucket/run/a0", task_execution_metadata=execution_metadata())

        request = aws.start_job_run.await_args.kwargs
        spark = request["job_driver"]["sparkSubmit"]
        assert spark["entryPoint"] == "s3://bucket/entrypoint.py"
        assert spark["entryPointArguments"] == list(task.container.args)
        assert spark["entryPointArguments"][0] == "a0"
        assert "--tgz" in spark["entryPointArguments"]
        assert "/usr/bin/python3.11" in spark["sparkSubmitParameters"]
        assert not meta.is_script_mode
        aws.update_application.assert_awaited_once_with(
            application_id="app123",
            image_configuration={"imageUri": "example.invalid/flyte:v1"},
        )

    async def test_started_pythonic_image_sync_is_deferred(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        aws.stop_application_if_needed.return_value = False
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        monkeypatch.setattr(
            connector,
            "_ensure_entrypoint_on_s3",
            lambda handler, config: "s3://bucket/entrypoint.py",
        )

        meta = await connector.create(
            template(EMRServerless(execution_role_arn="role", application_id="app123")),
            "s3://bucket/run/a0",
        )

        assert meta.pending_image_configuration == {"imageUri": "example.invalid/flyte:v1"}
        assert meta.pending_job_request is not None
        aws.update_application.assert_not_awaited()
        aws.start_application_if_needed.assert_not_awaited()
        aws.start_job_run.assert_not_awaited()

    async def test_pythonic_rejects_non_a0_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        with pytest.raises(ValueError, match="a0"):
            await connector.create(
                template(
                    EMRServerless(execution_role_arn="role", application_id="app123", sync_image=False),
                    args=["pyflyte-execute"],
                ),
                "s3://bucket/run/a0",
            )

    async def test_script_image_sync_requires_explicit_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        base = {
            "execution_role_arn": "role",
            "application_id": "app123",
            "spark_job_driver": EMRServerlessSparkJobDriver("s3://bucket/main.py"),
        }
        await connector.create(template(EMRServerless(**base)), "s3://bucket/run/a0")
        aws.get_application.assert_awaited_once()

        configured = EMRServerless(
            **base,
            image_configuration={"imageUri": "example.invalid/config:v1"},
        )
        await connector.create(template(configured, image="example.invalid/task:v2"), "s3://bucket/run/a0")
        aws.update_application.assert_awaited_once_with(
            application_id="app123",
            image_configuration={"imageUri": "example.invalid/task:v2"},
        )

    async def test_name_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        aws.find_application_by_name.return_value = "resolved123"
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        config = EMRServerless(
            execution_role_arn="role",
            application_id="analytics-app",
            spark_job_driver=EMRServerlessSparkJobDriver("s3://bucket/main.py"),
            sync_image=False,
        )
        meta = await connector.create(template(config), "s3://bucket/run/a0")
        assert meta.application_id == "resolved123"
        aws.find_application_by_name.assert_awaited_once_with("analytics-app")

    async def test_gated_create_and_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        config = EMRServerless(
            execution_role_arn="role",
            application_name="analytics",
            architecture="ARM64",
            initial_capacity={"DRIVER": {"workerCount": 1}},
            maximum_capacity={"cpu": "100vCPU"},
            network_configuration={"subnetIds": ["subnet-example"]},
            runtime_configuration=[{"classification": "spark"}],
            scheduler_configuration={"maxConcurrentRuns": 5},
            auto_stop_config={"enabled": True},
            image_configuration={"imageUri": "example.invalid/flyte:v1"},
        )
        monkeypatch.delenv(_ENV_ALLOW_CREATE_APPLICATION, raising=False)
        with pytest.raises(ValueError, match="disabled"):
            await connector.create(template(config), "s3://bucket/run/a0")

        monkeypatch.setenv(_ENV_ALLOW_CREATE_APPLICATION, "true")
        monkeypatch.setenv(_ENV_APPLICATION_NAME_PREFIX, "managed-")
        monkeypatch.setattr(connector, "_ensure_entrypoint_on_s3", lambda handler, config: "s3://bucket/e.py")
        meta = await connector.create(template(config), "s3://bucket/run/a0")
        assert meta.created_application
        create = aws.create_application.await_args.kwargs
        assert create["name"] == "managed-analytics"
        assert create["architecture"] == "ARM64"
        assert create["runtime_configuration"] == [{"classification": "spark"}]
        assert create["scheduler_configuration"] == {"maxConcurrentRuns": 5}
        assert create["auto_stop_config"] == {"enabled": True}

    async def test_script_application_create_does_not_infer_task_image(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        monkeypatch.setenv(_ENV_ALLOW_CREATE_APPLICATION, "true")
        config = EMRServerless(
            execution_role_arn="role",
            application_name="hive-e2e",
            application_type="HIVE",
            hive_job_driver=EMRServerlessHiveJobDriver(query="SELECT 1"),
        )

        await connector.create(
            template(config, image="example.invalid/control:v1"),
            "s3://bucket/run/a0",
        )

        assert aws.create_application.await_args.kwargs["image_configuration"] is None

    async def test_create_requires_id_or_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        config = EMRServerless(
            execution_role_arn="role",
            spark_job_driver=EMRServerlessSparkJobDriver("s3://bucket/main.py"),
        )
        with pytest.raises(ValueError, match="no application_name"):
            await connector.create(template(config), "s3://bucket/run/a0")

    async def test_cold_application_defers_exact_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        aws.start_application_if_needed.return_value = False
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        config = EMRServerless(
            execution_role_arn="role",
            application_id="app123",
            spark_job_driver=EMRServerlessSparkJobDriver("s3://bucket/main.py"),
            sync_image=False,
        )
        meta = await connector.create(
            template(config), "s3://bucket/run/a0", task_execution_metadata=execution_metadata()
        )
        assert meta.job_run_id == ""
        assert meta.pending_job_request is not None
        aws.start_job_run.assert_not_awaited()
        assert EMRServerlessJobMetadata.decode(meta.encode()) == meta
        assert isinstance(json.loads(meta.encode())["pending_job_request"]["execution_timeout_minutes"], int)


@pytest.mark.asyncio
class TestGetAndDelete:
    async def test_deferred_submit_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        aws.get_job_run.return_value = {"state": "SUCCESS", "stateDetails": "complete"}
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        request = {
            "execution_role_arn": "role",
            "job_driver": {"sparkSubmit": {"entryPoint": "s3://bucket/main.py"}},
            "client_token": "stable-token",
        }
        meta = EMRServerlessJobMetadata(
            application_id="app123",
            region="us-west-2",
            is_script_mode=True,
            pending_job_request=request,
        )
        result = await connector.get(meta)
        assert result.phase == TaskExecution.SUCCEEDED
        assert result.outputs == {}
        assert result.log_links[0].link_type == TaskLog.EXTERNAL
        aws.start_job_run.assert_awaited_once_with(application_id="app123", **request)

    async def test_deferred_image_sync_updates_then_restarts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        aws.start_application_if_needed.return_value = False
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        desired = {"imageUri": "example.invalid/flyte:v2"}
        meta = EMRServerlessJobMetadata(
            application_id="app123",
            pending_job_request={
                "execution_role_arn": "role",
                "client_token": "stable",
            },
            pending_image_configuration=desired,
        )

        result = await connector.get(meta)

        assert result.phase == TaskExecution.RUNNING
        aws.update_application.assert_awaited_once_with(
            application_id="app123",
            image_configuration=desired,
        )
        aws.start_application_if_needed.assert_awaited_once_with("app123")
        aws.start_job_run.assert_not_awaited()

    @pytest.mark.parametrize(
        "state,phase",
        [
            ("QUEUED", TaskExecution.RUNNING),
            ("PENDING", TaskExecution.RUNNING),
            ("RUNNING", TaskExecution.RUNNING),
            ("SUCCESS", TaskExecution.SUCCEEDED),
            ("FAILED", TaskExecution.FAILED),
            ("CANCELLED", TaskExecution.ABORTED),
        ],
    )
    async def test_state_mapping(self, state: str, phase: TaskExecution.Phase, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        aws.get_job_run.return_value = {"state": state, "stateDetails": "detail"}
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        result = await connector.get(EMRServerlessJobMetadata("app123", "job123", "us-west-2"))
        assert result.phase == phase
        assert "detail" in result.message

    async def test_retryable_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        aws.get_job_run.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
            "GetJobRun",
        )
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        result = await connector.get(EMRServerlessJobMetadata("app123", "job123", "us-west-2"))
        assert result.phase == TaskExecution.RETRYABLE_FAILED

    async def test_delete_before_submit_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        aws.start_application_if_needed.return_value = False
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        await connector.delete(
            EMRServerlessJobMetadata(
                application_id="app123",
                pending_job_request={"execution_role_arn": "role", "client_token": "stable"},
            )
        )
        aws.start_job_run.assert_not_awaited()
        aws.cancel_job_run.assert_not_awaited()

    async def test_delete_during_image_sync_does_not_submit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        await connector.delete(
            EMRServerlessJobMetadata(
                application_id="app123",
                pending_job_request={
                    "execution_role_arn": "role",
                    "client_token": "stable",
                },
                pending_image_configuration={"imageUri": "example.invalid/flyte:v2"},
            )
        )
        aws.start_job_run.assert_not_awaited()
        aws.cancel_job_run.assert_not_awaited()

    async def test_delete_submits_exact_request_then_cancels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        aws = handler()
        monkeypatch.setattr(connector, "_get_handler", lambda region: aws)
        request = {"execution_role_arn": "role", "client_token": "stable"}
        await connector.delete(EMRServerlessJobMetadata("app123", pending_job_request=request))
        aws.start_job_run.assert_awaited_once_with(application_id="app123", **request)
        aws.cancel_job_run.assert_awaited_once_with("app123", "job123")


class TestHelpers:
    def test_token_is_deterministic_and_request_sensitive(self) -> None:
        metadata = execution_metadata()
        assert _execution_token(metadata, {"one": 1}) == _execution_token(metadata, {"two": 2})
        assert _execution_token(None, {"one": 1}) != _execution_token(None, {"one": 2})

    def test_resource_meta_is_json_not_protobuf(self) -> None:
        meta = EMRServerlessJobMetadata("app123", "job123", "us-west-2")
        assert json.loads(meta.encode())["application_id"] == "app123"

    def test_client_region_fallback_shape(self) -> None:
        assert handler().client.meta.region_name == "us-west-2"


class TestEntrypointStaging:
    def test_bucket_resolution_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = EMRServerless(
            execution_role_arn="role",
            configuration_overrides={
                "monitoringConfiguration": {"s3MonitoringConfiguration": {"logUri": "s3://monitoring-bucket/logs"}}
            },
        )
        monkeypatch.setenv("FLYTE_AWS_S3_BUCKET", "fallback-bucket")
        monkeypatch.setenv(_ENV_ENTRYPOINT_BUCKET, "s3://explicit-bucket/prefix")
        assert EMRServerlessConnector._resolve_entrypoint_bucket(config) == "explicit-bucket"
        monkeypatch.delenv(_ENV_ENTRYPOINT_BUCKET)
        assert EMRServerlessConnector._resolve_entrypoint_bucket(config) == "monitoring-bucket"
        config.configuration_overrides = None
        assert EMRServerlessConnector._resolve_entrypoint_bucket(config) == "fallback-bucket"

    def test_content_addressed_upload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        client = MagicMock()
        client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        monkeypatch.setenv(_ENV_ENTRYPOINT_BUCKET, "staging-bucket")
        monkeypatch.setattr("flyteplugins.emr_serverless.connector.boto3.client", lambda *args, **kwargs: client)
        uri = connector._ensure_entrypoint_on_s3(
            SimpleNamespace(region="us-west-2"),
            EMRServerless(execution_role_arn="role"),
        )
        assert uri.startswith("s3://staging-bucket/flyte/emr-serverless/entrypoint-")
        put = client.put_object.call_args.kwargs
        assert put["Body"] == connector._read_entrypoint_bytes()
        assert put["ContentType"] == "text/x-python"

    def test_existing_entrypoint_is_not_uploaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connector = EMRServerlessConnector()
        client = MagicMock()
        monkeypatch.setenv(_ENV_ENTRYPOINT_BUCKET, "staging-bucket")
        monkeypatch.setattr("flyteplugins.emr_serverless.connector.boto3.client", lambda *args, **kwargs: client)
        first = connector._ensure_entrypoint_on_s3(
            SimpleNamespace(region="us-west-2"),
            EMRServerless(execution_role_arn="role"),
        )
        second = connector._ensure_entrypoint_on_s3(
            SimpleNamespace(region="us-west-2"),
            EMRServerless(execution_role_arn="role"),
        )
        assert first == second
        client.put_object.assert_not_called()
