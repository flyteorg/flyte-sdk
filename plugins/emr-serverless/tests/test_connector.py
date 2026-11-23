from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct

from flyteplugins.emr_serverless.connector import (
    EMRServerlessConnector,
    EMRServerlessMetadata,
)


def task_template() -> TaskTemplate:
    custom = Struct()
    ParseDict(
        {
            "region": "us-west-2",
            "request": {
                "applicationId": "app-123",
                "executionRoleArn": "arn:aws:iam::123456789012:role/emr-job",
                "jobDriver": {
                    "sparkSubmit": {
                        "entryPoint": "s3://example-bucket/jobs/main.py",
                        "entryPointArguments": ["10"],
                    }
                },
                "executionTimeoutMinutes": 10,
            },
        },
        custom,
    )
    return TaskTemplate(type="emr_serverless", custom=custom)


def client() -> MagicMock:
    value = MagicMock()
    value.meta = SimpleNamespace(region_name="us-west-2")
    return value


@pytest.mark.asyncio
async def test_cold_application_defers_submission_and_reuses_token(monkeypatch: pytest.MonkeyPatch) -> None:
    aws = client()
    aws.get_application.return_value = {"application": {"state": "STOPPED"}}
    connector = EMRServerlessConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)

    metadata = await connector.create(task_template(), output_prefix="s3://outputs")

    assert metadata.application_id == "app-123"
    assert metadata.job_run_id == ""
    assert metadata.pending_request is not None
    assert len(metadata.pending_request["clientToken"]) == 64
    assert metadata.pending_request["executionTimeoutMinutes"] == 10
    aws.start_application.assert_called_once_with(applicationId="app-123")
    aws.start_job_run.assert_not_called()

    repeated = await connector.create(task_template(), output_prefix="s3://outputs")
    assert repeated.pending_request is not None
    assert repeated.pending_request["clientToken"] == metadata.pending_request["clientToken"]

    aws.get_application.return_value = {"application": {"state": "STARTED"}}
    aws.start_job_run.return_value = {"jobRunId": "job-123"}
    aws.get_job_run.return_value = {"jobRun": {"state": "SUCCESS"}}

    first = await connector.get(metadata)
    second = await connector.get(metadata)

    assert first.phase == TaskExecution.SUCCEEDED
    assert first.outputs == {}
    assert first.log_links[0].link_type == TaskLog.EXTERNAL
    assert first.custom_info == {
        "application_id": "app-123",
        "job_run_id": "job-123",
    }
    assert second.phase == TaskExecution.SUCCEEDED
    submitted_tokens = [call.kwargs["clientToken"] for call in aws.start_job_run.call_args_list]
    assert submitted_tokens == [submitted_tokens[0], submitted_tokens[0]]


@pytest.mark.asyncio
async def test_get_waits_while_application_is_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    aws = client()
    aws.get_application.return_value = {"application": {"state": "STARTING"}}
    connector = EMRServerlessConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)
    metadata = EMRServerlessMetadata(
        application_id="app-123",
        region="us-west-2",
        pending_request={
            "applicationId": "app-123",
            "clientToken": "token",
        },
    )

    result = await connector.get(metadata)

    assert result.phase == TaskExecution.RUNNING
    aws.start_job_run.assert_not_called()


@pytest.mark.parametrize(
    ("state", "phase"),
    [
        ("PENDING", TaskExecution.RUNNING),
        ("RUNNING", TaskExecution.RUNNING),
        ("SUCCESS", TaskExecution.SUCCEEDED),
        ("FAILED", TaskExecution.FAILED),
        ("CANCELLED", TaskExecution.ABORTED),
    ],
)
@pytest.mark.asyncio
async def test_job_state_mapping(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    phase: TaskExecution.Phase,
) -> None:
    aws = client()
    aws.get_job_run.return_value = {
        "jobRun": {
            "state": state,
            "stateDetails": "details",
        }
    }
    connector = EMRServerlessConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)

    result = await connector.get(
        EMRServerlessMetadata(
            application_id="app-123",
            region="us-west-2",
            job_run_id="job-123",
        )
    )

    assert result.phase == phase
    assert "details" in result.message


@pytest.mark.asyncio
async def test_retryable_provider_error_requests_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    aws = client()
    aws.get_job_run.side_effect = ClientError(
        {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "try again",
            }
        },
        "GetJobRun",
    )
    connector = EMRServerlessConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)

    result = await connector.get(
        EMRServerlessMetadata(
            application_id="app-123",
            region="us-west-2",
            job_run_id="job-123",
        )
    )

    assert result.phase == TaskExecution.RETRYABLE_FAILED


@pytest.mark.asyncio
async def test_delete_before_submission_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    aws = client()
    aws.get_application.return_value = {"application": {"state": "STARTING"}}
    connector = EMRServerlessConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)
    metadata = EMRServerlessMetadata(
        application_id="app-123",
        region="us-west-2",
        pending_request={
            "applicationId": "app-123",
            "clientToken": "token",
        },
    )

    await connector.delete(metadata)

    aws.start_job_run.assert_not_called()
    aws.cancel_job_run.assert_not_called()


@pytest.mark.asyncio
async def test_delete_submits_idempotently_then_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    aws = client()
    aws.get_application.return_value = {"application": {"state": "STARTED"}}
    aws.start_job_run.return_value = {"jobRunId": "job-123"}
    aws.get_job_run.return_value = {"jobRun": {"state": "RUNNING"}}
    connector = EMRServerlessConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)
    metadata = EMRServerlessMetadata(
        application_id="app-123",
        region="us-west-2",
        pending_request={
            "applicationId": "app-123",
            "clientToken": "token",
        },
    )

    await connector.delete(metadata)

    aws.cancel_job_run.assert_called_once_with(
        applicationId="app-123",
        jobRunId="job-123",
    )


@pytest.mark.asyncio
async def test_delete_treats_missing_job_as_already_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    aws = client()
    aws.get_job_run.side_effect = ClientError(
        {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "not found",
            }
        },
        "GetJobRun",
    )
    connector = EMRServerlessConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)

    await connector.delete(
        EMRServerlessMetadata(
            application_id="app-123",
            region="us-west-2",
            job_run_id="job-123",
        )
    )

    aws.cancel_job_run.assert_not_called()
