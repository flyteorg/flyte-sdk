from datetime import datetime, timezone

import pytest
from conftest import aws_client, template
from flyteidl2.core.execution_pb2 import TaskExecution

from flyteplugins.sagemaker import SageMakerJobMetadata, SageMakerProcessingJobConnector


@pytest.mark.asyncio
async def test_create_get_delete_happy_path(monkeypatch):
    client = aws_client()
    client.create_processing_job.return_value = {}
    client.describe_processing_job.return_value = {
        "ProcessingJobStatus": "Completed",
        "ProcessingJobArn": "arn",
        "ProcessingJobName": "process",
        "ProcessingOutputConfig": {
            "Outputs": [
                {"OutputName": "s3", "S3Output": {"S3Uri": "s3://bucket/output"}},
                {"OutputName": "feature", "FeatureStoreOutput": {"FeatureGroupName": "group"}},
            ]
        },
        "ProcessingStartTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    connector = SageMakerProcessingJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    metadata = await connector.create(
        template(
            connector.task_type_name,
            {"region": "us-west-2", "config": {"ProcessingJobName": "process-{idempotence_token}"}},
        ),
        "s3://bucket/out",
    )
    resource = await connector.get(metadata)
    assert resource.phase == TaskExecution.SUCCEEDED
    assert resource.outputs["result"]["Outputs"] == [
        {"OutputName": "s3", "S3Uri": "s3://bucket/output"},
        {"OutputName": "feature", "FeatureGroupName": "group"},
    ]
    await connector.delete(metadata)
    client.stop_processing_job.assert_called_once()


@pytest.mark.parametrize(
    ("state", "phase"),
    [
        ("InProgress", TaskExecution.RUNNING),
        ("Stopping", TaskExecution.RUNNING),
        ("Completed", TaskExecution.SUCCEEDED),
        ("Failed", TaskExecution.FAILED),
        ("Stopped", TaskExecution.FAILED),
    ],
)
@pytest.mark.asyncio
async def test_phase_mapping(monkeypatch, state, phase):
    client = aws_client()
    client.describe_processing_job.return_value = {
        "ProcessingJobStatus": state,
        "FailureReason": "bad",
    }
    connector = SageMakerProcessingJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "process"}, "us-west-2"))
    assert result.phase == phase
    assert result.outputs is not None if state == "Completed" else result.outputs is None


@pytest.mark.asyncio
async def test_unknown_state_fails(monkeypatch):
    client = aws_client()
    client.describe_processing_job.return_value = {"ProcessingJobStatus": "NewState"}
    connector = SageMakerProcessingJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "process"}, "us-west-2"))
    assert result.phase == TaskExecution.FAILED
    assert "Unrecognized" in result.message
