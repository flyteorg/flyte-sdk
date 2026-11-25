from datetime import datetime, timezone

import pytest
from botocore.exceptions import ClientError
from conftest import aws_client, template
from flyteidl2.core.execution_pb2 import TaskExecution

from flyteplugins.sagemaker import SageMakerJobMetadata, SageMakerTransformJobConnector


@pytest.mark.asyncio
async def test_create_get_delete_happy_path(monkeypatch):
    client = aws_client()
    client.describe_transform_job.return_value = {
        "TransformJobStatus": "Completed",
        "TransformJobArn": "arn",
        "TransformJobName": "transform",
        "ModelName": "model",
        "TransformOutput": {"S3OutputPath": "s3://bucket/predictions"},
        "TransformEndTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    connector = SageMakerTransformJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    metadata = await connector.create(
        template(
            connector.task_type_name,
            {"region": "us-west-2", "config": {"TransformJobName": "transform-{idempotence_token}"}},
        ),
        "s3://bucket/out",
    )
    result = await connector.get(metadata)
    assert result.outputs["result"]["TransformOutput"]["S3OutputPath"] == "s3://bucket/predictions"
    assert result.outputs["result"]["TransformEndTime"].startswith("2026-01-01")
    await connector.delete(metadata)


@pytest.mark.parametrize(
    ("state", "phase"),
    [
        ("InProgress", TaskExecution.RUNNING),
        ("Stopping", TaskExecution.RUNNING),
        ("Failed", TaskExecution.FAILED),
        ("Stopped", TaskExecution.FAILED),
    ],
)
@pytest.mark.asyncio
async def test_phase_mapping(monkeypatch, state, phase):
    client = aws_client()
    client.describe_transform_job.return_value = {"TransformJobStatus": state, "FailureReason": "reason"}
    connector = SageMakerTransformJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "transform"}, "us-west-2"))
    assert result.phase == phase


@pytest.mark.parametrize("code", ["ResourceNotFound", "ResourceNotFoundException"])
@pytest.mark.asyncio
async def test_delete_swallows_resource_not_found(monkeypatch, code):
    client = aws_client()
    client.stop_transform_job.side_effect = ClientError(
        {"Error": {"Code": code, "Message": "missing"}},
        "StopTransformJob",
    )
    connector = SageMakerTransformJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    await connector.delete(SageMakerJobMetadata({"name": "transform"}, "us-west-2"))
