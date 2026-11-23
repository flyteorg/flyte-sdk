from datetime import datetime, timezone

import pytest
from botocore.exceptions import ClientError
from conftest import aws_client, template
from flyteidl2.core.execution_pb2 import TaskExecution

from flyteplugins.sagemaker import SageMakerJobMetadata, SageMakerTrainingJobConnector


@pytest.mark.asyncio
async def test_create_get_delete_happy_path(monkeypatch):
    client = aws_client()
    client.create_training_job.return_value = {}
    client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "TrainingJobArn": "arn",
        "TrainingJobName": "train",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://bucket/model.tar.gz"},
        "FinalMetricDataList": [
            {
                "MetricName": "loss",
                "Value": 0.1,
                "Timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        ],
    }
    connector = SageMakerTrainingJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    metadata = await connector.create(
        template(
            connector.task_type_name,
            {
                "region": "us-west-2",
                "config": {"TrainingJobName": "train-{idempotence_token}", "ResourceConfig": {"InstanceCount": 1}},
            },
        ),
        "s3://bucket/out",
    )
    resource = await connector.get(metadata)
    assert resource.phase == TaskExecution.SUCCEEDED
    assert resource.outputs["result"]["ModelArtifacts"]["S3ModelArtifacts"].endswith("model.tar.gz")
    assert resource.outputs["result"]["FinalMetricDataList"][0]["Timestamp"].startswith("2026-01-01")
    await connector.delete(metadata)
    client.stop_training_job.assert_called_once()


@pytest.mark.parametrize(
    ("state", "phase", "message"),
    [
        ("InProgress", TaskExecution.RUNNING, "Training"),
        ("Stopping", TaskExecution.RUNNING, None),
        ("Failed", TaskExecution.FAILED, "bad"),
        ("Stopped", TaskExecution.FAILED, "stopped"),
        ("Deleting", TaskExecution.FAILED, None),
    ],
)
@pytest.mark.asyncio
async def test_state_mapping(monkeypatch, state, phase, message):
    client = aws_client()
    client.describe_training_job.return_value = {
        "TrainingJobStatus": state,
        "SecondaryStatus": message,
        "FailureReason": message,
    }
    connector = SageMakerTrainingJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    resource = await connector.get(SageMakerJobMetadata({"name": "train"}, "us-west-2"))
    assert resource.phase == phase
    assert resource.message == message


@pytest.mark.asyncio
async def test_static_duplicate_name_propagates(monkeypatch):
    client = aws_client()
    client.create_training_job.side_effect = ClientError(
        {"Error": {"Code": "ResourceInUse", "Message": "exists"}},
        "CreateTrainingJob",
    )
    connector = SageMakerTrainingJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    with pytest.raises(RuntimeError):
        await connector.create(
            template(
                connector.task_type_name,
                {"region": "us-west-2", "config": {"TrainingJobName": "static"}},
            ),
            "s3://bucket/out",
        )
