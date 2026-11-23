from datetime import datetime, timezone

import pytest
from conftest import aws_client
from flyteidl2.core.execution_pb2 import TaskExecution

from flyteplugins.sagemaker import SageMakerHyperParameterTuningJobConnector, SageMakerJobMetadata


@pytest.mark.asyncio
async def test_get_completed_best_training_job(monkeypatch):
    client = aws_client()
    client.describe_hyper_parameter_tuning_job.return_value = {
        "HyperParameterTuningJobStatus": "Completed",
        "HyperParameterTuningJobArn": "arn:tuning",
        "HyperParameterTuningJobName": "tune",
        "BestTrainingJob": {
            "TrainingJobName": "best",
            "TrainingJobArn": "arn:best",
            "TrainingJobStatus": "Completed",
            "TunedHyperParameters": {"eta": "0.1"},
            "FinalHyperParameterTuningJobObjectiveMetric": {"MetricName": "accuracy", "Value": 0.9},
            "TrainingEndTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
        },
        "TrainingJobStatusCounters": {"Completed": 4},
        "ObjectiveStatusCounters": {"Succeeded": 4},
    }
    client.describe_training_job.return_value = {"ModelArtifacts": {"S3ModelArtifacts": "s3://bucket/model.tar.gz"}}
    connector = SageMakerHyperParameterTuningJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "tune"}, "us-west-2"))
    assert result.phase == TaskExecution.SUCCEEDED
    assert result.outputs["result"]["ModelArtifacts"]["S3ModelArtifacts"].endswith("model.tar.gz")
    assert result.outputs["result"]["BestTrainingJob"]["TunedHyperParameters"] == {"eta": "0.1"}


@pytest.mark.parametrize(
    ("state", "phase"),
    [
        ("InProgress", TaskExecution.RUNNING),
        ("Stopping", TaskExecution.RUNNING),
        ("Failed", TaskExecution.FAILED),
        ("Stopped", TaskExecution.FAILED),
        ("Deleting", TaskExecution.FAILED),
        ("DeleteFailed", TaskExecution.FAILED),
    ],
)
@pytest.mark.asyncio
async def test_phase_mapping_and_messages(monkeypatch, state, phase):
    client = aws_client()
    client.describe_hyper_parameter_tuning_job.return_value = {
        "HyperParameterTuningJobStatus": state,
        "FailureReason": "bad",
        "TrainingJobStatusCounters": {"Completed": 2, "InProgress": 1, "NonRetryableError": 1},
    }
    connector = SageMakerHyperParameterTuningJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "tune"}, "us-west-2"))
    assert result.phase == phase
    if state == "InProgress":
        assert result.message == "2 Completed / 1 InProgress / 1 Failed trials"


@pytest.mark.asyncio
async def test_completed_without_best_job(monkeypatch):
    client = aws_client()
    client.describe_hyper_parameter_tuning_job.return_value = {"HyperParameterTuningJobStatus": "Completed"}
    connector = SageMakerHyperParameterTuningJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "tune"}, "us-west-2"))
    assert result.outputs["result"]["ModelArtifacts"]["S3ModelArtifacts"] is None
    client.describe_training_job.assert_not_called()
