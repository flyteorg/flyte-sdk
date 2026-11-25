from datetime import datetime, timezone

import pytest
from conftest import aws_client
from flyteidl2.core.execution_pb2 import TaskExecution

from flyteplugins.sagemaker import SageMakerInferenceRecommenderJobConnector, SageMakerJobMetadata


@pytest.mark.asyncio
async def test_get_completed_projects_recommendations(monkeypatch):
    client = aws_client()
    client.describe_inference_recommendations_job.return_value = {
        "Status": "COMPLETED",
        "JobArn": "arn",
        "JobName": "recommend",
        "JobType": "Default",
        "CompletionTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "InferenceRecommendations": [
            {
                "RecommendationId": "one",
                "Metrics": {"CostPerHour": 1.2, "ModelLatency": 5},
                "EndpointConfiguration": {"InstanceType": "ml.c6i.large", "InitialInstanceCount": 1},
            }
        ],
        "EndpointPerformances": [{"Metrics": {"MaxInvocations": 100}, "EndpointInfo": {"EndpointName": "existing"}}],
    }
    connector = SageMakerInferenceRecommenderJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "recommend"}, "us-west-2"))
    output = result.outputs["result"]
    assert output["InferenceRecommendations"][0]["EndpointConfiguration"]["InstanceType"] == "ml.c6i.large"
    assert output["EndpointPerformances"][0]["EndpointInfo"]["EndpointName"] == "existing"
    assert output["CompletionTime"].startswith("2026-01-01")


@pytest.mark.parametrize(
    ("state", "phase"),
    [
        ("PENDING", TaskExecution.RUNNING),
        ("IN_PROGRESS", TaskExecution.RUNNING),
        ("STOPPING", TaskExecution.RUNNING),
        ("FAILED", TaskExecution.FAILED),
        ("STOPPED", TaskExecution.FAILED),
        ("DELETING", TaskExecution.FAILED),
        ("DELETED", TaskExecution.FAILED),
    ],
)
@pytest.mark.asyncio
async def test_state_mapping(monkeypatch, state, phase):
    client = aws_client()
    client.describe_inference_recommendations_job.return_value = {
        "Status": state,
        "FailureReason": "reason",
    }
    connector = SageMakerInferenceRecommenderJobConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "recommend"}, "us-west-2"))
    assert result.phase == phase
    if phase == TaskExecution.FAILED:
        assert result.message == "reason"
