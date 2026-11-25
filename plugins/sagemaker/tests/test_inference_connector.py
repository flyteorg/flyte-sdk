import pytest
from conftest import aws_client, template
from flyteidl2.core.execution_pb2 import TaskExecution

from flyteplugins.sagemaker import SageMakerEndpointConnector, SageMakerJobMetadata


@pytest.mark.asyncio
async def test_connector_create_get_delete(monkeypatch):
    client = aws_client()
    client.describe_endpoint.return_value = {
        "EndpointStatus": "InService",
        "EndpointArn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/demo",
    }
    connector = SageMakerEndpointConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    metadata = await connector.create(
        template(
            connector.task_type_name,
            {"region": "us-west-2", "config": {"EndpointName": "demo-{idempotence_token}"}},
        ),
        "s3://bucket/out",
    )
    result = await connector.get(metadata)
    assert result.phase == TaskExecution.SUCCEEDED
    assert result.outputs["result"]["EndpointArn"].endswith("/demo")
    await connector.delete(metadata)
    client.delete_endpoint.assert_called_once()


@pytest.mark.parametrize(
    ("state", "phase"),
    [
        ("Creating", TaskExecution.RUNNING),
        ("Updating", TaskExecution.RUNNING),
        ("SystemUpdating", TaskExecution.RUNNING),
        ("RollingBack", TaskExecution.RUNNING),
        ("Failed", TaskExecution.FAILED),
        ("OutOfService", TaskExecution.FAILED),
        ("Deleting", TaskExecution.RUNNING),
    ],
)
@pytest.mark.asyncio
async def test_endpoint_state_mapping(monkeypatch, state, phase):
    client = aws_client()
    client.describe_endpoint.return_value = {"EndpointStatus": state, "FailureReason": "reason"}
    connector = SageMakerEndpointConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    result = await connector.get(SageMakerJobMetadata({"name": "demo"}, "us-west-2"))
    assert result.phase == phase
