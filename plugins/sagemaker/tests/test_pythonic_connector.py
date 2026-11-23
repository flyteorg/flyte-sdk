import pytest
from botocore.exceptions import ClientError
from conftest import aws_client, template
from flyteidl2.core.execution_pb2 import TaskExecution

from flyteplugins.sagemaker import (
    PythonicJobMetadata,
    SageMakerProcessingTaskConnector,
    SageMakerTrainingTaskConnector,
)


@pytest.mark.parametrize(
    ("connector_cls", "task_type", "create_method"),
    [
        (SageMakerTrainingTaskConnector, "sagemaker-training-task", "create_training_job"),
        (SageMakerProcessingTaskConnector, "sagemaker-processing-task", "create_processing_job"),
    ],
)
@pytest.mark.asyncio
async def test_create_builds_a0_request(monkeypatch, connector_cls, task_type, create_method):
    client = aws_client()
    connector = connector_cls()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    custom = {
        "execution_role_arn": "arn:aws:iam::123456789012:role/sagemaker",
        "region": "us-west-2",
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "volume_size_in_gb": 30,
        "max_runtime_in_seconds": 600,
        "environment": {"USER_ENV": "yes"},
        "tags": {"team": "ml"},
    }
    metadata = await connector.create(
        template(
            task_type,
            custom,
            image="123456789012.dkr.ecr.us-west-2.amazonaws.com/task:latest",
            args=["a0", "--outputs-path", "s3://bucket/run/outputs.pb"],
            env={"TEMPLATE_ENV": "yes"},
        ),
        "s3://bucket/run",
    )
    request = getattr(client, create_method).call_args.kwargs
    assert "a0" in str(request)
    assert request["Environment"]["FLYTE_FAIL_ON_ERROR"] == "true"
    assert request["Environment"]["USER_ENV"] == "yes"
    assert request["Tags"] == [{"Key": "team", "Value": "ml"}]
    assert metadata.output_prefix == "s3://bucket/run"


@pytest.mark.parametrize(
    ("connector_cls", "describe_method", "status_key"),
    [
        (SageMakerTrainingTaskConnector, "describe_training_job", "TrainingJobStatus"),
        (SageMakerProcessingTaskConnector, "describe_processing_job", "ProcessingJobStatus"),
    ],
)
@pytest.mark.asyncio
async def test_completed_typed_return(monkeypatch, connector_cls, describe_method, status_key):
    client = aws_client()
    getattr(client, describe_method).return_value = {status_key: "Completed"}
    connector = connector_cls()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)

    async def exists(path):
        return path.endswith("outputs.pb")

    monkeypatch.setattr("flyteplugins.sagemaker.connectors.storage.exists", exists)
    result = await connector.get(PythonicJobMetadata("job", "s3://bucket/run", "us-west-2", True))
    assert result.phase == TaskExecution.SUCCEEDED
    assert result.outputs is None


@pytest.mark.parametrize("recoverable", [True, False])
@pytest.mark.asyncio
async def test_error_pb_controls_failure_kind(monkeypatch, recoverable):
    client = aws_client()
    client.describe_training_job.return_value = {"TrainingJobStatus": "Completed"}
    connector = SageMakerTrainingTaskConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)

    async def read_error(prefix):
        return ("boom", recoverable)

    monkeypatch.setattr(connector, "_read_error", read_error)
    result = await connector.get(PythonicJobMetadata("job", "s3://bucket/run", "us-west-2", True))
    assert result.phase == (TaskExecution.RETRYABLE_FAILED if recoverable else TaskExecution.FAILED)
    assert result.message == "boom"


@pytest.mark.parametrize(
    ("connector_cls", "stop_method"),
    [
        (SageMakerTrainingTaskConnector, "stop_training_job"),
        (SageMakerProcessingTaskConnector, "stop_processing_job"),
    ],
)
@pytest.mark.asyncio
async def test_delete_cancels(monkeypatch, connector_cls, stop_method):
    client = aws_client()
    connector = connector_cls()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    await connector.delete(PythonicJobMetadata("job", "s3://bucket/run", "us-west-2", False))
    getattr(client, stop_method).assert_called_once()


@pytest.mark.parametrize("code", ["ResourceNotFound", "ResourceNotFoundException"])
@pytest.mark.asyncio
async def test_delete_swallows_missing(monkeypatch, code):
    client = aws_client()
    client.stop_processing_job.side_effect = ClientError(
        {"Error": {"Code": code, "Message": "missing"}},
        "StopProcessingJob",
    )
    connector = SageMakerProcessingTaskConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: client)
    await connector.delete(PythonicJobMetadata("job", "s3://bucket/run", "us-west-2", False))


@pytest.mark.asyncio
async def test_processing_rejects_network_isolation(monkeypatch):
    connector = SageMakerProcessingTaskConnector()
    with pytest.raises(ValueError, match="EnableNetworkIsolation"):
        await connector.create(
            template(
                connector.task_type_name,
                {
                    "execution_role_arn": "role",
                    "region": "us-west-2",
                    "network_config": {"EnableNetworkIsolation": True},
                },
                image="repo/image",
            ),
            "s3://bucket/run",
        )


@pytest.mark.parametrize("args", [[], ["python", "task.py"]])
@pytest.mark.asyncio
async def test_create_rejects_invalid_container_entrypoint(args):
    connector = SageMakerTrainingTaskConnector()
    with pytest.raises(ValueError, match=r"a0|entrypoint"):
        await connector.create(
            template(
                connector.task_type_name,
                {"execution_role_arn": "role", "region": "us-west-2"},
                image="repo/image",
                args=args,
            ),
            "s3://bucket/run",
        )
