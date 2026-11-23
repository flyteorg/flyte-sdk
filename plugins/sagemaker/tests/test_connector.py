from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.interface_pb2 import TypedInterface, Variable, VariableEntry, VariableMap
from flyteidl2.core.tasks_pb2 import TaskTemplate
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct

from flyteplugins.sagemaker.connector import (
    SageMakerJobMetadata,
    SageMakerProcessingConnector,
    SageMakerTrainingConnector,
    SageMakerTransformConnector,
)

CASES = [
    (
        SageMakerTrainingConnector,
        "TrainingJobName",
        "create_training_job",
        "describe_training_job",
        "stop_training_job",
        "TrainingJobStatus",
        "training",
    ),
    (
        SageMakerProcessingConnector,
        "ProcessingJobName",
        "create_processing_job",
        "describe_processing_job",
        "stop_processing_job",
        "ProcessingJobStatus",
        "processing",
    ),
    (
        SageMakerTransformConnector,
        "TransformJobName",
        "create_transform_job",
        "describe_transform_job",
        "stop_transform_job",
        "TransformJobStatus",
        "transform",
    ),
]


def task_template(name_key: str, task_type: str, *, with_output: bool = False) -> TaskTemplate:
    custom = Struct()
    ParseDict(
        {
            "region": "us-west-2",
            "request": {
                name_key: "example-job",
                "Tags": [
                    {
                        "Key": "attempt",
                        "Value": "1",
                    }
                ],
            },
        },
        custom,
    )
    interface = TypedInterface()
    if with_output:
        output_type = LiteralType(simple=SimpleType.STRUCT)
        interface.outputs.CopyFrom(
            VariableMap(
                variables=[
                    VariableEntry(
                        key="result",
                        value=Variable(type=output_type),
                    )
                ]
            )
        )
    return TaskTemplate(type=task_type, custom=custom, interface=interface)


def client() -> MagicMock:
    value = MagicMock()
    value.meta = SimpleNamespace(region_name="us-west-2")
    return value


@pytest.mark.parametrize(
    (
        "connector_class",
        "name_key",
        "create_method",
        "describe_method",
        "_stop_method",
        "status_key",
        "_console_name",
    ),
    CASES,
)
@pytest.mark.asyncio
async def test_managed_job_create_and_success(
    monkeypatch: pytest.MonkeyPatch,
    connector_class: type,
    name_key: str,
    create_method: str,
    describe_method: str,
    _stop_method: str,
    status_key: str,
    _console_name: str,
) -> None:
    aws = client()
    getattr(aws, create_method).return_value = {}
    getattr(aws, describe_method).return_value = {
        status_key: "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://example-bucket/model.tar.gz"},
        "TrainingTimeInSeconds": Decimal(10),
        "CreationTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    connector = connector_class()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)

    metadata = await connector.create(
        task_template(name_key, connector.task_type_name, with_output=True),
        output_prefix="s3://outputs",
    )
    result = await connector.get(metadata)

    assert metadata == SageMakerJobMetadata(
        region="us-west-2",
        job_name="example-job",
        output_key="result",
    )
    getattr(aws, create_method).assert_called_once()
    assert result.phase == TaskExecution.SUCCEEDED
    assert result.outputs["result"]["job_name"] == "example-job"
    assert result.outputs["result"]["status"] == "Completed"
    assert result.log_links[0].link_type == TaskLog.EXTERNAL
    assert result.custom_info == {"job_name": "example-job"}


@pytest.mark.parametrize(
    (
        "connector_class",
        "name_key",
        "_create_method",
        "describe_method",
        "_stop_method",
        "status_key",
        "_console_name",
    ),
    CASES,
)
@pytest.mark.asyncio
async def test_managed_job_phase_mapping(
    monkeypatch: pytest.MonkeyPatch,
    connector_class: type,
    name_key: str,
    _create_method: str,
    describe_method: str,
    _stop_method: str,
    status_key: str,
    _console_name: str,
) -> None:
    aws = client()
    connector = connector_class()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)
    metadata = SageMakerJobMetadata(region="us-west-2", job_name="example-job")

    for state, phase in [
        ("InProgress", TaskExecution.RUNNING),
        ("Stopping", TaskExecution.RUNNING),
        ("Stopped", TaskExecution.ABORTED),
        ("Failed", TaskExecution.FAILED),
    ]:
        getattr(aws, describe_method).return_value = {
            status_key: state,
            "FailureReason": "reason",
        }
        result = await connector.get(metadata)
        assert result.phase == phase
        assert "reason" in result.message

    assert connector.definition.name_key == name_key


@pytest.mark.asyncio
async def test_duplicate_create_recovers_existing_job(monkeypatch: pytest.MonkeyPatch) -> None:
    aws = client()
    aws.create_training_job.side_effect = ClientError(
        {
            "Error": {
                "Code": "ResourceInUse",
                "Message": "already exists",
            }
        },
        "CreateTrainingJob",
    )
    aws.describe_training_job.return_value = {"TrainingJobStatus": "InProgress"}
    connector = SageMakerTrainingConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)

    metadata = await connector.create(
        task_template("TrainingJobName", connector.task_type_name),
        output_prefix="s3://outputs",
    )

    assert metadata.job_name == "example-job"
    aws.describe_training_job.assert_called_once_with(TrainingJobName="example-job")


@pytest.mark.asyncio
async def test_retryable_provider_error_requests_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    aws = client()
    aws.describe_training_job.side_effect = ClientError(
        {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "try again",
            }
        },
        "DescribeTrainingJob",
    )
    connector = SageMakerTrainingConnector()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)

    result = await connector.get(SageMakerJobMetadata(region="us-west-2", job_name="example-job"))

    assert result.phase == TaskExecution.RETRYABLE_FAILED


@pytest.mark.parametrize(
    (
        "connector_class",
        "name_key",
        "_create_method",
        "describe_method",
        "stop_method",
        "status_key",
        "_console_name",
    ),
    CASES,
)
@pytest.mark.asyncio
async def test_delete_stops_only_active_jobs(
    monkeypatch: pytest.MonkeyPatch,
    connector_class: type,
    name_key: str,
    _create_method: str,
    describe_method: str,
    stop_method: str,
    status_key: str,
    _console_name: str,
) -> None:
    aws = client()
    connector = connector_class()
    monkeypatch.setattr(connector, "_get_client", lambda region: aws)
    metadata = SageMakerJobMetadata(region="us-west-2", job_name="example-job")

    getattr(aws, describe_method).return_value = {status_key: "InProgress"}
    await connector.delete(metadata)
    getattr(aws, stop_method).assert_called_once_with(**{name_key: "example-job"})

    getattr(aws, stop_method).reset_mock()
    getattr(aws, describe_method).return_value = {status_key: "Completed"}
    await connector.delete(metadata)
    getattr(aws, stop_method).assert_not_called()
