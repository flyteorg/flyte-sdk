from datetime import datetime, timezone
from decimal import Decimal

import pytest
from botocore.exceptions import ClientError
from conftest import aws_client, template
from flyteidl2.core.execution_pb2 import TaskExecution

from flyteplugins.sagemaker.connectors import BotoConnector


@pytest.mark.asyncio
async def test_connector_renders_and_returns_json_safe_response(monkeypatch):
    client = aws_client()
    client.create_model.return_value = {
        "ModelArn": "arn:aws:sagemaker:us-west-2:123456789012:model/demo",
        "CreatedAt": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "Count": Decimal(2),
        "ResponseMetadata": {"RequestId": "x"},
    }
    connector = BotoConnector()
    monkeypatch.setattr(connector, "_client", lambda service, region: client)
    metadata = await connector.create(
        template(
            "boto",
            {
                "service": "sagemaker",
                "method": "create_model",
                "region": "us-west-2",
                "config": {"ModelName": "{inputs.name}-{idempotence_token}", "Count": 1},
            },
            outputs=("result", "idempotence_token"),
        ),
        "s3://bucket/out",
        inputs={"name": "demo"},
    )
    resource = await connector.get(metadata)
    assert client.create_model.call_args.kwargs["Count"] == 1
    assert client.create_model.call_args.kwargs["ModelName"].startswith("demo-")
    assert resource.phase == TaskExecution.SUCCEEDED
    assert resource.outputs["result"]["CreatedAt"] == "2026-01-01T00:00:00+00:00"
    assert resource.outputs["result"]["Count"] == 2


@pytest.mark.parametrize(
    ("service", "method"),
    [
        ("", "create_model"),
        ("SageMaker", "create_model"),
        ("sagemaker", "_private"),
        ("sagemaker", "CreateModel"),
    ],
)
@pytest.mark.asyncio
async def test_connector_rejects_invalid_operation_names(service, method):
    with pytest.raises(ValueError):
        await BotoConnector().create(
            template(
                "boto",
                {"service": service, "method": method, "region": "us-west-2", "config": {}},
            ),
            "s3://bucket/out",
        )


@pytest.mark.parametrize(
    ("method", "arn_key"),
    [("create_model", "ModelArn"), ("create_endpoint_config", "EndpointConfigArn")],
)
@pytest.mark.asyncio
async def test_existing_entity_is_idempotent(monkeypatch, method, arn_key):
    client = aws_client()
    arn = f"arn:aws:sagemaker:us-west-2:123456789012:{method.removeprefix('create_')}/demo"
    getattr(client, method).side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": f"Cannot create already existing {arn}"}},
        method,
    )
    connector = BotoConnector()
    monkeypatch.setattr(connector, "_client", lambda service, region: client)
    metadata = await connector.create(
        template(
            "boto",
            {
                "service": "sagemaker",
                "method": method,
                "region": "us-west-2",
                "config": {"Name": "demo-{idempotence_token}"},
            },
            outputs=("result", "idempotence_token"),
        ),
        "s3://bucket/out",
    )
    assert metadata.response[arn_key] == arn
