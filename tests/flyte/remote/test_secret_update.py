"""Tests for Secret.update and Secret.create: the request carries the scope, the name and the value."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.secret import definition_pb2, payload_pb2

from flyte.remote._secret import Secret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "field"),
    [("s3cr3t", "string_value"), (b"\x00\x01", "binary_value")],
)
async def test_update_sends_generic_spec_in_project_domain_scope(value, field):
    svc = MagicMock()
    svc.update_secret = AsyncMock(return_value=payload_pb2.UpdateSecretResponse())
    client = MagicMock(secrets_service=svc)
    cfg = MagicMock(org="o", project="p", domain="d")

    with (
        patch("flyte.remote._secret.ensure_client"),
        patch("flyte.remote._secret.get_client", return_value=client),
        patch("flyte.remote._secret.get_init_config", return_value=cfg),
    ):
        await Secret.update.aio("my-secret", value)

    svc.update_secret.assert_awaited_once()
    request = svc.update_secret.await_args.kwargs["request"]
    assert request.id == definition_pb2.SecretIdentifier(organization="o", project="p", domain="d", name="my-secret")
    assert request.secret_spec.type == definition_pb2.SecretType.SECRET_TYPE_GENERIC
    assert request.secret_spec.WhichOneof("value") == field
    assert getattr(request.secret_spec, field) == value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "field"),
    [("s3cr3t", "string_value"), (b"\x00\x01", "binary_value")],
)
async def test_create_sends_generic_spec_in_project_domain_scope(value, field):
    svc = MagicMock()
    svc.create_secret = AsyncMock(return_value=payload_pb2.CreateSecretResponse())
    client = MagicMock(secrets_service=svc)
    cfg = MagicMock(org="o", project="p", domain="d")

    with (
        patch("flyte.remote._secret.ensure_client"),
        patch("flyte.remote._secret.get_client", return_value=client),
        patch("flyte.remote._secret.get_init_config", return_value=cfg),
    ):
        await Secret.create.aio("my-secret", value)

    request = svc.create_secret.await_args.kwargs["request"]
    assert request.id == definition_pb2.SecretIdentifier(organization="o", project="p", domain="d", name="my-secret")
    assert request.secret_spec.type == definition_pb2.SecretType.SECRET_TYPE_GENERIC
    assert getattr(request.secret_spec, field) == value
