"""Tests for Secret.listall pagination (uses the scalar continuation token)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.secret import definition_pb2, payload_pb2

from flyte.remote._secret import Secret


@pytest.mark.asyncio
async def test_listall_paginates_via_scalar_token():
    # Page 1 returns a continuation token; page 2 returns an empty token (end).
    page1 = payload_pb2.ListSecretsResponse(
        secrets=[definition_pb2.Secret(), definition_pb2.Secret()], token="t1"
    )
    page2 = payload_pb2.ListSecretsResponse(secrets=[definition_pb2.Secret()], token="")
    svc = MagicMock()
    svc.list_secrets = AsyncMock(side_effect=[page1, page2])
    cfg = MagicMock(org="o", project="p", domain="d")

    with (
        patch("flyte.remote._secret.ensure_client"),
        patch("flyte.remote._secret.get_init_config", return_value=cfg),
        patch("flyte.remote._secret._secrets_service_for", AsyncMock(return_value=svc)),
    ):
        results = [s async for s in Secret.listall.aio(limit=2)]

    # All secrets across both pages are yielded — not just the first page.
    assert len(results) == 3
    # Two requests were made, and the cursor from page 1 was passed back on request 2.
    assert svc.list_secrets.await_count == 2
    assert svc.list_secrets.await_args_list[0].kwargs["request"].token == ""
    assert svc.list_secrets.await_args_list[1].kwargs["request"].token == "t1"


@pytest.mark.asyncio
async def test_listall_single_page_stops_when_token_empty():
    page = payload_pb2.ListSecretsResponse(secrets=[definition_pb2.Secret()], token="")
    svc = MagicMock()
    svc.list_secrets = AsyncMock(return_value=page)
    cfg = MagicMock(org="o", project="p", domain="d")

    with (
        patch("flyte.remote._secret.ensure_client"),
        patch("flyte.remote._secret.get_init_config", return_value=cfg),
        patch("flyte.remote._secret._secrets_service_for", AsyncMock(return_value=svc)),
    ):
        results = [s async for s in Secret.listall.aio(limit=2)]

    assert len(results) == 1
    assert svc.list_secrets.await_count == 1
