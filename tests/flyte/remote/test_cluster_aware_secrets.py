"""Tests for the ClusterAwareSecretService wrapper in flyte.remote._client.controlplane."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.cluster import payload_pb2 as cluster_payload_pb2
from flyteidl2.secret import definition_pb2 as secret_definition_pb2
from flyteidl2.secret import payload_pb2 as secret_payload_pb2

from flyte.remote._client.controlplane import ClusterAwareSecretService


def _make_wrapper(
    cluster_endpoint: str = "",
    own_endpoint: str = "dns:///localhost:8090",
):
    cluster_service = MagicMock()
    cluster_service.select_cluster = AsyncMock(
        return_value=cluster_payload_pb2.SelectClusterResponse(cluster_endpoint=cluster_endpoint)
    )
    session_config = MagicMock()
    session_config.endpoint = own_endpoint
    session_config.insecure = True
    session_config.insecure_skip_verify = False
    default_client = MagicMock()
    default_client.create_secret = AsyncMock(return_value=secret_payload_pb2.CreateSecretResponse())
    default_client.update_secret = AsyncMock(return_value=secret_payload_pb2.UpdateSecretResponse())
    default_client.get_secret = AsyncMock(return_value=secret_payload_pb2.GetSecretResponse())
    default_client.list_secrets = AsyncMock(return_value=secret_payload_pb2.ListSecretsResponse())
    default_client.delete_secret = AsyncMock(return_value=secret_payload_pb2.DeleteSecretResponse())
    return (
        ClusterAwareSecretService(
            cluster_service=cluster_service,
            session_config=session_config,
            default_client=default_client,
        ),
        cluster_service,
        default_client,
    )


def _secret_id(org="o", project="p", domain="d", name="s"):
    return secret_definition_pb2.SecretIdentifier(organization=org, project=project, domain=domain, name=name)


# --- Routing: project-scoped secrets ---


@pytest.mark.asyncio
async def test_create_secret_routes_by_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = secret_payload_pb2.CreateSecretRequest(id=_secret_id())

    await wrapper.create_secret(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_USE_SECRETS
    assert sent.WhichOneof("resource") == "project_id"
    assert sent.project_id.name == "p"
    assert sent.project_id.domain == "d"
    assert sent.project_id.organization == "o"
    default_client.create_secret.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_get_secret_routes_by_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = secret_payload_pb2.GetSecretRequest(id=_secret_id())

    await wrapper.get_secret(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.WhichOneof("resource") == "project_id"
    assert sent.project_id.name == "p"
    default_client.get_secret.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_update_secret_routes_by_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = secret_payload_pb2.UpdateSecretRequest(id=_secret_id())

    await wrapper.update_secret(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.WhichOneof("resource") == "project_id"
    default_client.update_secret.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_delete_secret_routes_by_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = secret_payload_pb2.DeleteSecretRequest(id=_secret_id())

    await wrapper.delete_secret(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.WhichOneof("resource") == "project_id"
    default_client.delete_secret.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_list_secrets_routes_by_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = secret_payload_pb2.ListSecretsRequest(organization="o", project="p", domain="d")

    await wrapper.list_secrets(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.WhichOneof("resource") == "project_id"
    assert sent.project_id.name == "p"
    assert sent.project_id.domain == "d"
    default_client.list_secrets.assert_awaited_once_with(req)


# --- Routing: domain-scoped secrets ---


@pytest.mark.asyncio
async def test_get_secret_domain_only_routes_by_domain_id():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = secret_payload_pb2.GetSecretRequest(id=_secret_id(project="", domain="d"))

    await wrapper.get_secret(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.WhichOneof("resource") == "domain_id"
    assert sent.domain_id.name == "d"
    assert sent.domain_id.organization == "o"
    default_client.get_secret.assert_awaited_once_with(req)


# --- Routing: org-wide secrets ---


@pytest.mark.asyncio
async def test_get_secret_org_only_routes_by_org_id():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = secret_payload_pb2.GetSecretRequest(id=_secret_id(project="", domain=""))

    await wrapper.get_secret(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.WhichOneof("resource") == "org_id"
    assert sent.org_id.name == "o"
    default_client.get_secret.assert_awaited_once_with(req)


# --- Caching ---


@pytest.mark.asyncio
async def test_cache_hits_reuse_selected_client():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = secret_payload_pb2.GetSecretRequest(id=_secret_id())

    await wrapper.get_secret(req)
    await wrapper.get_secret(req)

    assert cluster_service.select_cluster.await_count == 1
    assert default_client.get_secret.await_count == 2


@pytest.mark.asyncio
async def test_different_projects_get_separate_cache_entries():
    wrapper, cluster_service, _ = _make_wrapper()

    await wrapper.get_secret(secret_payload_pb2.GetSecretRequest(id=_secret_id(project="p1")))
    await wrapper.get_secret(secret_payload_pb2.GetSecretRequest(id=_secret_id(project="p2")))

    assert cluster_service.select_cluster.await_count == 2


# --- Remote cluster ---


@pytest.mark.asyncio
async def test_remote_cluster_endpoint_creates_new_client():
    wrapper, cluster_service, default_client = _make_wrapper(cluster_endpoint="dns:///other:8090")

    new_client_inst = MagicMock()
    new_client_inst.get_secret = AsyncMock(return_value=secret_payload_pb2.GetSecretResponse())
    new_session_cfg = MagicMock()
    new_session_cfg.connect_kwargs.return_value = {}

    with (
        patch(
            "flyte.remote._client.controlplane.create_session_config",
            new=AsyncMock(return_value=new_session_cfg),
        ),
        patch(
            "flyte.remote._client.controlplane.SecretServiceClient",
            return_value=new_client_inst,
        ),
    ):
        req = secret_payload_pb2.GetSecretRequest(id=_secret_id())
        await wrapper.get_secret(req)
        await wrapper.get_secret(req)

    assert cluster_service.select_cluster.await_count == 1
    assert new_client_inst.get_secret.await_count == 2
    default_client.get_secret.assert_not_awaited()
