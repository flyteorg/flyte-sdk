"""Tests for the ClusterAwareImageService wrapper in flyte.remote._client.controlplane."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.cluster import payload_pb2 as cluster_payload_pb2
from flyteidl2.common import identifier_pb2
from flyteidl2.imagebuilder import definition_pb2 as image_definition_pb2
from flyteidl2.imagebuilder import payload_pb2 as image_payload_pb2

from flyte.remote._client.controlplane import ClusterAwareImageService

# The cluster-aware image wrapper relies on OPERATION_GET_IMAGE, which was added
# in a newer flyteidl2. Skip the suite when the installed flyteidl2 predates that
# change so CI passes until the dependency is bumped.
_select_cluster_ops = set(cluster_payload_pb2.SelectClusterRequest.Operation.DESCRIPTOR.values_by_name)
pytestmark = pytest.mark.skipif(
    "OPERATION_GET_IMAGE" not in _select_cluster_ops,
    reason="Installed flyteidl2 lacks SelectClusterRequest OPERATION_GET_IMAGE.",
)


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
    session_config.auth_kwargs = {}
    default_client = MagicMock()
    default_client.get_image = AsyncMock(return_value=image_payload_pb2.GetImageResponse())
    return (
        ClusterAwareImageService(
            cluster_service=cluster_service,
            session_config=session_config,
            default_client=default_client,
        ),
        cluster_service,
        default_client,
    )


def _get_image_request(org="o", project="p", domain="d", name="img:tag"):
    return image_payload_pb2.GetImageRequest(
        id=image_definition_pb2.ImageIdentifier(name=name),
        organization=org,
        project_id=identifier_pb2.ProjectIdentifier(organization=org, domain=domain, name=project),
    )


# --- Routing ---


@pytest.mark.asyncio
async def test_get_image_routes_by_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = _get_image_request()

    await wrapper.get_image(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_GET_IMAGE
    assert sent.WhichOneof("resource") == "project_id"
    assert sent.project_id.name == "p"
    assert sent.project_id.domain == "d"
    assert sent.project_id.organization == "o"
    default_client.get_image.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_get_image_without_project_routes_by_org_id():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = image_payload_pb2.GetImageRequest(
        id=image_definition_pb2.ImageIdentifier(name="img:tag"),
        organization="o",
    )

    await wrapper.get_image(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_GET_IMAGE
    assert sent.WhichOneof("resource") == "org_id"
    assert sent.org_id.name == "o"
    default_client.get_image.assert_awaited_once_with(req)


# --- Endpoint selection ---


@pytest.mark.asyncio
async def test_empty_endpoint_uses_default_client():
    wrapper, _, default_client = _make_wrapper(cluster_endpoint="")

    await wrapper.get_image(_get_image_request())

    default_client.get_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_own_endpoint_uses_default_client():
    wrapper, _, default_client = _make_wrapper(
        cluster_endpoint="dns:///localhost:8090",
        own_endpoint="dns:///localhost:8090",
    )

    await wrapper.get_image(_get_image_request())

    default_client.get_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_different_endpoint_builds_per_cluster_client():
    wrapper, _, default_client = _make_wrapper(cluster_endpoint="https://cluster-1.dp.example.com")

    new_cfg = MagicMock()
    new_cfg.connect_kwargs.return_value = {}
    per_cluster_client = MagicMock()
    per_cluster_client.get_image = AsyncMock(return_value=image_payload_pb2.GetImageResponse())

    with (
        patch(
            "flyte.remote._client.controlplane.create_session_config",
            new=AsyncMock(return_value=new_cfg),
        ) as create_cfg,
        patch(
            "flyte.remote._client.controlplane.ImageServiceClient",
            return_value=per_cluster_client,
        ),
    ):
        await wrapper.get_image(_get_image_request())

    assert create_cfg.await_args[0][0] == "https://cluster-1.dp.example.com"
    per_cluster_client.get_image.assert_awaited_once()
    default_client.get_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_select_cluster_failure_raises_runtime_error():
    wrapper, cluster_service, _ = _make_wrapper()
    cluster_service.select_cluster = AsyncMock(side_effect=Exception("boom"))

    with pytest.raises(RuntimeError, match="SelectCluster failed for OPERATION_GET_IMAGE"):
        await wrapper.get_image(_get_image_request())


# --- Caching ---


@pytest.mark.asyncio
async def test_cache_hits_reuse_selected_client():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = _get_image_request()

    await wrapper.get_image(req)
    await wrapper.get_image(req)

    assert cluster_service.select_cluster.await_count == 1
    assert default_client.get_image.await_count == 2


@pytest.mark.asyncio
async def test_different_projects_get_separate_cache_entries():
    wrapper, cluster_service, _ = _make_wrapper()

    await wrapper.get_image(_get_image_request(project="p1"))
    await wrapper.get_image(_get_image_request(project="p2"))

    assert cluster_service.select_cluster.await_count == 2
