"""Tests for the ClusterAwareDataProxy wrapper in flyte.remote._client.controlplane."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.cluster import payload_pb2 as cluster_payload_pb2
from flyteidl2.common import identifier_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2

from flyte.remote._client.controlplane import ClusterAwareDataProxy


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
    default_client.create_upload_location = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(signed_url="https://signed/")
    )
    default_client.upload_inputs = AsyncMock(return_value=dataproxy_service_pb2.UploadInputsResponse())
    return (
        ClusterAwareDataProxy(
            cluster_service=cluster_service,
            session_config=session_config,
            default_client=default_client,
        ),
        cluster_service,
        default_client,
    )


@pytest.mark.asyncio
async def test_create_upload_location_routes_by_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename="f")

    await wrapper.create_upload_location(req)

    cluster_service.select_cluster.assert_awaited_once()
    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_CREATE_UPLOAD_LOCATION
    assert sent.WhichOneof("resource") == "project_id"
    assert sent.project_id.name == "p"
    assert sent.project_id.domain == "d"
    assert sent.project_id.organization == "o"
    default_client.create_upload_location.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_upload_inputs_with_project_id_routes_by_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    pid = identifier_pb2.ProjectIdentifier(name="p", domain="d", organization="o")
    req = dataproxy_service_pb2.UploadInputsRequest(project_id=pid)

    await wrapper.upload_inputs(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_UPLOAD_INPUTS
    assert sent.project_id == pid
    default_client.upload_inputs.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_upload_inputs_with_run_id_routes_by_derived_project():
    wrapper, cluster_service, default_client = _make_wrapper()
    run_id = identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="r")
    req = dataproxy_service_pb2.UploadInputsRequest(run_id=run_id)

    await wrapper.upload_inputs(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.WhichOneof("resource") == "project_id"
    assert sent.project_id.name == "p"
    assert sent.project_id.domain == "d"
    assert sent.project_id.organization == "o"
    default_client.upload_inputs.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_upload_inputs_requires_id_oneof():
    wrapper, _, _ = _make_wrapper()
    with pytest.raises(ValueError):
        await wrapper.upload_inputs(dataproxy_service_pb2.UploadInputsRequest())


@pytest.mark.asyncio
async def test_cache_hits_reuse_selected_client():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename="f")

    await wrapper.create_upload_location(req)
    await wrapper.create_upload_location(req)

    assert cluster_service.select_cluster.await_count == 1
    assert default_client.create_upload_location.await_count == 2


@pytest.mark.asyncio
async def test_cache_keyed_on_operation_and_resource():
    wrapper, cluster_service, _ = _make_wrapper()

    await wrapper.create_upload_location(
        dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o")
    )
    await wrapper.upload_inputs(
        dataproxy_service_pb2.UploadInputsRequest(
            project_id=identifier_pb2.ProjectIdentifier(name="p", domain="d", organization="o"),
        )
    )
    await wrapper.create_upload_location(
        dataproxy_service_pb2.CreateUploadLocationRequest(project="p2", domain="d", org="o")
    )

    assert cluster_service.select_cluster.await_count == 3


@pytest.mark.asyncio
async def test_remote_cluster_endpoint_creates_new_client():
    wrapper, cluster_service, default_client = _make_wrapper(cluster_endpoint="dns:///other:8090")

    new_client_inst = MagicMock()
    new_client_inst.create_upload_location = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(signed_url="https://remote/")
    )
    new_session_cfg = MagicMock()
    new_session_cfg.connect_kwargs.return_value = {}

    with (
        patch(
            "flyte.remote._client.controlplane.create_session_config",
            new=AsyncMock(return_value=new_session_cfg),
        ),
        patch(
            "flyte.remote._client.controlplane.DataProxyServiceClient",
            return_value=new_client_inst,
        ),
    ):
        req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o")
        await wrapper.create_upload_location(req)
        # Cached for a subsequent call
        await wrapper.create_upload_location(req)

    assert cluster_service.select_cluster.await_count == 1
    assert new_client_inst.create_upload_location.await_count == 2
    default_client.create_upload_location.assert_not_awaited()
