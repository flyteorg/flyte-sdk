"""Tests for the LocalRunService client wiring and the routed local-run upload path."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.cluster import payload_pb2 as cluster_payload_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.workflow.local_run_service_connect import LocalRunServiceClient

from flyte.remote._client._protocols import DataProxyService, LocalRunService
from flyte.remote._client.controlplane import ClusterAwareDataProxy, Console


def _make_wrapper(select_cluster_response: cluster_payload_pb2.SelectClusterResponse | None = None):
    cluster_service = MagicMock()
    cluster_service.select_cluster = AsyncMock(
        return_value=select_cluster_response or cluster_payload_pb2.SelectClusterResponse()
    )
    session_config = MagicMock()
    session_config.endpoint = "dns:///localhost:8090"
    session_config.insecure = True
    session_config.insecure_skip_verify = False
    session_config.auth_kwargs = {}
    default_client = MagicMock()
    default_client.create_upload_location = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(signed_url="https://signed/")
    )
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
async def test_local_run_upload_routes_via_local_run_data_operation():
    """The routed upload resolves via SelectCluster's OPERATION_LOCAL_RUN_DATA keyed by
    the request's org/project/domain; an empty response means the control plane serves
    the upload — default client used, cluster ""."""
    wrapper, cluster_service, default_client = _make_wrapper()
    req = dataproxy_service_pb2.CreateUploadLocationRequest(
        org="o",
        project="p",
        domain="d",
        filename_root="local-runs/local-x/a0",
        filename="inputs.pb",
    )

    resp, cluster = await wrapper.create_local_run_upload_location(req)

    assert resp.signed_url == "https://signed/"
    assert cluster == ""
    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_LOCAL_RUN_DATA
    assert sent.WhichOneof("resource") == "project_id"
    assert sent.project_id.name == "p"
    assert sent.project_id.domain == "d"
    assert sent.project_id.organization == "o"
    default_client.create_upload_location.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_local_run_upload_same_endpoint_short_circuits_to_default_client():
    """SelectCluster returning the session's own endpoint short-circuits: default
    client, cluster "" — even when the response names a cluster."""
    wrapper, _cluster_service, default_client = _make_wrapper(
        cluster_payload_pb2.SelectClusterResponse(cluster_endpoint="dns:///localhost:8090", cluster="c1")
    )
    req = dataproxy_service_pb2.CreateUploadLocationRequest(org="o", project="p", domain="d")

    resp, cluster = await wrapper.create_local_run_upload_location(req)

    assert resp.signed_url == "https://signed/"
    assert cluster == ""
    default_client.create_upload_location.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_local_run_upload_routes_to_cluster_and_propagates_cluster_name():
    """A different endpoint builds a per-cluster session/client (cached) and the
    SelectCluster response's cluster name is propagated to the caller."""
    wrapper, cluster_service, default_client = _make_wrapper(
        cluster_payload_pb2.SelectClusterResponse(cluster_endpoint="dns:///other:8090", cluster="cluster-a")
    )

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
        req = dataproxy_service_pb2.CreateUploadLocationRequest(org="o", project="p", domain="d")
        resp, cluster = await wrapper.create_local_run_upload_location(req)
        # Cached for a subsequent call: one SelectCluster, one client build.
        resp2, cluster2 = await wrapper.create_local_run_upload_location(req)

    assert resp.signed_url == resp2.signed_url == "https://remote/"
    assert cluster == cluster2 == "cluster-a"
    assert cluster_service.select_cluster.await_count == 1
    assert new_client_inst.create_upload_location.await_count == 2
    default_client.create_upload_location.assert_not_awaited()


def test_local_run_service_client_satisfies_protocol():
    """The generated connect client provides every method the LocalRunService protocol uses."""
    for method in (
        "create_run",
        "report_actions",
        "get_run_details",
        "watch_run_details",
        "list_runs",
        "watch_actions",
        "get_action_details",
    ):
        assert hasattr(LocalRunService, method)
        assert callable(getattr(LocalRunServiceClient, method))


def test_dataproxy_protocol_includes_local_run_upload():
    assert hasattr(DataProxyService, "create_local_run_upload_location")
    assert callable(ClusterAwareDataProxy.create_local_run_upload_location)


def test_console_local_run_url():
    console = Console("dns:///example.com", insecure=False)
    assert (
        console.local_run_url(project="proj", domain="dev", run_name="local-abc")
        == "https://example.com/v2/domain/dev/project/proj/local-runs/local-abc"
    )
