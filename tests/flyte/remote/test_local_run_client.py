"""Tests for the LocalRunService client wiring and the dataproxy upload_metadata passthrough."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.workflow.local_run_service_connect import LocalRunServiceClient

from flyte.remote._client._protocols import DataProxyService, LocalRunService
from flyte.remote._client.controlplane import ClusterAwareDataProxy, Console


def _make_wrapper():
    cluster_service = MagicMock()
    cluster_service.select_cluster = AsyncMock()
    session_config = MagicMock()
    session_config.endpoint = "dns:///localhost:8090"
    default_client = MagicMock()
    default_client.upload_metadata = AsyncMock(
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
async def test_upload_metadata_bypasses_cluster_routing():
    """Local-run metadata uploads must never SelectCluster — always the control-plane client."""
    wrapper, cluster_service, default_client = _make_wrapper()
    req = dataproxy_service_pb2.UploadMetadataRequest(
        artifact_type=dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS,
        content_md5=b"0123456789abcdef",
    )

    resp = await wrapper.upload_metadata(req)

    assert resp.signed_url == "https://signed/"
    cluster_service.select_cluster.assert_not_called()
    cluster_service.select_cluster.assert_not_awaited()
    default_client.upload_metadata.assert_awaited_once_with(req)


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


def test_dataproxy_protocol_includes_upload_metadata():
    assert hasattr(DataProxyService, "upload_metadata")
    assert callable(ClusterAwareDataProxy.upload_metadata)


def test_console_local_run_url():
    console = Console("dns:///example.com", insecure=False)
    assert (
        console.local_run_url(project="proj", domain="dev", run_name="local-abc")
        == "https://example.com/v2/domain/dev/project/proj/local-runs/local-abc"
    )
