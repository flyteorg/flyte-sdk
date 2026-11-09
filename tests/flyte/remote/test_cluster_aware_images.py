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


@pytest.mark.asyncio
async def test_get_image_routes_by_the_run_queue():
    """Inside a run sent to a queue, the lookup asks a cluster of that queue's pool: its builds may
    push to another registry than the project/domain default."""
    from flyte._upload_queue import upload_queue

    wrapper, cluster_service, default_client = _make_wrapper()
    req = _get_image_request()

    with upload_queue("shared-gpu"):
        await wrapper.get_image(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_GET_IMAGE
    assert sent.WhichOneof("resource") == "queue_id"
    assert (sent.queue_id.organization, sent.queue_id.project, sent.queue_id.domain, sent.queue_id.name) == (
        "o",
        "p",
        "d",
        "shared-gpu",
    )
    default_client.get_image.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_get_image_by_queue_falls_back_only_when_queue_routing_is_unsupported():
    """A backend that cannot route by queue (UNIMPLEMENTED) still answers, routed by project, and
    the fallback is not cached under the queue: the next lookup asks by queue again."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from flyte._upload_queue import upload_queue

    wrapper, cluster_service, default_client = _make_wrapper()
    calls = []

    async def select(req):
        calls.append(req.WhichOneof("resource"))
        if req.WhichOneof("resource") == "queue_id":
            raise ConnectError(Code.UNIMPLEMENTED, "queue_id not supported")
        return cluster_payload_pb2.SelectClusterResponse(cluster_endpoint="")

    cluster_service.select_cluster = AsyncMock(side_effect=select)

    with upload_queue("q"):
        await wrapper.get_image(_get_image_request())
        await wrapper.get_image(_get_image_request())

    assert calls == ["queue_id", "project_id", "queue_id"], "project routing is cached, the queue is asked again"
    assert default_client.get_image.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("code_name", ["PERMISSION_DENIED", "UNAUTHENTICATED"])
async def test_get_image_by_queue_does_not_hide_authorization_failures(code_name):
    """An unauthorized queue-routed SelectCluster is not silently rerouted to the project's cluster."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from flyte._upload_queue import upload_queue

    wrapper, cluster_service, default_client = _make_wrapper()
    cluster_service.select_cluster = AsyncMock(side_effect=ConnectError(getattr(Code, code_name), "no"))

    with upload_queue("q"), pytest.raises(RuntimeError):
        await wrapper.get_image(_get_image_request())
    assert cluster_service.select_cluster.await_count == 1, "no project fallback"
    default_client.get_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_image_by_queue_falls_back_for_an_unknown_queue():
    """A backend that does not know the queue (NOT_FOUND) is asked by project, as before queue routing."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from flyte._upload_queue import upload_queue

    wrapper, cluster_service, default_client = _make_wrapper()
    calls = []

    async def select(req):
        calls.append(req.WhichOneof("resource"))
        if req.WhichOneof("resource") == "queue_id":
            raise ConnectError(Code.NOT_FOUND, "no such queue")
        return cluster_payload_pb2.SelectClusterResponse(cluster_endpoint="")

    cluster_service.select_cluster = AsyncMock(side_effect=select)
    with upload_queue("q"):
        await wrapper.get_image(_get_image_request())
    assert calls == ["queue_id", "project_id"]
    default_client.get_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_image_by_queue_retries_a_transient_failure(monkeypatch):
    """A transient failure is retried by queue; if it persists the lookup falls back to project routing
    (uncached), so an existing image is not reported missing and rebuilt."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    import flyte.remote._client.controlplane as cp
    from flyte._upload_queue import upload_queue

    monkeypatch.setattr(cp, "_QUEUE_ROUTING_RETRY_DELAYS", (0, 0))
    wrapper, cluster_service, _default_client = _make_wrapper()
    calls = []

    async def flaky_once(req):
        calls.append(req.WhichOneof("resource"))
        if len(calls) == 1:
            raise ConnectError(Code.UNAVAILABLE, "blip")
        return cluster_payload_pb2.SelectClusterResponse(cluster_endpoint="")

    cluster_service.select_cluster = AsyncMock(side_effect=flaky_once)
    with upload_queue("q"):
        await wrapper.get_image(_get_image_request())
    assert calls == ["queue_id", "queue_id"], "recovered by retrying the queue route"

    calls.clear()
    wrapper2, cluster_service2, default_client2 = _make_wrapper()

    async def always_down(req):
        calls.append(req.WhichOneof("resource"))
        if req.WhichOneof("resource") == "queue_id":
            raise ConnectError(Code.UNAVAILABLE, "down")
        return cluster_payload_pb2.SelectClusterResponse(cluster_endpoint="")

    cluster_service2.select_cluster = AsyncMock(side_effect=always_down)
    with upload_queue("q"):
        await wrapper2.get_image(_get_image_request())
    assert calls == ["queue_id", "queue_id", "queue_id", "project_id"], "retries, then routes by project"
    default_client2.get_image.assert_awaited_once()


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
