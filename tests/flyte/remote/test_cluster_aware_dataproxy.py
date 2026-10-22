"""Tests for the ClusterAwareDataProxy wrapper in flyte.remote._client.controlplane."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.cluster import payload_pb2 as cluster_payload_pb2
from flyteidl2.common import identifier_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2

from flyte._upload_queue import upload_queue
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
    session_config.auth_kwargs = {}
    default_client = MagicMock()
    default_client.create_upload_location = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(signed_url="https://signed/")
    )
    default_client.upload_inputs = AsyncMock(return_value=dataproxy_service_pb2.UploadInputsResponse())

    async def _stream_one(_request):
        yield dataproxy_service_pb2.TailLogsResponse()

    default_client.tail_logs = MagicMock(side_effect=_stream_one)
    default_client.get_action_data = AsyncMock(return_value=dataproxy_service_pb2.GetActionDataResponse())
    default_client.create_download_link = AsyncMock(return_value=dataproxy_service_pb2.CreateDownloadLinkResponse())
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
async def test_upload_trigger_routes_by_upload_trigger_operation():
    wrapper, cluster_service, default_client = _make_wrapper()
    pid = identifier_pb2.ProjectIdentifier(name="p", domain="d", organization="o")
    req = dataproxy_service_pb2.UploadInputsRequest(project_id=pid)

    await wrapper.upload_trigger(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_UPLOAD_TRIGGER
    assert sent.project_id == pid
    # The actual upload is still the UploadInputs RPC on the resolved cluster.
    default_client.upload_inputs.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_upload_trigger_propagates_unimplemented():
    """SelectCluster UNIMPLEMENTED (zero trust off) propagates with its code, not wrapped."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    wrapper, cluster_service, _default_client = _make_wrapper()
    cluster_service.select_cluster = AsyncMock(side_effect=ConnectError(Code.UNIMPLEMENTED, "no zero trust"))
    req = dataproxy_service_pb2.UploadInputsRequest(
        project_id=identifier_pb2.ProjectIdentifier(name="p", domain="d", organization="o")
    )

    with pytest.raises(ConnectError) as exc:
        await wrapper.upload_trigger(req)
    assert exc.value.code == Code.UNIMPLEMENTED


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


@pytest.mark.asyncio
async def test_remote_cluster_endpoint_propagates_auth_kwargs():
    """Per-cluster sessions must inherit the parent session's auth configuration.

    Regression test for the case where ``init_passthrough`` set ``auth_type="Passthrough"``
    on the main ClientSet, but the first cluster-routed dataproxy call rebuilt a
    SessionConfig without forwarding ``auth_type`` and silently downgraded to PKCE
    (triggering the OAuth browser flow from a server-side request handler).
    """
    wrapper, _, _ = _make_wrapper(cluster_endpoint="dns:///other:8090")
    wrapper._session_config.auth_kwargs = {
        "auth_type": "Passthrough",
        "ca_cert_file_path": "/etc/ssl/ca.pem",
        "client_id": "demo-uctl",
    }

    new_session_cfg = MagicMock()
    new_session_cfg.connect_kwargs.return_value = {}
    new_client_inst = MagicMock()
    new_client_inst.create_upload_location = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(signed_url="https://remote/")
    )

    create_session_config_mock = AsyncMock(return_value=new_session_cfg)
    with (
        patch(
            "flyte.remote._client.controlplane.create_session_config",
            new=create_session_config_mock,
        ),
        patch(
            "flyte.remote._client.controlplane.DataProxyServiceClient",
            return_value=new_client_inst,
        ),
    ):
        await wrapper.create_upload_location(
            dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o")
        )

    create_session_config_mock.assert_awaited_once()
    forwarded_kwargs = create_session_config_mock.await_args.kwargs
    assert forwarded_kwargs["auth_type"] == "Passthrough"
    assert forwarded_kwargs["ca_cert_file_path"] == "/etc/ssl/ca.pem"
    assert forwarded_kwargs["client_id"] == "demo-uctl"
    # auth_endpoint must remain the parent session's endpoint so OAuth/passthrough
    # metadata stays anchored to the routing endpoint, not the cluster endpoint.
    assert forwarded_kwargs["auth_endpoint"] == "dns:///localhost:8090"


@pytest.mark.asyncio
async def test_clientsecret_session_does_not_downgrade_to_pkce_at_cluster_endpoint():
    """End-to-end: a ClientSecret session must stay ClientSecret at the per-cluster endpoint.

    Ties together the two halves of the fix that are otherwise only covered in
    isolation: ``create_session_config`` *capturing* ``auth_kwargs`` (verified in
    test_session.py) and ``_select_and_build`` *forwarding* them (verified above
    with ``create_session_config`` mocked). Here the REAL ``create_session_config``
    runs for both the control-plane session and the rebuilt per-cluster session,
    so the test fails if either half regresses.
    """
    from flyte.remote._client.auth import _session as session_mod
    from flyte.remote._client.auth._session import create_session_config

    # Record the auth_type that each create_session_config build hands to the
    # auth-interceptor factory. Missing/absent auth_type defaults to "Pkce" — the
    # downgrade we are guarding against. Both auth flows are stubbed so no network
    # or browser interaction occurs.
    recorded_auth_types: list[str] = []

    def _record_auth_interceptors(*, endpoint, http_client=None, **kwargs):
        recorded_auth_types.append(kwargs.get("auth_type", "Pkce"))
        return []

    per_cluster_client = MagicMock()
    per_cluster_client.create_upload_location = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(signed_url="https://cluster/")
    )

    with (
        patch.object(session_mod, "create_auth_interceptors", side_effect=_record_auth_interceptors),
        patch.object(session_mod, "create_proxy_auth_interceptors", return_value=[]),
        patch.object(session_mod, "get_async_session", return_value=MagicMock()),
        patch(
            "flyte.remote._client.controlplane.DataProxyServiceClient",
            return_value=per_cluster_client,
        ),
    ):
        # Real control-plane session configured for ClientSecret (no api_key, the
        # config-file case where there is nothing else to fall back on).
        main_cfg = await create_session_config(
            "dns:///controlplane.example.com:443",
            None,
            insecure=False,
            auth_type="ClientSecret",
            client_id="demo-uctl",
            client_credentials_secret="shhh",
        )
        # Capture half: auth_type must be snapshotted for the per-cluster rebuild.
        assert main_cfg.auth_kwargs["auth_type"] == "ClientSecret"

        cluster_service = MagicMock()
        cluster_service.select_cluster = AsyncMock(
            return_value=cluster_payload_pb2.SelectClusterResponse(cluster_endpoint="dns:///cluster-a.example.com:443")
        )
        wrapper = ClusterAwareDataProxy(
            cluster_service=cluster_service,
            session_config=main_cfg,
            default_client=MagicMock(),
        )

        await wrapper.create_upload_location(
            dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o")
        )

    # Exactly two real sessions were built — the control plane and the per-cluster
    # endpoint — and NEITHER downgraded to Pkce. Before the fix the second entry
    # would be "Pkce".
    assert recorded_auth_types == ["ClientSecret", "ClientSecret"]
    per_cluster_client.create_upload_location.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_reuses_cached_client_across_calls():
    """Repeated resolution for the same key issues one SelectCluster and reuses the client."""
    wrapper, cluster_service, default_client = _make_wrapper()
    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename="f")

    await wrapper.create_upload_location(req)
    await wrapper.create_upload_location(req)

    assert cluster_service.select_cluster.await_count == 1
    assert default_client.create_upload_location.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_resolves_all_succeed():
    """Concurrent callers for the same key each get a working client (no corruption).

    The cache stores resolved values, so a key that is already cached is served from the
    single shared mapping; a miss just computes under the lock guard. Either way every
    caller dispatches to the same default client.
    """
    wrapper, _, default_client = _make_wrapper()

    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename="f")
    tasks = [asyncio.create_task(wrapper.create_upload_location(req)) for _ in range(10)]
    await asyncio.gather(*tasks)

    assert default_client.create_upload_location.await_count == 10


@pytest.mark.asyncio
async def test_failed_resolve_is_evicted_so_retries_can_succeed():
    wrapper, cluster_service, default_client = _make_wrapper()
    cluster_service.select_cluster = AsyncMock(
        side_effect=[
            RuntimeError("transient"),
            cluster_payload_pb2.SelectClusterResponse(cluster_endpoint=""),
        ]
    )

    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o")
    with pytest.raises(RuntimeError):
        await wrapper.create_upload_location(req)
    # Second call should retry SelectCluster, not return the cached failure.
    await wrapper.create_upload_location(req)

    assert cluster_service.select_cluster.await_count == 2
    assert default_client.create_upload_location.await_count == 1


@pytest.mark.asyncio
async def test_tail_logs_routes_by_action_id():
    wrapper, cluster_service, default_client = _make_wrapper()
    action_id = identifier_pb2.ActionIdentifier(
        run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="r"),
        name="a",
    )
    req = dataproxy_service_pb2.TailLogsRequest(action_id=action_id, attempt=1)

    received = [resp async for resp in wrapper.tail_logs(req)]

    assert len(received) == 1
    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_TAIL_LOGS
    assert sent.WhichOneof("resource") == "action_id"
    assert sent.action_id == action_id
    default_client.tail_logs.assert_called_once_with(req)


@pytest.mark.asyncio
async def test_get_action_data_routes_by_action_id():
    wrapper, cluster_service, default_client = _make_wrapper()
    action_id = identifier_pb2.ActionIdentifier(
        run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="r"),
        name="a",
    )
    req = dataproxy_service_pb2.GetActionDataRequest(action_id=action_id)

    await wrapper.get_action_data(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_GET_ACTION_DATA
    assert sent.WhichOneof("resource") == "action_id"
    assert sent.action_id == action_id
    default_client.get_action_data.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_create_download_link_routes_by_action_attempt_id():
    wrapper, cluster_service, default_client = _make_wrapper()
    action_id = identifier_pb2.ActionIdentifier(
        run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="r"),
        name="a",
    )
    req = dataproxy_service_pb2.CreateDownloadLinkRequest(
        artifact_type=dataproxy_service_pb2.ARTIFACT_TYPE_REPORT,
        action_attempt_id=identifier_pb2.ActionAttemptIdentifier(action_id=action_id, attempt=1),
    )

    await wrapper.create_download_link(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_CREATE_DOWNLOAD_LINK
    assert sent.WhichOneof("resource") == "action_id"
    assert sent.action_id == action_id
    default_client.create_download_link.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_create_download_link_routes_by_task_id():
    from flyteidl2.task import task_definition_pb2

    wrapper, cluster_service, default_client = _make_wrapper()
    task_id = task_definition_pb2.TaskIdentifier(org="o", project="p", domain="d", name="t", version="v")
    req = dataproxy_service_pb2.CreateDownloadLinkRequest(
        artifact_type=dataproxy_service_pb2.ARTIFACT_TYPE_CODE_BUNDLE,
        task_id=task_id,
    )

    await wrapper.create_download_link(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_CREATE_DOWNLOAD_LINK
    assert sent.WhichOneof("resource") == "project_id"
    assert sent.project_id.name == "p"
    default_client.create_download_link.assert_awaited_once_with(req)


def test_resolve_cache_survives_event_loop_change(recwarn):
    """The per-cluster client cache must outlive the loop that filled it.

    Flyte drives one ClientSet from several loops in a process (the CLI's `asyncio.run`
    loop, the syncify background loop, the controller thread, the tracked-run reporter
    thread). A task-based cache such as `alru_cache` clears itself on the first loop
    change — re-running SelectCluster and printing `AlruCacheLoopResetWarning`.
    """
    wrapper, cluster_service, _ = _make_wrapper()
    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename="f")

    asyncio.run(wrapper.create_upload_location(req))
    asyncio.run(wrapper.create_upload_location(req))

    cluster_service.select_cluster.assert_awaited_once()
    assert [w for w in recwarn if "event loop change" in str(w.message)] == []


@pytest.mark.asyncio
async def test_create_upload_location_routes_by_upload_queue():
    """The upload queue routes the signed URL to that queue's cluster, so e.g. a code bundle for an
    Azure-pool queue isn't signed by the project's default (GCP) dataplane."""
    wrapper, cluster_service, default_client = _make_wrapper()
    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename="f")

    with upload_queue("not-default"):
        await wrapper.create_upload_location(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_CREATE_UPLOAD_LOCATION
    assert sent.WhichOneof("resource") == "queue_id"
    assert sent.queue_id == identifier_pb2.QueueIdentifier(
        organization="o", domain="d", project="p", name="not-default"
    )
    default_client.create_upload_location.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_upload_inputs_routes_by_upload_queue():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = dataproxy_service_pb2.UploadInputsRequest(
        project_id=identifier_pb2.ProjectIdentifier(name="p", domain="d", organization="o")
    )

    with upload_queue("q1"):
        await wrapper.upload_inputs(req)

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_UPLOAD_INPUTS
    assert sent.WhichOneof("resource") == "queue_id"
    assert sent.queue_id.name == "q1"
    default_client.upload_inputs.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_queue_routing_is_cached_per_queue():
    """Switching queues re-selects the cluster; repeating a queue reuses the cached client."""
    wrapper, cluster_service, _default_client = _make_wrapper()
    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename="f")

    with upload_queue("a"):
        await wrapper.create_upload_location(req)
        await wrapper.create_upload_location(req)
    with upload_queue("b"):
        await wrapper.create_upload_location(req)
    # Outside the block the queue is cleared: back to project routing.
    await wrapper.create_upload_location(req)

    resources = [c[0][0].WhichOneof("resource") for c in cluster_service.select_cluster.await_args_list]
    queues = [c[0][0].queue_id.name for c in cluster_service.select_cluster.await_args_list]
    assert resources == ["queue_id", "queue_id", "project_id"]
    assert queues[:2] == ["a", "b"]


@pytest.mark.asyncio
async def test_queue_routing_falls_back_to_project_when_unsupported():
    """A backend that can't route by queue (pre queue_id) falls back to project routing."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    wrapper, cluster_service, default_client = _make_wrapper()
    project_resp = cluster_payload_pb2.SelectClusterResponse(cluster_endpoint="")

    async def _select(req):
        if req.WhichOneof("resource") == "queue_id":
            raise ConnectError(Code.INVALID_ARGUMENT, "resource is required")
        return project_resp

    cluster_service.select_cluster = AsyncMock(side_effect=_select)
    req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename="f")

    with upload_queue("q"):
        await wrapper.create_upload_location(req)

    resources = [c[0][0].WhichOneof("resource") for c in cluster_service.select_cluster.await_args_list]
    assert resources == ["queue_id", "project_id"]
    default_client.create_upload_location.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_upload_queue_is_isolated_between_concurrent_tasks():
    """Two runs submitted concurrently with different queues each route to their own queue."""
    wrapper, cluster_service, _default_client = _make_wrapper()

    async def _upload(queue: str, filename: str):
        with upload_queue(queue):
            await asyncio.sleep(0)  # interleave with the other task
            req = dataproxy_service_pb2.CreateUploadLocationRequest(project="p", domain="d", org="o", filename=filename)
            await wrapper.create_upload_location(req)

    await asyncio.gather(_upload("a", "fa"), _upload("b", "fb"))

    queues = sorted(c[0][0].queue_id.name for c in cluster_service.select_cluster.await_args_list)
    assert queues == ["a", "b"]
