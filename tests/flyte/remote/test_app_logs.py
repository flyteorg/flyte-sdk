"""Tests for app log streaming: _iter_app_log_lines, AppLogs.tail, App.show_logs,
and the ClusterAwareAppLogsService wrapper."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.app import app_definition_pb2, app_logs_payload_pb2, replica_definition_pb2
from flyteidl2.cluster import payload_pb2 as cluster_payload_pb2
from flyteidl2.logs.dataplane import payload_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from flyte.errors import LogsNotYetAvailableError
from flyte.remote._app import App
from flyte.remote._client.controlplane import ClusterAwareAppLogsService
from flyte.remote._logs import AppLogs, _iter_app_log_lines, _ReplayFilter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_id(name: str = "my-app") -> app_definition_pb2.Identifier:
    return app_definition_pb2.Identifier(org="o", project="p", domain="d", name=name)


def _make_app(name: str = "my-app") -> App:
    pb2 = app_definition_pb2.App(metadata=app_definition_pb2.Meta(id=_make_app_id(name)))
    return App(pb2=pb2)


def _make_logline(
    message: str,
    originator: payload_pb2.LogLineOriginator = payload_pb2.LogLineOriginator.USER,
    ts: datetime.datetime | None = None,
) -> payload_pb2.LogLine:
    logline = payload_pb2.LogLine(message=message, originator=originator)
    if ts is not None:
        timestamp = Timestamp()
        timestamp.FromDatetime(ts)
        logline.timestamp.CopyFrom(timestamp)
    return logline


# ---------------------------------------------------------------------------
# _iter_app_log_lines (pure-function tests — no mocking needed)
# ---------------------------------------------------------------------------


class TestIterAppLogLines:
    def test_replicas_case_yields_nothing(self):
        resp = app_logs_payload_pb2.TailLogsResponse(
            replicas=app_logs_payload_pb2.ReplicaIdentifierList(
                replicas=[replica_definition_pb2.ReplicaIdentifier(app_id=_make_app_id(), name="r0")]
            )
        )
        assert list(_iter_app_log_lines(resp)) == []

    def test_log_lines_structured(self):
        resp = app_logs_payload_pb2.TailLogsResponse(
            log_lines=payload_pb2.LogLines(structured_lines=[_make_logline("one"), _make_logline("two")])
        )
        results = list(_iter_app_log_lines(resp))
        assert [r.message for r in results] == ["one\n", "two\n"]

    def test_log_lines_raw_strings_wrapped(self):
        resp = app_logs_payload_pb2.TailLogsResponse(log_lines=payload_pb2.LogLines(lines=["raw one", "raw two"]))
        results = list(_iter_app_log_lines(resp))
        assert [r.message for r in results] == ["raw one\n", "raw two\n"]
        assert all(isinstance(r, payload_pb2.LogLine) for r in results)

    def test_log_lines_structured_preferred_over_raw_duplicates(self):
        # The backend sends the same content as both structured and raw lines;
        # only the structured ones should be yielded.
        resp = app_logs_payload_pb2.TailLogsResponse(
            log_lines=payload_pb2.LogLines(lines=["raw"], structured_lines=[_make_logline("structured")])
        )
        results = list(_iter_app_log_lines(resp))
        assert [r.message for r in results] == ["structured\n"]

    def test_batches_prefixes_replica_name(self):
        ts = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        resp = app_logs_payload_pb2.TailLogsResponse(
            batches=app_logs_payload_pb2.LogLinesBatch(
                logs=[
                    app_logs_payload_pb2.LogLines(
                        replica_id=replica_definition_pb2.ReplicaIdentifier(app_id=_make_app_id(), name="r0"),
                        structured_lines=[_make_logline("structured", ts=ts)],
                        lines=["raw"],
                    )
                ]
            )
        )
        results = list(_iter_app_log_lines(resp))
        assert [r.message for r in results] == ["[r0] structured\n"]
        # Timestamp and originator preserved on the prefixed copy.
        assert results[0].timestamp.ToDatetime(tzinfo=datetime.timezone.utc) == ts
        assert results[0].originator == payload_pb2.LogLineOriginator.USER

    def test_batches_does_not_mutate_source_proto(self):
        line = _make_logline("original")
        resp = app_logs_payload_pb2.TailLogsResponse(
            batches=app_logs_payload_pb2.LogLinesBatch(
                logs=[
                    app_logs_payload_pb2.LogLines(
                        replica_id=replica_definition_pb2.ReplicaIdentifier(name="r0"),
                        structured_lines=[line],
                    )
                ]
            )
        )
        list(_iter_app_log_lines(resp))
        assert resp.batches.logs[0].structured_lines[0].message == "original"

    def test_batches_raw_lines_used_when_no_structured(self):
        resp = app_logs_payload_pb2.TailLogsResponse(
            batches=app_logs_payload_pb2.LogLinesBatch(
                logs=[
                    app_logs_payload_pb2.LogLines(
                        replica_id=replica_definition_pb2.ReplicaIdentifier(name="r0"),
                        lines=["raw only"],
                    )
                ]
            )
        )
        results = list(_iter_app_log_lines(resp))
        assert [r.message for r in results] == ["[r0] raw only\n"]

    def test_batches_without_replica_name_not_prefixed(self):
        resp = app_logs_payload_pb2.TailLogsResponse(
            batches=app_logs_payload_pb2.LogLinesBatch(
                logs=[app_logs_payload_pb2.LogLines(structured_lines=[_make_logline("no replica")])]
            )
        )
        results = list(_iter_app_log_lines(resp))
        assert [r.message for r in results] == ["no replica\n"]

    def test_batches_multiple_replicas(self):
        def _batch(replica: str, message: str) -> app_logs_payload_pb2.LogLines:
            return app_logs_payload_pb2.LogLines(
                replica_id=replica_definition_pb2.ReplicaIdentifier(name=replica),
                structured_lines=[_make_logline(message)],
            )

        resp = app_logs_payload_pb2.TailLogsResponse(
            batches=app_logs_payload_pb2.LogLinesBatch(logs=[_batch("r0", "from r0"), _batch("r1", "from r1")])
        )
        results = list(_iter_app_log_lines(resp))
        assert [r.message for r in results] == ["[r0] from r0\n", "[r1] from r1\n"]

    def test_existing_trailing_newline_not_doubled(self):
        resp = app_logs_payload_pb2.TailLogsResponse(
            log_lines=payload_pb2.LogLines(structured_lines=[_make_logline("already terminated\n")])
        )
        results = list(_iter_app_log_lines(resp))
        assert [r.message for r in results] == ["already terminated\n"]

    def test_empty_response_yields_nothing(self):
        assert list(_iter_app_log_lines(app_logs_payload_pb2.TailLogsResponse())) == []


# ---------------------------------------------------------------------------
# _ReplayFilter
# ---------------------------------------------------------------------------


class TestReplayFilter:
    def test_replays_dropped_new_lines_kept(self):
        t1 = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        t2 = datetime.datetime(2024, 6, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)
        f = _ReplayFilter()
        assert f.is_new(_make_logline("a", ts=t1))
        assert f.is_new(_make_logline("b", ts=t2))
        # Replayed backlog: older timestamp, or same timestamp + same message.
        assert not f.is_new(_make_logline("a", ts=t1))
        assert not f.is_new(_make_logline("b", ts=t2))
        # A new message within the newest second is not a replay.
        assert f.is_new(_make_logline("c", ts=t2))
        assert not f.is_new(_make_logline("c", ts=t2))

    def test_lines_without_timestamp_always_pass(self):
        t1 = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        f = _ReplayFilter()
        assert f.is_new(_make_logline("a", ts=t1))
        assert f.is_new(_make_logline("no ts"))
        assert f.is_new(_make_logline("no ts"))


# ---------------------------------------------------------------------------
# AppLogs.tail
# ---------------------------------------------------------------------------


def _response(*messages: str, ts: datetime.datetime | None = None) -> app_logs_payload_pb2.TailLogsResponse:
    return app_logs_payload_pb2.TailLogsResponse(
        log_lines=payload_pb2.LogLines(structured_lines=[_make_logline(m, ts=ts) for m in messages])
    )


def _make_client(responses: list[app_logs_payload_pb2.TailLogsResponse]):
    client = MagicMock()

    async def _stream(_request):
        for resp in responses:
            yield resp

    client.app_logs_service.tail_logs = MagicMock(side_effect=_stream)
    return client


def _client_with_streams(*streams):
    """Client whose tail_logs returns each given line-list as one connection."""

    async def _gen(lines):
        for line in lines:
            yield app_logs_payload_pb2.TailLogsResponse(log_lines=payload_pb2.LogLines(structured_lines=[line]))

    client = MagicMock()
    client.app_logs_service.tail_logs = MagicMock(side_effect=[_gen(s) for s in streams])
    return client


class TestAppLogsTail:
    @pytest.mark.asyncio
    async def test_yields_lines_from_stream(self):
        responses = [_response("one"), _response("two")]
        client = _make_client(responses)
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
        ):
            results = [line async for line in AppLogs.tail.aio(app_id=_make_app_id(), follow=False)]

        assert [r.message for r in results] == ["one\n", "two\n"]
        assert client.app_logs_service.tail_logs.call_count == 1

    @pytest.mark.asyncio
    async def test_request_uses_app_id(self):
        client = _make_client([])
        app_id = _make_app_id()
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
        ):
            async for _ in AppLogs.tail.aio(app_id=app_id, follow=False):
                pass

        request = client.app_logs_service.tail_logs.call_args[0][0]
        assert request.WhichOneof("target") == "app_id"
        assert request.app_id == app_id

    @pytest.mark.asyncio
    async def test_request_uses_replica_id_when_replica_name_given(self):
        client = _make_client([])
        app_id = _make_app_id()
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
        ):
            async for _ in AppLogs.tail.aio(app_id=app_id, replica_name="r0", follow=False):
                pass

        request = client.app_logs_service.tail_logs.call_args[0][0]
        assert request.WhichOneof("target") == "replica_id"
        assert request.replica_id.app_id == app_id
        assert request.replica_id.name == "r0"

    @pytest.mark.asyncio
    async def test_not_found_raises_after_retries(self):
        client = MagicMock()

        async def _not_found(_request):
            raise ConnectError(Code.NOT_FOUND, "no stream")
            yield  # make it an async generator

        client.app_logs_service.tail_logs = MagicMock(side_effect=_not_found)
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
        ):
            with pytest.raises(LogsNotYetAvailableError):
                async for _ in AppLogs.tail.aio(app_id=_make_app_id(), retry=1):
                    pass

    @pytest.mark.asyncio
    async def test_follow_reconnects_across_rollout_and_dedups_backlog(self):
        """Each reconnect replays the persisted backlog; only new lines are
        yielded, and the tail ends after idle_reconnects reconnects with
        nothing new (the scale-to-zero signature)."""
        t1 = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        t2 = datetime.datetime(2024, 6, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)
        t3 = datetime.datetime(2024, 6, 1, 12, 0, 2, tzinfo=datetime.timezone.utc)
        a, b, c = _make_logline("a", ts=t1), _make_logline("b", ts=t2), _make_logline("c", ts=t3)

        client = _client_with_streams(
            [a, b],  # initial connection: backlog
            [a, b, c],  # rollout: replayed backlog + the new replica's line
            [a, b, c],  # idle reconnect 1: replay only
            [a, b, c],  # idle reconnect 2
            [a, b, c],  # idle reconnect 3 -> stop
        )
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
            patch("flyte.remote._logs.asyncio.sleep", new=AsyncMock()),
        ):
            results = [line async for line in AppLogs.tail.aio(app_id=_make_app_id())]

        assert [r.message for r in results] == ["a\n", "b\n", "c\n"]
        assert client.app_logs_service.tail_logs.call_count == 5

    @pytest.mark.asyncio
    async def test_follow_mid_stream_disconnect_reconnects(self):
        t1 = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        a = _make_logline("a", ts=t1)

        async def _drop(_request):
            yield app_logs_payload_pb2.TailLogsResponse(log_lines=payload_pb2.LogLines(structured_lines=[a]))
            raise ConnectError(Code.UNAVAILABLE, "connection dropped")

        async def _replay(_request):
            yield app_logs_payload_pb2.TailLogsResponse(log_lines=payload_pb2.LogLines(structured_lines=[a]))

        client = MagicMock()
        client.app_logs_service.tail_logs = MagicMock(
            side_effect=[_drop(None), _replay(None), _replay(None), _replay(None)]
        )
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
            patch("flyte.remote._logs.asyncio.sleep", new=AsyncMock()),
        ):
            results = [line async for line in AppLogs.tail.aio(app_id=_make_app_id())]

        assert [r.message for r in results] == ["a\n"]
        assert client.app_logs_service.tail_logs.call_count == 4

    @pytest.mark.asyncio
    async def test_not_found_after_data_ends_cleanly(self):
        """NOT_FOUND once data has streamed means the app is gone (deleted or
        deactivated) — the tail ends instead of raising."""

        async def _one(_request):
            yield _response("one")

        async def _gone(_request):
            raise ConnectError(Code.NOT_FOUND, "app deleted")
            yield

        client = MagicMock()
        client.app_logs_service.tail_logs = MagicMock(side_effect=[_one(None), _gone(None)])
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
            patch("flyte.remote._logs.asyncio.sleep", new=AsyncMock()),
        ):
            results = [line async for line in AppLogs.tail.aio(app_id=_make_app_id())]

        assert [r.message for r in results] == ["one\n"]

    @pytest.mark.asyncio
    async def test_no_follow_returns_on_stream_close(self):
        client = _make_client([])
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
        ):
            results = [line async for line in AppLogs.tail.aio(app_id=_make_app_id(), follow=False)]

        assert results == []
        assert client.app_logs_service.tail_logs.call_count == 1

    @pytest.mark.asyncio
    async def test_no_follow_mid_stream_disconnect_raises(self):
        async def _drop(_request):
            yield _response("one")
            raise ConnectError(Code.UNAVAILABLE, "connection dropped")

        client = MagicMock()
        client.app_logs_service.tail_logs = MagicMock(side_effect=_drop)
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
        ):
            with pytest.raises(ConnectError) as exc:
                async for _ in AppLogs.tail.aio(app_id=_make_app_id(), follow=False):
                    pass
        assert exc.value.code == Code.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_other_connect_errors_propagate(self):
        client = MagicMock()

        async def _denied(_request):
            raise ConnectError(Code.PERMISSION_DENIED, "nope")
            yield

        client.app_logs_service.tail_logs = MagicMock(side_effect=_denied)
        with (
            patch("flyte.remote._logs.ensure_client"),
            patch("flyte.remote._logs.get_client", return_value=client),
        ):
            with pytest.raises(ConnectError) as exc:
                async for _ in AppLogs.tail.aio(app_id=_make_app_id()):
                    pass
        assert exc.value.code == Code.PERMISSION_DENIED


# ---------------------------------------------------------------------------
# App.show_logs
# ---------------------------------------------------------------------------


class TestAppShowLogs:
    @pytest.mark.asyncio
    async def test_delegates_to_create_viewer(self):
        app = _make_app()
        with patch("flyte.remote._logs.AppLogs.create_viewer", new=AsyncMock()) as mock_viewer:
            await app.show_logs.aio(max_lines=10, show_ts=True, raw=True, filter_system=True, replica_name="r0")

        mock_viewer.assert_awaited_once_with(
            app_id=app.pb2.metadata.id,
            max_lines=10,
            show_ts=True,
            raw=True,
            filter_system=True,
            replica_name="r0",
        )


# ---------------------------------------------------------------------------
# ClusterAwareAppLogsService
# ---------------------------------------------------------------------------


def _make_wrapper(cluster_endpoint: str = "", own_endpoint: str = "dns:///localhost:8090"):
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

    async def _stream_one(_request):
        yield app_logs_payload_pb2.TailLogsResponse()

    default_client.tail_logs = MagicMock(side_effect=_stream_one)
    return (
        ClusterAwareAppLogsService(
            cluster_service=cluster_service,
            session_config=session_config,
            default_client=default_client,
        ),
        cluster_service,
        default_client,
    )


@pytest.mark.asyncio
async def test_app_tail_logs_routes_by_app_id():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = app_logs_payload_pb2.TailLogsRequest(app_id=_make_app_id())

    results = [resp async for resp in wrapper.tail_logs(req)]

    assert len(results) == 1
    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.operation == cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_TAIL_LOGS
    assert sent.WhichOneof("resource") == "app_id"
    assert sent.app_id == _make_app_id()
    default_client.tail_logs.assert_called_once_with(req)


@pytest.mark.asyncio
async def test_app_tail_logs_with_replica_id_routes_by_parent_app():
    wrapper, cluster_service, default_client = _make_wrapper()
    req = app_logs_payload_pb2.TailLogsRequest(
        replica_id=replica_definition_pb2.ReplicaIdentifier(app_id=_make_app_id(), name="r0")
    )

    async for _ in wrapper.tail_logs(req):
        pass

    sent = cluster_service.select_cluster.await_args[0][0]
    assert sent.app_id == _make_app_id()
    default_client.tail_logs.assert_called_once_with(req)


@pytest.mark.asyncio
async def test_app_tail_logs_falls_back_to_default_on_select_cluster_failure():
    wrapper, cluster_service, default_client = _make_wrapper()
    cluster_service.select_cluster = AsyncMock(side_effect=ConnectError(Code.UNIMPLEMENTED, "no app routing"))
    req = app_logs_payload_pb2.TailLogsRequest(app_id=_make_app_id())

    results = [resp async for resp in wrapper.tail_logs(req)]

    assert len(results) == 1
    default_client.tail_logs.assert_called_once_with(req)
