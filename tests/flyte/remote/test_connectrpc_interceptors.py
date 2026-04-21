import asyncio
import re
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from flyte._context import internal_ctx
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.remote._client.auth._authenticators.base import AuthHeaders
from flyte.remote._client.auth._interceptors.auth import (
    AuthBidiStreamInterceptor,
    AuthClientStreamInterceptor,
    AuthServerStreamInterceptor,
    AuthUnaryInterceptor,
    _is_auth_retriable,
)
from flyte.remote._client.auth._interceptors.default_metadata import (
    DefaultMetadataInterceptor,
    _generate_request_id,
)
from flyte.remote._client.auth._interceptors.retry import (
    RetryServerStreamInterceptor,
    RetryUnaryInterceptor,
)
from flyte.report import Report


@pytest.fixture
def task_context():
    """Create a TaskContext for testing with context."""
    return TaskContext(
        action=ActionID(
            name="test-action",
            run_name="test-run",
            project="test-project",
            domain="test-domain",
        ),
        run_base_dir="/tmp/test",
        output_path="/tmp/test/outputs",
        raw_data_path=RawDataPath(path="/tmp/test/raw"),
        version="v1",
        report=Report("test-report"),
    )


class TestRequestIdGeneration:
    """Tests for request ID generation logic."""

    def test_generate_request_id_without_context(self):
        request_id = _generate_request_id()
        try:
            UUID(request_id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False
        assert is_valid_uuid, f"Expected UUID4, got: {request_id}"

    def test_generate_request_id_with_context(self, task_context):
        ctx = internal_ctx()
        with ctx.replace_task_context(task_context):
            request_id = _generate_request_id()
            assert "test-project" in request_id
            assert "test-domain" in request_id
            assert "test-run" in request_id
            assert "test-action" in request_id
            pattern = r"test-project-test-domain-test-run-test-action-[a-z0-9]{4}"
            assert re.match(pattern, request_id)

    def test_generate_request_id_different_salts(self, task_context):
        ctx = internal_ctx()
        with ctx.replace_task_context(task_context):
            r1 = _generate_request_id()
            r2 = _generate_request_id()
            assert r1 != r2


def _make_ctx_mock():
    """Create a mock RequestContext where request_headers() returns a mutable dict."""
    headers = {}
    ctx = Mock()
    ctx.request_headers.return_value = headers
    return ctx, headers


class TestDefaultMetadataInterceptor:
    @pytest.mark.asyncio
    async def test_injects_request_id(self):
        interceptor = DefaultMetadataInterceptor()
        ctx, headers = _make_ctx_mock()
        await interceptor.on_start(ctx)
        assert "x-request-id" in headers
        assert len(headers["x-request-id"]) > 0

    @pytest.mark.asyncio
    async def test_does_not_inject_accept_header(self):
        """ConnectRPC handles content negotiation — no accept header needed."""
        interceptor = DefaultMetadataInterceptor()
        ctx, headers = _make_ctx_mock()
        await interceptor.on_start(ctx)
        assert "accept" not in headers

    @pytest.mark.asyncio
    async def test_on_end_is_noop(self):
        interceptor = DefaultMetadataInterceptor()
        await interceptor.on_end(None, None, None)  # Should not raise

    @pytest.mark.asyncio
    async def test_with_task_context(self, task_context):
        interceptor = DefaultMetadataInterceptor()
        ctx, headers = _make_ctx_mock()
        ictx = internal_ctx()
        with ictx.replace_task_context(task_context):
            await interceptor.on_start(ctx)
            request_id = headers["x-request-id"]
            assert "test-project" in request_id

    @pytest.mark.asyncio
    async def test_preserves_existing_request_id(self):
        """If a caller already set x-request-id, the interceptor must not overwrite it."""
        interceptor = DefaultMetadataInterceptor()
        ctx, headers = _make_ctx_mock()
        headers["x-request-id"] = "caller-correlation-id"
        await interceptor.on_start(ctx)
        assert headers["x-request-id"] == "caller-correlation-id"


def _make_mock_authenticator(headers=None):
    """Create a mock authenticator that returns given headers."""
    auth = AsyncMock()
    if headers:
        auth.get_auth_headers.return_value = AuthHeaders(creds_id="test-creds", headers=headers)
    else:
        auth.get_auth_headers.return_value = None
    return auth


class TestAuthUnaryInterceptor:
    @pytest.mark.asyncio
    async def test_injects_auth_headers(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token123"})
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(return_value="response")
        ctx, headers = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "request", ctx)

        assert result == "response"
        assert headers["authorization"] == "Bearer token123"
        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_unauthenticated(self):
        auth = _make_mock_authenticator({"authorization": "Bearer old"})
        # After refresh, return new headers
        auth.get_auth_headers.side_effect = [
            AuthHeaders(creds_id="old", headers={"authorization": "Bearer old"}),  # initial
            AuthHeaders(creds_id="new", headers={"authorization": "Bearer new"}),  # after refresh
        ]
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.UNAUTHENTICATED, "expired"),
                "success",
            ]
        )
        ctx, _headers = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "request", ctx)

        assert result == "success"
        auth.refresh_credentials.assert_called_once_with(creds_id="old")
        assert call_next.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_unknown(self):
        auth = _make_mock_authenticator({"authorization": "Bearer old"})
        auth.get_auth_headers.side_effect = [
            AuthHeaders(creds_id="old", headers={"authorization": "Bearer old"}),
            AuthHeaders(creds_id="new", headers={"authorization": "Bearer new"}),
        ]
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.UNKNOWN, "unknown"),
                "success",
            ]
        )
        ctx, _headers = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "request", ctx)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_does_not_retry_other_errors(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token"})
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(side_effect=ConnectError(Code.NOT_FOUND, "not found"))
        ctx, _headers = _make_ctx_mock()

        with pytest.raises(ConnectError) as exc_info:
            await interceptor.intercept_unary(call_next, "request", ctx)
        assert exc_info.value.code == Code.NOT_FOUND
        auth.refresh_credentials.assert_not_called()

    @pytest.mark.asyncio
    async def test_lazy_authenticator_init(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token"})
        factory = Mock(return_value=auth)
        interceptor = AuthUnaryInterceptor(factory)

        # Factory not called yet
        factory.assert_not_called()

        call_next = AsyncMock(return_value="response")
        ctx, _ = _make_ctx_mock()
        await interceptor.intercept_unary(call_next, "request", ctx)

        # Now factory is called
        factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_headers_when_auth_returns_none(self):
        auth = _make_mock_authenticator(headers=None)
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(return_value="response")
        ctx, headers = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "request", ctx)
        assert result == "response"
        assert "authorization" not in headers

    @pytest.mark.asyncio
    async def test_removes_stale_header_on_retry_when_key_changes(self):
        """When refresh changes the header key (e.g. authorization → flyte-authorization),
        the old header must be removed so the retry doesn't send both.
        Non-auth headers must be preserved."""
        auth = AsyncMock()
        auth.get_auth_headers.side_effect = [
            AuthHeaders(creds_id="old", headers={"authorization": "Bearer expired"}),
            AuthHeaders(creds_id="new", headers={"flyte-authorization": "Bearer fresh"}),
        ]
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.UNAUTHENTICATED, "expired"),
                "success",
            ]
        )
        ctx, headers = _make_ctx_mock()
        headers["x-request-id"] = "req-123"
        headers["content-type"] = "application/proto"

        result = await interceptor.intercept_unary(call_next, "request", ctx)

        assert result == "success"
        assert "authorization" not in headers, "stale auth header should be removed on retry"
        assert headers["flyte-authorization"] == "Bearer fresh"
        assert headers["x-request-id"] == "req-123", "non-auth headers must be preserved"
        assert headers["content-type"] == "application/proto", "non-auth headers must be preserved"


class TestIsAuthRetriable:
    """Tests for _is_auth_retriable which detects auth errors by code and message."""

    def test_unauthenticated_code(self):
        assert _is_auth_retriable(ConnectError(Code.UNAUTHENTICATED, "expired"))

    def test_unknown_code(self):
        assert _is_auth_retriable(ConnectError(Code.UNKNOWN, "something"))

    def test_unavailable_with_unauthorized_message(self):
        """JSON 401 responses with unrecognized code strings fall back to UNAVAILABLE
        but keep the 'Unauthorized' message from the HTTP status."""
        assert _is_auth_retriable(ConnectError(Code.UNAVAILABLE, "Unauthorized"))

    def test_unavailable_with_unauthenticated_message(self):
        assert _is_auth_retriable(ConnectError(Code.UNAVAILABLE, "unauthenticated"))

    def test_unavailable_without_auth_message(self):
        assert not _is_auth_retriable(ConnectError(Code.UNAVAILABLE, "service down"))

    def test_not_found_not_retriable(self):
        assert not _is_auth_retriable(ConnectError(Code.NOT_FOUND, "not found"))

    def test_permission_denied_not_retriable(self):
        assert not _is_auth_retriable(ConnectError(Code.PERMISSION_DENIED, "forbidden"))


class TestAuthUnaryInterceptorMessageFallback:
    """Tests that the unary interceptor retries on misclassified 401 errors."""

    @pytest.mark.asyncio
    async def test_retries_on_unavailable_unauthorized_message(self):
        """When a JSON 401 has a non-standard code field, ConnectRPC maps it to
        UNAVAILABLE. The interceptor should still retry via message matching."""
        auth = _make_mock_authenticator({"authorization": "Bearer old"})
        auth.get_auth_headers.side_effect = [
            AuthHeaders(creds_id="old", headers={"authorization": "Bearer old"}),
            AuthHeaders(creds_id="new", headers={"authorization": "Bearer new"}),
        ]
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.UNAVAILABLE, "Unauthorized"),
                "success",
            ]
        )
        ctx, _headers = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "request", ctx)
        assert result == "success"
        auth.refresh_credentials.assert_called_once()
        assert call_next.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_unavailable_non_auth(self):
        """UNAVAILABLE errors without auth keywords should not trigger auth retry."""
        auth = _make_mock_authenticator({"authorization": "Bearer token"})
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(side_effect=ConnectError(Code.UNAVAILABLE, "service down"))
        ctx, _headers = _make_ctx_mock()

        with pytest.raises(ConnectError) as exc_info:
            await interceptor.intercept_unary(call_next, "request", ctx)
        assert exc_info.value.code == Code.UNAVAILABLE
        auth.refresh_credentials.assert_not_called()


class TestAuthServerStreamInterceptor:
    @pytest.mark.asyncio
    async def test_injects_auth_headers_and_streams(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token123"})
        interceptor = AuthServerStreamInterceptor(lambda: auth)

        async def mock_call_next(request, ctx):
            yield "chunk1"
            yield "chunk2"

        ctx, headers = _make_ctx_mock()

        results = []
        async for item in interceptor.intercept_server_stream(mock_call_next, "request", ctx):
            results.append(item)

        assert results == ["chunk1", "chunk2"]
        assert headers["authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_retries_on_unauthenticated(self):
        auth = _make_mock_authenticator({"authorization": "Bearer old"})
        auth.get_auth_headers.side_effect = [
            AuthHeaders(creds_id="old", headers={"authorization": "Bearer old"}),
            AuthHeaders(creds_id="new", headers={"authorization": "Bearer new"}),
        ]
        interceptor = AuthServerStreamInterceptor(lambda: auth)

        call_count = 0

        async def mock_call_next(request, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectError(Code.UNAUTHENTICATED, "expired")
            yield "success"

        ctx, _headers = _make_ctx_mock()

        results = []
        async for item in interceptor.intercept_server_stream(mock_call_next, "request", ctx):
            results.append(item)

        assert results == ["success"]
        auth.refresh_credentials.assert_called_once_with(creds_id="old")

    @pytest.mark.asyncio
    async def test_does_not_retry_other_errors(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token"})
        interceptor = AuthServerStreamInterceptor(lambda: auth)

        async def mock_call_next(request, ctx):
            raise ConnectError(Code.NOT_FOUND, "not found")
            yield  # make it a generator

        ctx, _headers = _make_ctx_mock()

        with pytest.raises(ConnectError) as exc_info:
            async for _ in interceptor.intercept_server_stream(mock_call_next, "request", ctx):
                pass
        assert exc_info.value.code == Code.NOT_FOUND


class TestAuthClientStreamInterceptor:
    @pytest.mark.asyncio
    async def test_injects_auth_headers(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token123"})
        interceptor = AuthClientStreamInterceptor(lambda: auth)

        call_next = AsyncMock(return_value="response")
        ctx, headers = _make_ctx_mock()

        async def request_iter():
            yield "req1"
            yield "req2"

        result = await interceptor.intercept_client_stream(call_next, request_iter(), ctx)

        assert result == "response"
        assert headers["authorization"] == "Bearer token123"
        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_unauthenticated(self):
        auth = _make_mock_authenticator({"authorization": "Bearer old"})
        auth.get_auth_headers.side_effect = [
            AuthHeaders(creds_id="old", headers={"authorization": "Bearer old"}),
            AuthHeaders(creds_id="new", headers={"authorization": "Bearer new"}),
        ]
        interceptor = AuthClientStreamInterceptor(lambda: auth)

        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.UNAUTHENTICATED, "expired"),
                "success",
            ]
        )
        ctx, _headers = _make_ctx_mock()

        async def request_iter():
            yield "req1"

        result = await interceptor.intercept_client_stream(call_next, request_iter(), ctx)

        assert result == "success"
        auth.refresh_credentials.assert_called_once_with(creds_id="old")
        assert call_next.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_other_errors(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token"})
        interceptor = AuthClientStreamInterceptor(lambda: auth)

        call_next = AsyncMock(side_effect=ConnectError(Code.NOT_FOUND, "not found"))
        ctx, _ = _make_ctx_mock()

        async def request_iter():
            yield "req1"

        with pytest.raises(ConnectError) as exc_info:
            await interceptor.intercept_client_stream(call_next, request_iter(), ctx)
        assert exc_info.value.code == Code.NOT_FOUND
        auth.refresh_credentials.assert_not_called()


class TestAuthBidiStreamInterceptor:
    @pytest.mark.asyncio
    async def test_injects_auth_headers_and_streams(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token123"})
        interceptor = AuthBidiStreamInterceptor(lambda: auth)

        async def mock_call_next(request, ctx):
            yield "chunk1"
            yield "chunk2"

        ctx, headers = _make_ctx_mock()

        async def request_iter():
            yield "req1"

        results = []
        async for item in interceptor.intercept_bidi_stream(mock_call_next, request_iter(), ctx):
            results.append(item)

        assert results == ["chunk1", "chunk2"]
        assert headers["authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_retries_on_unauthenticated(self):
        auth = _make_mock_authenticator({"authorization": "Bearer old"})
        auth.get_auth_headers.side_effect = [
            AuthHeaders(creds_id="old", headers={"authorization": "Bearer old"}),
            AuthHeaders(creds_id="new", headers={"authorization": "Bearer new"}),
        ]
        interceptor = AuthBidiStreamInterceptor(lambda: auth)

        call_count = 0

        async def mock_call_next(request, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectError(Code.UNAUTHENTICATED, "expired")
            yield "success"

        ctx, _ = _make_ctx_mock()

        async def request_iter():
            yield "req1"

        results = []
        async for item in interceptor.intercept_bidi_stream(mock_call_next, request_iter(), ctx):
            results.append(item)

        assert results == ["success"]
        auth.refresh_credentials.assert_called_once_with(creds_id="old")

    @pytest.mark.asyncio
    async def test_does_not_retry_other_errors(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token"})
        interceptor = AuthBidiStreamInterceptor(lambda: auth)

        async def mock_call_next(request, ctx):
            raise ConnectError(Code.NOT_FOUND, "not found")
            yield  # make it a generator

        ctx, _ = _make_ctx_mock()

        async def request_iter():
            yield "req1"

        with pytest.raises(ConnectError) as exc_info:
            async for _ in interceptor.intercept_bidi_stream(mock_call_next, request_iter(), ctx):
                pass
        assert exc_info.value.code == Code.NOT_FOUND


class TestRetryUnaryInterceptor:
    @pytest.mark.asyncio
    async def test_succeeds_first_attempt(self):
        interceptor = RetryUnaryInterceptor(max_attempts=3)
        call_next = AsyncMock(return_value="ok")
        ctx, _ = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "req", ctx)
        assert result == "ok"
        assert call_next.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_unavailable(self):
        interceptor = RetryUnaryInterceptor(max_attempts=3, initial_backoff=0.001)
        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.UNAVAILABLE, "unavailable"),
                "ok",
            ]
        )
        ctx, _ = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "req", ctx)
        assert result == "ok"
        assert call_next.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_resource_exhausted(self):
        interceptor = RetryUnaryInterceptor(max_attempts=3, initial_backoff=0.001)
        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.RESOURCE_EXHAUSTED, "exhausted"),
                "ok",
            ]
        )
        ctx, _ = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "req", ctx)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_internal(self):
        interceptor = RetryUnaryInterceptor(max_attempts=3, initial_backoff=0.001)
        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.INTERNAL, "internal"),
                "ok",
            ]
        )
        ctx, _ = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "req", ctx)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_does_not_retry_not_found(self):
        interceptor = RetryUnaryInterceptor(max_attempts=3)
        call_next = AsyncMock(side_effect=ConnectError(Code.NOT_FOUND, "nope"))
        ctx, _ = _make_ctx_mock()

        with pytest.raises(ConnectError) as exc_info:
            await interceptor.intercept_unary(call_next, "req", ctx)
        assert exc_info.value.code == Code.NOT_FOUND
        assert call_next.call_count == 1

    @pytest.mark.asyncio
    async def test_respects_max_attempts(self):
        interceptor = RetryUnaryInterceptor(max_attempts=3, initial_backoff=0.001)
        call_next = AsyncMock(side_effect=ConnectError(Code.UNAVAILABLE, "down"))
        ctx, _ = _make_ctx_mock()

        with pytest.raises(ConnectError) as exc_info:
            await interceptor.intercept_unary(call_next, "req", ctx)
        assert exc_info.value.code == Code.UNAVAILABLE
        assert call_next.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_with_jitter(self):
        """Verify backoff doubles each retry with jitter applied."""
        interceptor = RetryUnaryInterceptor(max_attempts=4, initial_backoff=1.0, max_backoff=5.0, multiplier=2.0)
        call_next = AsyncMock(
            side_effect=[
                ConnectError(Code.UNAVAILABLE, "1"),
                ConnectError(Code.UNAVAILABLE, "2"),
                ConnectError(Code.UNAVAILABLE, "3"),
                "ok",
            ]
        )
        ctx, _ = _make_ctx_mock()

        sleep_durations = []

        async def mock_sleep(duration):
            sleep_durations.append(duration)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", mock_sleep)
            result = await interceptor.intercept_unary(call_next, "req", ctx)

        assert result == "ok"
        assert len(sleep_durations) == 3
        # Jitter applies factor of (0.5 + random()) to each backoff level (1.0, 2.0, 4.0)
        # so each sleep is in [0.5*base, 1.5*base)
        assert 0.5 <= sleep_durations[0] < 1.5  # base=1.0
        assert 1.0 <= sleep_durations[1] < 3.0  # base=2.0
        assert 2.0 <= sleep_durations[2] < 6.0  # base=4.0


class TestRetryServerStreamInterceptor:
    @pytest.mark.asyncio
    async def test_streams_without_retry(self):
        interceptor = RetryServerStreamInterceptor(max_attempts=3)

        async def call_next(req, ctx):
            yield "a"
            yield "b"

        ctx, _ = _make_ctx_mock()
        results = [item async for item in interceptor.intercept_server_stream(call_next, "req", ctx)]
        assert results == ["a", "b"]

    @pytest.mark.asyncio
    async def test_retries_stream_on_unavailable(self):
        interceptor = RetryServerStreamInterceptor(max_attempts=3, initial_backoff=0.001)
        call_count = 0

        async def call_next(req, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectError(Code.UNAVAILABLE, "down")
            yield "ok"

        ctx, _ = _make_ctx_mock()
        results = [item async for item in interceptor.intercept_server_stream(call_next, "req", ctx)]
        assert results == ["ok"]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_after_partial_yield(self):
        """If the stream already yielded data, retries restart the full stream (items may duplicate)."""
        interceptor = RetryServerStreamInterceptor(max_attempts=3, initial_backoff=0.001)
        call_count = 0

        async def call_next(req, ctx):
            nonlocal call_count
            call_count += 1
            yield "first"
            if call_count == 1:
                raise ConnectError(Code.UNAVAILABLE, "mid-stream failure")

        ctx, _ = _make_ctx_mock()
        results = []
        async for item in interceptor.intercept_server_stream(call_next, "req", ctx):
            results.append(item)
        assert results == ["first", "first"]
        assert call_count == 2
