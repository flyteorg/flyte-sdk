import re
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from flyte.remote._client.auth._authenticators.base import AuthHeaders
from flyte.remote._client.auth._interceptors.auth import (
    AuthServerStreamInterceptor,
    AuthUnaryInterceptor,
)

from flyte._context import internal_ctx
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.remote._client.auth._interceptors.default_metadata import (
    DefaultMetadataInterceptor,
    _generate_request_id,
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
        ctx, headers = _make_ctx_mock()

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
        ctx, headers = _make_ctx_mock()

        result = await interceptor.intercept_unary(call_next, "request", ctx)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_does_not_retry_other_errors(self):
        auth = _make_mock_authenticator({"authorization": "Bearer token"})
        interceptor = AuthUnaryInterceptor(lambda: auth)

        call_next = AsyncMock(side_effect=ConnectError(Code.NOT_FOUND, "not found"))
        ctx, headers = _make_ctx_mock()

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

        ctx, headers = _make_ctx_mock()

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
            yield  # make it a generator  # noqa: RUF027

        ctx, headers = _make_ctx_mock()

        with pytest.raises(ConnectError) as exc_info:
            async for _ in interceptor.intercept_server_stream(mock_call_next, "request", ctx):
                pass
        assert exc_info.value.code == Code.NOT_FOUND
