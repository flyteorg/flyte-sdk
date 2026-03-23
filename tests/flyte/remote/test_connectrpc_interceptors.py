import re
from unittest.mock import Mock
from uuid import UUID

import pytest

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
