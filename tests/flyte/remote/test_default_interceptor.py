import re
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from grpc.aio import ClientCallDetails, Metadata

from flyte._context import internal_ctx
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.remote._client.auth._grpc_utils.default_metadata_interceptor import (
    DefaultMetadataStreamStreamInterceptor,
    DefaultMetadataStreamUnaryInterceptor,
    DefaultMetadataUnaryStreamInterceptor,
    DefaultMetadataUnaryUnaryInterceptor,
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


@pytest.fixture
def mock_call_details():
    """Create mock ClientCallDetails for testing."""
    return ClientCallDetails(
        method="/test.Service/TestMethod",
        timeout=30.0,
        metadata=Metadata(),
        credentials=None,
        wait_for_ready=False,
    )


@pytest.fixture
def mock_call_details_with_metadata():
    """Create mock ClientCallDetails with existing metadata."""
    return ClientCallDetails(
        method="/test.Service/TestMethod",
        timeout=30.0,
        metadata=Metadata(("existing-key", "existing-value")),
        credentials=None,
        wait_for_ready=False,
    )


class TestRequestIdGeneration:
    """Tests for request ID generation logic."""

    def test_generate_request_id_without_context(self):
        """Test that request ID falls back to UUID4 when no context is available."""
        # Generate request ID without context
        request_id = _generate_request_id()

        # Should be a valid UUID4
        try:
            UUID(request_id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False

        assert is_valid_uuid, f"Expected UUID4, got: {request_id}"

    def test_generate_request_id_with_context(self, task_context):
        """Test that request ID uses context information when available."""
        ctx = internal_ctx()
        with ctx.replace_task_context(task_context):
            request_id = _generate_request_id()

            # Should contain the context information
            assert "test-project" in request_id
            assert "test-domain" in request_id
            assert "test-run" in request_id
            assert "test-action" in request_id

            # Should match pattern: project-domain-run_name-action_name-salt
            # Salt is 4 alphanumeric characters
            pattern = r"test-project-test-domain-test-run-test-action-[a-z0-9]{4}"
            assert re.match(pattern, request_id), f"Request ID doesn't match expected pattern: {request_id}"

    def test_generate_request_id_with_context_has_salt(self, task_context):
        """Test that request ID generates different salts for consecutive calls."""
        ctx = internal_ctx()
        with ctx.replace_task_context(task_context):
            request_id_1 = _generate_request_id()
            request_id_2 = _generate_request_id()

            # Both should have the same base but different salts
            assert request_id_1.startswith("test-project-test-domain-test-run-test-action-")
            assert request_id_2.startswith("test-project-test-domain-test-run-test-action-")

            # The salts should be different (very high probability)
            assert request_id_1 != request_id_2


class TestDefaultMetadataUnaryUnaryInterceptor:
    """Tests for UnaryUnary interceptor."""

    @pytest.mark.asyncio
    async def test_intercept_adds_default_metadata(self, mock_call_details):
        """Test that interceptor adds accept and x-request-id headers."""
        interceptor = DefaultMetadataUnaryUnaryInterceptor()

        # Create mock continuation that captures the call details
        captured_call_details = None

        async def continuation(call_details, request):
            nonlocal captured_call_details
            captured_call_details = call_details
            # Return an awaitable mock (gRPC continuation returns a Call which is awaitable)
            mock_call = AsyncMock()
            mock_call.return_value = Mock()  # The actual response
            return mock_call()

        # Intercept the call
        request = Mock()
        await interceptor.intercept_unary_unary(continuation, mock_call_details, request)

        # Verify continuation was called
        assert captured_call_details is not None

        # Check metadata was added - convert to dict properly
        metadata_dict = {k: v for k, v in captured_call_details.metadata}
        assert "accept" in metadata_dict
        assert metadata_dict["accept"] == "application/grpc"
        assert "x-request-id" in metadata_dict
        assert len(metadata_dict["x-request-id"]) > 0

    @pytest.mark.asyncio
    async def test_intercept_preserves_existing_metadata(self, mock_call_details_with_metadata):
        """Test that interceptor preserves existing metadata."""
        interceptor = DefaultMetadataUnaryUnaryInterceptor()

        # Create mock continuation that captures the call details
        captured_call_details = None

        async def continuation(call_details, request):
            nonlocal captured_call_details
            captured_call_details = call_details
            # Return an awaitable mock
            mock_call = AsyncMock()
            mock_call.return_value = Mock()
            return mock_call()

        # Intercept the call
        request = Mock()
        await interceptor.intercept_unary_unary(continuation, mock_call_details_with_metadata, request)

        # Verify continuation was called
        assert captured_call_details is not None

        # Check metadata was added and existing was preserved
        metadata_dict = {k: v for k, v in captured_call_details.metadata}
        assert "existing-key" in metadata_dict
        assert metadata_dict["existing-key"] == "existing-value"
        assert "accept" in metadata_dict
        assert "x-request-id" in metadata_dict

    @pytest.mark.asyncio
    async def test_intercept_with_task_context(self, mock_call_details, task_context):
        """Test that interceptor uses task context for request ID."""
        interceptor = DefaultMetadataUnaryUnaryInterceptor()

        # Create mock continuation that captures the call details
        captured_call_details = None

        async def continuation(call_details, request):
            nonlocal captured_call_details
            captured_call_details = call_details
            # Return an awaitable mock
            mock_call = AsyncMock()
            mock_call.return_value = Mock()
            return mock_call()

        # Set up context
        ctx = internal_ctx()
        with ctx.replace_task_context(task_context):
            # Intercept the call
            request = Mock()
            await interceptor.intercept_unary_unary(continuation, mock_call_details, request)

            # Verify continuation was called
            assert captured_call_details is not None

            # Check request ID contains context information
            metadata_dict = {k: v for k, v in captured_call_details.metadata}
            request_id = metadata_dict["x-request-id"]
            assert "test-project" in request_id
            assert "test-domain" in request_id
            assert "test-run" in request_id
            assert "test-action" in request_id


class TestDefaultMetadataUnaryStreamInterceptor:
    """Tests for UnaryStream interceptor."""

    @pytest.mark.asyncio
    async def test_intercept_adds_metadata(self, mock_call_details):
        """Test that UnaryStream interceptor adds metadata."""
        interceptor = DefaultMetadataUnaryStreamInterceptor()

        # Create mock continuation
        mock_stream = AsyncMock()

        async def continuation(call_details, request):
            return mock_stream

        # Intercept the call
        request = Mock()
        result = await interceptor.intercept_unary_stream(continuation, mock_call_details, request)

        # Verify stream was returned
        assert result == mock_stream


class TestDefaultMetadataStreamUnaryInterceptor:
    """Tests for StreamUnary interceptor."""

    @pytest.mark.asyncio
    async def test_intercept_adds_metadata(self, mock_call_details):
        """Test that StreamUnary interceptor adds metadata."""
        interceptor = DefaultMetadataStreamUnaryInterceptor()

        # Create mock continuation
        mock_response = Mock()

        async def continuation(call_details, request_iterator):
            return mock_response

        # Intercept the call
        request_iterator = iter([Mock(), Mock()])
        result = await interceptor.intercept_stream_unary(continuation, mock_call_details, request_iterator)

        # Verify response was returned
        assert result == mock_response


class TestDefaultMetadataStreamStreamInterceptor:
    """Tests for StreamStream interceptor."""

    @pytest.mark.asyncio
    async def test_intercept_adds_metadata(self, mock_call_details):
        """Test that StreamStream interceptor adds metadata."""
        interceptor = DefaultMetadataStreamStreamInterceptor()

        # Create mock continuation
        mock_stream = AsyncMock()

        async def continuation(call_details, request_iterator):
            return mock_stream

        # Intercept the call
        request_iterator = iter([Mock(), Mock()])
        result = await interceptor.intercept_stream_stream(continuation, mock_call_details, request_iterator)

        # Verify stream was returned
        assert result == mock_stream


class TestAllInterceptorTypes:
    """Test that all interceptor types properly inject metadata."""

    @pytest.mark.asyncio
    async def test_all_interceptors_inject_both_headers(self, mock_call_details, task_context):
        """Test that all 4 interceptor types inject both accept and x-request-id headers."""
        interceptors = [
            DefaultMetadataUnaryUnaryInterceptor(),
            DefaultMetadataUnaryStreamInterceptor(),
            DefaultMetadataStreamUnaryInterceptor(),
            DefaultMetadataStreamStreamInterceptor(),
        ]

        ctx = internal_ctx()
        with ctx.replace_task_context(task_context):
            for interceptor in interceptors:
                # Test _inject_default_metadata directly
                updated_call_details = await interceptor._inject_default_metadata(mock_call_details)

                # Verify both headers are present
                metadata_dict = {k: v for k, v in updated_call_details.metadata}
                assert "accept" in metadata_dict, f"Missing 'accept' header in {interceptor.__class__.__name__}"
                assert metadata_dict["accept"] == "application/grpc"
                assert "x-request-id" in metadata_dict, (
                    f"Missing 'x-request-id' header in {interceptor.__class__.__name__}"
                )

                # Verify request ID contains context
                request_id = metadata_dict["x-request-id"]
                assert "test-project" in request_id
                assert "test-domain" in request_id
                assert "test-run" in request_id
                assert "test-action" in request_id
