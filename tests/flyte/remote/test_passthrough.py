import asyncio
from unittest import mock

import pytest
from grpc.aio import Metadata

from flyte.remote._client.auth._authenticators.passthrough import PassthroughAuthenticator
from flyte.remote._client.auth._keyring import Credentials


@pytest.fixture
def endpoint():
    """Fixture for test endpoint."""
    return "dns:///test-endpoint.com"


@pytest.fixture
def authenticator(endpoint):
    """Fixture for PassthroughAuthenticator instance."""
    return PassthroughAuthenticator(endpoint=endpoint)


def test_initialization(authenticator, endpoint):
    """Test PassthroughAuthenticator initialization."""
    assert authenticator._endpoint == endpoint
    assert authenticator._creds is None
    assert authenticator._creds_id == "passthrough"


def test_get_credentials(authenticator):
    """Test get_credentials returns a dummy credential."""
    creds = authenticator.get_credentials()

    assert isinstance(creds, Credentials)
    assert creds.access_token == "passthrough"
    # Note: for_endpoint is transformed by strip_scheme validator
    assert creds.for_endpoint == "test-endpoint.com"


@pytest.mark.asyncio
async def test_get_grpc_call_auth_metadata_with_context(authenticator):
    """Test get_grpc_call_auth_metadata when metadata is present in context."""
    # Mock the context to return metadata
    with mock.patch("flyte.remote._auth_metadata.get_auth_metadata") as mock_get_metadata:
        mock_get_metadata.return_value = (("key1", "value1"), ("key2", "value2"))

        result = await authenticator.get_grpc_call_auth_metadata()

        # Verify the result
        assert result is not None
        assert result.creds_id == "passthrough"
        assert isinstance(result.pairs, Metadata)

        # Convert metadata to list for comparison
        metadata_list = list(result.pairs)
        assert metadata_list == [("key1", "value1"), ("key2", "value2")]


@pytest.mark.asyncio
async def test_get_grpc_call_auth_metadata_without_context(authenticator):
    """Test get_grpc_call_auth_metadata when no metadata is in context."""
    # Mock the context to return None
    with mock.patch("flyte.remote._auth_metadata.get_auth_metadata") as mock_get_metadata:
        mock_get_metadata.return_value = None

        result = await authenticator.get_grpc_call_auth_metadata()

        # Should return None when no metadata is available
        assert result is None


@pytest.mark.asyncio
async def test_do_refresh_credentials(authenticator):
    """Test _do_refresh_credentials returns a dummy credential."""
    creds = await authenticator._do_refresh_credentials()

    assert isinstance(creds, Credentials)
    assert creds.access_token == "passthrough"
    # Note: for_endpoint is transformed by strip_scheme validator
    assert creds.for_endpoint == "test-endpoint.com"


@pytest.mark.asyncio
async def test_auth_metadata_context_integration(endpoint):
    """Test that auth_metadata context manager integrates with PassthroughAuthenticator."""
    from flyte._context import internal_ctx
    from flyte.remote._auth_metadata import auth_metadata

    authenticator = PassthroughAuthenticator(endpoint=endpoint)

    # Use the auth_metadata context manager
    with auth_metadata(("api-key", "test-key-123"), ("user-id", "user-456")):
        # Verify context has the metadata
        ctx = internal_ctx()
        assert ctx.data.metadata is not None
        assert len(ctx.data.metadata) == 2

        result = await authenticator.get_grpc_call_auth_metadata()

        # Verify the authenticator extracted the metadata correctly
        assert result is not None
        metadata_list = list(result.pairs)
        assert metadata_list == [("api-key", "test-key-123"), ("user-id", "user-456")]


@pytest.mark.asyncio
async def test_auth_metadata_empty_context(endpoint):
    """Test PassthroughAuthenticator outside auth_metadata context."""
    authenticator = PassthroughAuthenticator(endpoint=endpoint)

    # Outside the auth_metadata context, should return None
    result = await authenticator.get_grpc_call_auth_metadata()
    assert result is None


@pytest.mark.asyncio
async def test_concurrent_auth_metadata_contexts_isolated(endpoint):
    """
    Test that multiple concurrent coroutines can set context.metadata differently
    without polluting each other. This ensures context is properly isolated per coroutine.
    """
    from flyte.remote._auth_metadata import auth_metadata

    authenticator = PassthroughAuthenticator(endpoint=endpoint)
    results = {}

    async def coroutine_with_metadata(coroutine_id: str, metadata_tuples: tuple):
        """
        Coroutine that sets its own metadata and verifies it's not polluted by other coroutines.
        """
        # Set this coroutine's specific metadata
        with auth_metadata(*metadata_tuples):
            # Small delay to ensure coroutines overlap in execution
            await asyncio.sleep(0.01)

            result = await authenticator.get_grpc_call_auth_metadata()

            # Store result for verification
            results[coroutine_id] = result

            # Add another small delay to ensure overlap
            await asyncio.sleep(0.01)

        return coroutine_id

    # Create multiple coroutines with different metadata
    coro1 = coroutine_with_metadata("coro1", (("user", "alice"), ("role", "admin")))
    coro2 = coroutine_with_metadata("coro2", (("user", "bob"), ("role", "user")))
    coro3 = coroutine_with_metadata("coro3", (("user", "charlie"), ("role", "guest")))

    # Run all coroutines concurrently
    await asyncio.gather(coro1, coro2, coro3)

    # Verify each coroutine got its own isolated metadata
    assert "coro1" in results
    assert "coro2" in results
    assert "coro3" in results

    # Verify coro1's metadata
    metadata1 = list(results["coro1"].pairs)
    assert metadata1 == [("user", "alice"), ("role", "admin")]

    # Verify coro2's metadata
    metadata2 = list(results["coro2"].pairs)
    assert metadata2 == [("user", "bob"), ("role", "user")]

    # Verify coro3's metadata
    metadata3 = list(results["coro3"].pairs)
    assert metadata3 == [("user", "charlie"), ("role", "guest")]


def test_concurrent_auth_metadata_contexts_isolated_with_syncify(endpoint):
    """
    Test that multiple concurrent operations using @syncify maintain context isolation.
    This verifies that context variables are properly carried through the syncify thread boundary.
    """
    import threading

    from flyte.remote._auth_metadata import auth_metadata
    from flyte.syncify import syncify

    authenticator = PassthroughAuthenticator(endpoint=endpoint)
    results = {}
    results_lock = threading.Lock()

    @syncify
    async def async_operation_with_metadata(operation_id: str, metadata_tuples: tuple):
        """
        Async operation wrapped with @syncify that sets its own metadata.
        """
        # Set this operation's specific metadata
        with auth_metadata(*metadata_tuples):
            # Small delay to ensure operations overlap
            await asyncio.sleep(0.01)

            result = await authenticator.get_grpc_call_auth_metadata()

            # Store result for verification (thread-safe)
            with results_lock:
                results[operation_id] = result

            # Add another small delay to ensure overlap
            await asyncio.sleep(0.01)

        return operation_id

    # Create threads that will run syncified operations with different metadata
    def run_operation(operation_id: str, metadata_tuples: tuple):
        # Call the syncified function (which will run async code in the background loop)
        async_operation_with_metadata(operation_id, metadata_tuples)

    thread1 = threading.Thread(target=run_operation, args=("thread1", (("user", "alice"), ("role", "admin"))))
    thread2 = threading.Thread(target=run_operation, args=("thread2", (("user", "bob"), ("role", "user"))))
    thread3 = threading.Thread(target=run_operation, args=("thread3", (("user", "charlie"), ("role", "guest"))))

    # Start all threads
    thread1.start()
    thread2.start()
    thread3.start()

    # Wait for all threads to complete
    thread1.join()
    thread2.join()
    thread3.join()

    # Verify each thread got its own isolated metadata
    assert "thread1" in results
    assert "thread2" in results
    assert "thread3" in results

    # Verify thread1's metadata
    metadata1 = list(results["thread1"].pairs)
    assert metadata1 == [("user", "alice"), ("role", "admin")]

    # Verify thread2's metadata
    metadata2 = list(results["thread2"].pairs)
    assert metadata2 == [("user", "bob"), ("role", "user")]

    # Verify thread3's metadata
    metadata3 = list(results["thread3"].pairs)
    assert metadata3 == [("user", "charlie"), ("role", "guest")]
