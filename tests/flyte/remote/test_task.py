import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

import flyte
import flyte.errors
from flyte.remote._task import LazyEntity, TaskDetails


class TestLazyEntity:
    """Test suite for LazyEntity class focusing on call behavior and override functionality."""

    @pytest.fixture
    def mock_task_details(self):
        """Create a mock TaskDetails object."""
        # Create an AsyncMock that can be called directly
        mock_task = AsyncMock()
        mock_task.name = "test_task"
        mock_task.required_args = ("arg1",)
        return mock_task

    @pytest.fixture
    def mock_getter(self, mock_task_details):
        """Create a mock async getter function."""

        async def getter():
            return mock_task_details

        return getter

    @pytest.fixture
    def lazy_entity(self, mock_getter):
        """Create a LazyEntity instance for testing."""
        return LazyEntity("test_task", mock_getter)

    def test_lazy_entity_initialization(self, lazy_entity):
        """Test LazyEntity proper initialization."""
        assert lazy_entity.name == "test_task"
        assert lazy_entity._task is None
        assert isinstance(lazy_entity._mutex, asyncio.Lock)
        assert callable(lazy_entity._getter)

    @pytest.mark.asyncio
    async def test_fetch_caches_task(self, lazy_entity, mock_task_details):
        """Test that fetch method caches the task after first call."""
        # First fetch should call the getter
        task = await lazy_entity.fetch.aio()
        assert task == mock_task_details
        assert lazy_entity._task == mock_task_details

        # Second fetch should return cached task without calling getter again
        task2 = await lazy_entity.fetch.aio()
        assert task2 == mock_task_details
        assert task2 is task  # Same object reference

    @pytest.mark.asyncio
    async def test_fetch_raises_on_none_task(self):
        """Test that fetch raises RuntimeError when getter returns None."""

        async def failing_getter():
            return None

        lazy_entity = LazyEntity("failing_task", failing_getter)

        with pytest.raises(RuntimeError, match="Error downloading the task failing_task"):
            await lazy_entity.fetch.aio()

    @pytest.mark.asyncio
    async def test_call_local_execution_raises_error(self, lazy_entity, mock_task_details):
        """Test that calling LazyEntity locally raises RemoteTaskUsageError."""
        # Configure the mock task to raise RemoteTaskUsage when called
        mock_task_details.side_effect = flyte.errors.RemoteTaskUsageError(
            "Remote tasks [test_task] cannot be executed locally, only remotely."
        )

        with pytest.raises(flyte.errors.RemoteTaskUsageError, match="cannot be executed locally"):
            await lazy_entity(arg1="test_value")

    @pytest.mark.asyncio
    async def test_call_remote_execution_with_controller(self, lazy_entity, mock_task_details):
        """Test that calling LazyEntity in task context with controller submits to controller."""
        expected_result = "controller_result"

        # Configure the mock task to return the expected result when called
        mock_task_details.return_value = expected_result

        # Call the lazy entity
        result = await lazy_entity(arg1="test_value")

        # Verify the call was forwarded to the task
        mock_task_details.assert_called_once_with(arg1="test_value")
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_call_forwards_args_and_kwargs(self, lazy_entity, mock_task_details):
        """Test that __call__ properly forwards arguments and keyword arguments."""
        expected_result = "forwarded_result"
        mock_task_details.return_value = expected_result

        # Test with both args and kwargs
        result = await lazy_entity("pos_arg", kwarg1="value1", kwarg2="value2")

        mock_task_details.assert_called_once_with("pos_arg", kwarg1="value1", kwarg2="value2")
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_override_method(self, lazy_entity, mock_task_details):
        """Test that override method calls TaskDetails.override and returns self."""
        # Mock the override method on TaskDetails
        mock_task_details.override = Mock(return_value=mock_task_details)

        # Call override with some parameters (note: not awaited since override.aio() returns sync)
        result = await lazy_entity.override.aio(short_name="new_name", retries=3)

        # Verify override was called on the task details
        mock_task_details.override.assert_called_once_with(short_name="new_name", retries=3)

        # Verify it returns the different LazyEntity instance
        assert result is not lazy_entity
        assert result.name is lazy_entity.name

    @pytest.mark.asyncio
    async def test_override_fetches_task_if_not_cached(self, mock_getter, mock_task_details):
        """Test that override fetches the task if it's not already cached."""
        lazy_entity = LazyEntity("test_task", mock_getter)
        mock_task_details.override = Mock(return_value=mock_task_details)

        # Ensure task is not cached initially
        assert lazy_entity._task is None

        # Call override
        await lazy_entity.override.aio(short_name="new_name")

        # Verify task was fetched and cached
        assert lazy_entity._task == mock_task_details
        mock_task_details.override.assert_called_once_with(short_name="new_name")

    def test_repr_and_str_methods(self, lazy_entity):
        """Test string representation methods."""
        assert str(lazy_entity) == "Future for task with name test_task"
        assert repr(lazy_entity) == str(lazy_entity)

    def test_override_sync_wrapper(self, lazy_entity, mock_task_details):
        """Test that override method works with sync wrapper."""
        mock_task_details.override = Mock(return_value=mock_task_details)

        # Test the sync version of override
        result = lazy_entity.override(short_name="new_name")
        assert result is not lazy_entity
        assert result.name == lazy_entity.name
        mock_task_details.override.assert_called_once_with(short_name="new_name")


class TestLazyEntityIntegration:
    """Integration tests for LazyEntity with real TaskDetails behavior."""

    @pytest.fixture
    def mock_task_pb2(self):
        """Create a mock task protobuf for TaskDetails."""
        task_pb2 = Mock()
        task_pb2.task_id.name = "integration_task"
        task_pb2.task_id.version = "v1.0.0"
        task_pb2.spec.default_inputs = []

        # Mock the task template interface
        task_pb2.spec.task_template.interface.inputs = {}
        task_pb2.spec.task_template.interface.outputs = {}

        return task_pb2

    @pytest.fixture
    def task_details(self, mock_task_pb2):
        """Create a real TaskDetails instance for integration testing."""
        return TaskDetails(mock_task_pb2)

    @pytest.fixture
    def integration_lazy_entity(self, task_details):
        """Create LazyEntity with real TaskDetails."""

        async def getter():
            return task_details

        return LazyEntity("integration_task", getter)

    @pytest.mark.asyncio
    async def test_integration_call_local_execution(self, integration_lazy_entity):
        """Integration test for local execution error."""
        with pytest.raises(flyte.errors.RemoteTaskUsageError, match="cannot be executed locally"):
            await integration_lazy_entity()

    @pytest.mark.asyncio
    async def test_integration_fetch_with_real_taskdetails(self, integration_lazy_entity, task_details):
        """Integration test for fetch with real TaskDetails."""
        fetched_task = await integration_lazy_entity.fetch.aio()
        assert fetched_task is task_details
        assert fetched_task.name == "integration_task"
