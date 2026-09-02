import asyncio
import contextvars
import threading

import pytest

from flyte._utils.asyncify import run_sync_in_thread

# Context variable for testing context preservation
test_context_var: contextvars.ContextVar[str] = contextvars.ContextVar("test_context_var")


@pytest.mark.asyncio
async def test_basic_sync_function():
    """Test that a basic sync function can be called and returns the correct result."""

    def sync_add(a: int, b: int) -> int:
        return a + b

    result = await run_sync_in_thread(sync_add, 5, 7)
    assert result == 12


@pytest.mark.asyncio
async def test_sync_function_with_kwargs():
    """Test that kwargs are properly passed to the sync function."""

    def sync_multiply(x: int, y: int, multiplier: int = 1) -> int:
        return x * y * multiplier

    result = await run_sync_in_thread(sync_multiply, 3, 4, multiplier=2)
    assert result == 24


@pytest.mark.asyncio
async def test_context_variable_preservation():
    """Test that context variables are preserved when calling the sync function."""
    test_context_var.set("test_value")

    def get_context_value() -> str:
        return test_context_var.get()

    result = await run_sync_in_thread(get_context_value)
    assert result == "test_value"


@pytest.mark.asyncio
async def test_raises_error_on_async_function():
    """Test that TypeError is raised when trying to run an async function."""

    async def async_function():
        return 42

    with pytest.raises(TypeError) as exc_info:
        await run_sync_in_thread(async_function)

    assert "Cannot call run_sync_in_thread with async function" in str(exc_info.value)
    assert "async_function" in str(exc_info.value)


@pytest.mark.asyncio
async def test_sync_function_runs_as_plain_sync_code():
    """The sync function runs in its own thread with NO running event loop, so it can use asyncio.run()."""

    def get_execution_info() -> tuple:
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        # Plain sync code can spin up its own loop (e.g. third-party run_sync wrappers).
        async def _inner() -> int:
            return 42

        return has_running_loop, asyncio.run(_inner()), threading.current_thread().name

    has_running_loop, inner_result, thread_name = await run_sync_in_thread(get_execution_info)

    assert has_running_loop is False
    assert inner_result == 42
    # And it should be running in a dedicated thread
    assert "sync-executor" in thread_name


@pytest.mark.asyncio
async def test_thread_name_uniqueness():
    """Test that different invocations create threads with unique names."""
    thread_names = []

    def capture_thread_name() -> str:
        name = threading.current_thread().name
        thread_names.append(name)
        return name

    # Run multiple times
    name1 = await run_sync_in_thread(capture_thread_name)
    name2 = await run_sync_in_thread(capture_thread_name)

    # Thread names should be different due to random suffix
    assert name1 != name2
    assert "sync-executor" in name1
    assert "sync-executor" in name2
    assert "_from_" in name1
    assert "_from_" in name2


@pytest.mark.asyncio
async def test_exception_propagation():
    """Test that exceptions raised in sync functions are properly propagated."""

    def sync_function_that_raises():
        raise ValueError("Test error message")

    with pytest.raises(ValueError) as exc_info:
        await run_sync_in_thread(sync_function_that_raises)

    assert "Test error message" in str(exc_info.value)


@pytest.mark.asyncio
async def test_return_complex_types():
    """Test that complex return types are properly returned."""

    def sync_function_returning_dict() -> dict:
        return {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": True}}

    result = await run_sync_in_thread(sync_function_returning_dict)
    assert result == {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": True}}
