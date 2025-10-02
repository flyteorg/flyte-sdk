"""
Fallback Runner Pattern

A higher-order function that provides automatic fallback task execution when the
primary task fails with specific exceptions. This pattern is useful for:
- Implementing graceful degradation
- Providing alternative processing methods
- Handling different failure scenarios with different recovery strategies

Usage:
    @env.task
    async def primary_task(data: str) -> dict:
        # Primary processing that might fail
        return expensive_api_call(data)

    @env.task
    async def fallback_task(data: str) -> dict:
        # Fallback processing
        return local_processing(data)

    # Run with fallback
    result = await run_with_fallback(
        primary_task,
        fallback_task,
        data="input",
        fallback_exceptions=[APIError, TimeoutError]
    )
"""

from typing import Callable, List, Optional, Type, TypeVar

R = TypeVar("R")


async def run_with_fallback(
    primary_task: Callable[..., R],
    fallback_task: Callable[..., R],
    *args,
    fallback_exceptions: Optional[List[Type[Exception]]] = None,
    log_failures: bool = True,
    **kwargs,
) -> R:
    """
    Run a primary task with automatic fallback on specified exceptions.

    Args:
        primary_task: The main task to execute
        fallback_task: The task to run if primary fails
        *args: Positional arguments for both tasks
        fallback_exceptions: List of exception types that trigger fallback
                           (None means fallback on any exception)
        log_failures: Whether to log failure information
        **kwargs: Keyword arguments for both tasks

    Returns:
        Result from either primary or fallback task

    Raises:
        Exception: Re-raises exceptions that don't match fallback_exceptions
                  or exceptions from the fallback task
    """
    try:
        if log_failures:
            print("Attempting primary task...")
        result = await primary_task(*args, **kwargs)
        if log_failures:
            print("Primary task succeeded")
        return result

    except Exception as e:
        # Check if this exception should trigger fallback
        should_fallback = fallback_exceptions is None or any(
            isinstance(e, exc_type) for exc_type in fallback_exceptions
        )

        # If not a direct match, check if it's a Flyte error with a code that matches our target exceptions
        if not should_fallback and hasattr(e, "code") and fallback_exceptions:
            error_code = str(e.code)
            should_fallback = any(exc_type.__name__ == error_code for exc_type in fallback_exceptions)

        if should_fallback:
            if log_failures:
                print(f"Primary task failed with {type(e).__name__}: {e}")
                print("Running fallback task...")

            try:
                result = await fallback_task(*args, **kwargs)
                if log_failures:
                    print("Fallback task succeeded")
                return result
            except Exception as fallback_error:
                if log_failures:
                    print(f"Fallback task also failed with {type(fallback_error).__name__}: {fallback_error}")
                raise fallback_error
        else:
            if log_failures:
                print(f"Primary task failed with non-fallback exception {type(e).__name__}: {e}")
            raise e
