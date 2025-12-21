"""
Timer utilities for emitting timing metrics via structured logging.
"""

import functools
import inspect
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, overload

from flyte._logging import logger

# Context variable to track the stack of timer names for nested timers
_timer_stack: ContextVar[List[str]] = ContextVar("timer_stack", default=[])

F = TypeVar("F", bound=Callable[..., Any])


@asynccontextmanager
async def async_timer(metric_name: str, extra_fields: Optional[Dict[str, Any]] = None):
    """
    Async context manager for timing asynchronous code blocks.
    Emits timing metrics via structured logging.

    When timers are nested, metric names are automatically concatenated with dots.
    For example, if "outer" contains "inner", the inner metric will be named "outer.inner".

    Example:
        async with async_timer("download_inputs"):
            # async code to time
            await something()

    Nested example:
        async with async_timer("load_and_run"):
            async with async_timer("download_bundle"):
                # This will emit metric "load_and_run.download_bundle"
                await download()

    :param metric_name: Name of the metric to emit
    :param extra_fields: Additional fields to include in the log record
    """
    # Get current timer stack and add this timer to it
    stack = _timer_stack.get().copy()
    stack.append(metric_name)
    token = _timer_stack.set(stack)

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        full_metric_name = ".".join(stack)
        _emit_metric(full_metric_name, duration, extra_fields)
        # Restore previous stack
        _timer_stack.reset(token)


@contextmanager
def timer(metric_name: str, extra_fields: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing synchronous code blocks.
    Emits timing metrics via structured logging.

    When timers are nested, metric names are automatically concatenated with dots.
    For example, if "outer" contains "inner", the inner metric will be named "outer.inner".

    Example:
        with timer("download_code"):
            # code to time
            pass

    Nested example:
        with timer("load_task"):
            with timer("download_bundle"):
                # This will emit metric "load_task.download_bundle"
                pass

    :param metric_name: Name of the metric to emit
    :param extra_fields: Additional fields to include in the log record
    """
    # Get current timer stack and add this timer to it
    stack = _timer_stack.get().copy()
    stack.append(metric_name)
    token = _timer_stack.set(stack)

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        full_metric_name = ".".join(stack)
        _emit_metric(full_metric_name, duration, extra_fields)
        # Restore previous stack
        _timer_stack.reset(token)


def _emit_metric(metric_name: str, duration: float, extra_fields: Optional[Dict[str, Any]] = None):
    """
    Emit a timing metric via structured logging.

    :param metric_name: Name of the metric (may be hierarchical with dots)
    :param duration: Duration in seconds
    :param extra_fields: Additional fields to include in the log record
    """
    extra = {
        "metric_type": "timer",
        "metric_name": metric_name,
        "duration_seconds": duration,
    }
    if extra_fields:
        extra.update(extra_fields)

    logger.info(f"{metric_name} completed in {duration:.4f}s", extra=extra)


# Decorator overloads for better type hints
@overload
def timed(func: F) -> F: ...


@overload
def timed(
    name: Optional[str] = None, *, extra_fields: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]: ...


def timed(
    name_or_func: Optional[Union[str, F]] = None, *, extra_fields: Optional[Dict[str, Any]] = None
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to automatically time a function (works for both sync and async).

    Usage:
        # Use function name as metric name:
        @timed
        def my_function():
            pass

        @timed
        async def my_async_function():
            pass

        # Custom metric name:
        @timed("custom_name")
        def my_function():
            pass

        # With extra fields:
        @timed("custom_name", extra_fields={"key": "value"})
        async def my_function():
            pass

    :param name_or_func: Either the function to decorate, or a custom metric name
    :param extra_fields: Additional fields to include in the log record
    """

    def decorator(func: F) -> F:
        # Determine metric name
        if isinstance(name_or_func, str):
            metric_name = name_or_func
        else:
            metric_name = func.__name__

        # Wrap based on function type
        if inspect.isasyncgenfunction(func):
            # Async generator function
            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Get current timer stack and add this timer to it
                stack = _timer_stack.get().copy()
                stack.append(metric_name)
                token = _timer_stack.set(stack)

                start = time.perf_counter()
                try:
                    # Yield all items from the async generator
                    async for item in func(*args, **kwargs):
                        yield item
                finally:
                    duration = time.perf_counter() - start
                    full_metric_name = ".".join(stack)
                    _emit_metric(full_metric_name, duration, extra_fields)
                    _timer_stack.reset(token)

            return async_gen_wrapper  # type: ignore[return-value]

        elif inspect.iscoroutinefunction(func):
            # Regular async function
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with async_timer(metric_name, extra_fields=extra_fields):
                    result = await func(*args, **kwargs)

                    # Detect if this is a context manager factory (e.g., @asynccontextmanager)
                    if hasattr(result, "__aenter__") and hasattr(result, "__aexit__"):
                        raise TypeError(
                            f"@timed cannot be used on top of @asynccontextmanager for '{func.__name__}'. "
                            f"Use decorator order: @asynccontextmanager then @timed instead.\n"
                            f"Example:\n"
                            f"  @asynccontextmanager\n"
                            f"  @timed\n"
                            f"  async def {func.__name__}(): ..."
                        )

                    return result

            return async_wrapper  # type: ignore[return-value]

        else:
            # Sync function (including generators)
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with timer(metric_name, extra_fields=extra_fields):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    # Handle @timed vs @timed(...) syntax
    if callable(name_or_func) and not isinstance(name_or_func, str):
        # @timed (no parentheses) - name_or_func is the function itself
        return decorator(name_or_func)
    else:
        # @timed("name") or @timed() (with parentheses)
        return decorator
