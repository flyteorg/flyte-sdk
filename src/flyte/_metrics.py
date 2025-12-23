"""
Timer utilities for emitting timing metrics via structured logging.
"""

import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from flyte._logging import logger

# Context variable to track the stack of timer names for nested timers
_timer_stack: ContextVar[List[str]] = ContextVar("timer_stack", default=[])


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
