"""
Timer utilities for emitting timing metrics via structured logging.
"""

import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Optional

from flyte._logging import logger


@contextmanager
def timer(metric_name: str, extra_fields: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing synchronous code blocks.
    Emits timing metrics via structured logging.

    Example:
        with timer("download_code"):
            # code to time
            pass

    :param metric_name: Name of the metric to emit
    :param extra_fields: Additional fields to include in the log record
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        _emit_metric(metric_name, duration, extra_fields)


@asynccontextmanager
async def async_timer(metric_name: str, extra_fields: Optional[Dict[str, Any]] = None):
    """
    Async context manager for timing asynchronous code blocks.
    Emits timing metrics via structured logging.

    Example:
        async with async_timer("download_inputs"):
            # async code to time
            await something()

    :param metric_name: Name of the metric to emit
    :param extra_fields: Additional fields to include in the log record
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        _emit_metric(metric_name, duration, extra_fields)


def _emit_metric(metric_name: str, duration: float, extra_fields: Optional[Dict[str, Any]] = None):
    """
    Emit a timing metric via structured logging.

    :param metric_name: Name of the metric
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
