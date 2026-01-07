"""
Timer utilities for emitting timing metrics via structured logging.
"""

import time
from typing import Any, Dict, Optional

from flyte._logging import logger


class Stopwatch:
    """
    Simple stopwatch for timing code blocks.
    Emits timing metrics via structured logging when stopped.

    Example:
        sw = Stopwatch("download_inputs")
        sw.start()
        # code to time
        sw.stop()

    :param metric_name: Name of the metric to emit
    :param extra_fields: Additional fields to include in the log record
    """

    def __init__(self, metric_name: str, extra_fields: Optional[Dict[str, Any]] = None):
        self.metric_name = metric_name
        self.extra_fields = extra_fields
        self._start_time: Optional[float] = None

    def start(self):
        """Start the stopwatch."""
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the stopwatch and emit the timing metric."""
        if self._start_time is None:
            raise RuntimeError(f"Stopwatch '{self.metric_name}' was never started")
        duration = time.perf_counter() - self._start_time
        _emit_metric(self.metric_name, duration, self.extra_fields)


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

    logger.info(f"Stopwatch: {metric_name} completed in {duration:.4f}s", extra=extra)
