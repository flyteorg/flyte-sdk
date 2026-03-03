"""
Flyte durable utilities.

This module provides deterministic, crash-resilient replacements for time-related functions.
Usage of ``time.time()``, ``time.sleep()`` or ``asyncio.sleep()`` introduces non-determinism.
The utilities here persist state across crashes and restarts, making workflows durable.

- :func:`sleep` - a durable replacement for ``time.sleep`` / ``asyncio.sleep``
- :func:`time` - a durable replacement for ``time.time``
- :func:`now` - a durable replacement for ``datetime.now``
"""

from ._time import durable_now as now
from ._time import durable_sleep as sleep
from ._time import durable_time as time

__all__ = [
    "now",
    "sleep",
    "time",
]
