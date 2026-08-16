"""
## ML training utilities

This package contains helpers for common machine-learning training patterns
(e.g. early stopping, metric tracking) that are not part of the core
task-authoring API.
"""

__all__ = [
    "EarlyStopping",
    "MetricTracker",
]

from ._early_stopping import EarlyStopping
from ._metric_tracker import MetricTracker
