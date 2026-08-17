"""
## ML training utilities

This package contains helpers for common machine-learning training patterns
(e.g. early stopping, metric tracking, gradient accumulation) that are not part
of the core task-authoring API.
"""

__all__ = [
    "EarlyStopping",
    "GradientAccumulator",
    "MetricTracker",
]

from ._early_stopping import EarlyStopping
from ._gradient_accumulator import GradientAccumulator
from ._metric_tracker import MetricTracker
