"""
## ML training utilities

This package contains helpers for common machine-learning training patterns
(e.g. early stopping) that are not part of the core task-authoring API.
"""

__all__ = [
    "EarlyStopping",
]

from ._early_stopping import EarlyStopping
