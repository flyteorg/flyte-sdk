"""
Store utilities for Flyte.

This module provides functionality to store various artifacts from remote registries,
such as HuggingFace models.
"""

from ._hf_model import (
    HuggingFaceModelInfo,
    StoredModelInfo,
    ShardConfig,
    VLLMShardArgs,
    hf_model,
)

__all__ = [
    "HuggingFaceModelInfo",
    "StoredModelInfo",
    "ShardConfig",
    "VLLMShardArgs",
    "hf_model",
]
