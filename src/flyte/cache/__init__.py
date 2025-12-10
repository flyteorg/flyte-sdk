"""
Cache utilities for Flyte.

This module provides functionality to cache various artifacts from remote registries,
such as HuggingFace models.
"""

from ._hf_model import (
    CachedModelInfo,
    HuggingFaceModelInfo,
    ShardConfig,
    VLLMShardArgs,
    hf_model,
)

__all__ = [
    "CachedModelInfo",
    "HuggingFaceModelInfo",
    "ShardConfig",
    "VLLMShardArgs",
    "hf_model",
]

