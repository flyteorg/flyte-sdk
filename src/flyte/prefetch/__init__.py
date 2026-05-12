"""
Prefetch utilities for Flyte.

This module provides functionality to prefetch various artifacts from remote registries,
such as HuggingFace models.
"""

from ._hf_model import (
    HuggingFaceModelInfo,
    ShardConfig,
    StoredModelInfo,
    VLLMShardArgs,
    hf_model,
)

__all__ = [
    "HuggingFaceModelInfo",
    "ShardConfig",
    "StoredModelInfo",
    "VLLMShardArgs",
    "hf_model",
]
