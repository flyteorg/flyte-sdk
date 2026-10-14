"""
Flyte SDK for authoring compound AI applications, services and workflows.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ._cache import Cache, CachePolicy, CacheRequest
from ._checkpoint import BaseCheckpoint, Checkpoint, latest_checkpoint
from ._condition import ConditionWebhook, new_condition
from ._context import ctx
from ._custom_context import custom_context, get_custom_context
from ._doc import Documentation
from ._environment import Environment
from ._excepthook import custom_excepthook
from ._group import group
from ._image import Image
from ._initialize import (
    current_domain,
    current_project,
    init,
    init_from_api_key,
    init_from_config,
    init_in_cluster,
    init_passthrough,
    is_control_plane_available,
)
from ._interactive_run_context import load_interactive_ctx
from ._link import Link
from ._logging import logger as system_logger
from ._logging import user_logger as logger
from ._map import map
from ._pod import PodTemplate
from ._resources import AMD_GPU, GPU, HABANA_GAUDI, TPU, Device, DeviceClass, Neuron, Resources
from ._retry import Backoff, RetryStrategy
from ._reusable_environment import ReusePolicy
from ._run import rerun, run, with_runcontext
from ._run_python_script import load_plugin_config, run_python_script
from ._secret import Secret, SecretRequest
from ._serve import AppHandle, serve, with_servecontext
from ._task import AsyncFunctionTaskTemplate, TaskTemplate
from ._task_environment import TaskEnvironment
from ._timeout import Timeout, TimeoutType
from ._trace import trace
from ._trigger import Cron, FixedRate, OnArtifact, Trigger, TriggeredArtifact, TriggeredPartition, TriggerTime
from ._version import __version__

if TYPE_CHECKING:
    from ._build import ImageBuild, build
    from ._deploy import build_images, deploy

# Build and deploy machinery is only needed by whoever builds or deploys, never
# by a task running in a pod; importing it eagerly cost every task start
# ~0.25s. Resolved on first use instead (PEP 562).
_LAZY = {
    "ImageBuild": "._build",
    "build": "._build",
    "build_images": "._deploy",
    "deploy": "._deploy",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


sys.excepthook = custom_excepthook


def version() -> str:
    """
    Returns the version of the Flyte SDK.
    """
    return __version__


__all__ = [
    "AMD_GPU",
    "GPU",
    "HABANA_GAUDI",
    "TPU",
    "AppHandle",
    "AsyncFunctionTaskTemplate",
    "Backoff",
    "BaseCheckpoint",
    "Cache",
    "CachePolicy",
    "CacheRequest",
    "Checkpoint",
    "ConditionWebhook",
    "Cron",
    "Device",
    "DeviceClass",
    "Documentation",
    "Environment",
    "FixedRate",
    "Image",
    "ImageBuild",
    "Link",
    "Neuron",
    "OnArtifact",
    "PodTemplate",
    "Resources",
    "RetryStrategy",
    "ReusePolicy",
    "Secret",
    "SecretRequest",
    "TaskEnvironment",
    "TaskTemplate",
    "Timeout",
    "TimeoutType",
    "Trigger",
    "TriggerTime",
    "TriggeredArtifact",
    "TriggeredPartition",
    "__version__",
    "build",
    "build_images",
    "ctx",
    "current_domain",
    "current_project",
    "custom_context",
    "deploy",
    "get_custom_context",
    "group",
    "init",
    "init_from_api_key",
    "init_from_config",
    "init_in_cluster",
    "init_passthrough",
    "is_control_plane_available",
    "latest_checkpoint",
    "load_interactive_ctx",
    "load_plugin_config",
    "logger",
    "map",
    "new_condition",
    "rerun",
    "run",
    "run_python_script",
    "serve",
    "system_logger",
    "trace",
    "version",
    "with_runcontext",
    "with_servecontext",
]
