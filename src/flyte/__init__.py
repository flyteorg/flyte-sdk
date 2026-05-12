"""
Flyte SDK for authoring compound AI applications, services and workflows.
"""

from __future__ import annotations

import sys

from ._build import ImageBuild, build
from ._cache import Cache, CachePolicy, CacheRequest
from ._checkpoint import BaseCheckpoint, Checkpoint, latest_checkpoint
from ._context import ctx
from ._custom_context import custom_context, get_custom_context
from ._deploy import build_images, deploy
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
)
from ._link import Link
from ._logging import user_logger as logger
from ._map import map
from ._pod import PodTemplate
from ._resources import AMD_GPU, GPU, HABANA_GAUDI, TPU, Device, DeviceClass, Neuron, Resources
from ._retry import RetryStrategy
from ._reusable_environment import ReusePolicy
from ._run import run, with_runcontext
from ._run_python_script import run_python_script
from ._secret import Secret, SecretRequest
from ._serve import AppHandle, serve, with_servecontext
from ._task_environment import TaskEnvironment
from ._timeout import Timeout, TimeoutType
from ._trace import trace
from ._trigger import Cron, FixedRate, Trigger, TriggerTime
from ._version import __version__

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
    "BaseCheckpoint",
    "Cache",
    "CachePolicy",
    "CacheRequest",
    "Checkpoint",
    "Cron",
    "Device",
    "DeviceClass",
    "Environment",
    "FixedRate",
    "Image",
    "ImageBuild",
    "Link",
    "Neuron",
    "PodTemplate",
    "Resources",
    "RetryStrategy",
    "ReusePolicy",
    "Secret",
    "SecretRequest",
    "TaskEnvironment",
    "Timeout",
    "TimeoutType",
    "Trigger",
    "TriggerTime",
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
    "latest_checkpoint",
    "logger",
    "map",
    "run",
    "run_python_script",
    "serve",
    "trace",
    "version",
    "with_runcontext",
    "with_servecontext",
]
