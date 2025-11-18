"""
Flyte SDK for authoring compound AI applications, services and workflows.
"""

from __future__ import annotations

import sys

from ._build import build
from ._cache import Cache, CachePolicy, CacheRequest
from ._context import ctx
from ._custom_context import custom_context, get_custom_context
from ._deploy import build_images, deploy
from ._environment import Environment
from ._excepthook import custom_excepthook
from ._group import group
from ._image import Image
from ._initialize import current_domain, init, init_from_config, init_in_cluster
from ._logging import logger
from ._map import map
from ._pod import PodTemplate
from ._resources import AMD_GPU, GPU, HABANA_GAUDI, TPU, Device, DeviceClass, Neuron, Resources
from ._retry import RetryStrategy
from ._reusable_environment import ReusePolicy
from ._run import run, with_runcontext
from ._secret import Secret, SecretRequest
from ._task_environment import TaskEnvironment
from ._timeout import Timeout, TimeoutType
from ._trace import trace
from ._trigger import Cron, FixedRate, Trigger, TriggerTime
from ._version import __version__

sys.excepthook = custom_excepthook


def _silence_grpc_warnings():
    """
    Silences gRPC warnings that can clutter the output.
    """
    import os

    # Set environment variables for gRPC, this reduces log spew and avoids unnecessary warnings
    # before importing grpc
    if "GRPC_VERBOSITY" not in os.environ:
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        os.environ["GRPC_CPP_MIN_LOG_LEVEL"] = "ERROR"
        # Disable fork support (stops "skipping fork() handlers")
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
        # Reduce absl/glog verbosity
        os.environ["GLOG_minloglevel"] = "2"
        os.environ["ABSL_LOG"] = "0"


_silence_grpc_warnings()


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
    "Cache",
    "CachePolicy",
    "CacheRequest",
    "Cron",
    "Device",
    "DeviceClass",
    "Environment",
    "FixedRate",
    "Image",
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
    "custom_context",
    "deploy",
    "get_custom_context",
    "group",
    "init",
    "init_from_config",
    "init_in_cluster",
    "logger",
    "map",
    "run",
    "trace",
    "version",
    "with_runcontext",
]
