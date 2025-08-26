"""
Flyte SDK for authoring compound AI applications, services and workflows.
"""

from __future__ import annotations

import sys
from importlib import metadata

from ._build import build
from ._cache import Cache, CachePolicy, CacheRequest
from ._context import ctx
from ._deploy import build_images, deploy
from ._environment import Environment
from ._excepthook import custom_excepthook
from ._group import group
from ._image import Image
from ._initialize import init, init_from_config
from ._map import map
from ._pod import PodTemplate
from ._resources import GPU, TPU, Device, Resources
from ._retry import RetryStrategy
from ._reusable_environment import ReusePolicy
from ._run import run, with_runcontext
from ._secret import Secret, SecretRequest
from ._task_environment import TaskEnvironment
from ._timeout import Timeout, TimeoutType
from ._trace import trace
from ._version import __version__

sys.excepthook = custom_excepthook
_original_entry_points = metadata.entry_points


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


def _filtered_entry_points(*args, **kwargs):
    """Wrap importlib.metadata.entry_points to exclude a specific distribution."""
    eps = _original_entry_points(*args, **kwargs)
    excluded_distribution = ["union", "unionai"]

    return metadata.EntryPoints(ep for ep in eps if ep.dist is None or ep.dist.name not in excluded_distribution)


metadata.entry_points = _filtered_entry_points
_silence_grpc_warnings()


def version() -> str:
    """
    Returns the version of the Flyte SDK.
    """
    return __version__


__all__ = [
    "GPU",
    "TPU",
    "Cache",
    "CachePolicy",
    "CacheRequest",
    "Device",
    "Environment",
    "Image",
    "PodTemplate",
    "Resources",
    "RetryStrategy",
    "ReusePolicy",
    "Secret",
    "SecretRequest",
    "TaskEnvironment",
    "Timeout",
    "TimeoutType",
    "__version__",
    "build",
    "build_images",
    "ctx",
    "deploy",
    "group",
    "init",
    "init_from_config",
    "map",
    "run",
    "trace",
    "with_runcontext",
]
