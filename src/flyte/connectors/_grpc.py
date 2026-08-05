"""gRPC wrapper that silences noisy logging before the C extension loads.

Usage::

    from flyte.connectors._grpc import grpc

This replaces bare ``import grpc`` in the connectors package.
"""

import os

if "GRPC_VERBOSITY" not in os.environ:
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_CPP_MIN_LOG_LEVEL"] = "ERROR"
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    os.environ["GLOG_minloglevel"] = "2"
    os.environ["ABSL_LOG"] = "0"

import grpc

__all__ = ["grpc"]
