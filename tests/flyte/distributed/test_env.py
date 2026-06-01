"""Validation tests for ClusteredTaskEnvironment, TorchRun, and ClusterFailurePolicy."""

from __future__ import annotations

import warnings

import pytest

import flyte
from flyte.distributed._environment import (
    ClusteredTaskEnvironment,
    ClusterFailurePolicy,
    TorchRun,
)

# ---------------------------------------------------------------------------
# TorchRun
# ---------------------------------------------------------------------------


def test_torchrun_defaults():
    rt = TorchRun()
    assert rt.rdzv_backend == "static"
    assert rt.max_restarts == 0
    assert not hasattr(rt, "master_port"), "master_port must not exist on TorchRun (hardcoded in plugin)"


def test_torchrun_c10d():
    rt = TorchRun(rdzv_backend="c10d", max_restarts=3)
    assert rt.rdzv_backend == "c10d"
    assert rt.max_restarts == 3


# ---------------------------------------------------------------------------
# ClusterFailurePolicy
# ---------------------------------------------------------------------------


def test_cluster_failure_policy_defaults():
    fp = ClusterFailurePolicy()
    assert fp.max_restarts == 0
    assert fp.restart_on_host_maintenance is False


def test_cluster_failure_policy_custom():
    fp = ClusterFailurePolicy(max_restarts=5, restart_on_host_maintenance=True)
    assert fp.max_restarts == 5
    assert fp.restart_on_host_maintenance is True


# ---------------------------------------------------------------------------
# ClusteredTaskEnvironment — valid construction
# ---------------------------------------------------------------------------


def _make_env(**overrides):
    defaults = {
        "name": "clustered_env",
        "image": "python:3.11",
        "replicas": 2,
        "nproc_per_node": 4,
    }
    defaults.update(overrides)
    return ClusteredTaskEnvironment(**defaults)


def test_valid_construction():
    env = _make_env()
    assert env.replicas == 2
    assert env.nproc_per_node == 4
    assert isinstance(env.runtime, TorchRun)
    assert env.interconnect == "tcp"
    assert isinstance(env.failure_policy, ClusterFailurePolicy)
    assert env.ttl_seconds_after_finished is None


def test_clustered_env_is_task_environment():
    env = _make_env()
    assert isinstance(env, flyte.TaskEnvironment)


def test_optional_fields():
    env = _make_env(
        runtime=TorchRun(rdzv_backend="c10d", max_restarts=2),
        failure_policy=ClusterFailurePolicy(max_restarts=3, restart_on_host_maintenance=True),
        ttl_seconds_after_finished=3600,
    )
    assert env.runtime.rdzv_backend == "c10d"
    assert env.failure_policy.restart_on_host_maintenance is True
    assert env.ttl_seconds_after_finished == 3600


# ---------------------------------------------------------------------------
# ClusteredTaskEnvironment — validation failures
# ---------------------------------------------------------------------------


def test_replicas_lt_1_raises():
    with pytest.raises(ValueError, match="replicas must be >= 1"):
        _make_env(replicas=0)


def test_nproc_per_node_lt_1_raises():
    with pytest.raises(ValueError, match="nproc_per_node must be >= 1"):
        _make_env(nproc_per_node=0)


def test_gpu_int_lt_nproc_per_node_raises():
    with pytest.raises(ValueError, match=r"resources\.gpu"):
        _make_env(nproc_per_node=8, resources=flyte.Resources(gpu=4))


def test_gpu_accelerator_lt_nproc_per_node_raises():
    # Accelerator string "H100:4" → count 4 < nproc_per_node 8
    with pytest.raises(ValueError, match=r"resources\.gpu"):
        _make_env(nproc_per_node=8, resources=flyte.Resources(gpu="H100:4"))


def test_gpu_eq_nproc_per_node_ok():
    env = _make_env(nproc_per_node=4, resources=flyte.Resources(gpu=4))
    assert env.nproc_per_node == 4


def test_gpu_gt_nproc_per_node_ok():
    env = _make_env(nproc_per_node=4, resources=flyte.Resources(gpu=8))
    assert env.nproc_per_node == 4


def test_gpu_accelerator_eq_nproc_per_node_ok():
    env = _make_env(nproc_per_node=8, resources=flyte.Resources(gpu="H100:8"))
    assert env.nproc_per_node == 8


def test_unsupported_runtime_raises():
    with pytest.raises(TypeError, match="unsupported runtime type"):
        _make_env(runtime="invalid")  # type: ignore[arg-type]


def test_non_tcp_interconnect_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_env(interconnect="efa")
    assert any("Phase 3" in str(w.message) for w in caught)


def test_task_decorator_works():
    env = _make_env()

    @env.task
    async def my_task(x: int) -> int:
        return x * 2

    assert my_task is not None
    assert my_task.parent_env_name == "clustered_env"
