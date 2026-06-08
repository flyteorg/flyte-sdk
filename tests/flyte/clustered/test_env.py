"""Validation tests for ClusteredTaskEnvironment, TorchRun, and ClusterFailurePolicy."""

from __future__ import annotations

import pytest

import flyte
from flyte.clustered._environment import (
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


def test_non_tcp_interconnect_raises():
    with pytest.raises(ValueError, match="interconnect must be one of"):
        _make_env(interconnect="efa")  # type: ignore[arg-type]


def test_task_decorator_works():
    env = _make_env()

    @env.task
    async def my_task(x: int) -> int:
        return x * 2

    assert my_task is not None
    assert my_task.parent_env_name == "clustered_env"


def test_reusable_not_supported_raises():
    with pytest.raises(ValueError, match="does not support reusable"):
        _make_env(reusable=flyte.ReusePolicy(replicas=1, idle_ttl=60))


# ---------------------------------------------------------------------------
# Plugin wiring — env routes tasks to ClusteredTaskTemplate via plugin_config
# ---------------------------------------------------------------------------


def test_env_sets_clustered_plugin_config():
    from flyte.clustered._task import _ClusteredPlugin

    env = _make_env()
    assert isinstance(env.plugin_config, _ClusteredPlugin)


def test_task_decorator_returns_clustered_template():
    from flyte.clustered._task import ClusteredTaskTemplate

    env = _make_env()

    @env.task
    async def my_task(x: int) -> int:
        return x

    # The plugin registry must select ClusteredTaskTemplate, not the base AsyncFunctionTaskTemplate.
    assert isinstance(my_task, ClusteredTaskTemplate)
    assert my_task.task_type == "clustered-task"
    assert my_task.task_type_version == 1
    # container_command supplies the entrypoint wrapper (sctx is unused by this override).
    assert my_task.container_command(None) == ["python", "-m", "flyte.clustered._entrypoint"]


def test_clustered_template_custom_config_reads_env():
    """custom_config resolves the parent env via weakref and returns its to_custom_dict()."""
    env = _make_env()

    @env.task
    async def my_task(x: int) -> int:
        return x

    custom = my_task.custom_config(None)
    assert custom == env.to_custom_dict()
    assert custom["replicas"] == env.replicas


def test_base_task_template_container_command_default_empty():
    """A plain (non-clustered) task keeps the empty default command from the base TaskTemplate."""
    plain_env = flyte.TaskEnvironment(name="plain", image="python:3.11")

    @plain_env.task
    async def plain(x: int) -> int:
        return x

    assert plain.container_command(None) == []
