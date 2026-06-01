"""Serde tests for ClusteredTaskEnvironment."""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import flyte
from flyte._internal.runtime.task_serde import get_proto_task
from flyte.distributed._environment import (
    ClusteredTaskEnvironment,
    ClusterFailurePolicy,
    TorchRun,
)
from flyte.models import SerializationContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(**overrides):
    defaults = {
        "name": "train_env",
        "image": "pytorch/pytorch:2.3-cuda12.1",
        "replicas": 4,
        "nproc_per_node": 8,
    }
    defaults.update(overrides)
    return ClusteredTaskEnvironment(**defaults)


def _make_ctx():
    return SerializationContext(
        project="test-project",
        domain="test-domain",
        version="v1",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )


# ---------------------------------------------------------------------------
# Core serde tests (no proto required — to_custom_dict is mocked)
# ---------------------------------------------------------------------------


def _run_serde(env_overrides=None, task_overrides=None):
    """Build an env + task, mock to_custom_dict, call get_proto_task, return proto."""
    env = _make_env(**(env_overrides or {}))
    ctx = _make_ctx()

    @env.task
    async def train(x: int) -> int:
        return x

    fake_custom = {"replicas": env.replicas, "nprocPerNode": env.nproc_per_node}
    with patch.object(type(env), "to_custom_dict", return_value=fake_custom):
        # re-access the task's parent env via weakref; patch is on the class
        proto = get_proto_task(train, ctx)
    return proto


def test_task_type_is_clustered():
    proto = _run_serde()
    assert proto.type == "clustered-task"


def test_task_type_version_is_1():
    proto = _run_serde()
    assert proto.task_type_version == 1


def test_container_command_is_entrypoint():
    proto = _run_serde()
    assert list(proto.container.command) == ["python", "-m", "flyte.distributed._entrypoint"]


def test_container_args_contains_a0():
    proto = _run_serde()
    assert "a0" in proto.container.args


def test_custom_is_populated():
    proto = _run_serde()
    # custom is a google.protobuf.Struct; its fields map to our dict keys
    fields = dict(proto.custom.fields.items())
    assert "replicas" in fields
    assert "nprocPerNode" in fields


def test_pod_template_clustered_gets_entrypoint():
    """A clustered task with a pod_template must still get the entrypoint wrapper
    spliced into the k8s_pod's primary container (regression: command rewrite used to
    be skipped on the pod_template path, leaving command=[])."""
    from kubernetes.client import V1Container, V1PodSpec

    from flyte import PodTemplate

    pod_template = PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(containers=[V1Container(name="primary")]),
    )
    env = _make_env(pod_template=pod_template)
    ctx = _make_ctx()

    @env.task
    async def train(x: int) -> int:
        return x

    fake_custom = {"replicas": env.replicas}
    with patch.object(type(env), "to_custom_dict", return_value=fake_custom):
        proto = get_proto_task(train, ctx)

    assert proto.type == "clustered-task"
    assert not proto.HasField("container")  # pod_template path -> k8s_pod, not container
    primary = next(c for c in proto.k8s_pod.pod_spec["containers"] if c["name"] == "primary")
    assert primary["command"] == ["python", "-m", "flyte.distributed._entrypoint"]
    assert "a0" in primary["args"]


def test_non_clustered_task_unaffected():
    """A plain TaskEnvironment task must keep type='python' and empty command."""
    plain_env = flyte.TaskEnvironment(name="plain_env", image="python:3.11")
    ctx = _make_ctx()

    @plain_env.task
    async def plain(x: int) -> int:
        return x

    proto = get_proto_task(plain, ctx)
    assert proto.type == "python"
    assert proto.task_type_version == 0
    assert list(proto.container.command) == []


# ---------------------------------------------------------------------------
# Proto round-trip tests
# ---------------------------------------------------------------------------


def test_proto_roundtrip_basic():
    env = _make_env()
    d = env.to_custom_dict()

    from flyteidl2.plugins.clustered_pb2 import ClusteredTaskSpec
    from google.protobuf import json_format

    spec = json_format.ParseDict(d, ClusteredTaskSpec())
    assert spec.replicas == 4
    assert spec.nproc_per_node == 8


def test_proto_roundtrip_torchrun_c10d():
    env = _make_env(runtime=TorchRun(rdzv_backend="c10d", max_restarts=2))
    d = env.to_custom_dict()

    from flyteidl2.plugins.clustered_pb2 import ClusteredTaskSpec, RdzvBackend
    from google.protobuf import json_format

    spec = json_format.ParseDict(d, ClusteredTaskSpec())
    assert spec.runtime.torchrun.rdzv_backend == RdzvBackend.C10D
    assert spec.runtime.torchrun.max_restarts == 2


def test_proto_roundtrip_failure_policy():
    env = _make_env(failure_policy=ClusterFailurePolicy(max_restarts=3, restart_on_host_maintenance=True))
    d = env.to_custom_dict()

    from flyteidl2.plugins.clustered_pb2 import ClusteredTaskSpec
    from google.protobuf import json_format

    spec = json_format.ParseDict(d, ClusteredTaskSpec())
    assert spec.failure_policy.max_restarts == 3
    assert spec.failure_policy.restart_on_host_maintenance is True


def test_proto_roundtrip_ttl():
    env = _make_env(ttl_seconds_after_finished=7200)
    d = env.to_custom_dict()

    from flyteidl2.plugins.clustered_pb2 import ClusteredTaskSpec
    from google.protobuf import json_format

    spec = json_format.ParseDict(d, ClusteredTaskSpec())
    assert spec.ttl_seconds_after_finished.value == 7200


def test_full_serde_with_real_proto():
    """End-to-end: get_proto_task with real to_custom_dict (not mocked)."""
    env = _make_env(
        failure_policy=ClusterFailurePolicy(max_restarts=2, restart_on_host_maintenance=True),
    )
    ctx = _make_ctx()

    @env.task
    async def train(x: int) -> int:
        return x

    proto = get_proto_task(train, ctx)

    assert proto.type == "clustered-task"
    assert proto.task_type_version == 1
    assert list(proto.container.command) == ["python", "-m", "flyte.distributed._entrypoint"]
    assert "a0" in proto.container.args
    assert proto.custom.fields["replicas"].number_value == 4
