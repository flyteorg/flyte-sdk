"""Tests for ``PodTemplate.with_termination_grace_period`` — the helper that sets
``terminationGracePeriodSeconds`` on a pod template without needing the ``kubernetes`` package.
"""

import copy
import pathlib
from datetime import timedelta

import pytest
from kubernetes.client import V1Container, V1PodSpec

import flyte
from flyte._internal.runtime.task_serde import get_proto_task
from flyte._pod import PodTemplate
from flyte.models import SerializationContext


def _grace_seconds(task):
    """Serialize a task and return the pod spec's terminationGracePeriodSeconds (or None)."""
    ctx = SerializationContext(project="p", domain="d", org="o", version="v1", root_dir=pathlib.Path.cwd())
    proto = get_proto_task(task, ctx)
    if not proto.HasField("k8s_pod"):
        return None
    pod_spec = proto.k8s_pod.pod_spec  # google.protobuf.Struct: supports `in` / `[]`, not `.get`
    return pod_spec["terminationGracePeriodSeconds"] if "terminationGracePeriodSeconds" in pod_spec else None


# --------------------------------------------------------------------------- #
# PodTemplate.with_termination_grace_period
# --------------------------------------------------------------------------- #


class TestWithTerminationGracePeriod:
    def test_synthesizes_pod_spec_with_primary_no_kubernetes_needed(self):
        # Starting from a bare PodTemplate(), the helper builds the pod spec + primary container
        # internally, so the caller never touches kubernetes.client.
        pt = PodTemplate().with_termination_grace_period(42)
        assert pt.pod_spec.termination_grace_period_seconds == 42
        assert any(c.name == "primary" for c in pt.pod_spec.containers)

    def test_int_seconds(self):
        assert PodTemplate().with_termination_grace_period(30).pod_spec.termination_grace_period_seconds == 30

    def test_timedelta(self):
        pt = PodTemplate().with_termination_grace_period(timedelta(minutes=2))
        assert pt.pod_spec.termination_grace_period_seconds == 120

    def test_zero_is_kept(self):
        # 0 is a meaningful k8s value (immediate kill), not "unset".
        assert PodTemplate().with_termination_grace_period(0).pod_spec.termination_grace_period_seconds == 0

    def test_custom_primary_container_name(self):
        pt = PodTemplate(primary_container_name="worker").with_termination_grace_period(10)
        assert any(c.name == "worker" for c in pt.pod_spec.containers)
        assert pt.pod_spec.termination_grace_period_seconds == 10

    def test_preserves_existing_pod_spec_fields(self):
        base = PodTemplate(pod_spec=V1PodSpec(containers=[V1Container(name="primary", image="img")], hostname="h"))
        pt = base.with_termination_grace_period(99)
        assert pt.pod_spec.termination_grace_period_seconds == 99
        assert pt.pod_spec.containers[0].image == "img"
        assert pt.pod_spec.hostname == "h"

    def test_does_not_mutate_original(self):
        base = PodTemplate(pod_spec=V1PodSpec(containers=[V1Container(name="primary")]))
        snapshot = copy.deepcopy(base)
        base.with_termination_grace_period(99)
        assert base == snapshot

    def test_reapplying_overwrites(self):
        pt = PodTemplate().with_termination_grace_period(30).with_termination_grace_period(60)
        assert pt.pod_spec.termination_grace_period_seconds == 60

    def test_composes_with_capability_helpers(self):
        pt = PodTemplate().allow_fuse().with_termination_grace_period(60)
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        assert pt.pod_spec.termination_grace_period_seconds == 60
        assert primary.resources.requests["smarter-devices/fuse"] == "1"  # allow_fuse survived

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="not bool"):
            PodTemplate().with_termination_grace_period(True)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            PodTemplate().with_termination_grace_period(-1)

    def test_wrong_type_rejected(self):
        with pytest.raises(TypeError):
            PodTemplate().with_termination_grace_period("30")

    def test_none_rejected(self):
        with pytest.raises(TypeError):
            PodTemplate().with_termination_grace_period(None)


# --------------------------------------------------------------------------- #
# End-to-end: the helper output flows through task serialization
# --------------------------------------------------------------------------- #


env = flyte.TaskEnvironment(
    name="tgp_env",
    pod_template=PodTemplate().with_termination_grace_period(timedelta(minutes=10)),
)


@env.task
async def env_task() -> int:
    return 1


env_plain = flyte.TaskEnvironment(name="tgp_plain")


@env_plain.task
async def plain_task() -> int:
    return 1


@env_plain.task(pod_template=PodTemplate().with_termination_grace_period(300))
async def decorated_task() -> int:
    return 1


class TestSerialization:
    def test_environment_pod_template(self):
        assert _grace_seconds(env_task) == 600

    def test_no_pod_template_uses_container_path(self):
        assert _grace_seconds(plain_task) is None
        ctx = SerializationContext(project="p", domain="d", org="o", version="v1", root_dir=pathlib.Path.cwd())
        assert get_proto_task(plain_task, ctx).HasField("container")

    def test_decorator_pod_template(self):
        assert _grace_seconds(decorated_task) == 300

    def test_override_pod_template(self):
        overridden = plain_task.override(pod_template=PodTemplate().with_termination_grace_period(15))
        assert _grace_seconds(overridden) == 15
