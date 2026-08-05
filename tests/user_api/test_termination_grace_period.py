"""Tests for the ``termination_grace_period`` setting on TaskEnvironment / @env.task / override.

The value is sugar that lands as ``terminationGracePeriodSeconds`` on the task's pod spec at
serialization time (synthesizing a pod template when the task has none).
"""

import copy
import pathlib
from datetime import timedelta

import pytest
from kubernetes.client import V1Container, V1PodSpec

import flyte
from flyte._internal.runtime.task_serde import get_proto_task
from flyte._pod import (
    PodTemplate,
    apply_termination_grace_period,
    normalize_termination_grace_period,
)
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
# normalize_termination_grace_period
# --------------------------------------------------------------------------- #


class TestNormalize:
    def test_none(self):
        assert normalize_termination_grace_period(None) is None

    def test_int_seconds(self):
        assert normalize_termination_grace_period(30) == 30

    def test_zero_is_kept(self):
        # 0 is a meaningful k8s value (immediate kill), not "unset".
        assert normalize_termination_grace_period(0) == 0

    def test_timedelta_truncated_to_seconds(self):
        assert normalize_termination_grace_period(timedelta(minutes=5)) == 300
        assert normalize_termination_grace_period(timedelta(seconds=1.9)) == 1

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="not bool"):
            normalize_termination_grace_period(True)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            normalize_termination_grace_period(-1)

    def test_wrong_type_rejected(self):
        with pytest.raises(TypeError):
            normalize_termination_grace_period("30")


# --------------------------------------------------------------------------- #
# apply_termination_grace_period
# --------------------------------------------------------------------------- #


class TestApply:
    def test_no_grace_returns_input_unchanged(self):
        assert apply_termination_grace_period(None, None) is None
        pt = PodTemplate()
        assert apply_termination_grace_period(pt, None) is pt

    def test_synthesizes_pod_template_with_primary(self):
        pt = apply_termination_grace_period(None, 42)
        assert pt.pod_spec.termination_grace_period_seconds == 42
        assert any(c.name == "primary" for c in pt.pod_spec.containers)

    def test_timedelta(self):
        pt = apply_termination_grace_period(None, timedelta(minutes=2))
        assert pt.pod_spec.termination_grace_period_seconds == 120

    def test_preserves_existing_pod_spec_fields(self):
        existing = PodTemplate(pod_spec=V1PodSpec(containers=[V1Container(name="primary", image="img")], hostname="h"))
        pt = apply_termination_grace_period(existing, 99)
        assert pt.pod_spec.termination_grace_period_seconds == 99
        assert pt.pod_spec.containers[0].image == "img"
        assert pt.pod_spec.hostname == "h"

    def test_does_not_mutate_input(self):
        existing = PodTemplate(pod_spec=V1PodSpec(containers=[V1Container(name="primary")]))
        snapshot = copy.deepcopy(existing)
        apply_termination_grace_period(existing, 99)
        assert existing == snapshot

    def test_string_pod_template_raises(self):
        with pytest.raises(ValueError, match="named"):
            apply_termination_grace_period("my-template", 30)

    def test_string_pod_template_without_grace_is_passthrough(self):
        # No grace requested -> the named template is returned untouched.
        assert apply_termination_grace_period("my-template", None) == "my-template"


# --------------------------------------------------------------------------- #
# End-to-end: the three configuration levels
# --------------------------------------------------------------------------- #


env_grace = flyte.TaskEnvironment(name="tgp_env", termination_grace_period=600)


@env_grace.task
async def env_task() -> int:
    return 1


env_plain = flyte.TaskEnvironment(name="tgp_plain")


@env_plain.task
async def plain_task() -> int:
    return 1


@env_plain.task(termination_grace_period=timedelta(minutes=10))
async def decorated_task() -> int:
    return 1


env_with_pt = flyte.TaskEnvironment(
    name="tgp_with_pt",
    pod_template=PodTemplate(pod_spec=V1PodSpec(containers=[V1Container(name="primary")], hostname="h")),
    termination_grace_period=120,
)


@env_with_pt.task
async def pt_task() -> int:
    return 1


class TestEndToEnd:
    def test_environment_level(self):
        assert _grace_seconds(env_task) == 600

    def test_no_grace_uses_container_path(self):
        # Without a grace period (and no pod template), the task stays a plain container.
        assert _grace_seconds(plain_task) is None
        ctx = SerializationContext(project="p", domain="d", org="o", version="v1", root_dir=pathlib.Path.cwd())
        assert get_proto_task(plain_task, ctx).HasField("container")

    def test_decorator_level(self):
        assert _grace_seconds(decorated_task) == 600

    def test_decorator_overrides_environment(self):
        env = flyte.TaskEnvironment(name="tgp_ovr", termination_grace_period=30)

        @env.task(termination_grace_period=300)
        async def t() -> int:
            return 1

        assert _grace_seconds(t) == 300

    def test_override_at_call_time(self):
        overridden = env_task.override(termination_grace_period=15)
        assert _grace_seconds(overridden) == 15
        # original template is unchanged
        assert _grace_seconds(env_task) == 600

    def test_override_inherits_when_not_passed(self):
        overridden = env_task.override(retries=1)
        assert _grace_seconds(overridden) == 600

    def test_combined_with_pod_template_preserves_other_fields(self):
        assert _grace_seconds(pt_task) == 120
        ctx = SerializationContext(project="p", domain="d", org="o", version="v1", root_dir=pathlib.Path.cwd())
        assert get_proto_task(pt_task, ctx).k8s_pod.pod_spec["hostname"] == "h"


# --------------------------------------------------------------------------- #
# Validation & preservation on the environment
# --------------------------------------------------------------------------- #


class TestEnvironment:
    def test_invalid_value_raises_at_definition(self):
        with pytest.raises(ValueError, match="non-negative"):
            flyte.TaskEnvironment(name="tgp_bad", termination_grace_period=-5)

    def test_clone_with_preserves(self):
        cloned = env_grace.clone_with("tgp_clone")
        assert cloned.termination_grace_period == 600

    def test_named_pod_template_plus_grace_raises_at_serialize(self):
        env = flyte.TaskEnvironment(name="tgp_named", pod_template="named-ref", termination_grace_period=30)

        @env.task
        async def t() -> int:
            return 1

        ctx = SerializationContext(project="p", domain="d", org="o", version="v1", root_dir=pathlib.Path.cwd())
        with pytest.raises(ValueError, match="named"):
            get_proto_task(t, ctx)
