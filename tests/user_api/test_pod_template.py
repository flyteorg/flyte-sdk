import copy

import pytest

import flyte
from flyte._pod import PodTemplate


def test_pod_template_defaults():
    pt = PodTemplate()
    assert pt.pod_spec is None
    assert pt.primary_container_name == "primary"
    assert pt.labels is None
    assert pt.annotations is None


def test_pod_template_with_labels_and_annotations():
    pt = PodTemplate(
        labels={"team": "ml", "env": "prod"},
        annotations={"note": "testing"},
    )
    assert pt.labels == {"team": "ml", "env": "prod"}
    assert pt.annotations == {"note": "testing"}


def test_pod_template_custom_primary_container():
    pt = PodTemplate(primary_container_name="worker")
    assert pt.primary_container_name == "worker"


def test_pod_template_equality():
    pt1 = PodTemplate(labels={"a": "b"})
    pt2 = PodTemplate(labels={"a": "b"})
    assert pt1 == pt2


def test_pod_template_inequality():
    pt1 = PodTemplate(labels={"a": "b"})
    pt2 = PodTemplate(labels={"c": "d"})
    assert pt1 != pt2


def test_pod_template_to_k8s_pod():
    from kubernetes.client import V1Container, V1PodSpec

    pt = PodTemplate(
        pod_spec=V1PodSpec(containers=[V1Container(name="primary")]),
        labels={"team": "data"},
        annotations={"version": "v1"},
    )
    k8s_pod = pt.to_k8s_pod()
    assert k8s_pod.metadata.labels == {"team": "data"}
    assert k8s_pod.metadata.annotations == {"version": "v1"}
    assert k8s_pod.primary_container_name == "primary"


def test_pod_template_to_k8s_pod_with_empty_container():
    from kubernetes.client import V1Container, V1PodSpec

    pt = PodTemplate(pod_spec=V1PodSpec(containers=[V1Container(name="primary")]))
    k8s_pod = pt.to_k8s_pod()
    assert k8s_pod is not None


def test_pod_template_importable():
    assert flyte.PodTemplate is PodTemplate


# --------------------------------------------------------------------------- #
# PodTemplate.from_spec
# --------------------------------------------------------------------------- #


def _spec(*names: str):
    from kubernetes.client import V1Container, V1PodSpec

    return V1PodSpec(containers=[V1Container(name=n) for n in names])


class TestFromSpec:
    def test_explicit_primary(self):
        pt = PodTemplate.from_spec(_spec("worker", "sidecar"), primary_container_name="worker")
        assert pt.primary_container_name == "worker"

    def test_explicit_primary_missing_raises(self):
        with pytest.raises(ValueError, match="not found in pod spec"):
            PodTemplate.from_spec(_spec("worker", "sidecar"), primary_container_name="nope")

    def test_container_named_primary_wins(self):
        pt = PodTemplate.from_spec(_spec("sidecar", "primary"))
        assert pt.primary_container_name == "primary"

    def test_single_container_adopted(self):
        pt = PodTemplate.from_spec(_spec("worker"))
        assert pt.primary_container_name == "worker"

    def test_multiple_containers_ambiguous_raises(self):
        with pytest.raises(ValueError, match="Cannot determine the primary container"):
            PodTemplate.from_spec(_spec("a", "b"))

    def test_labels_and_annotations_passthrough(self):
        pt = PodTemplate.from_spec(_spec("primary"), labels={"team": "ml"}, annotations={"a": "b"})
        assert pt.labels == {"team": "ml"}
        assert pt.annotations == {"a": "b"}

    def test_input_spec_not_mutated(self):
        spec = _spec("primary")
        snapshot = copy.deepcopy(spec)
        pt = PodTemplate.from_spec(spec)
        pt.pod_spec.containers[0].name = "changed"
        assert spec == snapshot


# --------------------------------------------------------------------------- #
# Capability helpers: allow_fuse / allow_nested_sandboxing
# --------------------------------------------------------------------------- #

# (method name, kwargs) — every grant variant must satisfy the augmentor laws.
CAPABILITY_CALLS = [
    ("allow_fuse", {}),
    ("allow_fuse", {"privileged": False}),
    ("allow_nested_sandboxing", {}),
]
CAPABILITY_IDS = ["allow_fuse", "allow_fuse_unprivileged", "allow_nested_sandboxing"]


def _rich_template() -> PodTemplate:
    """A template with pre-existing sidecars, volumes, mounts, labels, and
    annotations that augmentation must preserve."""
    from kubernetes.client import (
        V1Container,
        V1EmptyDirVolumeSource,
        V1PodSpec,
        V1SecurityContext,
        V1Volume,
        V1VolumeMount,
    )

    return PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    volume_mounts=[V1VolumeMount(name="scratch", mount_path="/scratch")],
                    security_context=V1SecurityContext(run_as_user=1000),
                ),
                V1Container(name="sidecar"),
            ],
            volumes=[V1Volume(name="scratch", empty_dir=V1EmptyDirVolumeSource())],
        ),
        labels={"team": "ml"},
        annotations={"note": "keep-me"},
    )


@pytest.mark.parametrize(("method", "kwargs"), CAPABILITY_CALLS, ids=CAPABILITY_IDS)
class TestCapabilityLaws:
    def test_pure_original_unchanged(self, method, kwargs):
        original = _rich_template()
        snapshot = copy.deepcopy(original)
        getattr(original, method)(**kwargs)
        assert original == snapshot

    def test_idempotent(self, method, kwargs):
        once = getattr(_rich_template(), method)(**kwargs)
        twice = getattr(once, method)(**kwargs)
        assert once == twice

    def test_additive_preserves_existing_fields(self, method, kwargs):
        pt = getattr(_rich_template(), method)(**kwargs)
        container_names = [c.name for c in pt.pod_spec.containers]
        assert "sidecar" in container_names
        assert any(v.name == "scratch" for v in pt.pod_spec.volumes)
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        assert any(m.name == "scratch" for m in primary.volume_mounts)
        assert primary.security_context.run_as_user == 1000
        assert pt.labels == {"team": "ml"}
        assert pt.annotations["note"] == "keep-me"

    def test_standalone_creates_primary(self, method, kwargs):
        pt = getattr(PodTemplate(), method)(**kwargs)
        assert any(c.name == "primary" for c in pt.pod_spec.containers)

    def test_stamps_capability_annotation(self, method, kwargs):
        pt = getattr(PodTemplate(), method)(**kwargs)
        name = method.removeprefix("allow_").replace("_", "-")
        assert pt.annotations[f"flyte.org/capability-{name}"] == "true"

    def test_adds_sys_admin(self, method, kwargs):
        pt = getattr(_rich_template(), method)(**kwargs)
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        assert "SYS_ADMIN" in primary.security_context.capabilities.add

    def test_sys_admin_not_duplicated(self, method, kwargs):
        pt = getattr(getattr(_rich_template(), method)(**kwargs), method)(**kwargs)
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        assert primary.security_context.capabilities.add.count("SYS_ADMIN") == 1


class TestAllowFuse:
    def test_fuse_device_volume_and_mount(self):
        pt = PodTemplate().allow_fuse()
        vol = next(v for v in pt.pod_spec.volumes if v.name == "fuse-device")
        assert vol.host_path.path == "/dev/fuse"
        assert vol.host_path.type == "CharDevice"
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        mount = next(m for m in primary.volume_mounts if m.name == "fuse-device")
        assert mount.mount_path == "/dev/fuse"

    def test_privileged(self):
        pt = PodTemplate().allow_fuse()
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        assert primary.security_context.privileged is True

    def test_conflicts_with_explicit_unprivileged(self):
        from kubernetes.client import V1Container, V1PodSpec, V1SecurityContext

        pt = PodTemplate(
            pod_spec=V1PodSpec(
                containers=[V1Container(name="primary", security_context=V1SecurityContext(privileged=False))]
            )
        )
        with pytest.raises(ValueError, match="privileged=false"):
            pt.allow_fuse()

    def test_conflicts_allow_nested_sandboxing(self):
        with pytest.raises(ValueError, match="allowPrivilegeEscalation"):
            PodTemplate().allow_nested_sandboxing().allow_fuse()


class TestAllowFuseUnprivileged:
    def test_not_privileged(self):
        pt = PodTemplate().allow_fuse(privileged=False)
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        sc = primary.security_context
        assert sc.privileged is not True
        assert "SYS_ADMIN" in sc.capabilities.add
        # composability: must not pin allowPrivilegeEscalation either way
        assert sc.allow_privilege_escalation is None

    def test_device_volume_still_mounted(self):
        pt = PodTemplate().allow_fuse(privileged=False)
        assert any(v.name == "fuse-device" for v in pt.pod_spec.volumes)

    def test_sets_apparmor_unconfined(self):
        pt = PodTemplate(primary_container_name="worker").allow_fuse(privileged=False)
        assert pt.annotations["container.apparmor.security.beta.kubernetes.io/worker"] == "unconfined"

    def test_conflicts_with_other_apparmor_profile(self):
        pt = PodTemplate(annotations={"container.apparmor.security.beta.kubernetes.io/primary": "runtime/default"})
        with pytest.raises(ValueError, match="AppArmor"):
            pt.allow_fuse(privileged=False)

    def test_composes_with_sandboxing_in_both_orders(self):
        a = PodTemplate().allow_fuse(privileged=False).allow_nested_sandboxing()
        b = PodTemplate().allow_nested_sandboxing().allow_fuse(privileged=False)
        assert a == b  # order-free law holds for the composable pair
        primary = next(c for c in a.pod_spec.containers if c.name == "primary")
        sc = primary.security_context
        assert sc.capabilities.add == ["SYS_ADMIN"]
        assert sc.allow_privilege_escalation is False
        assert sc.privileged is not True
        assert any(v.name == "fuse-device" for v in a.pod_spec.volumes)
        assert a.annotations["container.apparmor.security.beta.kubernetes.io/primary"] == "unconfined"
        assert a.annotations["flyte.org/capability-fuse"] == "true"
        assert a.annotations["flyte.org/capability-nested-sandboxing"] == "true"


class TestAllowNestedSandboxing:
    def test_security_posture(self):
        pt = PodTemplate().allow_nested_sandboxing()
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        sc = primary.security_context
        assert sc.capabilities.add == ["SYS_ADMIN"]
        assert sc.allow_privilege_escalation is False
        assert sc.privileged is not True

    def test_apparmor_annotation_uses_primary_container_name(self):
        pt = PodTemplate(primary_container_name="worker").allow_nested_sandboxing()
        key = "container.apparmor.security.beta.kubernetes.io/worker"
        assert pt.annotations[key] == "unconfined"

    def test_conflicts_with_privileged(self):
        from kubernetes.client import V1Container, V1PodSpec, V1SecurityContext

        pt = PodTemplate(
            pod_spec=V1PodSpec(
                containers=[V1Container(name="primary", security_context=V1SecurityContext(privileged=True))]
            )
        )
        with pytest.raises(ValueError, match="privileged=true"):
            pt.allow_nested_sandboxing()

    def test_conflicts_allow_fuse(self):
        with pytest.raises(ValueError, match="privileged"):
            PodTemplate().allow_fuse().allow_nested_sandboxing()

    def test_conflicts_with_other_apparmor_profile(self):
        pt = PodTemplate(annotations={"container.apparmor.security.beta.kubernetes.io/primary": "runtime/default"})
        with pytest.raises(ValueError, match="AppArmor"):
            pt.allow_nested_sandboxing()

    def test_composes_with_user_spec_via_from_spec(self):
        pt = PodTemplate.from_spec(_spec("worker", "primary")).allow_nested_sandboxing()
        primary = next(c for c in pt.pod_spec.containers if c.name == "primary")
        assert "SYS_ADMIN" in primary.security_context.capabilities.add
        assert [c.name for c in pt.pod_spec.containers] == ["worker", "primary"]
