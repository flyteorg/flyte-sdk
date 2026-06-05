from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from flyteidl2.core.tasks_pb2 import K8sPod
    from kubernetes.client import V1Container, V1PodSpec


_PRIMARY_CONTAINER_NAME_FIELD = "primary_container_name"
_PRIMARY_CONTAINER_DEFAULT_NAME = "primary"

# Name used for the hostPath volume and its volumeMount added by
# ``PodTemplate.allow_fuse()``.
_FUSE_DEVICE_VOLUME_NAME = "fuse-device"
_FUSE_DEVICE_PATH = "/dev/fuse"

# Every capability helper stamps ``flyte.org/capability-<name>: "true"`` on the
# resulting template so the requested grant is auditable on the pod itself
# (admission controllers and humans can match on the annotation) regardless of
# how the template was built.
_CAPABILITY_ANNOTATION_PREFIX = "flyte.org/capability-"

_APPARMOR_ANNOTATION_PREFIX = "container.apparmor.security.beta.kubernetes.io/"
_APPARMOR_UNCONFINED = "unconfined"


@dataclass(init=True, repr=True, eq=True, frozen=False)
class PodTemplate(object):
    """Custom PodTemplate specification for a Task."""

    pod_spec: Optional["V1PodSpec"] = None
    primary_container_name: str = _PRIMARY_CONTAINER_DEFAULT_NAME
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

    @classmethod
    def from_spec(
        cls,
        pod_spec: "V1PodSpec",
        *,
        primary_container_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> PodTemplate:
        """
        Create a :class:`PodTemplate` from an existing ``V1PodSpec``.

        The spec is deep-copied, so later mutations of the input (or of the
        returned template) don't leak into each other.

        The primary container — the one Flyte injects the task image/command
        into — is resolved as follows:

        1. If ``primary_container_name`` is given, a container with that name
           must exist in the spec.
        2. Otherwise, a container named ``primary`` is used if present.
        3. Otherwise, if the spec has exactly one container, that container is
           adopted as the primary.
        4. Otherwise (multiple containers, none named ``primary``), a
           ``ValueError`` is raised — pass ``primary_container_name`` to pick
           one explicitly.

        :param pod_spec: The ``kubernetes.client.V1PodSpec`` to wrap.
        :param primary_container_name: Optional explicit name of the primary
            container within ``pod_spec``.
        :param labels: Optional pod labels.
        :param annotations: Optional pod annotations.
        """
        container_names = [getattr(c, "name", None) for c in (pod_spec.containers or [])]

        if primary_container_name is not None:
            if primary_container_name not in container_names:
                raise ValueError(
                    f"primary_container_name {primary_container_name!r} not found in pod spec; "
                    f"available containers: {container_names}"
                )
            resolved = primary_container_name
        elif _PRIMARY_CONTAINER_DEFAULT_NAME in container_names:
            resolved = _PRIMARY_CONTAINER_DEFAULT_NAME
        elif len(container_names) == 1 and container_names[0]:
            resolved = container_names[0]
        else:
            raise ValueError(
                f"Cannot determine the primary container: the pod spec has containers {container_names} and none "
                f"is named {_PRIMARY_CONTAINER_DEFAULT_NAME!r}. Pass primary_container_name= to pick one."
            )

        return cls(
            pod_spec=copy.deepcopy(pod_spec),
            primary_container_name=resolved,
            labels=dict(labels) if labels else None,
            annotations=dict(annotations) if annotations else None,
        )

    def allow_fuse(self, privileged: bool = True) -> PodTemplate:
        """
        Return a copy of this template granted everything a container needs to
        perform an in-process FUSE mount (e.g. for ``Volume`` support).

        Specifically, the copy:

        * adds a ``hostPath`` volume named ``fuse-device`` pointing at
          ``/dev/fuse`` on the node, mounted into the primary container;
        * adds ``CAP_SYS_ADMIN`` to the primary container so the ``mount``
          syscall is permitted;
        * with ``privileged=True`` (the default), sets ``privileged: true`` on
          the primary container; with ``privileged=False``, instead sets the
          ``container.apparmor.security.beta.kubernetes.io/<primary>:
          unconfined`` pod annotation (the default AppArmor profile would
          otherwise block ``mount``);
        * stamps the ``flyte.org/capability-fuse`` annotation for auditability.

        Why privileged is the default: opening ``/dev/fuse`` is gated by the
        container runtime's device-cgroup allowlist, which only ``privileged``
        bypasses — there is no pod-spec field to whitelist a device for a
        non-privileged container. Pass ``privileged=False`` **only** on
        clusters that permit ``/dev/fuse`` for non-privileged containers (e.g.
        via a FUSE device plugin or runtime device-allowlist configuration);
        otherwise the pod deploys fine but ``open("/dev/fuse")`` fails with
        ``EPERM`` at runtime. ``privileged=False`` composes with
        ``allow_nested_sandboxing()``; ``privileged=True`` does not (Kubernetes
        rejects privileged containers that set
        ``allowPrivilegeEscalation: false``).

        The original template is never mutated; existing volumes, mounts,
        sidecars, labels, annotations, and unrelated security-context fields
        are preserved. Re-applying with the same arguments is idempotent.

        Raises ``ValueError`` if the template already pins a conflicting
        security posture (with ``privileged=True``: ``privileged: false`` or
        ``allowPrivilegeEscalation: false`` pre-set; with ``privileged=False``:
        a different AppArmor profile for the primary container).
        """
        return _apply_fuse(self, privileged=privileged)

    def allow_nested_sandboxing(self) -> PodTemplate:
        """
        Return a copy of this template granted the prerequisites for creating
        nested sandboxes (e.g. the bubblewrap/``bwrap`` backend of
        ``SandboxEnvironment``) — and nothing more.

        bwrap runs as a non-root user via unprivileged user namespaces, but the
        containerd default seccomp profile only permits the ``mount`` /
        ``pivot_root`` / ``setns`` / ``unshare`` syscalls it needs when the
        container's capability set includes ``CAP_SYS_ADMIN``, and the default
        AppArmor profile must be unconfined so those calls aren't blocked.
        The copy carries exactly that:

        * ``CAP_SYS_ADMIN`` added to the primary container's capabilities;
        * ``allowPrivilegeEscalation: false`` (no other caps, not privileged);
        * the ``container.apparmor.security.beta.kubernetes.io/<primary>:
          unconfined`` pod annotation (on K8s >= 1.30 the
          ``securityContext.appArmorProfile: {type: Unconfined}`` field is the
          equivalent; the annotation is used here for version compatibility);
        * the ``flyte.org/capability-nested-sandboxing`` annotation for
          auditability.

        A cluster that already permits unprivileged user namespaces needs none
        of this for the ``userns`` sandbox backend — only use this when the
        seccomp/AppArmor defaults block the sandboxing syscalls.

        Composes with ``allow_fuse(privileged=False)``; not with the default
        ``allow_fuse()``, which makes the container privileged.

        The original template is never mutated; existing fields are preserved
        and re-applying is idempotent. Raises ``ValueError`` if the template
        already pins a conflicting security posture (``privileged: true``,
        ``allowPrivilegeEscalation: true``, or a different AppArmor profile
        for the primary container).
        """
        return _apply_sandboxing(self)

    def to_k8s_pod(self) -> "K8sPod":
        from flyteidl2.core.tasks_pb2 import K8sObjectMetadata, K8sPod
        from kubernetes.client import ApiClient, V1PodSpec

        if self.pod_spec is None:
            self.pod_spec = V1PodSpec()

        return K8sPod(
            metadata=K8sObjectMetadata(labels=self.labels, annotations=self.annotations),
            pod_spec=ApiClient().sanitize_for_serialization(self.pod_spec),
            primary_container_name=self.primary_container_name,
        )


def _clone_with_primary(pt: PodTemplate) -> PodTemplate:
    """Deep-copy ``pt``, ensuring it has a pod spec containing the primary container."""
    from kubernetes.client import V1Container, V1PodSpec

    pt = copy.deepcopy(pt)
    if pt.pod_spec is None:
        pt.pod_spec = V1PodSpec(containers=[V1Container(name=pt.primary_container_name)])

    containers = list(pt.pod_spec.containers or [])
    if not any(getattr(c, "name", None) == pt.primary_container_name for c in containers):
        containers.append(V1Container(name=pt.primary_container_name))
        pt.pod_spec.containers = containers
    return pt


def _get_primary_container(pt: PodTemplate) -> "V1Container":
    """Return the primary container of a template prepared by ``_clone_with_primary``."""
    assert pt.pod_spec is not None  # _clone_with_primary guarantees it
    primary = next((c for c in pt.pod_spec.containers if c.name == pt.primary_container_name), None)
    assert primary is not None  # _clone_with_primary guarantees it
    return primary


def _stamp_capability(pt: PodTemplate, name: str) -> None:
    """Record the granted capability as a pod annotation for auditability."""
    annotations = dict(pt.annotations or {})
    annotations[f"{_CAPABILITY_ANNOTATION_PREFIX}{name}"] = "true"
    pt.annotations = annotations


def _add_sys_admin(primary: "V1Container") -> None:
    """Set-merge ``CAP_SYS_ADMIN`` into the primary container's capability set."""
    from kubernetes.client import V1Capabilities, V1SecurityContext

    sc = primary.security_context or V1SecurityContext()
    caps = sc.capabilities or V1Capabilities()
    existing_add = list(caps.add or [])
    if "SYS_ADMIN" not in existing_add:
        existing_add.append("SYS_ADMIN")
    caps.add = existing_add
    sc.capabilities = caps
    primary.security_context = sc


def _set_apparmor_unconfined(pt: PodTemplate, caller: str) -> None:
    """Set the AppArmor-unconfined annotation for the primary container.

    Raises ``ValueError`` if a different AppArmor profile is already pinned.
    """
    apparmor_key = f"{_APPARMOR_ANNOTATION_PREFIX}{pt.primary_container_name}"
    existing_apparmor = (pt.annotations or {}).get(apparmor_key)
    if existing_apparmor is not None and existing_apparmor != _APPARMOR_UNCONFINED:
        raise ValueError(
            f"{caller} requires the AppArmor profile of the primary container to be "
            f"{_APPARMOR_UNCONFINED!r}, but the pod template sets {apparmor_key}={existing_apparmor!r}."
        )
    annotations = dict(pt.annotations or {})
    annotations[apparmor_key] = _APPARMOR_UNCONFINED
    pt.annotations = annotations


def _apply_fuse(pod_template: PodTemplate, *, privileged: bool = True) -> PodTemplate:
    """Augmentor behind :meth:`PodTemplate.allow_fuse`."""
    from kubernetes.client import V1HostPathVolumeSource, V1Volume, V1VolumeMount

    pt = _clone_with_primary(pod_template)
    primary = _get_primary_container(pt)

    if privileged:
        # Conflict checks: a privileged container cannot deny privilege
        # escalation (Kubernetes rejects the combination), and an explicit
        # privileged=False contradicts what this grant needs.
        sc = primary.security_context
        if sc is not None:
            if sc.privileged is False:
                raise ValueError(
                    f"allow_fuse() requires privileged=true on the primary container "
                    f"({pt.primary_container_name!r}), but the pod template explicitly sets privileged=false. "
                    "On clusters that permit /dev/fuse for non-privileged containers, use "
                    "allow_fuse(privileged=False)."
                )
            if sc.allow_privilege_escalation is False:
                raise ValueError(
                    "allow_fuse() makes the primary container privileged, which Kubernetes rejects when "
                    "allowPrivilegeEscalation=false is set (e.g. after allow_nested_sandboxing()). On clusters "
                    "that permit /dev/fuse for non-privileged containers, use allow_fuse(privileged=False), "
                    "which composes with allow_nested_sandboxing()."
                )
    else:
        # Non-privileged FUSE: mount needs CAP_SYS_ADMIN (added below) and an
        # unconfined AppArmor profile; the cluster must separately permit
        # /dev/fuse for non-privileged containers (device plugin or runtime
        # device-allowlist config). Leaves privileged/allowPrivilegeEscalation
        # untouched, so it composes with allow_nested_sandboxing().
        _set_apparmor_unconfined(pt, "allow_fuse(privileged=False)")

    # Volume + volumeMount for /dev/fuse — idempotent: skip if already present.
    assert pt.pod_spec is not None  # _clone_with_primary guarantees it
    volumes = list(pt.pod_spec.volumes or [])
    if not any(getattr(v, "name", None) == _FUSE_DEVICE_VOLUME_NAME for v in volumes):
        volumes.append(
            V1Volume(
                name=_FUSE_DEVICE_VOLUME_NAME,
                host_path=V1HostPathVolumeSource(path=_FUSE_DEVICE_PATH, type="CharDevice"),
            )
        )
        pt.pod_spec.volumes = volumes

    mounts = list(primary.volume_mounts or [])
    if not any(getattr(m, "name", None) == _FUSE_DEVICE_VOLUME_NAME for m in mounts):
        mounts.append(V1VolumeMount(name=_FUSE_DEVICE_VOLUME_NAME, mount_path=_FUSE_DEVICE_PATH))
        primary.volume_mounts = mounts

    # SecurityContext: CAP_SYS_ADMIN (+ privileged unless opted out). Preserve
    # any other security-context fields the caller set.
    _add_sys_admin(primary)
    if privileged:
        primary.security_context.privileged = True

    _stamp_capability(pt, "fuse")
    return pt


def _apply_sandboxing(pod_template: PodTemplate) -> PodTemplate:
    """Augmentor behind :meth:`PodTemplate.allow_nested_sandboxing`."""
    pt = _clone_with_primary(pod_template)
    primary = _get_primary_container(pt)

    # Conflict checks — refuse to silently widen or narrow a security posture
    # the user pinned explicitly.
    sc = primary.security_context
    if sc is not None:
        if sc.privileged is True:
            raise ValueError(
                "allow_nested_sandboxing() grants only CAP_SYS_ADMIN with allowPrivilegeEscalation=false, but the "
                "pod template sets privileged=true on the primary container, which Kubernetes rejects in "
                "combination with allowPrivilegeEscalation=false (note: this also means allow_nested_sandboxing() "
                "cannot be combined with allow_fuse() unless the latter uses allow_fuse(privileged=False))."
            )
        if sc.allow_privilege_escalation is True:
            raise ValueError(
                "allow_nested_sandboxing() sets allowPrivilegeEscalation=false on the primary container, but the pod "
                "template explicitly sets allowPrivilegeEscalation=true."
            )

    _set_apparmor_unconfined(pt, "allow_nested_sandboxing()")

    _add_sys_admin(primary)
    primary.security_context.allow_privilege_escalation = False

    _stamp_capability(pt, "nested-sandboxing")
    return pt
