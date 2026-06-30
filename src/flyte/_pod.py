from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from flyteidl2.core.tasks_pb2 import K8sPod
    from kubernetes.client import V1Container, V1PodSpec


_PRIMARY_CONTAINER_NAME_FIELD = "primary_container_name"
_PRIMARY_CONTAINER_DEFAULT_NAME = "primary"

# Name/path for the hostPath volume + volumeMount used by the legacy
# ``allow_fuse(privileged=True)`` escape hatch.
_FUSE_DEVICE_VOLUME_NAME = "fuse-device"
_FUSE_DEVICE_PATH = "/dev/fuse"

# Extended resource advertised by a FUSE device plugin (smarter-device-manager /
# fuse-device-plugin DaemonSet). Requesting it makes kubelet inject /dev/fuse
# into the container's devices-cgroup allowlist — granting an UNPRIVILEGED
# container access to the device, which a hostPath cannot do. This is what the
# default (device-plugin) path of ``PodTemplate.allow_fuse()`` uses.
_FUSE_DEVICE_RESOURCE = "smarter-devices/fuse"

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

    def allow_fuse(self, privileged: bool = False) -> PodTemplate:
        """
        Return a copy of this template granted everything an **unprivileged**
        container needs to perform an in-process FUSE mount (e.g. for ``Volume``
        support).

        By default (``privileged=False``) the copy:

        * requests the ``smarter-devices/fuse`` extended resource (request +
          limit) on the primary container, so the cluster's FUSE **device
          plugin** (smarter-device-manager / fuse-device-plugin DaemonSet)
          injects ``/dev/fuse`` into the container's devices-cgroup allowlist —
          making the node device *usable* from an unprivileged container; and
        * adds ``CAP_SYS_ADMIN`` to the primary container, required for the
          ``mount(2)`` syscall that attaches the FUSE filesystem;
        * stamps the ``flyte.org/capability-fuse`` annotation for auditability.

        It does **not** set ``privileged: true`` and does **not** add a
        ``/dev/fuse`` hostPath. A hostPath only makes the device *node* visible;
        the devices cgroup still denies ``open()`` with ``EPERM`` — the device
        plugin is what actually grants access. This default composes with
        ``allow_nested_sandboxing()``. The cluster must run a FUSE device plugin
        advertising ``smarter-devices/fuse`` (the Union dataplane chart ships an
        opt-in ``fuseDevicePlugin`` DaemonSet for this).

        ``privileged=True`` is a legacy escape hatch for clusters **without** a
        FUSE device plugin: it instead adds a ``/dev/fuse`` hostPath volume +
        mount and sets ``privileged: true`` on the primary container (the device
        cgroup is bypassed by privilege). It does not compose with
        ``allow_nested_sandboxing()`` (Kubernetes rejects privileged containers
        that set ``allowPrivilegeEscalation: false``).

        AppArmor note: on clusters that enforce a restrictive default AppArmor
        profile, the ``mount`` syscall may additionally need the primary
        container's profile set to ``unconfined`` — set that annotation on the
        template yourself if needed; it is not applied by default to keep this
        grant minimal.

        The original template is never mutated; existing volumes, mounts,
        resources, sidecars, labels, annotations, and unrelated security-context
        fields are preserved. Re-applying with the same arguments is idempotent.

        Raises ``ValueError`` if the template already pins a conflicting security
        posture (with ``privileged=True``: ``privileged: false`` or
        ``allowPrivilegeEscalation: false`` pre-set).
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


def _add_fuse_device_resource(primary: "V1Container") -> None:
    """Request the FUSE device-plugin resource on the primary container, merging
    into any resources the caller already set (request + limit, per the k8s rule
    that extended resources must have request == limit)."""
    from kubernetes.client import V1ResourceRequirements

    resources = primary.resources or V1ResourceRequirements()
    resources.requests = {**(resources.requests or {}), _FUSE_DEVICE_RESOURCE: "1"}
    resources.limits = {**(resources.limits or {}), _FUSE_DEVICE_RESOURCE: "1"}
    primary.resources = resources


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


def _apply_fuse(pod_template: PodTemplate, *, privileged: bool = False) -> PodTemplate:
    """Augmentor behind :meth:`PodTemplate.allow_fuse`. Dispatches to one of two
    clearly separate strategies; both grant CAP_SYS_ADMIN and stamp the fuse
    capability annotation."""
    pt = _clone_with_primary(pod_template)
    primary = _get_primary_container(pt)

    if privileged:
        _apply_fuse_privileged(pt, primary)
    else:
        _apply_fuse_device_plugin(primary)

    _add_sys_admin(primary)  # the mount(2) syscall needs it on both paths
    _stamp_capability(pt, "fuse")
    return pt


def _apply_fuse_device_plugin(primary: "V1Container") -> None:
    """Default, UNPRIVILEGED path: request the ``smarter-devices/fuse`` extended
    resource so a FUSE device plugin makes kubelet inject ``/dev/fuse`` into the
    container's devices-cgroup allowlist. No privileged, no hostPath — composes
    with ``allow_nested_sandboxing()``. Requires a FUSE device plugin on the
    cluster."""
    _add_fuse_device_resource(primary)


def _apply_fuse_privileged(pt: PodTemplate, primary: "V1Container") -> None:
    """Legacy escape hatch for clusters WITHOUT a FUSE device plugin: a
    ``/dev/fuse`` hostPath + ``privileged=true`` (privilege bypasses the device
    cgroup that would otherwise deny ``open()``)."""
    from kubernetes.client import V1HostPathVolumeSource, V1Volume, V1VolumeMount

    # Conflict checks: a privileged container cannot deny privilege escalation
    # (Kubernetes rejects the combination), and an explicit privileged=False
    # contradicts what this grant needs.
    sc = primary.security_context
    if sc is not None:
        if sc.privileged is False:
            raise ValueError(
                f"allow_fuse(privileged=True) requires privileged=true on the primary container "
                f"({pt.primary_container_name!r}), but the pod template explicitly sets privileged=false. "
                "On clusters with a FUSE device plugin, use allow_fuse() (the unprivileged default)."
            )
        if sc.allow_privilege_escalation is False:
            raise ValueError(
                "allow_fuse(privileged=True) makes the primary container privileged, which Kubernetes rejects "
                "when allowPrivilegeEscalation=false is set (e.g. after allow_nested_sandboxing()). Use the "
                "unprivileged default allow_fuse(), which composes with allow_nested_sandboxing()."
            )

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

    from kubernetes.client import V1SecurityContext

    primary.security_context = primary.security_context or V1SecurityContext()
    primary.security_context.privileged = True


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
