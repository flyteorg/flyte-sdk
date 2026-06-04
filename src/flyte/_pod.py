import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from flyteidl2.core.tasks_pb2 import K8sPod
    from kubernetes.client import V1PodSpec


_PRIMARY_CONTAINER_NAME_FIELD = "primary_container_name"
_PRIMARY_CONTAINER_DEFAULT_NAME = "primary"

# Name used for the hostPath volume and its volumeMount when
# ``enable_fuse_mount=True`` is set on an Environment.
_FUSE_DEVICE_VOLUME_NAME = "fuse-device"
_FUSE_DEVICE_PATH = "/dev/fuse"


@dataclass(init=True, repr=True, eq=True, frozen=False)
class PodTemplate(object):
    """Custom PodTemplate specification for a Task."""

    pod_spec: Optional["V1PodSpec"] = None
    primary_container_name: str = _PRIMARY_CONTAINER_DEFAULT_NAME
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

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


def pod_template_with_fuse_mount(
    pod_template: Optional[PodTemplate] = None,
    *,
    primary_container_name: str = _PRIMARY_CONTAINER_DEFAULT_NAME,
    privileged_only: bool = False,
) -> PodTemplate:
    """Return a :class:`PodTemplate` augmented with everything a container
    needs to perform an in-process FUSE mount.

    Specifically, this:

    * Adds a ``hostPath`` volume named ``fuse-device`` pointing at
      ``/dev/fuse`` on the node.
    * Mounts that volume into the primary container at ``/dev/fuse``.
    * Sets the primary container's ``securityContext`` to ``privileged=True``
      with ``CAP_SYS_ADMIN`` so the ``mount`` syscall is permitted.

    With ``privileged_only=True``, only ``privileged=True`` is set — no
    hostPath volume/mount and no explicit capability add. A privileged
    container already sees every host device (including ``/dev/fuse``)
    and holds all capabilities, so those fields are redundant; they only
    exist for defense-in-depth on raw task pods. Knative's admission
    webhook rejects hostPath volumes (no support before Serving 1.17,
    feature-gated after) and gates ``capabilities.add``, so App pods
    must use this mode.

    If ``pod_template`` is provided, the returned template is a deep copy
    with the FUSE bits *added* — existing volumes, mounts, security
    context fields, sidecars, labels, and annotations are preserved.
    Re-applying to an already-FUSE-enabled template is idempotent.

    Used by the SDK at serialization time when a ``TaskEnvironment`` or
    ``AppEnvironment`` has ``enable_fuse_mount=True``. Callers normally
    set the flag instead of touching this helper directly; it's exposed
    for advanced cases (custom serializers, manual K8sPod construction).
    """
    from kubernetes.client import (
        V1Capabilities,
        V1Container,
        V1HostPathVolumeSource,
        V1PodSpec,
        V1SecurityContext,
        V1Volume,
        V1VolumeMount,
    )

    if pod_template is None:
        pt = PodTemplate(
            pod_spec=V1PodSpec(containers=[V1Container(name=primary_container_name)]),
            primary_container_name=primary_container_name,
        )
    else:
        pt = copy.deepcopy(pod_template)
        if pt.pod_spec is None:
            pt.pod_spec = V1PodSpec(containers=[V1Container(name=pt.primary_container_name)])

    pod_spec = pt.pod_spec
    assert pod_spec is not None  # set above

    # Ensure the primary container exists; create a minimal one if not.
    containers = list(pod_spec.containers or [])
    primary: Optional[V1Container] = next((c for c in containers if c.name == pt.primary_container_name), None)
    if primary is None:
        primary = V1Container(name=pt.primary_container_name)
        containers.append(primary)
        pod_spec.containers = containers

    # Volume + volumeMount for /dev/fuse — idempotent: skip if already present.
    # Skipped entirely in privileged_only mode (privileged containers already
    # see host devices; Knative rejects hostPath volumes).
    if not privileged_only:
        volumes = list(pod_spec.volumes or [])
        if not any(getattr(v, "name", None) == _FUSE_DEVICE_VOLUME_NAME for v in volumes):
            volumes.append(
                V1Volume(
                    name=_FUSE_DEVICE_VOLUME_NAME,
                    host_path=V1HostPathVolumeSource(path=_FUSE_DEVICE_PATH, type="CharDevice"),
                )
            )
            pod_spec.volumes = volumes

        mounts = list(primary.volume_mounts or [])
        if not any(getattr(m, "name", None) == _FUSE_DEVICE_VOLUME_NAME for m in mounts):
            mounts.append(V1VolumeMount(name=_FUSE_DEVICE_VOLUME_NAME, mount_path=_FUSE_DEVICE_PATH))
            primary.volume_mounts = mounts

    # SecurityContext: privileged (+ CAP_SYS_ADMIN unless privileged_only —
    # privileged already implies all capabilities, and Knative gates
    # capabilities.add). Preserve any other security-context fields the
    # caller set.
    sc = primary.security_context or V1SecurityContext()
    sc.privileged = True
    if not privileged_only:
        caps = sc.capabilities or V1Capabilities()
        existing_add = list(caps.add or [])
        if "SYS_ADMIN" not in existing_add:
            existing_add.append("SYS_ADMIN")
        caps.add = existing_add
        sc.capabilities = caps
    primary.security_context = sc

    return pt
