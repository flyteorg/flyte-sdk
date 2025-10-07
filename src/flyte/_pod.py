from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from flyteidl2.core.tasks_pb2 import K8sPod
    from kubernetes.client import V1PodSpec


_PRIMARY_CONTAINER_NAME_FIELD = "primary_container_name"
_PRIMARY_CONTAINER_DEFAULT_NAME = "primary"


@dataclass(init=True, repr=True, eq=True, frozen=False)
class PodTemplate(object):
    """Custom PodTemplate specification for a Task."""

    pod_spec: Optional["V1PodSpec"] = field(default_factory=lambda: V1PodSpec())
    primary_container_name: str = _PRIMARY_CONTAINER_DEFAULT_NAME
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

    def to_k8s_pod(self) -> "K8sPod":
        from flyteidl2.core.tasks_pb2 import K8sObjectMetadata, K8sPod
        from kubernetes.client import ApiClient

        return K8sPod(
            metadata=K8sObjectMetadata(labels=self.labels, annotations=self.annotations),
            pod_spec=ApiClient().sanitize_for_serialization(self.pod_spec),
            primary_container_name=self.primary_container_name,
        )
