"""
GKE ProvisioningRequest plumbing: manifest builders (pure, unit-testable), the Kubernetes calls the
probe makes from inside the target cluster, and the pod-template merge that makes the placed worker
consume the provisioned nodes.

Contract (GKE docs, "Run a large-scale workload with flex-start with queued provisioning"):
  - The ProvisioningRequest must be in the same namespace as the pods that consume it.
  - Each `podSets[]` entry references a `PodTemplate` object; nodes come up atomically, in one zone.
  - Once `Provisioned=True`, the pods have 10 minutes to start before the autoscaler removes the nodes.
  - Consuming pods carry two annotations naming the request and the provisioning class.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from flyte._internal.runtime.resources_serde import ACCELERATOR_DEVICE_MAP
from flyte._pod import PodTemplate

from .._shape import PodSet

PROVREQ_GROUP = "autoscaling.x-k8s.io"
PROVREQ_VERSION = "v1"
PROVREQ_PLURAL = "provisioningrequests"
QUEUED_PROVISIONING_CLASS = "queued-provisioning.gke.io"

CONSUME_ANNOTATION = "autoscaling.x-k8s.io/consume-provisioning-request"
CLASS_ANNOTATION = "autoscaling.x-k8s.io/provisioning-class-name"

FLEX_START_LABEL = "cloud.google.com/gke-flex-start"
ACCELERATOR_LABEL = "cloud.google.com/gke-accelerator"
QUEUED_TAINT = "cloud.google.com/gke-queued"
# Required on the PodTemplate so GKE applies its warden policies to the simulated pods.
WARDEN_LABEL = "cloud.google.com/apply-warden-policies"

# The PodTemplate's container is only ever simulated by the autoscaler, never run.
PAUSE_IMAGE = "registry.k8s.io/pause:3.9"

SERVICE_ACCOUNT_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

# Marks every object the probe creates, so they can be found if a release is ever missed.
MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
MANAGED_BY = "flyteplugins-autoplace"

JSON_PATCH = "application/json-patch+json"


def accelerator_label(podset: PodSet, override: Optional[str] = None) -> Optional[str]:
    """
    The `cloud.google.com/gke-accelerator` value for the podset's GPU, or None for CPU-only pods and
    GPU names the SDK does not map.

    The ProvisioningRequest's PodTemplate pins the accelerator so the cluster autoscaler simulates the
    fit against the right flex-start pool. It has to be the value the worker pod will select on, or
    the nodes come up and the worker cannot land on them. The backend derives that selector from the
    task's serialized GPU device, which the SDK fills from `ACCELERATOR_DEVICE_MAP`, so the same map
    is used here. A name the SDK leaves unmapped is normalized by backend config the SDK cannot see,
    so it gets no label (the autoscaler picks any flex-start pool that fits) unless `override` says.
    """
    if not podset.gpu_count:
        return None
    if override:
        return override
    return ACCELERATOR_DEVICE_MAP.get(podset.gpu_type or "")


def owner_reference(pod: Dict[str, Any]) -> Dict[str, Any]:
    """
    An ownerReference to the probe's own pod. Garbage collection then deletes the PodTemplates and
    the ProvisioningRequest whenever the pod goes away, which is how a cancelled probe (losing
    cluster, aborted run) withdraws its request without any code of ours running. A probe that wins
    hands ownership over (`handover_patches`) before it exits, since its pod is deleted soon after.
    """
    meta = pod["metadata"]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "name": meta["name"],
        "uid": meta["uid"],
        "blockOwnerDeletion": False,
    }


def pod_template_manifest(
    name: str, namespace: str, podset: PodSet, accelerator: Optional[str], owner: Dict[str, Any]
) -> Dict[str, Any]:
    requests: Dict[str, str] = {"cpu": podset.cpu, "memory": podset.memory}
    if podset.gpu_count:
        requests["nvidia.com/gpu"] = str(podset.gpu_count)
    node_selector = {FLEX_START_LABEL: "true"}
    if accelerator:
        node_selector[ACCELERATOR_LABEL] = accelerator
    return {
        "apiVersion": "v1",
        "kind": "PodTemplate",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {WARDEN_LABEL: "true", MANAGED_BY_LABEL: MANAGED_BY},
            "ownerReferences": [owner],
        },
        "template": {
            "spec": {
                "nodeSelector": node_selector,
                # Tolerate everything: the dataplane taints its pools (flyte.org/node-role, nvidia.com/gpu,
                # the gke-queued taint) and tolerates them from config the SDK cannot see. The template is
                # simulated, never scheduled, so a blanket toleration only keeps the fit from failing on
                # a taint the real pod would have tolerated anyway.
                "tolerations": [{"operator": "Exists"}],
                "containers": [
                    {
                        "name": "primary",
                        "image": PAUSE_IMAGE,
                        "resources": {"requests": requests, "limits": requests},
                    }
                ],
                "restartPolicy": "Never",
            }
        },
    }


def provisioning_request_manifest(
    name: str,
    namespace: str,
    podsets: Sequence[Tuple[str, int]],
    max_run_seconds: int,
    owner: Dict[str, Any],
) -> Dict[str, Any]:
    """`podsets` pairs each PodTemplate name with its pod count."""
    return {
        "apiVersion": f"{PROVREQ_GROUP}/{PROVREQ_VERSION}",
        "kind": "ProvisioningRequest",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {MANAGED_BY_LABEL: MANAGED_BY},
            "ownerReferences": [owner],
        },
        "spec": {
            "provisioningClassName": QUEUED_PROVISIONING_CLASS,
            # GKE takes the duration as a string; it cannot be changed after creation.
            "parameters": {"maxRunDurationSeconds": str(max_run_seconds)},
            "podSets": [{"count": count, "podTemplateRef": {"name": template}} for template, count in podsets],
        },
    }


def request_owner_reference(request: Dict[str, Any]) -> Dict[str, Any]:
    """An ownerReference to a ProvisioningRequest, for the PodTemplates its podSets point at."""
    meta = request["metadata"]
    return {
        "apiVersion": f"{PROVREQ_GROUP}/{PROVREQ_VERSION}",
        "kind": "ProvisioningRequest",
        "name": meta["name"],
        "uid": meta["uid"],
        "blockOwnerDeletion": False,
    }


def handover_patches(request: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    JSON Patches that move a won reservation off the probe's pod: the PodTemplates become owned by the
    ProvisioningRequest (so deleting the request takes them along), and the request owned by nothing,
    so it outlives the probe pod (deleted seconds after the probe exits) until it is released. Returns
    (patch for each PodTemplate, patch for the request). JSON Patch rather than a merge, which for
    ownerReferences would add the new owner next to the pod instead of replacing it.
    """
    templates = [{"op": "replace", "path": "/metadata/ownerReferences", "value": [request_owner_reference(request)]}]
    req = [{"op": "remove", "path": "/metadata/ownerReferences"}]
    return templates, req


def provisioned_at(conditions: Dict[str, Dict[str, Any]]) -> float:
    """When GKE marked the request Provisioned (epoch seconds), 0 if it has not or did not say."""
    from datetime import datetime

    stamp = conditions.get("Provisioned", {}).get("lastTransitionTime")
    if not stamp or conditions.get("Provisioned", {}).get("status") != "True":
        return 0.0
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def consume_pod_template(base: Optional[PodTemplate], provreq_name: str) -> PodTemplate:
    """
    Return a copy of `base` (or a fresh template) whose pods consume `provreq_name`: the two GKE
    annotations, the flex-start node selector, and a toleration for the queued-provisioning taint.
    Everything already on `base` is kept; `override(pod_template=)` replaces rather than merges, so
    this is where the environment's own template and the placement meet.
    """
    from kubernetes.client import V1PodSpec, V1Toleration

    pt = copy.deepcopy(base) if base is not None else PodTemplate()
    pt.annotations = {
        **(pt.annotations or {}),
        CONSUME_ANNOTATION: provreq_name,
        CLASS_ANNOTATION: QUEUED_PROVISIONING_CLASS,
    }
    if pt.pod_spec is None:
        # No containers: task_serde clones in the primary container from the task's resources.
        pt.pod_spec = V1PodSpec(containers=[])
    pt.pod_spec.node_selector = {**(pt.pod_spec.node_selector or {}), FLEX_START_LABEL: "true"}
    tolerations = list(pt.pod_spec.tolerations or [])
    if not any(getattr(t, "key", None) == QUEUED_TAINT for t in tolerations):
        tolerations.append(V1Toleration(key=QUEUED_TAINT, operator="Exists", effect="NoSchedule"))
    pt.pod_spec.tolerations = tolerations
    return pt


# --- Kubernetes calls. Synchronous (the official client is), run via asyncio.to_thread by the probe. ---


def in_cluster_namespace() -> str:
    with open(SERVICE_ACCOUNT_NAMESPACE_FILE) as f:
        return f.read().strip()


def load_clients():
    from kubernetes import client, config

    config.load_incluster_config()
    return client.CoreV1Api(), client.CustomObjectsApi()


def read_pod(core, namespace: str, name: str) -> Dict[str, Any]:
    from kubernetes.client import ApiClient

    return ApiClient().sanitize_for_serialization(core.read_namespaced_pod(name, namespace))


def request_name(search: str, queue: str) -> str:
    """
    The ProvisioningRequest name a search's probe uses in `queue`: fixed by the search and the queue,
    so the search can release it without hearing back from the probe (a cancelled probe may have won).
    """
    import hashlib

    return f"autoplace-{hashlib.sha256(f'{search}/{queue}'.encode()).hexdigest()[:10]}"


def _exists(e: Exception) -> bool:
    return getattr(e, "status", None) == 409


def create_pod_template(core, manifest: Dict[str, Any]) -> None:
    """Create the template; one left by an earlier attempt of the same probe is reused."""
    try:
        core.create_namespaced_pod_template(manifest["metadata"]["namespace"], manifest)
    except Exception as e:
        if not _exists(e):
            raise


def create_provisioning_request(custom, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the request and return it as stored (with its uid). One left by an earlier attempt of the
    same probe is reused rather than filed twice.
    """
    namespace, name = manifest["metadata"]["namespace"], manifest["metadata"]["name"]
    try:
        return custom.create_namespaced_custom_object(
            PROVREQ_GROUP, PROVREQ_VERSION, namespace, PROVREQ_PLURAL, manifest
        )
    except Exception as e:
        if not _exists(e):
            raise
        return custom.get_namespaced_custom_object(PROVREQ_GROUP, PROVREQ_VERSION, namespace, PROVREQ_PLURAL, name)


def hand_over(core, custom, request: Dict[str, Any], template_names: Sequence[str]) -> None:
    """Apply `handover_patches`: templates first, so they are never left without an owner."""
    namespace = request["metadata"]["namespace"]
    template_patch, request_patch = handover_patches(request)
    for template in template_names:
        core.patch_namespaced_pod_template(template, namespace, template_patch, _content_type=JSON_PATCH)
    custom.patch_namespaced_custom_object(
        PROVREQ_GROUP,
        PROVREQ_VERSION,
        namespace,
        PROVREQ_PLURAL,
        request["metadata"]["name"],
        request_patch,
        _content_type=JSON_PATCH,
    )


def read_conditions(custom, namespace: str, name: str) -> Dict[str, Dict[str, Any]]:
    """The request's status conditions keyed by type (`Accepted`, `Provisioned`, `Failed`, ...)."""
    obj = custom.get_namespaced_custom_object(PROVREQ_GROUP, PROVREQ_VERSION, namespace, PROVREQ_PLURAL, name)
    conditions: List[Dict[str, Any]] = (obj.get("status") or {}).get("conditions") or []
    return {c["type"]: c for c in conditions if "type" in c}


def delete_provisioning_request(custom, namespace: str, name: str) -> None:
    from kubernetes.client.exceptions import ApiException

    try:
        custom.delete_namespaced_custom_object(PROVREQ_GROUP, PROVREQ_VERSION, namespace, PROVREQ_PLURAL, name)
    except ApiException as e:
        if e.status != 404:
            raise
