"""
The GKE capacity probe: a small task that runs *inside* a candidate cluster, files a
ProvisioningRequest for the worker's pod shape in its own namespace, and reports whether the
capacity came through.

It is a task rather than a `flyte.trace` for three reasons: the request must live in the pods'
namespace of the *target* cluster (so the driver would need a kubeconfig per cluster), a trace
replays a recorded error on every driver retry instead of searching again, and a trace cannot
fan out child tasks. As a child action the probe is durable (a retried driver reuses a completed
probe), cancellable (cancelling the awaited call aborts the action, deleting the pod, whose
ownerReference garbage-collects the request), and credential-free (in-cluster service account).

`probe_env` is module-level, like `hitl`'s task environment, because the default task resolver
imports `<module>.<attribute>` in the probe container; an environment built per
`AutoPlaceEnvironment` would not be importable there.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

import flyte

from .._provider import Reservation
from .._shape import PodSet
from . import _provreq as pr

logger = logging.getLogger(__name__)

PROBE_ENV_NAME = "gke-autoplace"
POLL_SECONDS = 10

probe_image = flyte.Image.from_debian_base(flyte_version=flyte.version(), name="gke-autoplace-probe").with_pip_packages(
    f"flyteplugins-autoplace=={flyte.version()}"
)

probe_env = flyte.TaskEnvironment(
    name=PROBE_ENV_NAME,
    image=probe_image,
    resources=flyte.Resources(cpu="250m", memory="256Mi"),
    # A cached probe would hand back the name of a request whose nodes are long gone.
    cache="disable",
    # Narrow RBAC (create/get/delete provisioningrequests and podtemplates, get pods) binds to this
    # account only; ordinary tasks never get it.
    service_account="autoplace",
)


def _condition_reason(cond: dict) -> str:
    reason = cond.get("reason") or ""
    message = cond.get("message") or ""
    return f"{reason}: {message}" if reason and message else reason or message


def evaluate_conditions(conditions: Dict[str, dict]) -> Tuple[Optional[bool], str]:
    """
    Read a ProvisioningRequest's conditions: `(True, "")` once provisioned, `(False, reason)` when GKE
    has given up (`Failed=True`), else `(None, reason)` to keep waiting, where `reason` is the latest
    thing GKE said. Both `Provisioned=False` (NotProvisioned, QuotaExceeded, ResourcePoolExhausted)
    and `Accepted=False` (e.g. NoQueuedNodepoolAvailable: no flex-start pool for this shape) are
    waited out rather than treated as final, since a pool or quota can appear; the reason is kept so a
    timeout says why.
    """
    provisioned = conditions.get("Provisioned", {})
    if provisioned.get("status") == "True":
        return True, ""
    failed = conditions.get("Failed", {})
    if failed.get("status") == "True":
        return False, _condition_reason(failed) or "Failed"
    for cond in (provisioned, conditions.get("Accepted", {})):
        if cond.get("status") == "False":
            reason = _condition_reason(cond)
            if reason:
                return None, reason
    return None, ""


@probe_env.task
async def reserve_capacity(
    queue: str, podsets: List[PodSet], max_run_seconds: int, wait_seconds: int, accelerator: Optional[str]
) -> Reservation:
    """
    File a ProvisioningRequest for `podsets` in this cluster and wait up to `wait_seconds` for
    `Provisioned=True`. `queue` is only echoed back so the caller can tell which probe answered;
    `accelerator` overrides the GPU-name lookup for the request's node selector.
    """
    result = Reservation(queue=queue)
    namespace = pr.in_cluster_namespace()
    name = f"autoplace-{uuid.uuid4().hex[:10]}"
    created = False
    try:
        core, custom = await asyncio.to_thread(pr.load_clients)
        pod = await asyncio.to_thread(pr.read_pod, core, namespace, os.environ["HOSTNAME"])
        owner = pr.owner_reference(pod)

        refs = []
        for i, podset in enumerate(podsets):
            template_name = f"{name}-{i}"
            manifest = pr.pod_template_manifest(
                template_name, namespace, podset, pr.accelerator_label(podset, accelerator), owner
            )
            await asyncio.to_thread(pr.create_pod_template, core, manifest)
            refs.append((template_name, podset.count))
        await asyncio.to_thread(
            pr.create_provisioning_request,
            custom,
            pr.provisioning_request_manifest(name, namespace, refs, max_run_seconds, owner),
        )
        created = True
        logger.info(f"Filed ProvisioningRequest {namespace}/{name} for {podsets}; waiting up to {wait_seconds}s")

        deadline = time.monotonic() + wait_seconds
        last = ""
        while True:
            conditions = await asyncio.to_thread(pr.read_conditions, custom, namespace, name)
            verdict, reason = evaluate_conditions(conditions)
            if verdict is True:
                result.ok, result.name, result.namespace = True, name, namespace
                return result
            if verdict is False:
                result.reason = reason
                return result
            if reason:
                last = reason
            if time.monotonic() >= deadline:
                result.reason = f"no capacity within {wait_seconds}s" + (f" ({last})" if last else "")
                return result
            await asyncio.sleep(POLL_SECONDS)
    except Exception as e:
        logger.warning(f"Probe for queue {queue} failed: {e!r}")
        result.reason = f"probe error: {e!r}"
        return result
    finally:
        # The ownerReference already covers cancellation and crashes; this keeps a request that we
        # know is not going to be consumed from holding a place in the queue a moment longer.
        if created and not result.ok:
            try:
                await asyncio.to_thread(pr.delete_provisioning_request, custom, namespace, name)
            except Exception as e:
                logger.warning(f"Could not delete ProvisioningRequest {namespace}/{name}: {e!r}")
