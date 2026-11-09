"""
The GKE capacity probe: a small task that runs *inside* a candidate cluster, files a
ProvisioningRequest for the worker's pod shape in its own namespace, and returns once GKE has
provisioned it.

It is a task rather than a `flyte.trace` because it has to run in the *target* cluster: it uses that
cluster's in-cluster credentials and RBAC, and it reaches the cluster by being submitted to the
cluster's queue. As a child action it is also durable (a retried parent reuses a completed probe) and
cancellable (cancelling the awaited call aborts the action, deleting the pod, whose ownerReference
garbage-collects the request). The search bounds it with a Flyte deadline rather than a timer of its own.

A probe that wins hands its request over before exiting: the request stops being owned by the probe
pod (which is deleted seconds after the probe finishes) and its PodTemplates become owned by the
request. It then lives until `release_capacity` deletes it, which takes the templates along.

`probe_env` is module-level, like `hitl`'s task environment, because the default task resolver
imports `<module>.<attribute>` in the probe container; an environment built per
`AutoPlaceEnvironment` would not be importable there.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple

import flyte

from .._shape import PodSet
from . import _provreq as pr

logger = logging.getLogger(__name__)

PROBE_ENV_NAME = "gke-autoplace"
POLL_SECONDS = 10

probe_image = flyte.Image.from_debian_base(flyte_version=flyte.version(), name="gke-autoplace-probe").with_pip_packages(
    f"flyteplugins-autoplace=={flyte.version()}"
)

# The service account is the AutoPlaceEnvironment's, applied per call with `override`.
probe_env = flyte.TaskEnvironment(
    name=PROBE_ENV_NAME,
    image=probe_image,
    resources=flyte.Resources(cpu="250m", memory="256Mi"),
    # A cached probe would hand back the name of a request whose nodes are long gone.
    cache="disable",
)


class CapacityRequestFailed(RuntimeError):
    """GKE gave up on the ProvisioningRequest (`Failed=True`)."""


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
    waited out rather than treated as final, since a pool or quota can appear.
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
async def reserve_capacity(name: str, podsets: List[PodSet], max_run_seconds: int, accelerator: Optional[str]) -> None:
    """
    File a ProvisioningRequest called `name` for `podsets` in this cluster and return once it is
    provisioned; raise if GKE gives up on it. `accelerator` overrides the GPU-name lookup for the
    request's node selector.
    """
    namespace = pr.in_cluster_namespace()
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
    request = await asyncio.to_thread(
        pr.create_provisioning_request,
        custom,
        pr.provisioning_request_manifest(name, namespace, refs, max_run_seconds, owner),
    )
    logger.info(f"Filed ProvisioningRequest {namespace}/{name} for {podsets}")

    won = False
    try:
        while True:
            conditions = await asyncio.to_thread(pr.read_conditions, custom, namespace, name)
            verdict, reason = evaluate_conditions(conditions)
            if verdict is True:
                won = True
                break
            if verdict is False:
                raise CapacityRequestFailed(f"GKE gave up on ProvisioningRequest {namespace}/{name}: {reason}")
            if reason:
                logger.info(f"ProvisioningRequest {namespace}/{name} waiting: {reason}")
            await asyncio.sleep(POLL_SECONDS)
    finally:
        # The ownerReference already covers cancellation and crashes; this keeps a request that we
        # know is not going to be consumed from holding a place in the queue a moment longer.
        if not won:
            try:
                await asyncio.to_thread(pr.delete_provisioning_request, custom, namespace, name)
            except Exception as e:
                logger.warning(f"Could not delete ProvisioningRequest {namespace}/{name}: {e!r}")

    try:
        await asyncio.to_thread(pr.hand_over, core, custom, request, [t for t, _ in refs])
    except Exception as e:
        # Without the handover the request is deleted with this pod, possibly before the placed task
        # binds; GKE has been seen to keep the nodes regardless, so report the win rather than give it
        # up. Usually missing `patch` RBAC.
        logger.warning(f"Could not hand over ProvisioningRequest {namespace}/{name}: {e!r}")


@probe_env.task
async def release_capacity(name: str) -> None:
    """
    Delete the ProvisioningRequest `name` in this cluster; its PodTemplates go with it (they are owned
    by the request once handed over). Already gone, or never created, is fine.
    """
    namespace = pr.in_cluster_namespace()
    _, custom = await asyncio.to_thread(pr.load_clients)
    await asyncio.to_thread(pr.delete_provisioning_request, custom, namespace, name)
    logger.info(f"Released ProvisioningRequest {namespace}/{name}")
