"""
Smoke test for `flyteplugins.autoplace.AutoPlaceEnvironment` against a Union tenant.

Two modes, picked by the driver's `mode` input:

- `real`: the GKE provider's probe files a ProvisioningRequest in each queue's cluster. On a cluster
  without the CRD (AWS) every probe reports a probe error and the driver returns NoCapacityError;
  on GKE without a flex-start pool the request is Accepted but never Provisioned, so the search
  times out. Either way it exercises deploy, the resolver import of the probe, RBAC, manifests,
  polling and cleanup.
- `fake`: `FakeProvider` below stands in for the cloud. Its probe answers "reserved" (fast for the
  first queue, slowly for the rest), so first-wins, cancellation of the losers, the consume
  annotations on the placed pod and the booking-window fail-fast path run without GKE. The placed
  pod cannot schedule (it selects flex-start nodes), so after `booking_window` it fails with
  MaxQueuedTimeExceeded, the search runs again, and the driver returns NoCapacityError after
  `max_placements` rounds. It is also the smallest example of writing a provider.

Run (dogfood org, AWS clusters), from the plugin's own environment so the plugin is importable:
    cd plugins/autoplace && uv run flyte --config ~/.flyte/config-dogfood.yaml run examples/autoplace_smoke.py main \
        --mode fake --queues '["dogfood-1","dogfood-2"]'

The images bake the local flyte and flyteplugins-autoplace wheels from dist/ (`make dist` and
`FLYTE_PLUGIN_DIST=plugins/autoplace make dist-plugins` first).
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import replace
from datetime import timedelta
from typing import List, Optional, Sequence

import flyte

from flyteplugins.autoplace import AutoPlaceEnvironment, NoCapacityError, PodSet, Reservation
from flyteplugins.autoplace.gke import GKEFlexStart, probe_env

image = (
    flyte.Image.from_debian_base(python_version=(3, 12)).with_local_v2().with_local_v2_plugins("flyteplugins-autoplace")
)

# The probe's image is read from its environment at deploy time, so the local wheels have to be baked
# into it too. Its service account is a provider setting (see below).
probe_env.image = image

env = AutoPlaceEnvironment(
    name="autoplace-smoke",
    image=image,
    # One L4: the shape dogfood-gcp's flex-start pool (gcpg2standard4l4flex, max 2 nodes) offers.
    resources=flyte.Resources(gpu="L4:1", cpu=2, memory="8Gi"),
    clusters=["dogfood-1", "dogfood-2"],  # replaced by the driver's `queues` input
    # Flex-start provisioning takes minutes; the booking window keeps the GKE default (10 min hold - 2).
    # The probe runs as `autoplace` (probe_env's default): on dogfood-gcp that SA has the probe RBAC and a
    # Workload Identity grant on the task bucket. Flex-start L4 supply is bursty, so the search waits an hour.
    provider=GKEFlexStart(max_run=timedelta(hours=1)),
    search_timeout=timedelta(hours=1),
    max_placements=2,
)


@env.task
async def work(x: int) -> str:
    return f"{x} ran on {socket.gethostname()}"


fake_env = flyte.TaskEnvironment(
    name="autoplace-smoke-fakeprobe",
    image=image,
    resources=flyte.Resources(cpu="250m", memory="256Mi"),
    cache="disable",
)


@fake_env.task
async def fake_probe(queue: str, podsets: List[PodSet], wait_seconds: int) -> Reservation:
    """The queue whose name ends in 1 answers in 5s; the others take 2 minutes, so they get cancelled."""
    await asyncio.sleep(5 if queue.endswith("1") else 120)
    return Reservation(queue=queue, ok=True, name=f"fake-{queue}", namespace="n/a")


class FakeProvider:
    """A CapacityProvider that always says yes; placement metadata is borrowed from the GKE provider."""

    name = "fake"
    capacity_hold = timedelta(minutes=10)
    environments: Sequence[flyte.TaskEnvironment] = (fake_env,)

    def validate(self, podsets: Sequence[PodSet]) -> None:
        pass

    def reserve(self, queue: str, podsets: Sequence[PodSet], wait: timedelta):
        return fake_probe.override(queue=queue)(queue, list(podsets), int(wait.total_seconds()))

    def consume(self, pod_template, reservation: Reservation):
        return GKEFlexStart().consume(pod_template, reservation)


driver_env = flyte.TaskEnvironment(
    name="autoplace-smoke-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[env, fake_env],
)


@driver_env.task
async def main(mode: str = "real", queues: Optional[List[str]] = None) -> str:
    task = work
    settings = task.autoplace
    if queues:
        settings = replace(settings, clusters=tuple(queues))
    if mode == "fake":
        settings = replace(settings, provider=FakeProvider())
    task.autoplace = settings
    try:
        return await task(7)
    except NoCapacityError as e:
        return f"NoCapacityError: {e}"


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, mode="fake")
    print(r.url)
