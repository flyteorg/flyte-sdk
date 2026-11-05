"""
Smoke test for `flyteplugins.autoplace.AutoPlaceEnvironment` against a deployment with one queue per cluster.

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

Run from the plugin's own environment so the plugin is importable, passing your queue names:
    cd plugins/autoplace && uv run flyte run examples/autoplace_smoke.py main \
        --mode fake --queues '["cluster-a","cluster-b"]'

The images bake the local flyte and flyteplugins-autoplace wheels from dist/ (`make dist` and
`FLYTE_PLUGIN_DIST=plugins/autoplace make dist-plugins` first).
"""

from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import replace
from datetime import timedelta
from typing import List, Optional, Sequence

import flyte

from flyteplugins.autoplace import AutoPlaceEnvironment, NoCapacityError, PodSet, Reservation, Selection, race
from flyteplugins.autoplace.gke import GKEFlexStart, probe_env, select_env

image = (
    flyte.Image.from_debian_base(python_version=(3, 12)).with_local_v2().with_local_v2_plugins("flyteplugins-autoplace")
)

# The provider's environments (the probe, and the search that races the probes) read their image at deploy
# time, so the local wheels have to be baked into them too. The probe's service account is a provider setting.
probe_env.image = image
select_env.image = image

env = AutoPlaceEnvironment(
    name="autoplace-smoke",
    image=image,
    # One T4: flex-start T4 supply is deep, so this lands quickly. Match it to a flex-start pool you have.
    resources=flyte.Resources(gpu="T4:1", cpu=2, memory="8Gi"),
    clusters=["cluster-a", "cluster-b"],  # replaced by the driver's `queues` input
    # Flex-start provisioning takes minutes; the booking window keeps the GKE default (10 min hold - 2).
    # The probe runs as `autoplace` (probe_env's default), which needs the probe RBAC and, on GKE with Workload
    # Identity, read/write on the task buckets. Flex-start supply can be bursty, so the search waits an hour.
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
async def fake_probe(queue: str, delay_seconds: int) -> Reservation:
    """Say yes after `delay_seconds`, without asking any cloud."""
    await asyncio.sleep(delay_seconds)
    return Reservation(queue=queue, ok=True, name=f"fake-{queue}", namespace="n/a", provisioned_at=time.time())


@fake_env.task
async def fake_select(queues: List[str]) -> Selection:
    """The first queue answers in 5s, the others in 2 minutes, so they are cancelled."""

    def probe(i: int, queue: str):
        return lambda: fake_probe.override(queue=queue).aio(queue, 5 if i == 0 else 120)

    return await race({q: probe(i, q) for i, q in enumerate(queues)})


@fake_env.task
async def fake_release(name: str) -> None:
    """Nothing to give back: no cloud was asked."""


class FakeProvider:
    """
    A CapacityProvider that always says yes; placement metadata is borrowed from the GKE provider. It is
    also the smallest example of writing one: `select` and `release` are tasks, so a retried driver
    replays them.
    """

    name = "fake"
    capacity_hold = timedelta(minutes=10)
    environments: Sequence[flyte.TaskEnvironment] = (fake_env,)

    def validate(self, podsets: Sequence[PodSet]) -> None:
        pass

    def select(self, queues: Sequence[str], podsets: Sequence[PodSet], wait: timedelta):
        return fake_select(list(queues))

    def consume(self, pod_template, reservation: Reservation):
        return GKEFlexStart().consume(pod_template, reservation)

    def release(self, reservation: Reservation):
        return fake_release.override(queue=reservation.queue)(reservation.name)


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
