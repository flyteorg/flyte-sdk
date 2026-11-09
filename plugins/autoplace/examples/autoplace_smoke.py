"""
Smoke test for `flyteplugins.autoplace.AutoPlaceEnvironment` on GKE: one T4 task, searched across the
queues passed to the driver (one queue per cluster). Each cluster needs the setup in the README ("GKE
per-cluster setup"), with the RBAC bound to the environment's `service_account`.

Run from the plugin's own environment so the plugin is importable, passing your queue names:
    cd plugins/autoplace && uv run flyte run examples/autoplace_smoke.py main \
        --queues '["cluster-a","cluster-b"]'

The images bake the local flyte and flyteplugins-autoplace wheels from dist/ (`make dist` and
`FLYTE_PLUGIN_DIST=plugins/autoplace make dist-plugins` first).
"""

from __future__ import annotations

import socket
from dataclasses import replace
from datetime import timedelta
from typing import List, Optional

import flyte

from flyteplugins.autoplace import AutoPlaceEnvironment
from flyteplugins.autoplace.gke import GKEFlexStart, probe_env

image = (
    flyte.Image.from_debian_base(python_version=(3, 12)).with_local_v2().with_local_v2_plugins("flyteplugins-autoplace")
)

# The probe environment reads its image at deploy time, so the local wheels have to be baked into it too.
probe_env.image = image

env = AutoPlaceEnvironment(
    name="autoplace-smoke",
    image=image,
    # One T4: flex-start T4 supply is deep, so this lands quickly. Match it to a flex-start pool you have.
    resources=flyte.Resources(gpu="T4:1", cpu=2, memory="8Gi"),
    cluster_queues=["cluster-a", "cluster-b"],  # replaced by the driver's `queues` input
    service_account="autoplace",
    provider=GKEFlexStart(max_run=timedelta(hours=1)),
    search_deadline=timedelta(hours=1),
)


@env.task
async def work(x: int) -> str:
    return f"{x} ran on {socket.gethostname()}"


driver_env = flyte.TaskEnvironment(
    name="autoplace-smoke-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[env],
)


@driver_env.task
async def main(queues: Optional[List[str]] = None) -> str:
    task = work
    if queues:
        task = work.override(plugin_config=replace(work.autoplace, cluster_queues=tuple(queues)))
    return await task(7)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
