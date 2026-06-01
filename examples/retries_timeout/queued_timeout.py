"""
Exercises ``flyte.Timeout(max_queued_time=...)`` — the budget for time the
plugin spends pre-Running on the leaseworker (pod scheduling, image pull,
container init). Anchored on the leasor-stamped ``lease.EnqueuedAt``.

The k3d-devbox cluster has no GPU nodes, so asking for a GPU keeps the pod
permanently Pending. The plugin stays in PluginPhase=Queued and the
leaseworker fires ``queued_timeout`` after the budget elapses. CPU-only
resource requests aren't reliable on k3d because the single-node cluster
doesn't enforce them strictly; the GPU request is honored because nodes
must advertise the resource explicitly via device plugins.
"""

import asyncio
from datetime import timedelta

import flyte

env = flyte.TaskEnvironment(
    name="queued_timeout_demo",
    # No node in k3d-devbox advertises nvidia.com/gpu, so the pod stays
    # Pending forever — perfect for exercising the queued_timeout path.
    resources=flyte.Resources(cpu=1, memory="250Mi", gpu="A100:1"),
)


@env.task(
    timeout=flyte.Timeout(max_queued_time=timedelta(seconds=10)),
)
async def too_slow_to_start() -> str:
    print("too_slow_to_start: user code reached (unexpected — queued_timeout should have fired)")
    await asyncio.sleep(1)
    return "ran to completion"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(too_slow_to_start)
    print(run.name)
    print(run.url)
