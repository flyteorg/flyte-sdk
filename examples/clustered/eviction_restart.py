"""
Free restart on host maintenance / eviction.

When a node is reclaimed by the cloud provider (spot reclaim, host maintenance, drain) the pod is
evicted through no fault of the user's code. With ``restart_on_host_maintenance=True`` the JobSet
treats that as a FREE restart: the whole set restarts (``restart_attempt`` increments) but the
``max_restarts`` budget is left UNCHANGED, so a flaky cluster can't burn the user's real
failure budget.

Mechanism : the inner Job's ``podFailurePolicy`` matches the
``DisruptionTarget`` pod condition and ``FailJob``s; the JobSet's failure policy then matches that
``PodFailurePolicy`` job-failure reason and restarts with ``RestartJobSetAndIgnoreMaxRestarts``.

This is hard to trigger purely from task code — you induce the eviction from the cluster side. The
task just runs long enough for you to drain its node and watch the counters.

Runbook:
  1. Start the run; wait for the worker pods `<jobset>-workers-0-*` to be Running.
  2. Find a worker's node:  kubectl get pod <jobset>-workers-0-1 -n <ns> -o wide
  3. Drain it:             kubectl drain <node> --ignore-daemonsets --delete-emptydir-data
  4. Observe: the JobSet restarts, `ctx.restart_attempt` climbs, but `failurePolicy.maxRestarts`
     budget is not decremented (compare against a non-maintenance failure, which would consume it).

    uv run python examples/clustered/eviction_restart.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

image = (
    flyte.Image.from_debian_base(name="eviction_restart")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch")
)

env = ClusteredTaskEnvironment(
    name="eviction_env",
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    replicas=2,
    nproc_per_node=1,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    # The flag under test: evictions become free restarts, not budget-consuming failures.
    failure_policy=ClusterFailurePolicy(max_restarts=1, restart_on_host_maintenance=True),
)


@env.task
async def long_running(minutes: int = 10) -> int:
    """Idle long enough (printing heartbeats) for you to drain a worker's node mid-run."""
    import asyncio

    import torch.distributed as dist

    ctx = flyte.ctx()
    attempt = ctx.restart_attempt or 0
    rank = ctx.rank or 0

    dist.init_process_group(backend="gloo")
    print(f"[rank {rank}] restart_attempt={attempt} world_size={ctx.world_size} — draining now is safe", flush=True)

    heartbeats = minutes * 6  # one every 10s
    for hb in range(heartbeats):
        if rank == 0 and hb % 6 == 0:
            print(f"[rank 0] heartbeat {hb // 6} min, restart_attempt={attempt}", flush=True)
        await asyncio.sleep(10)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] finished after {minutes} min on attempt {attempt}", flush=True)
    return attempt


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(long_running, minutes=10)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
