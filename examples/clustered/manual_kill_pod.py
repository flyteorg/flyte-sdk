"""
Long-running CPU job for MANUAL chaos testing — kill a pod yourself and watch (CPU / gloo).

This task does light DDP work and then idles with heartbeats for a long time (default ~20 min) so
YOU can delete one of its worker pods by hand and observe how the clustered runtime reacts. It
checkpoints a step counter periodically, so after the set restarts you can see it resume rather than
start over.

What to expect when you kill a worker pod:
    kubectl delete pod <jobset>-workers-0-1-xxxxx
      ->  that pod's Job fails immediately (backoffLimit=0)
      ->  the JobSet restarts ALL pods together with a fresh rendezvous id
      ->  new pods come up; ctx.restart_attempt increments (printed in the heartbeats)
      ->  training resumes from the last checkpoint
    Repeat up to max_restarts; beyond that the run becomes a RetryableFailure.

RUNBOOK (read-only setup + the one mutating kill):
    export CLOUD_REPO=/Users/adil/Documents/Git/cloud
    export AWS_CONFIG_FILE=$CLOUD_REPO/gen/cli-config/aws
    export KUBECONFIG=$HOME/.kube/config:$CLOUD_REPO/gen/cli-config/kubeconfig
    C=<your-context>           # e.g. org-dogfood-dogfood-1
    NS=<project>-<domain>      # e.g. flytesnacks-development

    # 1. Watch the JobSet + its pods (leave this running in one terminal):
    kubectl get pods -n $NS --context=$C -w | grep workers
    # 2. Pick a NON-rank-0 worker (rank-0 is <jobset>-workers-0-0) and delete it:
    kubectl delete pod <jobset>-workers-0-1-xxxxx -n $NS --context=$C
    # 3. Watch: the whole set is recreated; heartbeats show restart_attempt climb.

Tip: killing rank-0 (<jobset>-workers-0-0) is also interesting — it's the rendezvous master, so the
whole set must re-form around the new pod-0.

    uv run python examples/clustered/manual_kill_pod.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# --- Knobs ---------------------------------------------------------------------------------------
RUN_MINUTES = 20  # how long to idle (with heartbeats) so you have time to kill a pod
MAX_RESTARTS = 3  # how many manual kills the set will recover from before giving up

image = (
    flyte.Image.from_debian_base(name="manual_kill_pod")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch")
)

env = ClusteredTaskEnvironment(
    name="manual_kill_env",
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    replicas=2,  # 2 worker pods so there's a non-rank-0 pod to kill
    nproc_per_node=1,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=MAX_RESTARTS),
)


@env.task
async def idle_for_chaos(minutes: int = RUN_MINUTES) -> int:
    """Idle with heartbeats so a pod can be killed by hand; checkpoint + resume across restarts."""
    import asyncio

    import torch.distributed as dist

    ctx = flyte.ctx()
    attempt = ctx.restart_attempt or 0
    rank = ctx.rank or 0

    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    print(
        f"[rank {rank}/{world_size}] restart_attempt={attempt} — ready. Kill a worker pod now; watch the set restart.",
        flush=True,
    )

    # Resume the heartbeat counter from a previous attempt's checkpoint (best-effort).
    cp = ctx.checkpoint
    start = 0
    if cp is not None:
        prev = await cp.load()
        if prev is not None:
            try:
                payload = prev / "payload" if prev.is_dir() else prev
                start = int(payload.read_text().strip())
                print(f"[rank {rank}] resumed at heartbeat {start}", flush=True)
            except (ValueError, OSError):
                start = 0

    total = minutes * 4  # one heartbeat every 15s
    for hb in range(start, total):
        # Every rank participates in a collective each heartbeat, so a killed pod is noticed promptly
        # (the survivors block here until the JobSet restarts the set).
        dist.barrier()
        if rank == 0:
            mins = hb / 4
            print(f"[rank 0] heartbeat {hb}/{total} (~{mins:.2f} min)  restart_attempt={attempt}", flush=True)
            if cp is not None and hb % 4 == 0:
                await cp.save(str(hb).encode())  # checkpoint progress so the next attempt resumes
        await asyncio.sleep(15)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] finished {minutes} min on attempt {attempt}", flush=True)
    return attempt


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(idle_for_chaos, minutes=RUN_MINUTES)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
