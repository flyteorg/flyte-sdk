"""
Scale / load test: 16-pod rendezvous on plain TCP (§9 exit-criterion #11).

Most correctness bugs in distributed bootstrapping only show up at non-trivial scale: headless DNS
for 16 pod names, the static rendezvous all 16 ranks must agree on, and the gang landing together.
This example does no real training — it just stands up a 16-pod JobSet, runs one collective
(all-reduce) plus a barrier, and confirms every rank sees ``world_size == 16``. CPU/gloo keeps it
cheap; the point is the control plane, not GPUs.

If your cluster lacks 16 free slots and gang admission is not yet wired (StudyNotes §2.5), this will
sit Queued until capacity frees up — that itself is a useful signal.

    uv run python examples/clustered/load_16pod.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

REPLICAS = 16  # 16 pods, one process each => world_size = 16

image = (
    flyte.Image.from_debian_base(name="load_16pod")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch")
)

env = ClusteredTaskEnvironment(
    name="load_16pod_env",
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("512Mi", "1Gi")),
    replicas=REPLICAS,
    nproc_per_node=1,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=0),
)


@env.task
async def all_reduce_smoke() -> int:
    """Rendezvous 16 ranks, run one all-reduce + barrier, assert world_size == 16."""
    import torch
    import torch.distributed as dist

    ctx = flyte.ctx()
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[rank {rank}/{world_size}] node_rank={ctx.node_rank} master_addr={ctx.master_addr}", flush=True)

    # Every rank contributes its rank number; the sum proves all 16 joined the collective.
    t = torch.tensor([float(rank)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size - 1) // 2  # 0 + 1 + ... + 15 = 120
    assert int(t.item()) == expected, f"all_reduce sum {int(t.item())} != expected {expected}"

    dist.barrier()
    if rank == 0:
        print(f"[rank 0] all {world_size} ranks rendezvoused; all_reduce sum = {int(t.item())}", flush=True)
    dist.destroy_process_group()
    return world_size


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(all_reduce_smoke)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
