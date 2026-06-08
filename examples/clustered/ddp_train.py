"""
End-to-end DDP training on a ClusteredTaskEnvironment.

Trains a tiny linear-regression model with PyTorch DistributedDataParallel across
``replicas x nproc_per_node`` workers. The workers are bootstrapped by ``torchrun``
via the clustered entrypoint (``python -m flyte.clustered._entrypoint``), which the
Go ``clustered`` plugin wires into a Kubernetes JobSet. Backend is ``gloo`` (CPU) so it
needs no GPUs — swap to ``nccl`` + ``resources.gpu`` for real GPU training.

This exercises the full path:
    ClusteredTaskEnvironment  ->  task_serde (type=clustered-task, command rewrite)
      ->  JobSet (N pods)  ->  entrypoint DNS wait + torchrun  ->  Nxa0 per pod
      ->  torch.distributed rendezvous  ->  DDP training  ->  rank-0 uploads outputs.

Run (registers + runs on the configured cluster):
    uv run python examples/clustered/ddp_train.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# Image carries the LOCAL flyte build (so the container has flyte.clustered._entrypoint
# and the clustered runtime fixes), plus torch for the actual DDP workload.
image = (
    flyte.Image.from_debian_base(name="ddp_train6")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch", "numpy")
)

env = ClusteredTaskEnvironment(
    name="ddp_env",
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    replicas=2,  # 2 pods (== 2 nodes)
    nproc_per_node=2,  # 2 worker processes per pod  => world_size = 4
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=1),
)


@env.task
async def train_ddp(steps: int = 50, lr: float = 0.05) -> float:
    """Run DDP training and return the final (rank-0) training loss."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    ctx = flyte.ctx()

    # torchrun has already populated RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        f"[rank {rank}/{world_size}] ctx.rank={ctx.rank} ctx.world_size={ctx.world_size} "
        f"ctx.node_rank={ctx.node_rank} ctx.nnodes={ctx.nnodes} master_addr={ctx.master_addr}",
        flush=True,
    )

    # Tiny model the workers train cooperatively: learn y = x · [1,1,1,1].
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Each rank trains on its own shard of synthetic data.
    g = torch.Generator().manual_seed(rank)
    x = torch.randn(64, 4, generator=g)
    y = x.sum(dim=1, keepdim=True)

    last_loss = 0.0
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if rank == 0 and step % 10 == 0:
            print(f"[rank 0] step {step:3d}  loss {last_loss:.5f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] done — final loss {last_loss:.5f}", flush=True)
    return last_loss


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_ddp, steps=50)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
