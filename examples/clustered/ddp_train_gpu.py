"""
Multi-node GPU DDP training on a ClusteredTaskEnvironment (NCCL backend).

The GPU sibling of ``ddp_train.py`` (which uses CPU/gloo). Trains a tiny linear-regression model
with PyTorch DistributedDataParallel across ``replicas x nproc_per_node`` GPU workers, each pinned
to its local GPU. The workers are bootstrapped by ``torchrun`` via the dedicated ``clustered``
runtime entrypoint, which the Go ``clustered`` plugin wires into a Kubernetes JobSet.

This doubles as the §9 exit-criteria GPU smoke (#1 1x1, #2 2x1): assert a JobSet is created, the
gang lands, ``torchrun`` -> ``clustered`` workers rendezvous over NCCL, ``world_size`` matches, and
only rank-0 uploads outputs.

Set GPU_DEVICE / REPLICAS / NPROC_PER_NODE to match your cluster's GPU capacity, then run on a GPU
cluster (e.g. dogfood, domain=development):
    uv run python examples/clustered/ddp_train_gpu.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# --- Knobs ---------------------------------------------------------------------------------------
GPU_DEVICE = "L4"  # one of flyte._resources.GPUType device names; match the cluster's GPUs
REPLICAS = 2  # pods (== nodes). Set to 1 for the 1x1 smoke, 2 for 2x1.
NPROC_PER_NODE = 1  # GPUs (processes) per pod; must be <= the GPU count on the device

# Image carries the LOCAL flyte build (so the container has the `clustered` runtime entrypoint),
# plus torch. The PyPI `torch` wheel bundles CUDA + NCCL, so this same image runs on GPU nodes —
# no separate CUDA base image needed.
image = (
    flyte.Image.from_debian_base(name="ddp_train_gpu")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch", "numpy")
)

env = ClusteredTaskEnvironment(
    name="ddp_gpu_env",
    image=image,
    resources=flyte.Resources(cpu=(2, 4), memory=("4Gi", "8Gi"), gpu=flyte.GPU(GPU_DEVICE, NPROC_PER_NODE)),
    replicas=REPLICAS,
    nproc_per_node=NPROC_PER_NODE,  # world_size = REPLICAS * NPROC_PER_NODE
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=1),
)


@env.task
async def train_ddp_gpu(steps: int = 50, lr: float = 0.05) -> float:
    """Run NCCL DDP training on GPUs and return the final (rank-0) training loss."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    ctx = flyte.ctx()
    assert torch.cuda.is_available(), "no CUDA device visible — is this running on a GPU node?"

    # torchrun has already populated RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT. Pin this rank
    # to its local GPU BEFORE init_process_group so NCCL binds to the right device.
    local_rank = ctx.local_rank or 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        f"[rank {rank}/{world_size}] device={device} ctx.rank={ctx.rank} "
        f"ctx.local_rank={ctx.local_rank} ctx.world_size={ctx.world_size} "
        f"ctx.node_rank={ctx.node_rank} ctx.nnodes={ctx.nnodes} master_addr={ctx.master_addr}",
        flush=True,
    )

    # Tiny model the workers train cooperatively: learn y = x · [1,1,1,1].
    torch.manual_seed(0)
    model = nn.Linear(4, 1).to(device)
    ddp = DDP(model, device_ids=[local_rank])
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Each rank trains on its own shard of synthetic data.
    g = torch.Generator().manual_seed(rank)
    x = torch.randn(64, 4, generator=g).to(device)
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
    run = flyte.run(train_ddp_gpu, steps=50)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
