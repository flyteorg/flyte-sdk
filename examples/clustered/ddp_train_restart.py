"""
DDP training that FAILS on attempt 0 and SUCCEEDS on the JobSet restart.

This is a regression test for the clustered-restart fix on this branch: when a
JobSet restarts (``JOBSET_RESTART_ATTEMPT`` > 0), ``upload_outputs`` must first
delete the stale ``error.pb`` written by the previous failed attempt. Otherwise
the successful retry's outputs land alongside a leftover error file and the
execution is still reported as FAILED.

Mechanics:
    - ``ClusterFailurePolicy(max_restarts=1)`` lets the JobSet restart once.
    - Attempt 0 (``JOBSET_RESTART_ATTEMPT`` unset / "0") raises -> writes error.pb.
    - Attempt 1 (``JOBSET_RESTART_ATTEMPT`` == "1") runs DDP and uploads outputs.
      With the fix, the stale error.pb is cleared and the run ends SUCCEEDED.
      Without the fix, the run ends FAILED despite the successful retry.

Run:
    uv run python examples/clustered/ddp_train_restart.py
"""

from __future__ import annotations

import os

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

image = (
    flyte.Image.from_debian_base(name="ddp_train_restart_1")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch", "numpy")
)

# --- Knobs ---------------------------------------------------------------------------------------
USE_GPU = True
GPU_DEVICE = "L4"  # one of flyte._resources.Accelerators device names; match the cluster's GPUs
REPLICAS = 2  # pods (== nodes)
NPROC_PER_NODE = 1  # processes (one per GPU) per pod  => world_size = REPLICAS * NPROC_PER_NODE

_BACKEND = "nccl" if USE_GPU else "gloo"

resources = (
    flyte.Resources(cpu=(2, 4), memory=("4Gi", "8Gi"), gpu=f"{GPU_DEVICE}:{NPROC_PER_NODE}")
    if USE_GPU
    else flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi"))
)

env = ClusteredTaskEnvironment(
    name="ddp_restart_env",
    image=image,
    resources=resources,
    replicas=REPLICAS,
    nproc_per_node=NPROC_PER_NODE,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=1),  # allow ONE JobSet restart
)


@env.task
async def train_ddp_with_restart(steps: int = 50, lr: float = 0.05) -> float:
    """Fail on the first JobSet attempt, then train + return loss on the restart."""
    restart_attempt = int(os.environ.get("JOBSET_RESTART_ATTEMPT", "0") or "0")
    rank = os.environ.get("RANK", "0")
    print(f"[rank {rank}] JOBSET_RESTART_ATTEMPT={restart_attempt}", flush=True)

    # Attempt 0 fails on every worker -> writes error.pb for the execution.
    if restart_attempt == 0:
        raise RuntimeError("Intentional failure on attempt 0 to force a JobSet restart")

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    ctx = flyte.ctx()

    # Bind this rank to its local GPU BEFORE init_process_group so NCCL binds the right device.
    if _BACKEND == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank or 0)
        device = torch.device(f"cuda:{ctx.local_rank or 0}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=_BACKEND)
    rank_i = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        f"[rank {rank_i}/{world_size}] device={device} restart attempt {restart_attempt} — training",
        flush=True,
    )

    torch.manual_seed(0)
    model = nn.Linear(4, 1).to(device)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    g = torch.Generator().manual_seed(rank_i)
    x = torch.randn(64, 4, generator=g).to(device)
    y = x.sum(dim=1, keepdim=True)

    last_loss = 0.0
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if rank_i == 0 and step % 10 == 0:
            print(f"[rank 0] step {step:3d}  loss {last_loss:.5f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank_i}] done — final loss {last_loss:.5f}", flush=True)
    return last_loss


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_ddp_with_restart, steps=50)
    print("Run URL:", run.url)
    run.wait()
    # Expected WITH the fix: SUCCEEDED. Without it: FAILED (stale error.pb).
    print("Final phase:", run.phase)
