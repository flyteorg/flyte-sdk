"""
DDP training that FAILS on the first torchrun attempt and SUCCEEDS on the in-pod restart.

Regression test for the stale ``error.pb`` cleanup in the clustered runtime
(``flyte._internal.runtime.io.clear_stale_clustered_error``): whenever a clustered rank-0
worker starts, it removes any ``error.pb`` an earlier restart left under the attempt's output
prefix, before the task body runs. Without that cleanup a successful restart uploads
``outputs.pb`` next to the leftover ``error.pb`` and the executor still reports the run FAILED.

The stale file is produced here without any backend help:
    - ``ClusterFailurePolicy(max_restarts=0)`` makes every attempt look terminal to the SDK's
      terminal-attempt gate (``JOBSET_RESTART_ATTEMPT 0 >= JOBSET_MAX_RESTARTS 0``), so the first
      failure writes ``error.pb`` right away and the worker exits 1.
    - ``PET_MAX_RESTARTS=1`` lets torchrun restart the worker group once inside the same pod.
      (``TorchRun(max_restarts=...)`` is not wired through to torchrun yet, so the env var is set
      directly; torchrun reads ``PET_<FLAG>`` for every CLI flag.)
    - Attempt 0 (``TORCHELASTIC_RESTART_COUNT`` == "0") raises on every rank.
    - Attempt 1 (``TORCHELASTIC_RESTART_COUNT`` == "1") trains and uploads outputs. Rank-0 logs
      "Removed stale ... error.pb" at startup.

Expected: run phase SUCCEEDED. Without the cleanup: FAILED with the attempt-0 error.

A JobSet-level restart (``ClusterFailurePolicy(max_restarts >= 1)``) goes through the same cleanup,
but without free host-maintenance restarts the gate never writes a premature ``error.pb``, so that
variant passes with or without the fix and is not a useful regression test.

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
USE_GPU = False
REPLICAS = 1  # one pod: torchrun's in-pod restart then needs no cross-node re-rendezvous
NPROC_PER_NODE = 2  # processes per pod  => world_size = REPLICAS * NPROC_PER_NODE

_BACKEND = "nccl" if USE_GPU else "gloo"

resources = (
    flyte.Resources(cpu=(2, 4), memory=("4Gi", "8Gi"), gpu="L4:2")  # one GPU per process (NPROC_PER_NODE)
    if USE_GPU
    else flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi"))
)

env = ClusteredTaskEnvironment(
    name="ddp_restart_env",
    image=image,
    resources=resources,
    replicas=REPLICAS,
    nproc_per_node=NPROC_PER_NODE,
    # max_restarts here is not wired through to torchrun yet; PET_MAX_RESTARTS below is what works today.
    runtime=TorchRun(rdzv_backend="static", max_restarts=1),
    failure_policy=ClusterFailurePolicy(max_restarts=0),  # every attempt looks terminal to the SDK gate
    env_vars={"PET_MAX_RESTARTS": "1"},  # ONE in-pod torchrun restart (see module docstring)
)


@env.task
async def train_ddp_with_restart(steps: int = 50, lr: float = 0.05) -> float:
    """Fail on the first torchrun attempt, then train + return loss on the in-pod restart."""
    restart_attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0") or "0")
    rank = os.environ.get("RANK", "0")
    print(
        f"[rank {rank}] TORCHELASTIC_RESTART_COUNT={restart_attempt} "
        f"JOBSET_RESTART_ATTEMPT={flyte.ctx().restart_attempt}",
        flush=True,
    )

    # Attempt 0 fails on every rank -> rank-0 writes error.pb (the SDK gate sees 0 >= 0) and every
    # worker exits 1, so torchrun restarts the worker group in-pod.
    if restart_attempt == 0:
        raise RuntimeError("Intentional failure on torchrun attempt 0 to leave a stale error.pb behind")

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
    # Expected WITH the cleanup: SUCCEEDED. Without it: FAILED (stale error.pb from attempt 0).
    print("Final phase:", run.phase)
