"""
Kill a single rank mid-training and watch the whole-set restart (CPU / gloo).

The "same stuff" as ``ddp_train.py`` — real PyTorch DDP — but with a deliberate failure injected
into one rank so you can SEE the cluster's failure handling. The core fact of distributed training:
when one rank dies, every other rank blocks forever in the next collective, so the only safe fix is
to kill them all and restart the whole set. The clustered runtime makes that happen:

    one rank fails  ->  torchrun on that pod tears down its siblings, pod exits != 0
      ->  Job fails immediately (backoffLimit=0)
      ->  JobSet restarts ALL pods with a fresh rendezvous id (RDZV_ID rotates)
      ->  ctx.restart_attempt increments; training resumes

Pick the rank, the step, and the FAILURE MODE to compare signatures:
  * "exit"      — os._exit(1): a hard, uncatchable death (closest to a real crash / OOM-killer).
  * "exception" — raise RuntimeError: the a0 runtime catches it, writes a typed error, exits != 0.

Unlike ``failure_cascade.py`` (which adds checkpoint-resume and exhausts the restart budget to force
a RetryableFailure), this one is the minimal "watch the kill propagate and the set recover" demo.

    uv run python examples/clustered/cpu_kill_rank.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# --- Knobs ---------------------------------------------------------------------------------------
KILL_RANK = 2  # which global rank dies (0..world_size-1). With the env below, world_size = 4.
KILL_AT_STEP = 20  # ~mid-run, given the 1s/step pacing below
KILL_MODE = "exit"  # "exit" (hard os._exit) or "exception" (clean raise)

image = (
    flyte.Image.from_debian_base(name="cpu_kill_rank1")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch", "numpy")
)

env = ClusteredTaskEnvironment(
    name="cpu_kill_rank_env",
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    replicas=2,
    # world_size = replicas * nproc_per_node = 4 (ranks 0-3): pod 0 holds ranks 0-1, pod 1 holds
    # ranks 2-3. So KILL_RANK=2 dies on pod 1 (a cross-pod kill); use KILL_RANK<2 to kill on pod 0.
    nproc_per_node=2,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),  # no in-pod retries; restart the set
    # Allow a couple of whole-set restarts so you can watch it recover rather than fail outright.
    failure_policy=ClusterFailurePolicy(max_restarts=2),
)


@env.task
async def train_and_kill(steps: int = 40, lr: float = 0.05) -> float:
    """Run DDP; on the FIRST attempt, KILL_RANK dies at KILL_AT_STEP. Later attempts run clean."""
    import asyncio
    import os

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    ctx = flyte.ctx()
    assert ctx is not None  # always set inside a task
    attempt = ctx.restart_attempt or 0
    rank = ctx.rank or 0

    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    print(f"[rank {rank}/{world_size}] restart_attempt={attempt} node_rank={ctx.node_rank}", flush=True)

    torch.manual_seed(0)
    ddp = DDP(nn.Linear(4, 1))
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x = torch.randn(64, 4, generator=torch.Generator().manual_seed(rank))
    y = x.sum(dim=1, keepdim=True)

    last_loss = 0.0
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        await asyncio.sleep(1)  # pace the run so the kill lands mid-training and is observable

        if rank == 0 and step % 10 == 0:
            print(f"[rank 0] step {step:3d}  loss {last_loss:.5f}  (attempt {attempt})", flush=True)

        # Inject the failure on the first attempt only, so the restart succeeds and you see recovery.
        if attempt == 0 and rank == KILL_RANK and step == KILL_AT_STEP:
            print(f"[rank {rank}] KILLING SELF at step {step} via mode={KILL_MODE!r}", flush=True)
            if KILL_MODE == "exception":
                raise RuntimeError(f"injected failure on rank {rank} at step {step}")
            os._exit(1)  # default: hard exit, uncatchable

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] completed {steps} steps on attempt {attempt} — final loss {last_loss:.5f}", flush=True)
    return last_loss


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_and_kill, steps=40)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
