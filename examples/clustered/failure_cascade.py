"""
Failure cascade + whole-set restart + checkpoint resume.

The core fact of distributed training: if ONE rank dies, every other rank hangs forever in the next
collective. So the clustered runtime sets ``backoffLimit=0`` and the JobSet's failure policy
restarts the WHOLE set with a fresh rendezvous id, and the user code resumes from its last
checkpoint. This example deliberately breaks the system to prove that loop works:

  * ``ClusterFailurePolicy(max_restarts=2)`` — the JobSet may restart twice before giving up.
  * On the FIRST attempt only, rank-1 hard-exits (``os._exit(1)``) partway through.
  * Every attempt checkpoints its step counter to S3 and resumes from it, so progress is not lost.
  * ``ctx.restart_attempt`` is printed each attempt — you should see it climb 0 -> 1 (-> 2).

Expected outcomes:
  * With the crash firing once, the set restarts, resumes, and SUCCEEDS on attempt 1.
  * If you set ``ALWAYS_CRASH=True``, every attempt crashes, restarts are exhausted, and Flyte
    surfaces a RetryableFailure (DownstreamSystemError) after attempt 2.

CPU/gloo is sufficient — this is about the control plane, not the math.
    uv run python examples/clustered/failure_cascade.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# --- Knobs ---------------------------------------------------------------------------------------
ALWAYS_CRASH = False  # True => exhaust restarts and force a RetryableFailure
CRASH_RANK = 2
CRASH_AT_STEP = 15  # ~mid-run; with the 1s/step sleep below this is well into training

image = (
    flyte.Image.from_debian_base(name="failure_cascade")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch")
)

env = ClusteredTaskEnvironment(
    name="failure_cascade_env",
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    replicas=2,
    nproc_per_node=2,  # world_size = 4 (ranks 0-3); CRASH_RANK=2 lives on pod 1
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),  # no in-pod retries; restart the set
    failure_policy=ClusterFailurePolicy(max_restarts=2),
)


@env.task
async def train_with_crash(steps: int = 40) -> int:
    """Train; crash rank-1 once (or always); checkpoint + resume across JobSet restarts."""
    import asyncio
    import os

    import torch
    import torch.distributed as dist
    import torch.nn as nn

    ctx = flyte.ctx()
    attempt = ctx.restart_attempt or 0
    rank = ctx.rank or 0

    dist.init_process_group(backend="gloo")
    print(f"[rank {rank}] restart_attempt={attempt} world_size={ctx.world_size}", flush=True)

    # Resume the step counter from the checkpoint written by a previous attempt (best-effort).
    cp = ctx.checkpoint
    start_step = 0
    if cp is not None:
        prev = await cp.load()
        if prev is not None:
            try:
                payload = prev / "payload" if prev.is_dir() else prev
                start_step = int(payload.read_text().strip())
                print(f"[rank {rank}] resumed from checkpoint at step {start_step}", flush=True)
            except (ValueError, OSError):
                start_step = 0

    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x = torch.randn(32, 4)
    y = x.sum(dim=1, keepdim=True)

    for step in range(start_step, steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        await asyncio.sleep(1)  # stretch the run so the crash lands mid-training and restarts are visible

        # Checkpoint progress (rank-0 owns the upload) so the next attempt can resume.
        if rank == 0 and cp is not None and step % 5 == 0:
            await cp.save(str(step).encode())

        # Inject the failure. On the first attempt (or always), rank-1 dies hard -> all ranks hang
        # -> Job fails (backoffLimit=0) -> JobSet restarts the whole set with a fresh rendezvous id.
        should_crash = ALWAYS_CRASH or attempt == 0
        if should_crash and rank == CRASH_RANK and step == CRASH_AT_STEP:
            print(f"[rank {rank}] INJECTED CRASH at step {step} (attempt {attempt})", flush=True)
            os._exit(1)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] completed all {steps} steps on attempt {attempt}", flush=True)
    return steps


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_with_crash, steps=40)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
