"""
Composability: pretrain -> evaluate, two sequential JobSets (§9 exit-criterion #6).

A realistic pipeline chains a distributed pretrain step into a distributed evaluate step. Each
clustered task emits its OWN JobSet, and they run sequentially when one awaits the other.

THE IMPORTANT CONSTRAINT: a clustered task runs with NO controller (the torchrun worker calls the
runtime with ``controller_enabled=False``), so it **cannot enqueue subtasks**. The orchestration
must therefore live in a *regular* ``flyte.TaskEnvironment`` task — the ``driver`` below — which
awaits the two clustered tasks. Putting the ``await``s inside a clustered task would fail at
runtime; that boundary is documented and asserted in the unit tests.

Topology:
    driver (plain task, has a controller)
      ├─ await pretrain()    -> JobSet #1 (N pods)
      └─ await evaluate(...)  -> JobSet #2 (N pods)

    uv run python examples/clustered/compose_pretrain_eval.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

image = (
    flyte.Image.from_debian_base(name="compose_clustered")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch")
)

# --- Knobs ---------------------------------------------------------------------------------------
USE_GPU = True
GPU_DEVICE = "L4"  # one of flyte._resources.Accelerators device names; match the cluster's GPUs
REPLICAS = 2  # pods (== nodes)
NPROC_PER_NODE = 1  # processes (one per GPU) per pod  => world_size = REPLICAS * NPROC_PER_NODE

_BACKEND = "nccl" if USE_GPU else "gloo"

_clustered_resources = (
    flyte.Resources(cpu=(2, 4), memory=("4Gi", "8Gi"), gpu=f"{GPU_DEVICE}:{NPROC_PER_NODE}")
    if USE_GPU
    else flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi"))
)

# Both clustered steps share ONE ClusteredTaskEnvironment (UC7): each @task on it emits a JobSet.
clustered_env = ClusteredTaskEnvironment(
    name="compose_clustered_env",
    image=image,
    resources=_clustered_resources,
    replicas=REPLICAS,
    nproc_per_node=NPROC_PER_NODE,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=1),
)

# The driver is a PLAIN environment — it is the only one allowed to launch subtasks. It must declare
# depends_on=[clustered_env] so the clustered env's image is registered in the image cache the driver
# sees at runtime; otherwise awaiting pretrain()/evaluate() fails with "Environment ... not found in
# image cache".
driver_env = flyte.TaskEnvironment(
    name="compose_driver_env",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[clustered_env],
)


@clustered_env.task
async def pretrain(steps: int = 20) -> float:
    """Distributed 'pretrain' step — emits its own JobSet. Returns a final loss."""
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
    print(f"[pretrain rank {ctx.rank}/{ctx.world_size}] device={device} starting", flush=True)

    torch.manual_seed(0)
    ddp = DDP(nn.Linear(4, 1).to(device), device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    x = torch.randn(64, 4, generator=torch.Generator().manual_seed(ctx.rank or 0)).to(device)
    y = x.sum(dim=1, keepdim=True)

    last = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()
        opt.step()
        last = float(loss.detach())

    dist.barrier()
    dist.destroy_process_group()
    print(f"[pretrain rank {ctx.rank}] done — loss {last:.5f}", flush=True)
    return last


@clustered_env.task
async def evaluate(trained_loss: float) -> float:
    """Distributed 'evaluate' step — emits a SECOND JobSet. Returns a synthetic eval metric."""
    import torch
    import torch.distributed as dist

    ctx = flyte.ctx()

    # Bind this rank to its local GPU BEFORE init_process_group so NCCL binds the right device.
    if _BACKEND == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank or 0)
    dist.init_process_group(backend=_BACKEND)
    print(
        f"[evaluate rank {ctx.rank}/{ctx.world_size}] evaluating with train_loss={trained_loss:.5f}",
        flush=True,
    )

    # Stand-in metric; in a real pipeline you'd load the pretrain checkpoint and run a held-out set.
    metric = 1.0 / (1.0 + trained_loss)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[evaluate rank {ctx.rank}] done — metric {metric:.5f}", flush=True)
    return metric


@driver_env.task
async def pretrain_then_eval(steps: int = 20) -> float:
    """Plain orchestrator: await pretrain (JobSet #1), then evaluate (JobSet #2), sequentially."""
    loss = await pretrain(steps=steps)
    metric = await evaluate(trained_loss=loss)
    print(f"[driver] pretrain_loss={loss:.5f} eval_metric={metric:.5f}", flush=True)
    return metric


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pretrain_then_eval, steps=20)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
