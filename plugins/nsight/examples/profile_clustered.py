"""Profile a distributed (multi-GPU) training task with Nsight Systems.

`ClusteredTaskEnvironment` runs the task under torchrun: `replicas` pods x `nproc_per_node` procs,
one process per GPU. Adding `@nsys_profile` profiles the global primary worker only (RANK 0) — the
runtime wraps just that one process under nsys and leaves every other rank untouched, so the
profiler never sits in the rendezvous / NCCL path. Rank 0 is representative of a data-parallel job,
and its trace still shows the NCCL collectives (the all-reduce in backward), which is usually what
you profile a distributed run to see.

The report and the .nsys-rep come back exactly as in the single-GPU example, but reflect rank 0.
The other ranks run normally and are not profiled.

This runs remotely against local, unreleased code, so the image bakes the flyte and
flyteplugins-nsight wheels from ./dist. Build the wheels once, then run:

    make dist && FLYTE_PLUGIN_DIST=plugins/nsight make dist-plugins
    flyte run plugins/nsight/examples/profile_clustered.py profile_clustered
"""

from __future__ import annotations

import flyte
from flyte.clustered import ClusteredTaskEnvironment

from flyteplugins.nsight import nsys_profile, nvtx

# Same base-image setup as profile_training.py (NGC PyTorch for nsys + torch, local wheels baked in,
# system-site-packages flipped so the venv sees NGC's torch). NGC's torch also provides `torchrun`,
# which the clustered launcher execs. See profile_training.py for the full rationale.
image = (
    flyte.Image.from_base("nvcr.io/nvidia/pytorch:24.08-py3")
    .clone(extendable=True, name="nsight-clustered", python_version=(3, 10))
    .with_pip_packages("uv", "kubernetes")
    .with_local_v2()
    .with_local_v2_plugins(["flyteplugins-nsight"])
    .with_commands(
        ["sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' /opt/venv/pyvenv.cfg"]
    )
)

# replicas=2, nproc_per_node=2 -> two nodes, four GPUs (four DDP ranks). resources.gpu must be >=
# nproc_per_node, so request T4:4.
env = ClusteredTaskEnvironment(
    name="nsight-clustered",
    image=image,
    resources=flyte.Resources(cpu="8", memory="32Gi", gpu="T4:4"),
    replicas=2,
    nproc_per_node=2,
)


@nsys_profile(trace=["cuda", "nvtx"])
@env.task
async def profile_clustered(steps: int = 30, width: int = 4096, batch: int = 512) -> str:
    """DDP training across the pod's GPUs; nsys profiles rank 0 only.

    Every rank runs this body (torchrun sets RANK/LOCAL_RANK/WORLD_SIZE); rank 0's return value is
    the task output. Only rank 0 is under nsys, so the NVTX ranges and the report come from it.
    """
    import os

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    # env:// rendezvous — MASTER_ADDR / RANK / WORLD_SIZE are set by the clustered launcher + torchrun.
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    dev = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(dev)

    model = DDP(
        nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)).to(dev),
        device_ids=[local_rank],
    )
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Warm up outside the NVTX loop so the profiled steps reflect steady state, not first-iteration
    # lazy loading + NCCL/DDP bucket setup. (See profile_region.py for the capture="manual"
    # alternative that excludes warmup from the trace entirely.)
    for _ in range(3):
        loss_fn(model(torch.randn(batch, width, device=dev)), torch.randn(batch, width, device=dev)).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    loss = torch.tensor(0.0)
    for step in range(steps):
        with nvtx.range(f"step_{step}"):
            x = torch.randn(batch, width, device=dev)
            y = torch.randn(batch, width, device=dev)
            with nvtx.range("forward"):
                out = model(x)
                loss = loss_fn(out, y)
            with nvtx.range("backward"):
                # DDP all-reduces gradients here — the NCCL collectives show up on rank 0's timeline.
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
    torch.cuda.synchronize()

    dist.destroy_process_group()
    return f"rank {rank}: training done, final loss: {loss.item():.4f}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(profile_clustered)
    print(run.url)
