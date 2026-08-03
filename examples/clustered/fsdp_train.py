"""
Fully Sharded Data Parallel (FSDP) training on a ClusteredTaskEnvironment.

FSDP shards a model's parameters across ranks, so each GPU holds only a slice — the way you train
models too big to fit one device. The cluster shape is identical to DDP (one JobSet, N rank-uniform
pods); only the in-task wrapping differs (``FullyShardedDataParallel`` instead of ``DistributedDataParallel``),
which is exactly the design's "one backend serves all" point.

This example also demonstrates the **checkpoint-to-S3** pattern that FSDP requires (JobSets.md §7):
pod-local disk is wiped on restart, so the model state must be persisted to durable storage. Because
each rank only holds a shard, you either (a) gather a FULL state dict onto rank-0 and save it (fine
for small models — shown here), or (b) have every rank write its own shard (scales to huge models —
sketched in comments).

Run on a GPU cluster (set USE_GPU=True) or smoke it on CPU/gloo:
    uv run python examples/clustered/fsdp_train.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# --- Knobs ---------------------------------------------------------------------------------------
USE_GPU = True  # FSDP is meant for GPU; CPU/gloo here is only a wiring smoke
REPLICAS = 2
NPROC_PER_NODE = 1

_BACKEND = "nccl" if USE_GPU else "gloo"

image = (
    flyte.Image.from_debian_base(name="fsdp_train")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch")
)

resources = (
    flyte.Resources(cpu=(2, 4), memory=("4Gi", "8Gi"), gpu="L4:1")
    if USE_GPU
    else flyte.Resources(cpu=(1, 2), memory=("2Gi", "4Gi"))
)

env = ClusteredTaskEnvironment(
    name="fsdp_env",
    image=image,
    resources=resources,
    replicas=REPLICAS,
    nproc_per_node=NPROC_PER_NODE,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=1),
)


@env.task
async def train_fsdp(steps: int = 30, lr: float = 0.01) -> float:
    """Train a small transformer under FSDP and checkpoint the gathered state to S3."""
    import io

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    ctx = flyte.ctx()

    if _BACKEND == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank or 0)
        device = torch.device(f"cuda:{ctx.local_rank or 0}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=_BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        f"[rank {rank}/{world_size}] device={device} ctx.rank={ctx.rank} "
        f"ctx.world_size={ctx.world_size} ctx.nnodes={ctx.nnodes}",
        flush=True,
    )

    # A small transformer encoder — each FSDP unit's parameters are sharded across ranks.
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True),
        nn.Linear(64, 1),
    ).to(device)
    fsdp_model = FSDP(model, device_id=device.index if device.type == "cuda" else None)
    opt = torch.optim.AdamW(fsdp_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Synthetic per-rank data: learn to sum a sequence's features.
    g = torch.Generator().manual_seed(rank)
    x = torch.randn(16, 8, 32, generator=g).to(device)  # (batch, seq, features)
    y = x.mean(dim=(1, 2), keepdim=True).squeeze(-1)[:, :1]

    last_loss = 0.0
    for step in range(steps):
        opt.zero_grad()
        out = fsdp_model(x).mean(dim=1)  # pool over sequence -> (batch, 1)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if rank == 0 and step % 10 == 0:
            print(f"[rank 0] step {step:3d}  loss {last_loss:.5f}", flush=True)

    # --- Checkpoint-to-S3: gather a FULL state dict onto rank-0 and persist it. -------------------
    # For models too large to gather, switch to StateDictType.SHARDED_STATE_DICT and have EVERY rank
    # save its shard under a rank-suffixed key (e.g. f"shard-{rank}") instead of the rank-0-only path.
    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_cfg):
        full_state = fsdp_model.state_dict()  # populated only on rank-0

    cp = ctx.checkpoint
    if rank == 0 and cp is not None:
        buf = io.BytesIO()
        torch.save(full_state, buf)
        await cp.save(buf.getvalue())
        print(f"[rank 0] saved FSDP checkpoint ({buf.tell()} bytes) to {cp.path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] done — final loss {last_loss:.5f}", flush=True)
    return last_loss


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_fsdp, steps=30)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
