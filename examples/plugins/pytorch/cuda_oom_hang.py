"""
Reproducer: CUDA OOM in one DDP worker causes the entire job to hang.

Run on a machine with 2 GPUs (e.g. 2x T4):

    python examples/plugins/pytorch/cuda_oom_hang.py

What happens:
    1. elastic_launch spawns 2 workers (nproc_per_node=2), each on its own GPU
    2. Both workers init a DDP model and start training
    3. On epoch 2, rank 1 hits CUDA OOM
    4. Rank 1 catches the OOM and skips the batch — a common but WRONG recovery pattern
    5. This causes rank 1 to skip a DDP all-reduce (inside loss.backward())
    6. The ranks are now permanently out of sync on NCCL collectives
    7. When rank 1 finishes training (one fewer backward call), rank 0 is still
       blocked in its final loss.backward() waiting for rank 1 in all-reduce
    8. Rank 0 hangs for NCCL_TIMEOUT (default 1800s = 30 minutes) before timing out
    9. The elastic agent restarts the group, and rank 1 OOMs again
   10. Total hang: up to (max_restarts+1) * NCCL_TIMEOUT

With the Flyte PyTorch plugin's NCCL timeout knobs:
    - nccl_enable_monitoring=True: activates NCCL's monitoring thread (enabled by default)
    - nccl_collective_timeout_sec=60: collective timeout reduced from 600s to 60s
    - nccl_heartbeat_timeout_sec=60: heartbeat watchdog aborts ~60s after collective fires
    - nccl_async_error_handling=True: stuck collectives abort asynchronously
    - max_restarts=0: fail immediately on first OOM, no restart cycles
    - Total failure time: ~2 min instead of 90 min
"""

import typing

import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from flyteplugins.pytorch.task import Elastic
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import flyte
from flyte._image import DIST_FOLDER, PythonWheels

image = (
    flyte.Image.from_debian_base(name="torch")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyteplugins-pytorch", pre=True))
    .with_pip_packages("numpy")
)

torch_env = flyte.TaskEnvironment(
    name="torch_oom_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi"), gpu=2),
    plugin_config=Elastic(
        nproc_per_node=2,
        nnodes=1,
        max_restarts=0,
        rdzv_configs={"timeout": 900, "join_timeout": 900},
        nccl_heartbeat_timeout_sec=60,
        nccl_async_error_handling=True,
        nccl_collective_timeout_sec=60,
    ),
    image=image,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train_loop(epochs: int = 10) -> float:
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=torch.distributed.get_world_size(), rank=rank)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    final_loss = 0.0

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            bx, by = batch_x.to(device), batch_y.to(device)

            outputs = model(bx)
            loss = criterion(outputs, by)
            optimizer.zero_grad()

            # ---- OOM trigger ----
            # On epoch 2, rank 1 hits OOM. The common (but wrong) recovery pattern
            # is to catch the error and skip the batch. This causes rank 1 to skip
            # the DDP all-reduce inside loss.backward(), permanently desyncing the
            # ranks on NCCL collectives.
            if epoch == 2 and rank == 1:
                try:
                    print(f"[rank {rank}] About to trigger CUDA OOM...")
                    # ~20 GB — exceeds a T4's 16 GB VRAM
                    _ = torch.empty(5_000_000_000, device="cuda", dtype=torch.float32)
                except torch.cuda.OutOfMemoryError:
                    print(f"[rank {rank}] Caught OOM, skipping batch...")
                    torch.cuda.empty_cache()
                    continue  # WRONG: skips backward → misses DDP all-reduce → ranks desync

            loss.backward()  # DDP all-reduce happens here; rank 0 eventually has no match → HANG
            optimizer.step()

            final_loss = loss.item()

        if rank == 0:
            print(f"[rank {rank}] epoch {epoch} loss={final_loss:.4f}")

    # Ideally you should do this
    torch.distributed.destroy_process_group()
    return final_loss


@torch_env.task
def torch_oom_train(epochs: int) -> typing.Optional[float]:
    return train_loop(epochs=epochs)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(torch_oom_train, epochs=10)
    print("run name:", run.name)
    print("run url:", run.url)
