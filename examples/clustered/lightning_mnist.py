"""
PyTorch Lightning MNIST training on a ClusteredTaskEnvironment.

This is the clustered-backend port of the classic Kubeflow `Elastic` Lightning example (the v1
`flytekitplugins-kfpytorch` MNIST autoencoder). It shows that Lightning — which has its own
multi-process launching machinery — also rides the torchrun contract: we let the ``clustered`` ->
``torchrun`` -> ``clustered`` chain start one process per rank, and configure Lightning's
``Trainer`` to use the EXISTING process group rather than spawning its own.

The key is ``strategy="ddp"`` with ``devices=<local ranks>`` / ``num_nodes=<nodes>``: Lightning
attaches to the torchrun-provided env (RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT)
instead of re-launching. Do NOT use the spawn strategies here — they would fork a second launcher
inside each rank.

Run on a GPU cluster (USE_GPU=True) or smoke on CPU:
    uv run python examples/clustered/lightning_mnist.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# --- Knobs ---------------------------------------------------------------------------------------
USE_GPU = True
REPLICAS = 2  # nodes
NPROC_PER_NODE = 1  # processes (GPUs) per node

image = (
    flyte.Image.from_debian_base(name="lightning_mnist")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch", "torchvision", "lightning")
)

resources = (
    flyte.Resources(cpu=(2, 4), memory=("4Gi", "8Gi"), gpu="L4:1")
    if USE_GPU
    else flyte.Resources(cpu=(2, 4), memory=("2Gi", "4Gi"))
)

env = ClusteredTaskEnvironment(
    name="lightning_env",
    image=image,
    resources=resources,
    replicas=REPLICAS,
    nproc_per_node=NPROC_PER_NODE,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=1),
)


@env.task
async def train_lightning(max_steps: int = 50) -> float:
    """Train a tiny MNIST autoencoder with Lightning DDP over the JobSet's torchrun ranks."""
    import lightning as L
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    ctx = flyte.ctx()
    print(
        f"[rank {ctx.rank}/{ctx.world_size}] node_rank={ctx.node_rank} nnodes={ctx.nnodes} "
        f"local_rank={ctx.local_rank} master_addr={ctx.master_addr}",
        flush=True,
    )

    class AutoEncoder(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        def training_step(self, batch, _):
            x, _y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            loss = F.mse_loss(self.decoder(z), x)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    # Synthetic MNIST-shaped data so the example needs no dataset download; swap in torchvision MNIST
    # for the real thing. Lightning's DistributedSampler shards this across ranks automatically.
    torch.manual_seed(0)
    images = torch.rand(512, 1, 28, 28)
    labels = torch.zeros(512, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=True)

    # Attach to the torchrun-provided process group: ddp (not ddp_spawn), devices per node and
    # num_nodes from the clustered context. Lightning reads RANK/WORLD_SIZE/MASTER_ADDR from env.
    trainer = L.Trainer(
        accelerator="gpu" if USE_GPU else "cpu",
        devices=NPROC_PER_NODE,
        num_nodes=ctx.nnodes or REPLICAS,
        strategy="ddp",
        max_steps=max_steps,
        enable_checkpointing=False,
        logger=False,
    )
    model = AutoEncoder()
    trainer.fit(model, loader)

    final_loss = float(trainer.callback_metrics.get("train_loss", torch.tensor(0.0)))
    print(f"[rank {ctx.rank}] done — final loss {final_loss:.5f}", flush=True)
    return final_loss


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_lightning, max_steps=50)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
