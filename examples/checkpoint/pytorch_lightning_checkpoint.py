# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "flyte",
#   "torch>=2.0",
#   "lightning>=2.0",
# ]
# ///

"""
PyTorch Lightning + Flyte ``AsyncCheckpoint``
=============================================

Trains a tiny regression model with Lightning's :class:`~lightning.pytorch.callbacks.ModelCheckpoint`,
persisting the checkpoint directory through the task's Flyte checkpoint prefix so retries and
new attempts can resume from ``last.ckpt``.

Flow:

1. ``await checkpoint.load.aio()`` — download any prior checkpoint tree into :attr:`~flyte.AsyncCheckpoint.path`.
2. If ``last.ckpt`` exists under that tree, pass it to :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`
   as ``ckpt_path``.
3. After training, ``await checkpoint.save.aio(...)`` uploads the directory that holds ``last.ckpt``.

**Note:** Install this repository editable (``pip install -e .``) so ``TaskContext.checkpoint`` exists;
see ``generic_data_checkpoint.py``.
"""

from __future__ import annotations

import pathlib

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

import flyte

env = flyte.TaskEnvironment(name="checkpoint_lightning")

FEATURES = 16


def find_last_checkpoint(root: pathlib.Path) -> str | None:
    matches = list(root.rglob("last.ckpt"))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0])


class TinyModule(L.LightningModule):
    def __init__(self, lr: float = 0.02):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Linear(FEATURES, 32), nn.ReLU(), nn.Linear(32, 1))
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def _make_loaders(batch: int = 32, batches: int = 8) -> DataLoader:
    g = torch.Generator().manual_seed(42)
    x = torch.randn(batches * batch, FEATURES, generator=g)
    y = torch.randn(batches * batch, 1, generator=g)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch, shuffle=True)


@env.task()
async def train_lightning(max_epochs: int = 3) -> float:
    tctx = flyte.ctx()
    assert tctx is not None
    checkpoint = tctx.checkpoint
    assert checkpoint is not None

    await checkpoint.load.aio()
    root = checkpoint.path
    ckpt_dir = root / "pl_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume = find_last_checkpoint(root)

    model = TinyModule()
    mc = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="last",
        save_last=True,
        save_top_k=0,
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=True,
        callbacks=[mc],
        enable_progress_bar=True,
        logger=False,
        accelerator="cpu",
        devices=1,
    )
    loader = _make_loaders()
    trainer.fit(model, loader, ckpt_path=resume)

    await checkpoint.save.aio(local_path=ckpt_dir)

    with torch.no_grad():
        x = torch.ones(1, FEATURES)
        return float(model(x).squeeze().item())


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local").run(train_lightning, max_epochs=2)
    print(run.outputs())
