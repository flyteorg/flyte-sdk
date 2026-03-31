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

Flow (aligned with ``huggingface_trainer_checkpoint.py``):

1. ``checkpoint_path = await checkpoint.load.aio()`` — restore prior state into the checkpoint workspace when a
   previous attempt exists.
2. Resolve ``ckpt_dir`` from ``checkpoint_path`` (see task body); if ``last.ckpt`` exists, pass it to ``Trainer.fit``
   as ``ckpt_path``.
3. :class:`FlyteLightningCheckpointCallback` — subclasses :class:`~lightning.pytorch.callbacks.ModelCheckpoint`
   and, after each epoch checkpoint write, uploads ``dirpath`` with blocking :meth:`~flyte.AsyncCheckpoint.save`.
4. Optional final ``await checkpoint.save.aio(ckpt_dir)`` if training can stop before the next epoch boundary.

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
from typing_extensions import override

import flyte

env = flyte.TaskEnvironment(name="checkpoint_lightning")

FEATURES = 16


class FlyteLightningCheckpointCallback(ModelCheckpoint):
    """
    :class:`~lightning.pytorch.callbacks.ModelCheckpoint` that mirrors ``dirpath`` to Flyte after each
    on-disk checkpoint cycle in :meth:`~ModelCheckpoint.on_train_epoch_end`.

    Calls ``super()`` first so ``last.ckpt`` exists locally, then uses blocking
    :meth:`~flyte.AsyncCheckpoint.save` (safe inside ``Trainer.fit``).
    """

    def __init__(self, flyte_checkpoint: flyte.AsyncCheckpoint, *, dirpath: str | pathlib.Path, **kwargs) -> None:
        super().__init__(dirpath=str(dirpath), **kwargs)
        self._flyte_checkpoint = flyte_checkpoint

    @override
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self.dirpath:
            self._flyte_checkpoint.save(pathlib.Path(self.dirpath))


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
    ctx = flyte.ctx()
    assert ctx is not None
    checkpoint = ctx.checkpoint
    assert checkpoint is not None

    checkpoint_path: pathlib.Path | None = await checkpoint.load.aio()
    if checkpoint_path is None:
        ckpt_dir = pathlib.Path("pl_checkpoints")
    else:
        ckpt_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent / "pl_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume = find_last_checkpoint(ckpt_dir)

    model = TinyModule()
    mc = FlyteLightningCheckpointCallback(
        checkpoint,
        dirpath=ckpt_dir,
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

    await checkpoint.save.aio(ckpt_dir)

    with torch.no_grad():
        x = torch.ones(1, FEATURES)
        return float(model(x).squeeze().item())


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local").run(train_lightning, max_epochs=2)
    print(run.outputs())
