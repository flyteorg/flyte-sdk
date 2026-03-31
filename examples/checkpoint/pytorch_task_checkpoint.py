# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "flyte",
#   "torch>=2.0",
# ]
# ///

"""
PyTorch checkpoint via TaskContext
==================================

Saves ``training.pt`` (state dict + optimizer + epoch) using :class:`~flyte.AsyncCheckpoint`,
following the same load / persist pattern as ``huggingface_trainer_checkpoint.py`` (resolve a local
root from ``await checkpoint.load.aio()``, sync :meth:`~flyte.AsyncCheckpoint.save` at each epoch end,
then a final ``await checkpoint.save.aio(...)``).

**Note:** Use an editable install of this SDK (or a release that includes
``TaskContext.checkpoint``); see ``generic_data_checkpoint.py`` for details.
"""

from __future__ import annotations

import pathlib

import torch
import torch.nn as nn

import flyte

env = flyte.TaskEnvironment(name="checkpoint_pytorch")

CHECKPOINT_FILE = "training.pt"


def weights_path(root: pathlib.Path) -> pathlib.Path:
    direct = root / CHECKPOINT_FILE
    if direct.exists():
        return direct
    found = list(root.rglob(CHECKPOINT_FILE))
    return found[0] if found else direct


@env.task()
async def train_linear(epochs: int = 3) -> float:
    ctx = flyte.ctx()
    assert ctx is not None
    ck = ctx.checkpoint
    assert ck is not None

    checkpoint_path: pathlib.Path | None = await ck.load.aio()
    if checkpoint_path is None:
        train_root = pathlib.Path("pytorch_linear")
    else:
        train_root = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent / "pytorch_linear"
    train_root.mkdir(parents=True, exist_ok=True)
    wpath = weights_path(train_root)

    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    start = 0

    if wpath.exists():
        blob = torch.load(wpath, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start = int(blob["epoch"]) + 1

    for epoch in range(start, epochs):
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        wpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
            wpath,
        )
        ck.save(wpath)

    await ck.save.aio(wpath)
    with torch.no_grad():
        return float(model(torch.ones(1, 4)).squeeze().item())


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local").run(train_linear, epochs=2)
    print(run.outputs())
