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

Saves ``model.pt`` (state dict + optimizer + epoch) under the Flyte checkpoint
prefix using :class:`~flyte.AsyncCheckpoint`. Use :meth:`~flyte.AsyncCheckpoint.load`
and :meth:`~flyte.AsyncCheckpoint.save` (sync), or ``await ...load.aio()`` /
``await ...save.aio(...)`` from async code.

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
    tctx = flyte.ctx()
    assert tctx is not None
    ck = tctx.checkpoint
    assert ck is not None

    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    start = 0

    await ck.load.aio()
    wpath = weights_path(ck.path)
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
        {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epochs - 1},
        wpath,
    )
    await ck.save.aio(local_path=wpath)
    with torch.no_grad():
        return float(model(torch.ones(1, 4)).squeeze().item())


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local").run(train_linear, epochs=2)
    print(run.outputs())
