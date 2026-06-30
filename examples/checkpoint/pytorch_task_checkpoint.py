# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "flyte",
#   "torch>=2.0",
# ]
# ///

"""
PyTorch checkpoints
===================

Saves `training.pt` (state dict + optimizer + epoch) using `flyte.Checkpoint`,
following the same load / persist pattern as `huggingface_trainer_checkpoint.py` (resolve a local
root from `await checkpoint.load()`, `await checkpoint.save(...)` each epoch end,
then a final `await checkpoint.save(...)`).
"""

from __future__ import annotations

import pathlib

import torch
import torch.nn as nn

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_pytorch",
    image=flyte.Image.from_debian_base().with_pip_packages("torch"),
)

RETRIES = 3


@env.task(retries=RETRIES)
async def train_linear(epochs: int = 3) -> float:
    assert epochs > RETRIES
    ctx = flyte.ctx()
    assert ctx is not None
    cp = ctx.checkpoint
    assert cp is not None

    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    prev_cp_path: pathlib.Path | None = await cp.load()
    if prev_cp_path:
        # load model, optimizer, and epoch from previous checkpoint
        blob = torch.load(prev_cp_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start = int(blob["epoch"]) + 1
    else:
        start = 0

    # create local checkpoint directory
    wpath = pathlib.Path("pytorch_linear") / "training.pt"
    wpath.parent.mkdir(parents=True, exist_ok=True)

    failure_interval = epochs // RETRIES
    print(f"Start at epoch {start} of {epochs}")
    for epoch in range(start, epochs):
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch > start and epoch % failure_interval == 0:
            raise RuntimeError(f"Failed at epoch {epoch}, failure_interval {failure_interval}.")

        wpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
            wpath,
        )
        # save checkpoint to object storage, which can be loaded by await cp.load() at the top of the script on the
        # next attempt
        await cp.save(wpath)

    with torch.no_grad():
        return float(model(torch.ones(1, 4)).squeeze().item())


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(train_linear, epochs=10)
    print(run.url)
