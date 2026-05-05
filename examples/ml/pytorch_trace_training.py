# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.9",
#    "pydantic>=2.0",
#    "torch==2.7.1",
# ]
# ///

"""
PyTorch Training with S3 Checkpointing
=======================================

Each epoch saves a checkpoint to a stable remote path. On any failure — whether
a retry within the same run or a completely new run — training resumes from the
last saved epoch automatically.

The checkpoint path is derived from a hash of the training config so each unique
set of hyperparams gets its own path automatically — no checkpoint_dir parameter
needed. The path is rooted at the storage bucket from raw_data_path (always the right
environment) with a hash of the training config as the key — stable across
retries and across runs with the same hyperparameters.
"""

import asyncio
import hashlib
import io
import os
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset

import flyte
import flyte.errors
from flyte.io import File

image = flyte.Image.from_uv_script(__file__, name="pytorch-checkpoint-training")

env = flyte.TaskEnvironment(
    name="pytorch_checkpoint_training",
    image=image,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)


def get_attempt_number() -> int:
    return int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    num_epochs: int
    batch_size: int
    learning_rate: float


class EpochMetrics(BaseModel):
    epoch: int
    loss: float


class CheckpointMeta(BaseModel):
    config: TrainingConfig
    epoch: int  # last completed epoch
    history: list[EpochMetrics]


def checkpoint_path(config: TrainingConfig) -> str:
    """Derive a stable checkpoint path from the storage bucket + config hash.

    The bucket is pulled from raw_data_path so it's always the right environment.
    The config hash makes each unique set of hyperparams its own checkpoint,
    and the path is stable across retries and across runs.
    """
    raw = flyte.ctx().raw_data_path.path  # type: ignore[union-attr]
    parsed = urlparse(raw)
    bucket_root = f"{parsed.scheme}://{parsed.netloc}"
    config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:12]
    return f"{bucket_root}/checkpoints/{config_hash}/checkpoint.pt"


# ---------------------------------------------------------------------------
# Minimal model + data (not meant to be realistic)
# ---------------------------------------------------------------------------

INPUT_DIM = 16


def build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(INPUT_DIM, 64), nn.ReLU(), nn.Linear(64, 1))


def make_loader(batch_size: int) -> DataLoader:
    g = torch.Generator().manual_seed(42)
    X = torch.randn(1000, INPUT_DIM, generator=g)
    y = X.sum(dim=1, keepdim=True)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


async def load_checkpoint(path: str, config: TrainingConfig) -> tuple[CheckpointMeta, dict, dict] | None:
    """Load checkpoint if it exists, otherwise return None to start fresh."""
    f = File.from_existing_remote(path)
    if not await f.exists():
        return None

    async with f.open("rb") as fh:
        # weights_only=False because the checkpoint contains Pydantic-serialized
        # metadata alongside tensors — we own everything that was saved here.
        raw = torch.load(io.BytesIO(bytes(await fh.read())), weights_only=False)

    meta = CheckpointMeta.model_validate(raw["meta"])
    return meta, raw["model"], raw["optimizer"]


async def save_checkpoint(path: str, meta: CheckpointMeta, model: nn.Module, optimizer: optim.Optimizer) -> None:
    buf = io.BytesIO()
    torch.save(
        {
            "meta": meta.model_dump(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        buf,
    )
    async with File.from_existing_remote(path).open("wb") as fh:
        await fh.write(buf.getvalue())


# ---------------------------------------------------------------------------
# Training task
# ---------------------------------------------------------------------------


@env.task(retries=3)
async def train(
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> list[EpochMetrics]:
    """Training loop that resumes from the last checkpoint.

    The checkpoint path is derived from the training config — no external path
    needed. Same hyperparams across runs will always find the same checkpoint.
    """
    config = TrainingConfig(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    ckpt_path = checkpoint_path(config)
    print(f"[attempt {get_attempt_number()}] checkpoint path: {ckpt_path}", flush=True)

    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    meta = CheckpointMeta(config=config, epoch=0, history=[])

    ckpt = await load_checkpoint(ckpt_path, config)
    if ckpt is not None:
        meta, model_state, opt_state = ckpt
        model.load_state_dict(model_state)
        optimizer.load_state_dict(opt_state)
        print(f"Resuming from epoch {meta.epoch}", flush=True)

    loader = make_loader(batch_size)
    criterion = nn.MSELoss()

    for epoch in range(meta.epoch + 1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X)

        await asyncio.sleep(2.0)  # simulate work

        metrics = EpochMetrics(epoch=epoch, loss=float(total_loss / 1000))
        meta = CheckpointMeta(config=config, epoch=epoch, history=[*meta.history, metrics])

        print(f"[epoch {epoch:02d}] loss={metrics.loss:.4f}", flush=True)
        await save_checkpoint(ckpt_path, meta, model, optimizer)

        # Simulated failure at epoch 10 on the first attempt only.
        # On retry the checkpoint from epoch 10 exists, so training resumes at 11.
        if epoch == 10 and get_attempt_number() == 0:
            raise flyte.errors.RuntimeSystemError("simulated", "Simulated failure after epoch 10")

    return meta.history


if __name__ == "__main__":
    flyte.init_from_config()
    result = flyte.run(train)
    print(result.url)

    # Run with:
    # uv run examples/ml/pytorch_trace_training.py
