# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.9",
#    "torch==2.7.1",
# ]
# ///

"""
PyTorch Training Loop with Trace-Based Checkpointing
=====================================================

Demonstrates using @flyte.trace to checkpoint each training epoch so that on
any failure and retry, completed epochs are replayed instantly from their
recorded outputs — no wasted compute.

The core pattern is a *state chain*: each epoch trace receives the previous
epoch's model/optimizer state as input and returns the updated state as output.
Because trace replay is keyed on (function, inputs), deterministic inputs
guarantee the checkpoint is found on retry:

    initial_state
        → [epoch 1 trace] → state_1   (checkpointed)
        → [epoch 2 trace] → state_2   (checkpointed)
        → ...crash at epoch K...
    retry:
        → epoch 1: replayed instantly  (no GPU work)
        → epoch 2: replayed instantly
        → epoch K: resumes with correct state_K-1

Key Flyte concepts:
- @flyte.trace     — per-epoch checkpoint; replayed on retry
- flyte.group()    — groups epoch traces under a named span for observability
- @env.task(retries=N) — allows the task to retry after failure
"""

import asyncio
import base64
import io
import os

import flyte
import flyte.errors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

image = flyte.Image.from_uv_script(__file__, name="pytorch-trace-training")


def get_attempt_number() -> int:
    return int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))

env = flyte.TaskEnvironment(
    name="pytorch_trace_training",
    image=image,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 1


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    )


# ---------------------------------------------------------------------------
# State serialization
#
# Model/optimizer state dicts are serialized to base64-encoded strings so they
# can be passed as trace inputs/outputs.  Base64 strings are JSON-serializable
# and safe to store in Flyte's checkpoint system.
# ---------------------------------------------------------------------------


def state_to_b64(state_dict: dict) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_state(b64: str) -> dict:
    return torch.load(io.BytesIO(base64.b64decode(b64)), weights_only=True)


# ---------------------------------------------------------------------------
# Dataset
#
# Synthetic regression: y = sum(x) + noise.  Fixed seed makes the dataset
# identical across retries, which keeps trace replay inputs deterministic.
# ---------------------------------------------------------------------------


def make_loaders(batch_size: int, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(2000, INPUT_DIM, generator=g)
    y = X.sum(dim=1, keepdim=True) + 0.05 * torch.randn(2000, 1, generator=g)
    ds = TensorDataset(X, y)
    train_ds, val_ds = torch.utils.data.random_split(ds, [1600, 400], generator=g)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Traced training epoch
#
# Each call to train_epoch is a Flyte checkpoint.  On retry, any epoch whose
# (function, inputs) tuple was previously recorded is replayed from the stored
# output — the function body does not re-execute.
#
# Why base64 strings for state?
#   Trace inputs/outputs must be serializable.  Encoding model state as a b64
#   string keeps every value in the trace signature a JSON primitive, which is
#   guaranteed to round-trip through Flyte's checkpoint store correctly.
# ---------------------------------------------------------------------------


@flyte.trace
async def train_epoch(
    epoch: int,
    model_state_b64: str,
    optimizer_state_b64: str,
    batch_size: int,
    learning_rate: float,
) -> dict:
    """Run one training epoch; return updated state + metrics.

    On retry this function is not re-executed — Flyte returns the previously
    recorded output dict directly, skipping all GPU work.
    """
    model = build_model()
    model.load_state_dict(b64_to_state(model_state_b64))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(b64_to_state(optimizer_state_b64))

    train_loader, val_loader = make_loaders(batch_size)
    criterion = nn.MSELoss()

    # Training pass
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X)
    train_loss /= 1600

    # Validation pass
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            val_loss += criterion(model(X), y).item() * len(X)
    val_loss /= 400

    print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)

    # Simulate processing time so the speedup from trace replay is obvious
    # when the task is retried: replayed epochs skip this entirely.
    await asyncio.sleep(2.0)

    return {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        # Updated state is passed forward to the next epoch's trace inputs,
        # making that epoch's checkpoint reachable on replay.
        "model_state_b64": state_to_b64(model.state_dict()),
        "optimizer_state_b64": state_to_b64(optimizer.state_dict()),
    }


# ---------------------------------------------------------------------------
# Main training task
# ---------------------------------------------------------------------------


@env.task(retries=3)
async def train(
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Training loop using @flyte.trace for per-epoch checkpointing.

    The task is configured with retries=3.  If it fails mid-training, Flyte
    will retry the whole task — but because each epoch is a trace, all epochs
    that already succeeded are replayed from their checkpoints.  Only the
    failed epoch (and any after it) actually re-execute.

    Determinism note:
        Both the model initialization and the dataset are seeded with `seed`.
        This makes the initial trace inputs identical across retries, which is
        required for Flyte to find and replay the correct checkpoints.
    """
    # Initialize with fixed seed so the first epoch's trace inputs are
    # reproducible across any number of retries.
    torch.manual_seed(seed)
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_state_b64 = state_to_b64(model.state_dict())
    optimizer_state_b64 = state_to_b64(optimizer.state_dict())

    history: list[dict] = []

    # flyte.group gives all epoch traces a shared parent span in the UI,
    # making it easy to inspect per-epoch timing and metrics at a glance.
    with flyte.group("training-loop"):
        for epoch in range(1, num_epochs + 1):
            result = await train_epoch(
                epoch=epoch,
                model_state_b64=model_state_b64,
                optimizer_state_b64=optimizer_state_b64,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            # Simulated failure: crash after epoch 10 on the first attempt only.
            # Epoch 10's trace has already been checkpointed by this point, so on
            # retry epochs 1-10 are all replayed instantly and training resumes at 11.
            if epoch == 10 and get_attempt_number() == 0:
                raise flyte.errors.RuntimeSystemError(
                    "simulated", "Simulated failure after epoch 10 on attempt 0"
                )

            # Thread state forward: this epoch's output becomes next epoch's input.
            # On replay the chain stays intact because each replayed trace returns
            # the same state bytes it recorded originally.
            model_state_b64 = result["model_state_b64"]
            optimizer_state_b64 = result["optimizer_state_b64"]
            history.append(
                {
                    "epoch": result["epoch"],
                    "train_loss": result["train_loss"],
                    "val_loss": result["val_loss"],
                }
            )

    final = history[-1]
    print(
        f"Training complete after {num_epochs} epochs. "
        f"Final val_loss={final['val_loss']:.4f}",
        flush=True,
    )
    return {"history": history, "final_val_loss": final["val_loss"]}


if __name__ == "__main__":
    flyte.init_from_config()
    result = flyte.run(train)
    print(result.url)

    # Run with:
    # uv run examples/ml/pytorch_trace_training.py