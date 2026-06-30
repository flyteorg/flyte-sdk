"""
Distributed Training with Callback-Driven Evaluation on Checkpoints
===================================================================

This example demonstrates a pattern for distributed training where
evaluation is triggered directly from a Lightning checkpoint callback.

How it works:
- Training saves checkpoints to a remote directory via ``ModelCheckpoint``.
- ``EvalOnCheckpointCallback`` detects each new checkpoint and launches
  the eval task as a separate Flyte run.
- The eval task checks if training is still alive via
  ``flyte.remote.Run.get`` — if training has failed, completed, or been
  aborted, eval returns early.
- If the model converges, eval writes ``stop_signal.json`` which training
  polls for via ``StopSignalCallback``.
"""

import json
from datetime import datetime, timezone

import lightning as L
import torch
import torch.distributed as dist
from flyteplugins.pytorch.task import Elastic
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset

import flyte
from flyte.io import File

image = flyte.Image.from_debian_base(name="lightning-eval").with_pip_packages(
    "lightning==2.6.1", "flyteplugins-pytorch==2.0.3"
)

# Multi-node training: 2 nodes, 1 process per node
train_env = flyte.TaskEnvironment(
    name="distributed-train",
    resources=flyte.Resources(cpu=4, memory="25Gi", gpu="T4:1"),
    plugin_config=Elastic(
        nproc_per_node=1,
        nnodes=2,
    ),
    image=image,
)

# Single-GPU eval on a cheaper instance
eval_env = flyte.TaskEnvironment(
    name="eval",
    resources=flyte.Resources(cpu=2, memory="4Gi", gpu="T4:1"),
    image=image,
)


def _write_json_to_remote(path: str, data: dict) -> None:
    """Write a JSON dict to a remote file."""
    f = File.from_existing_remote(path)
    with f.open_sync("wb") as fh:
        fh.write(json.dumps(data).encode("utf-8"))


def _read_json_from_remote(path: str) -> dict | None:
    """Read a JSON dict from a remote file, or return None if it doesn't exist."""
    f = File.from_existing_remote(path)
    if not f.exists_sync():
        return None
    with f.open_sync("rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


class StreamingRegressionDataset(IterableDataset):
    def __init__(self, dim: int = 512):
        self.dim = dim

    def __iter__(self):
        while True:
            x = torch.randn(self.dim)
            y = x.sum().unsqueeze(0) + torch.randn(1) * 0.1
            yield x, y


class SyntheticDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 2048):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = StreamingRegressionDataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


class SimpleModel(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


class StopSignalCallback(L.Callback):
    """Polls for a remote stop signal file after each training epoch.

    When the eval workflow determines the model has converged, it writes
    ``stop_signal.json`` to the checkpoint directory.  This callback checks
    for that file and requests a graceful stop via ``trainer.should_stop``.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        stop_path = f"{self.checkpoint_dir}/stop_signal.json"
        stop = _read_json_from_remote(stop_path)
        if stop is not None:
            print(
                f"[train] Stop signal received: {stop.get('reason', 'unknown')}. "
                f"Stopping after epoch {trainer.current_epoch}."
            )
            trainer.should_stop = True


def log_rank0(msg: str):
    if not dist.is_available() or not dist.is_initialized():
        print(msg)
    elif dist.get_rank() == 0:
        print(msg)


class TrainingLoggingCallback(L.Callback):
    def on_fit_start(self, trainer, pl_module):
        log_rank0(
            f"[train] Starting training | world_size={trainer.world_size} "
            f"| num_nodes={trainer.num_nodes} | device={pl_module.device}"
        )

    def on_train_epoch_start(self, trainer, pl_module):
        log_rank0(f"[train] Epoch {trainer.current_epoch} started")

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        loss = metrics.get("train_loss")
        if loss is not None:
            loss = loss.item()

        log_rank0(f"[train] Epoch {trainer.current_epoch} finished | loss={loss} | global_step={trainer.global_step}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        log_rank0(f"[train] Checkpoint saved | epoch={trainer.current_epoch} | step={trainer.global_step}")

    def on_fit_end(self, trainer, pl_module):
        log_rank0("[train] Training finished")


class EvalModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that launches an eval task after each save.

    Subclasses ``ModelCheckpoint`` so that eval is triggered *after* the
    checkpoint is written.  ``flyte.run()`` submits a separate, independent
    and returns immediately (fire-and-forget).
    """

    def __init__(self, *args, checkpoint_dir: str, training_run_name: str, **kwargs):
        super().__init__(*args, dirpath=checkpoint_dir, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.training_run_name = training_run_name

    def _remove_checkpoint(self, trainer, filepath):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        # Save the checkpoint first
        super().on_train_epoch_end(trainer, pl_module)

        # Launch eval on global rank 0 if a new checkpoint was saved
        if trainer.is_global_zero:
            current_path = self.best_model_path
            if current_path:
                try:
                    result = flyte.run(
                        run_eval,
                        training_run_name=self.training_run_name,
                        checkpoint_path=current_path,
                        checkpoint_dir=self.checkpoint_dir,
                    )
                    log_rank0(f"[eval-callback] Eval run submitted: {result.url}")
                except Exception as e:
                    log_rank0(f"[eval-callback] Failed to launch eval: {e}")


@train_env.task
def train(checkpoint_dir: str, max_epochs: int = 20) -> str | None:
    flyte.init_in_cluster()  # this is for eval

    model = SimpleModel()
    data = SyntheticDataModule()

    training_run_name = flyte.ctx().action.run_name

    log_rank0("[train] Task started")
    log_rank0(f"[train] checkpoint_dir={checkpoint_dir}")
    log_rank0(f"[train] training_run_name={training_run_name}")
    log_rank0(f"[train] max_epochs={max_epochs}")

    checkpoint_callback = EvalModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        training_run_name=training_run_name,
        save_top_k=1,
        filename="latest-{epoch}-{step}",
        every_n_epochs=1,
    )

    stop_signal_callback = StopSignalCallback(checkpoint_dir)
    logging_callback = TrainingLoggingCallback()

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        strategy="ddp",
        num_nodes=2,
        callbacks=[
            logging_callback,
            checkpoint_callback,
            stop_signal_callback,
        ],
        enable_progress_bar=False,
        limit_train_batches=2000,
    )

    log_rank0("[train] Calling trainer.fit()")
    trainer.fit(model, data)

    log_rank0("[train] trainer.fit() completed")

    return "Training completed!"


@eval_env.task
def run_eval(
    training_run_name: str,
    checkpoint_path: str,
    checkpoint_dir: str,
    convergence_threshold: float = 0.05,
) -> str:
    """Evaluates a specific checkpoint.

    Launched by ``EvalOnCheckpointCallback`` inside the training task.
    """
    checkpoint_name = checkpoint_path.rsplit("/", maxsplit=1)[-1]
    print(f"[eval] Evaluating checkpoint: {checkpoint_name}")

    # Check if already evaluated
    record_path = f"{checkpoint_dir}/eval/{checkpoint_name}.json"
    record_file = File.from_existing_remote(record_path)
    if record_file.exists_sync():
        print("[eval] Already evaluated — skipping.")
        return f"already_evaluated_{checkpoint_name}"

    # Load checkpoint and run evaluation on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_ckpt = File.from_existing_remote(checkpoint_path).download_sync()
    ckpt = torch.load(local_ckpt, weights_only=True, map_location=device)

    model = SimpleModel()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    n = 200_000
    dim = 512

    x_val = torch.randn(n, dim, device=device)
    y_val = x_val.sum(dim=1, keepdim=True) + torch.randn(n, 1, device=device) * 0.1

    with torch.no_grad():
        preds = model(x_val)
        val_loss = nn.functional.mse_loss(preds, y_val).item()

    print(f"[eval] checkpoint={checkpoint_name}  val_loss={val_loss:.4f}")

    now = datetime.now(timezone.utc).isoformat()
    _write_json_to_remote(
        record_path,
        {
            "checkpoint": checkpoint_path,
            "val_loss": val_loss,
            "evaluated_at": now,
            "run_id": training_run_name,
            "converged": val_loss < convergence_threshold,
        },
    )

    # If converged, write stop signal
    if val_loss < convergence_threshold:
        print(f"[eval] Converged! val_loss={val_loss:.4f} < {convergence_threshold}. Writing stop signal.")

        _write_json_to_remote(
            f"{checkpoint_dir}/stop_signal.json",
            {
                "reason": "converged",
                "checkpoint": checkpoint_path,
                "val_loss": val_loss,
                "signaled_at": now,
            },
        )

        return f"converged_{checkpoint_name}"

    return f"evaluated_{checkpoint_name}_loss_{val_loss:.4f}"


if __name__ == "__main__":
    flyte.init_from_config()

    result = flyte.run(train, checkpoint_dir="s3://lightning/training-run-002/checkpoints")
    print(f"Training run: {result.url}")

    flyte.deploy(eval_env)
