"""
Distributed Training with Fully Decoupled Periodic Evaluation
=============================================================

This example demonstrates a pattern for distributed training with
periodic, independent evaluation of the latest checkpoint.

Why two workflows?
- Eval is much slower than training and must not block the training loop.
- If training fails, eval naturally stops on the next cron tick (no checkpoint
  to evaluate) and disables itself.
- If the model converges, eval writes a stop signal that training polls for.

How eval finds training:
1. The eval trigger receives the training run name as an input,
   set when the trigger is created after training starts.
2. Eval lists checkpoint files in that directory using ``Dir.from_existing_remote``,
   and reads/writes coordination files alongside them:
   - ``stop_signal.json`` — written by eval when converged, polled by training
   - ``eval/{checkpoint_name}.json`` — written by eval to track already-evaluated checkpoints

Evaluation discovers and monitors training, while training remains unaware of evaluation.
"""

import json
import re
from datetime import datetime

import lightning as L
import torch
import torch.distributed as dist
from flyteplugins.pytorch.task import Elastic
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset

import flyte
import flyte.remote
from flyte.io import Dir, File
from flyte.models import ActionPhase, NativeInterface

image = flyte.Image.from_debian_base(name="lightning-eval").with_pip_packages(
    "lightning==2.6.1", "flyteplugins-pytorch==2.0.2"
)

# Multi-node training: 2 nodes, 1 process per node
train_env = flyte.TaskEnvironment(
    name="distributed-train",
    resources=flyte.Resources(cpu=4, memory="25Gi", gpu="L4:1"),
    plugin_config=Elastic(
        nproc_per_node=1,
        nnodes=2,
    ),
    image=image,
)

# Single-GPU eval on a cheaper instance, runs independently via Flyte trigger
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


@train_env.task
def train(checkpoint_dir: str, max_epochs: int = 20) -> str | None:
    model = SimpleModel()
    data = SyntheticDataModule()

    log_rank0("[train] Task started")
    log_rank0(f"[train] checkpoint_dir={checkpoint_dir}")
    log_rank0(f"[train] max_epochs={max_epochs}")

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}",
        save_top_k=-1,
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
            checkpoint_callback,
            stop_signal_callback,
            logging_callback,
        ],
        enable_progress_bar=False,
        limit_train_batches=2000,
    )

    log_rank0("[train] Calling trainer.fit()")
    trainer.fit(model, data)

    log_rank0("[train] trainer.fit() completed")

    return "Training completed!"


@eval_env.task(triggers=flyte.Trigger.minutely(name="lightning-eval", trigger_time_input_key="trigger_time"))
def run_eval(
    trigger_time: datetime,
    training_run_name: str,
    convergence_threshold: float = 0.05,
) -> str:
    """Periodic evaluation of the latest training checkpoint.

    Receives the training run name as a fixed trigger input, looks up the
    run to read its ``checkpoint_dir`` and evaluates the latest checkpoint.
    """
    print(f"[eval] Triggered at {trigger_time.isoformat()}")
    print(f"[eval] Training run: {training_run_name}")

    # 1. Look up the training run and check its status
    training_run = flyte.remote.Run.get(name=training_run_name)

    if training_run.phase.is_terminal:
        print("[eval] Training run has reached a terminal state — deactivating eval trigger.")
        flyte.remote.Trigger.update(name="lightning-eval", task_name=f"{eval_env.name}.run_eval", active=False)
        return "training_failed"
    elif training_run.phase != ActionPhase.RUNNING:
        print("[eval] Training run is still initializing.")
        return "training_initializing"

    # 2. Read the checkpoint_dir from the training run's inputs
    inputs = training_run.inputs()
    checkpoint_dir = inputs["checkpoint_dir"]
    print(f"[eval] Checkpoint dir from training inputs: {checkpoint_dir}")

    # 3. List checkpoint files uploaded by ModelCheckpoint
    ckpt_dir = Dir.from_existing_remote(checkpoint_dir)
    if not ckpt_dir.exists_sync():
        print("[eval] Checkpoint directory does not exist yet — training may not have written one.")
        return "no_checkpoint"

    ckpt_files = [f for f in ckpt_dir.list_files_sync() if f.path.endswith(".ckpt")]
    if not ckpt_files:
        print("[eval] No checkpoint files found yet.")
        return "no_checkpoint"

    def extract_step(path: str) -> int:
        m = re.search(r"step=(\d+)", path)
        return int(m.group(1)) if m else -1

    latest_ckpt = max(ckpt_files, key=lambda f: extract_step(f.path))
    checkpoint_path = latest_ckpt.path
    checkpoint_name = checkpoint_path.split("/")[-1]

    print(f"[eval] Latest checkpoint: {checkpoint_path}")

    record_path = f"{checkpoint_dir}/eval/{checkpoint_name}.json"
    record_file = File.from_existing_remote(record_path)

    if record_file.exists_sync():
        print("[eval] Already evaluated — skipping.")
        return f"already_evaluated_{checkpoint_name}"

    # 5. Load checkpoint and run evaluation on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_ckpt = latest_ckpt.download_sync()
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

    print(f"[eval] checkpoint={checkpoint_path}  val_loss={val_loss:.4f}")

    _write_json_to_remote(
        record_path,
        {
            "checkpoint": checkpoint_path,
            "val_loss": val_loss,
            "evaluated_at": trigger_time.isoformat(),
            "run_id": training_run_name,
            "converged": val_loss < convergence_threshold,
        },
    )

    # 7. If converged, write stop signal
    if val_loss < convergence_threshold:
        print(f"[eval] Converged! val_loss={val_loss:.4f} < {convergence_threshold}. Writing stop signal.")

        stop_path = f"{checkpoint_dir}/stop_signal.json"
        _write_json_to_remote(
            stop_path,
            {
                "reason": "converged",
                "checkpoint": checkpoint_path,
                "val_loss": val_loss,
                "signaled_at": trigger_time.isoformat(),
            },
        )

        return f"converged_{checkpoint_name}"

    return f"evaluated_{checkpoint_name}_loss_{val_loss:.4f}"


if __name__ == "__main__":
    flyte.init_from_config()

    result = flyte.run(train, checkpoint_dir="s3://lightning/training-run-001/checkpoints")
    print(f"Training run URL: {result.url}")

    inputs = dict(run_eval.interface.inputs)
    inputs["training_run_name"] = (str, result.name)
    run_eval.interface = NativeInterface(inputs, dict(run_eval.interface.outputs))

    flyte.deploy(eval_env)

    print("Eval workflow will fire via Flyte trigger.")
