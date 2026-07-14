# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "flyte",
#   "torch>=2.0",
#   "transformers>=4.38",
#   "accelerate>=1.1.0",
# ]
# ///

"""
Hugging Face `transformers.Trainer` + Flyte `Checkpoint`
===========================================================

Trains a tiny BERT classifier with `transformers.Trainer`, mirroring
`pytorch_task_checkpoint.py` and `pytorch_lightning_checkpoint.py`.

**Async task (like `pytorch_task_checkpoint`):** use `cp = ctx.checkpoint`, `prev_cp_path = await cp.load()`,
and `await cp.save(...)` so object storage stays in sync with local HF checkpoints.

**Layout (like `pytorch_lightning_checkpoint`):** resolve a fixed `output_dir` under the restored Flyte tree
(or `hf_trainer` when there is no prior checkpoint), then `resume_from_checkpoint` from
`transformers.trainer_utils.get_last_checkpoint` (not the raw Flyte path).

**Retries:** `chunks_start` from `trainer_state.json` `global_step` when resuming. Failure injection matches
sklearn / Lightning: `(global_step - 1) > chunks_start and (global_step - 1) % failure_interval == 0` with
`failure_interval = max_steps // RETRIES`.

**Callbacks (Lightning order):** `FailureInjectionCallback` first, then
`FlyteTrainerCheckpointCallback`, so a simulated failure happens before that step is mirrored to Flyte.
`FlyteTrainerCheckpointCallback` uses `flyte.Checkpoint.save_sync` on HF `on_save` (Trainer
callbacks are synchronous).

**Final persist:** `await cp.save(output_dir)` after `transformers.Trainer.train` (same idea as the last
`await cp.save(...)` in `pytorch_task_checkpoint`).

The model id `hf-internal-testing/tiny-random-bert` is small and intended for tests; the first run may
download weights from the Hub. `accelerate` is required for `transformers.Trainer` with PyTorch.
`use_cpu=True` keeps the example portable.
"""

from __future__ import annotations

import json
import pathlib

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_hf_trainer",
    image=flyte.Image.from_debian_base().with_pip_packages("transformers[torch]"),
)

MODEL_ID = "hf-internal-testing/tiny-random-bert"
RETRIES = 3


class ToyTextDataset(Dataset):
    """Synthetic binary classification examples."""

    def __init__(self, tokenizer, n: int = 64, max_length: int = 32):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._texts = [f"classification example {i} with enough tokens" for i in range(n)]
        self._labels = [i % 2 for i in range(n)]

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self._tokenizer(
            self._texts[idx],
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
        )
        enc["labels"] = self._labels[idx]
        return enc


class FailureInjectionCallback(TrainerCallback):
    """Sklearn-style step failures: `i > chunks_start and i % failure_interval == 0` for `i = global_step - 1`."""

    def __init__(self, chunks_start: int, failure_interval: int) -> None:
        self._chunks_start = chunks_start
        self._failure_interval = failure_interval

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        g = state.epoch
        i = g - 1
        if i > self._chunks_start and i % self._failure_interval == 0:
            raise RuntimeError(f"Failed after epoch {g} (i={i}), failure_interval {self._failure_interval}.")


class FlyteTrainerCheckpointCallback(TrainerCallback):
    """Mirror HF `output_dir` to Flyte after each on-disk save (sync `flyte.Checkpoint.save_sync`)."""

    def __init__(self, cp: flyte.Checkpoint, output_dir: pathlib.Path) -> None:
        self._cp = cp
        self._output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        # Trainer callbacks are sync — same pattern as FlyteLightningCheckpointCallback.save_sync
        self._cp.save_sync(self._output_dir)


def _chunks_start_from_hf_checkpoint(checkpoint_dir: str | None) -> int:
    if not checkpoint_dir:
        return 0
    p = pathlib.Path(checkpoint_dir) / "trainer_state.json"
    if not p.is_file():
        return 0
    return int(json.loads(p.read_text())["global_step"])


@env.task(retries=RETRIES)
def train_transformers(max_epochs: int = 24) -> float:
    assert max_epochs > RETRIES
    ctx = flyte.ctx()
    assert ctx is not None
    cp = ctx.checkpoint
    assert cp is not None

    ckpt_dir = pathlib.Path("hf_trainer")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    failure_interval = max_epochs // RETRIES

    prev_cp_path: pathlib.Path | None = cp.load_sync()
    hf_resume: str | None = None
    last_ckpt = None
    chunks_start = 0
    if prev_cp_path:
        last_ckpt = get_last_checkpoint(str(prev_cp_path))
        hf_resume = last_ckpt or None
        chunks_start = _chunks_start_from_hf_checkpoint(hf_resume)

    print(f"chunks_start={chunks_start}, prev_cp_path={prev_cp_path}, hf_resume={hf_resume}, last_ckpt={last_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    train_ds = ToyTextDataset(tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=4,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        seed=42,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[
            FailureInjectionCallback(chunks_start, failure_interval),
            FlyteTrainerCheckpointCallback(cp, ckpt_dir),
        ],
    )
    trainer.train(resume_from_checkpoint=hf_resume)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = tokenizer(
            "classification example for inference",
            return_tensors="pt",
            truncation=True,
            max_length=32,
            padding="max_length",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        return float(logits[0, 1].item())


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(train_transformers, max_epochs=10)
    print(run.url)
