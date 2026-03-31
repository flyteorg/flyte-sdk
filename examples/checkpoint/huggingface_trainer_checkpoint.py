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
Hugging Face ``transformers.Trainer`` + Flyte ``AsyncCheckpoint``
================================================================

Trains a tiny BERT classifier with :class:`transformers.Trainer`, keeping the Hugging Face
``output_dir`` (including ``checkpoint-*`` folders) under :attr:`~flyte.AsyncCheckpoint.path`
and syncing it through the task checkpoint prefix.

Flow:

1. ``await checkpoint.load.aio()`` — restore any prior tree from object storage.
2. :func:`transformers.trainer_utils.get_last_checkpoint` — pick up the latest ``checkpoint-*``
   folder after a restore (or ``None`` on the first run).
3. ``Trainer.train(resume_from_checkpoint=...)`` — resume training when a checkpoint exists.
4. :class:`FlyteTrainerCheckpointCallback` — at each epoch end, upload ``output_dir`` with
   :meth:`~flyte.AsyncCheckpoint.save` (sync; safe inside ``Trainer.train``).
5. ``await checkpoint.save.aio(output_dir)`` — final upload so the tail of the last epoch is persisted when
   training stops on ``max_steps`` before the next epoch boundary.

The model id ``hf-internal-testing/tiny-random-bert`` is small and intended for tests; the first
run may download weights from the Hub. Transformers may print a "LOAD REPORT" with
``UNEXPECTED`` keys for this checkpoint — that is normal when repurposing a random test weight.

``accelerate`` is required for :class:`transformers.Trainer` with PyTorch (see Transformers docs).
``use_cpu=True`` keeps the example portable on machines with CUDA/MPS.

**Note:** Install this repository editable (``pip install -e .``) so ``TaskContext.checkpoint`` exists;
see ``generic_data_checkpoint.py``.
"""

from __future__ import annotations

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

import flyte

env = flyte.TaskEnvironment(name="checkpoint_hf_trainer")

# Small test weights — first run downloads from the Hub (~few MB).
MODEL_ID = "hf-internal-testing/tiny-random-bert"


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


class FlyteTrainerCheckpointCallback(TrainerCallback):
    """
    After each training epoch, upload ``output_dir`` (HF checkpoints and trainer state) to Flyte.

    :class:`~transformers.Trainer` runs synchronously; use blocking
    :meth:`~flyte.AsyncCheckpoint.save`, not ``await ...save.aio()`` here.
    """

    def __init__(self, flyte_checkpoint: flyte.AsyncCheckpoint, output_dir: pathlib.Path) -> None:
        self._flyte_checkpoint = flyte_checkpoint
        self._output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        self._flyte_checkpoint.save(self._output_dir)


@env.task()
async def train_transformers(max_steps: int = 24) -> float:
    ctx = flyte.ctx()
    assert ctx is not None
    checkpoint = ctx.checkpoint

    checkpoint_path: pathlib.Path | None = await checkpoint.load.aio()
    if checkpoint_path is None:
        output_dir = pathlib.Path("hf_trainer")
    else:
        output_dir = checkpoint_path

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    train_ds = ToyTextDataset(tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    save_steps = max(1, max_steps // 2)
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=4,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        logging_steps=max(1, save_steps // 2),
        report_to="none",
        seed=42,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[FlyteTrainerCheckpointCallback(checkpoint, output_dir)],
    )
    trainer.train(resume_from_checkpoint=checkpoint_path)

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
    run = flyte.with_runcontext(mode="local").run(train_transformers, max_steps=16)
    print(run.outputs())
