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

Trains a tiny BERT classifier with :class:`transformers.Trainer`, keeping the Hugging Face ``output_dir``
under the task checkpoint prefix. Matches ``sklearn_partial_checkpoint.py``:

1. ``await checkpoint.load.aio()`` — restore prior tree from object storage.
2. ``resume_from_checkpoint`` from :func:`transformers.trainer_utils.get_last_checkpoint` (not the raw
   Flyte load path).
3. ``chunks_start`` from ``trainer_state.json`` ``global_step`` when resuming (steps completed before this
   attempt). Failure uses the same rule as sklearn:
   ``(global_step - 1) > chunks_start and (global_step - 1) % failure_interval == 0`` with
   ``failure_interval = max_steps // RETRIES``.
4. :class:`FlyteTrainerCheckpointCallback` — :meth:`~flyte.AsyncCheckpoint.save` on each HF ``on_save``
   (aligned with HF checkpoint writes).
5. ``await checkpoint.save.aio(output_dir)`` at the end.

The model id ``hf-internal-testing/tiny-random-bert`` is small and intended for tests; the first run may
download weights from the Hub. ``accelerate`` is required for :class:`transformers.Trainer` with PyTorch.
``use_cpu=True`` keeps the example portable.
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

env = flyte.TaskEnvironment(name="checkpoint_hf_trainer")

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
    """Sklearn-style step failures: ``i > chunks_start and i % failure_interval == 0`` for ``i = global_step - 1``."""

    def __init__(self, chunks_start: int, failure_interval: int) -> None:
        self._chunks_start = chunks_start
        self._failure_interval = failure_interval

    def on_step_end(self, args, state, control, **kwargs) -> None:
        g = state.global_step
        i = g - 1
        if i > self._chunks_start and i % self._failure_interval == 0:
            raise RuntimeError(
                f"Failed after optimizer step global_step={g} (i={i}), failure_interval {self._failure_interval}."
            )


class FlyteTrainerCheckpointCallback(TrainerCallback):
    """After each HF checkpoint save, upload ``output_dir`` to Flyte (blocking :meth:`~flyte.AsyncCheckpoint.save`)."""

    def __init__(self, flyte_checkpoint: flyte.AsyncCheckpoint, output_dir: pathlib.Path) -> None:
        self._flyte_checkpoint = flyte_checkpoint
        self._output_dir = output_dir

    def on_save(self, args, state, control, **kwargs) -> None:
        self._flyte_checkpoint.save(self._output_dir)


def _chunks_start_from_hf_checkpoint(checkpoint_dir: str | None) -> int:
    if not checkpoint_dir:
        return 0
    p = pathlib.Path(checkpoint_dir) / "trainer_state.json"
    if not p.is_file():
        return 0
    return int(json.loads(p.read_text())["global_step"])


@env.task(retries=RETRIES)
async def train_transformers(max_steps: int = 24) -> float:
    assert max_steps > RETRIES
    ctx = flyte.ctx()
    assert ctx is not None
    checkpoint = ctx.checkpoint
    assert checkpoint is not None

    checkpoint_path: pathlib.Path | None = await checkpoint.load.aio()
    if checkpoint_path is None:
        output_dir = pathlib.Path("hf_trainer")
    else:
        output_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent / "hf_trainer"
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_resume = get_last_checkpoint(str(output_dir))
    chunks_start = _chunks_start_from_hf_checkpoint(hf_resume)
    failure_interval = max_steps // RETRIES
    print(f"chunks_start={chunks_start}, max_steps={max_steps}, failure_interval={failure_interval}")

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
        save_total_limit=2,
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
        callbacks=[
            FailureInjectionCallback(chunks_start, failure_interval),
            FlyteTrainerCheckpointCallback(checkpoint, output_dir),
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
    run = flyte.with_runcontext(mode="remote").run(train_transformers, max_steps=24)
    print(run.url)
