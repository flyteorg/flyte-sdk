# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "flyte",
#   "torch>=2.0",
#   "unsloth",
#   "trl",
#   "datasets",
#   "transformers",
#   "accelerate",
# ]
# ///

"""
Unsloth + TRL ``SFTTrainer`` + Flyte ``Checkpoint``
=========================================================

LoRA fine-tuning with `Unsloth <https://unsloth.ai/docs>`__ and :class:`trl.trainer.SFTTrainer`, persisting
``output_dir`` through :class:`~flyte.Checkpoint` so Flyte retries can resume.

Follows ``sklearn_partial_checkpoint.py`` and ``huggingface_trainer_checkpoint.py``:

- ``await checkpoint.load()``, ``get_last_checkpoint``, ``trainer.train(resume_from_checkpoint=...)``.
- ``chunks_start`` from ``trainer_state.json`` and step failure rule
  ``(global_step - 1) > chunks_start and (global_step - 1) % failure_interval == 0`` with
  ``failure_interval = max_steps // RETRIES``.
- Flyte sync on HF ``on_save``.

**Hardware:** Unsloth requires an **NVIDIA, AMD, or Intel GPU** (not Apple Silicon / MPS). Imports from
``unsloth`` are deferred until the task body.
"""

from __future__ import annotations

import json
import pathlib

from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_unsloth_sft",
    image=flyte.Image.from_debian_base().with_pip_packages("unsloth"),
)

MODEL_NAME = "unsloth/Llama-3.2-1B-bnb-4bit"
RETRIES = 3


class FailureInjectionCallback(TrainerCallback):
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
    def __init__(self, flyte_checkpoint: flyte.Checkpoint, output_dir: pathlib.Path) -> None:
        self._flyte_checkpoint = flyte_checkpoint
        self._output_dir = output_dir

    def on_save(self, args, state, control, **kwargs) -> None:
        self._flyte_checkpoint.save_sync(self._output_dir)


def _chunks_start_from_hf_checkpoint(checkpoint_dir: str | None) -> int:
    if not checkpoint_dir:
        return 0
    p = pathlib.Path(checkpoint_dir) / "trainer_state.json"
    if not p.is_file():
        return 0
    return int(json.loads(p.read_text())["global_step"])


def _tiny_instruction_dataset():
    from datasets import Dataset

    rows = [
        "### Instruction:\nSay hello.\n\n### Response:\nHello!",
        "### Instruction:\nWhat is 1+1?\n\n### Response:\n2",
        "### Instruction:\nName a color.\n\n### Response:\nBlue",
    ] * 6
    return Dataset.from_dict({"text": rows})


@env.task(retries=RETRIES)
async def train_unsloth_sft(max_steps: int = 12) -> float:
    assert max_steps > RETRIES
    import unsloth  # noqa: F401
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    ctx = flyte.ctx()
    assert ctx is not None
    checkpoint = ctx.checkpoint
    assert checkpoint is not None

    checkpoint_path: pathlib.Path | None = await checkpoint.load()
    if checkpoint_path is None:
        output_dir = pathlib.Path("unsloth_sft")
    else:
        output_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent / "unsloth_sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_resume = get_last_checkpoint(str(output_dir))
    chunks_start = _chunks_start_from_hf_checkpoint(hf_resume)
    failure_interval = max_steps // RETRIES
    print(f"chunks_start={chunks_start}, max_steps={max_steps}, failure_interval={failure_interval}")

    max_seq_length = 512
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_dataset = _tiny_instruction_dataset()

    save_steps = max(1, max_steps // 2)
    args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=max(1, save_steps // 2),
        report_to="none",
        seed=42,
        dataset_text_field="text",
        max_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[
            FailureInjectionCallback(chunks_start, failure_interval),
            FlyteTrainerCheckpointCallback(checkpoint, output_dir),
        ],
    )
    trainer.train(resume_from_checkpoint=hf_resume)

    await checkpoint.save(output_dir)

    for h in reversed(trainer.state.log_history):
        if "loss" in h:
            return float(h["loss"])
        if "train_loss" in h:
            return float(h["train_loss"])
    return float(max_steps)


if __name__ == "__main__":
    flyte.init_from_config()
    try:
        run = flyte.with_runcontext(mode="remote").run(train_unsloth_sft, max_steps=12)
        print(run.url)
    except Exception as e:
        msg = str(e)
        if "Unsloth currently only works" in msg or "NVIDIA, AMD and Intel GPUs" in msg:
            raise SystemExit(
                "Unsloth requires an NVIDIA, AMD, or Intel GPU. Run on a supported GPU worker; see "
                "https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements\n"
                f"Details: {msg}"
            ) from e
        raise
