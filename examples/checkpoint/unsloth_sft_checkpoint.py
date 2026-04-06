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
Unsloth + TRL `trl.trainer.SFTTrainer` + Flyte `Checkpoint`
===================================================

LoRA fine-tuning with Unsloth (https://unsloth.ai/docs) and `trl.trainer.SFTTrainer`, using the same
checkpointing shape as `pytorch_task_checkpoint.py`, `pytorch_lightning_checkpoint.py`, and
`huggingface_trainer_checkpoint.py`.

**Async task:** `cp = ctx.checkpoint`, `prev_cp_path = await cp.load()`, fixed `unsloth_sft` under the
restored tree (or fresh), then `get_last_checkpoint` and `trainer.train(resume_from_checkpoint=...)`.

**Retries:** `chunks_start` from `trainer_state.json`; failures when
`(global_step - 1) > chunks_start and (global_step - 1) % failure_interval == 0` with
`failure_interval = max_steps // RETRIES`.

**Callbacks:** `FailureInjectionCallback` then `FlyteTrainerCheckpointCallback` (fail before Flyte
mirror). The latter calls `flyte.Checkpoint.save_sync` on HF `on_save` (sync callback).

**Final persist:** `await cp.save(output_dir)` after training, matching Hugging Face / PyTorch task examples.

**Hardware:** Unsloth requires an **NVIDIA, AMD, or Intel GPU** (not Apple Silicon / MPS). Imports from
`unsloth` are deferred until the task body.
"""

from __future__ import annotations

import json
import pathlib

try:
    import unsloth  # noqa: F401
except NotImplementedError:
    pass

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_unsloth_sft",
    image=flyte.Image.from_debian_base().with_pip_packages("unsloth"),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu="L4:1"),
)

MODEL_NAME = "unsloth/Llama-3.2-1B-bnb-4bit"
RETRIES = 3


class FailureInjectionCallback(TrainerCallback):
    def __init__(self, chunks_start: int, failure_interval: int) -> None:
        self._chunks_start = chunks_start
        self._failure_interval = failure_interval

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        g = state.epoch
        i = g - 1
        if i > self._chunks_start and i % self._failure_interval == 0:
            raise RuntimeError(
                f"Failed after optimizer step global_step={g} (i={i}), failure_interval {self._failure_interval}."
            )


class FlyteTrainerCheckpointCallback(TrainerCallback):
    def __init__(self, cp: flyte.Checkpoint, output_dir: pathlib.Path) -> None:
        self._cp = cp
        self._output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        self._cp.save_sync(self._output_dir)


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
def train_unsloth_sft(max_epochs: int = 10) -> float:
    assert max_epochs > RETRIES
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    ctx = flyte.ctx()
    assert ctx is not None
    cp = ctx.checkpoint
    assert cp is not None

    ckpt_dir = pathlib.Path("unsloth_sft")
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
        print("prev_cp_path contents")
        for file in prev_cp_path.iterdir():
            print(file.name)

    print(f"chunks_start={chunks_start}, prev_cp_path={prev_cp_path}, hf_resume={hf_resume}, last_ckpt={last_ckpt}")

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

    args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=2,
        logging_steps=1,
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
        return float(logits[0, 1].mean().item())


if __name__ == "__main__":
    flyte.init_from_config()
    try:
        run = flyte.with_runcontext(mode="remote").run(train_unsloth_sft, max_epochs=10)
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
