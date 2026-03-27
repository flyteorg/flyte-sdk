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
Unsloth + TRL ``SFTTrainer`` + Flyte ``AsyncCheckpoint``
=========================================================

LoRA fine-tuning with `Unsloth <https://unsloth.ai/docs>`__ and :class:`trl.trainer.SFTTrainer`,
while persisting the Hugging Face ``output_dir`` (checkpoint shards, adapter config, etc.) through
the task :class:`~flyte.AsyncCheckpoint` so Flyte retries can resume.

Unsloth's docs describe checkpointing with ``save_strategy`` / ``save_steps`` and
``trainer.train(resume_from_checkpoint=...)`` — see
`Finetuning from Last Checkpoint <https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint>`__
and the `continued pretraining / adapter notes
<https://unsloth.ai/docs/basics/continued-pretraining>`__ for loading LoRA adapters.

This example mirrors that pattern, but stores checkpoints under ``checkpoint.path`` and syncs them
to object storage with ``await checkpoint.save.aio(...)``.

**Hardware:** Unsloth currently requires an **NVIDIA, AMD, or Intel GPU** (not Apple Silicon / MPS).
See `Unsloth requirements <https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements>`__.
Imports from ``unsloth`` are deferred until the task body so the script can be parsed on any machine.

**Note:** Editable ``pip install -e .`` for this SDK so ``TaskContext.checkpoint`` exists; see
``generic_data_checkpoint.py``.
"""

from __future__ import annotations

import flyte

env = flyte.TaskEnvironment(name="checkpoint_unsloth_sft")

# Small 4-bit model from the Unsloth Hub (multi-GB download on first run).
MODEL_NAME = "unsloth/Llama-3.2-1B-bnb-4bit"


def _tiny_instruction_dataset():
    """A few instruction/response rows for a smoke fine-tune."""
    from datasets import Dataset

    rows = [
        "### Instruction:\nSay hello.\n\n### Response:\nHello!",
        "### Instruction:\nWhat is 1+1?\n\n### Response:\n2",
        "### Instruction:\nName a color.\n\n### Response:\nBlue",
    ] * 6
    return Dataset.from_dict({"text": rows})


@env.task()
async def train_unsloth_sft(max_steps: int = 12) -> float:
    # Unsloth must be imported before TRL/transformers (see Unsloth docs). Deferred so Mac hosts
    # without a supported GPU can still load this module.
    import unsloth  # noqa: F401
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    tctx = flyte.ctx()
    assert tctx is not None
    checkpoint = tctx.checkpoint
    assert checkpoint is not None

    await checkpoint.load.aio()
    root = checkpoint.path
    output_dir = root / "unsloth_sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = get_last_checkpoint(str(output_dir))

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
        save_total_limit=1,
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
    )
    trainer.train(resume_from_checkpoint=resume_path)

    await checkpoint.save.aio(local_path=output_dir)

    for h in reversed(trainer.state.log_history):
        if "loss" in h:
            return float(h["loss"])
        if "train_loss" in h:
            return float(h["train_loss"])
    return float(max_steps)


if __name__ == "__main__":
    flyte.init_from_config()
    try:
        run = flyte.with_runcontext(mode="local").run(train_unsloth_sft, max_steps=8)
        print(run.outputs())
    except Exception as e:
        msg = str(e)
        if "Unsloth currently only works" in msg or "NVIDIA, AMD and Intel GPUs" in msg:
            raise SystemExit(
                "Unsloth requires an NVIDIA, AMD, or Intel GPU. Run on a supported GPU worker; see "
                "https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements\n"
                f"Details: {msg}"
            ) from e
        raise
