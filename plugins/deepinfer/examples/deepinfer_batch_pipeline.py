"""Batch pipeline: cache the checkpoint once, then fan out over independent GPU nodes.

When the model fits on one node, the simplest way to scale past it is several
*independent* single-node jobs rather than one multi-node job: each shard is scheduled,
retried, and cached on its own. This pipeline:

1. downloads the checkpoint from the Hugging Face hub **once** into a cached
   `flyte.io.Dir`, so every shard (and every later run) reads it from your object
   store instead of the hub;
2. reads prompts from a JSONL `flyte.io.File` and splits them into `num_shards`;
3. scores the shards concurrently, one GPU node each;
4. writes `{"prompt", "completion"}` records back out as a JSONL `File`, in input order.

    flyte run examples/deepinfer_batch_pipeline.py pipeline --prompts_file prompts.jsonl --num_shards 4

Each line of `prompts.jsonl` is `{"prompt": "..."}`.
"""

import asyncio
import json
import os
import tempfile

import flyte
import flyte.io

from flyteplugins.deepinfer import DeepInfer, deepinfer_image

# --- Knobs ----------------------------------------------------------------------------
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
GPU = "B200:8"
CHECKPOINT_DISK = "1Ti"  # room for the checkpoint on the download and scoring nodes

image = deepinfer_image()

# The download only needs network and disk, not GPUs.
download_env = flyte.TaskEnvironment(
    name="deepinfer-download",
    resources=flyte.Resources(cpu=8, memory="32Gi", disk=CHECKPOINT_DISK),
    image=image,
    cache="auto",
)

gpu_env = flyte.TaskEnvironment(
    name="deepinfer-shard",
    resources=flyte.Resources(gpu=GPU, disk=CHECKPOINT_DISK),
    image=image,
    cache="auto",
)

driver = flyte.TaskEnvironment(
    name="deepinfer-pipeline",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=image,
    depends_on=[download_env, gpu_env],
)


@download_env.task
async def download_checkpoint(model: str) -> flyte.io.Dir:
    """Snapshot a Hugging Face model into the object store; cached on `model`."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(model, local_dir=os.path.join(tempfile.mkdtemp(), "model"))
    return await flyte.io.Dir.from_local(local_dir)


@gpu_env.task
async def score_shard(checkpoint: flyte.io.Dir, prompts: list[str], max_tokens: int) -> list[str]:
    """Score one shard on one node: `DeepInfer` downloads the checkpoint and runs one engine per GPU."""
    with DeepInfer(checkpoint) as llm:
        return await llm.generate.aio(prompts, temperature=0, max_tokens=max_tokens)


@driver.task
async def pipeline(
    prompts_file: flyte.io.File,
    model: str = MODEL,
    num_shards: int = 2,
    max_tokens: int = 512,
) -> flyte.io.File:
    checkpoint = await download_checkpoint(model)

    async with prompts_file.open("rb") as f:
        lines = bytes(await f.read()).decode("utf-8").splitlines()
    prompts = [json.loads(line)["prompt"] for line in lines if line.strip()]

    # Contiguous shards keep input order trivial to restore.
    size = -(-len(prompts) // num_shards)  # ceil division
    shards = [prompts[i : i + size] for i in range(0, len(prompts), size)]
    with flyte.group("score-shards"):
        completions = await asyncio.gather(*(score_shard(checkpoint, shard, max_tokens) for shard in shards))

    out = flyte.io.File.new_remote(file_name="completions.jsonl")
    async with out.open("wb") as fh:
        for prompt, completion in zip(prompts, (c for shard in completions for c in shard)):
            await fh.write((json.dumps({"prompt": prompt, "completion": completion}) + "\n").encode("utf-8"))
    return out


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pipeline, prompts_file=flyte.io.File.from_local_sync("prompts.jsonl"), num_shards=2)
    print("Run URL:", run.url)
