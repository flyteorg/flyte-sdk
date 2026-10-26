"""End-to-end batch pipeline: cache the checkpoint once, fan out over independent GPU nodes.

When the model fits on one node, the simplest way to scale past it is to run several
*independent* single-node jobs in parallel rather than one multi-node job: each shard
is retried, cached, and scheduled on its own. This pipeline:

1. downloads the checkpoint from the Hugging Face hub **once** into a cached
   `flyte.io.Dir`, so shards (and later runs) read it from your object store instead of
   the hub;
2. reads prompts from a JSONL `flyte.io.File` and splits them into `num_shards`;
3. scores the shards concurrently, each on its own 8-GPU node with one engine per GPU;
4. writes the completions back out as a JSONL `File`, in input order.

    flyte run examples/deepinfer_batch_pipeline.py pipeline --prompts_file prompts.jsonl --num_shards 4

Each line of `prompts.jsonl` is `{"prompt": "..."}`.
"""

import asyncio
import functools
import json
import os
import tempfile

import flyte
import flyte.io

from flyteplugins.deepinfer import (
    create_llm,
    deepinfer_image_from_source,
    get_actual_num_gpus,
    run_engines,
    sampling_params,
    setup_common_env,
)

image = deepinfer_image_from_source()

# The checkpoint download only needs network and disk, not GPUs. Size `disk` to your
# checkpoint.
download_env = flyte.TaskEnvironment(
    name="deepinfer-download",
    resources=flyte.Resources(cpu=8, memory="32Gi", disk="1Ti"),
    image=image,
    secrets=flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    cache="auto",
)

gpu_env = flyte.TaskEnvironment(
    name="deepinfer-shard",
    resources=flyte.Resources(gpu=8, disk="1Ti"),
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
    """Snapshot an HF model into the object store; cached on `model`."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(model, local_dir=os.path.join(tempfile.mkdtemp(), "model"))
    return await flyte.io.Dir.from_local(local_dir)


def _generate_shard(model_path: str, prompts: list[str], params: dict, dp_size: int, dp_rank: int, out) -> None:
    # Module-level so it pickles into each spawned engine process.
    from deepinfer.sampling_params import SamplingParams

    llm = create_llm(model_path, dp_rank)
    indices = list(range(dp_rank, len(prompts), dp_size))
    if indices:
        outputs = llm.generate(
            prompts=[prompts[i] for i in indices],
            sampling_params=[SamplingParams(**params) for _ in indices],
            use_tqdm=False,
        )
        tokenizer = llm.get_tokenizer()
        out.extend(
            (i, tokenizer.decode([tok.token_id for tok in output], skip_special_tokens=True))
            for i, output in zip(indices, outputs)
        )
    llm.close()


@gpu_env.task
async def score_shard(checkpoint: flyte.io.Dir, prompts: list[str], max_tokens: int) -> list[str]:
    """Score one shard on one node, one DeepInfer engine per GPU."""
    model_path = await checkpoint.download()  # a local dir with config.json
    params = sampling_params(temperature=0, max_tokens=max_tokens)
    results = run_engines(
        functools.partial(_generate_shard, model_path, prompts, params),
        dp_size=get_actual_num_gpus(),
        initializer=setup_common_env,
    )
    texts = dict(results)
    return [texts[i] for i in range(len(prompts))]


@driver.task
async def pipeline(
    prompts_file: flyte.io.File,
    model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    num_shards: int = 2,
    max_tokens: int = 512,
) -> flyte.io.File:
    checkpoint = await download_checkpoint(model)

    async with prompts_file.open("rb") as f:
        lines = bytes(await f.read()).decode("utf-8").splitlines()
    prompts = [json.loads(line)["prompt"] for line in lines if line.strip()]

    # Contiguous shards keep input order trivial to restore when concatenating.
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
