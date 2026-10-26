"""Single-node data-parallel batch inference: one DeepInfer engine per GPU.

DeepInfer's highest-throughput layout on one node is *data parallelism*: each GPU runs
its own engine process (its own `CUDA_VISIBLE_DEVICES` slice), and the MoE experts are
shared across all of them through expert parallelism. `run_engines` spawns one engine
per rank, and `create_llm` defaults `ep_size` to every visible GPU and
`data_parallel_size` to one engine per GPU, so the task body only decides how to split
the prompts.

Use this when you have a large, one-off batch to push through a single 8-GPU node. For
repeated small batches on warm containers, see `deepinfer_offline_inference.py`; for
more than one node, see `deepinfer_batch_pipeline.py` and `deepinfer_multinode.py`.

    flyte run examples/deepinfer_data_parallel.py batch_generate \\
        --prompts '["Summarize MoE in one line.", "Write a haiku about GPUs."]'
"""

import functools

import flyte

from flyteplugins.deepinfer import (
    create_llm,
    deepinfer_image_from_source,
    get_actual_num_gpus,
    run_engines,
    sampling_params,
    setup_common_env,
)

env = flyte.TaskEnvironment(
    name="deepinfer-data-parallel",
    # Swap in your platform's accelerator token if one is exposed, e.g. gpu="SM100:8".
    resources=flyte.Resources(gpu=8),
    # Builds DeepInfer from source; point at a prebuilt image instead to skip the build
    # (see deepinfer_multinode.py).
    image=deepinfer_image_from_source(),
)


def generate_shard(model: str, prompts: list[str], params: dict, dp_size: int, dp_rank: int, out) -> None:
    """Run one engine over its strided shard of `prompts` and emit `(index, text)` pairs.

    Runs in a `spawn`ed child process, so it lives at module level and receives its
    inputs through `functools.partial` (closures cannot be pickled across processes).
    """
    from deepinfer.sampling_params import SamplingParams

    llm = create_llm(model, dp_rank, gpu_memory_utilization=0.85)
    indices = list(range(dp_rank, len(prompts), dp_size))
    if not indices:
        llm.close()
        return
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


@env.task
async def batch_generate(
    prompts: list[str],
    model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    temperature: float = 0.6,
    max_tokens: int = 512,
) -> list[str]:
    """Generate one completion per prompt, in input order."""
    params = sampling_params(temperature=temperature, max_tokens=max_tokens)
    results = run_engines(
        functools.partial(generate_shard, model, prompts, params),
        dp_size=get_actual_num_gpus(),
        # Runtime flags (NextN decode, EPLB, ...) are set in each engine process before
        # deepinfer is imported; pass keyword overrides with functools.partial if needed.
        initializer=setup_common_env,
    )
    texts = dict(results)
    return [texts[i] for i in range(len(prompts))]


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(batch_generate, prompts=["Summarize MoE in one line.", "Write a haiku about GPUs."])
    print("Run URL:", run.url)
