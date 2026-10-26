"""Run DeepSeek V4.1-Flash offline inference with DeepInfer as a Flyte task.

Demonstrates the flyteplugins-deepinfer plugin on a **reusable** task environment:
`reusable=flyte.ReusePolicy(...)` keeps a warm pool of containers alive across task
invocations, so the DeepInfer engine (a heavy model + CUDA warm-up) is loaded only
once per container instead of on every call. The engine is cached with
`functools.lru_cache`; because the container's Python process persists between
invocations, cache hits skip `create_llm` entirely.

Requires an image containing a built `deepinfer` package (see the plugin README) on
NVIDIA SM100 GPUs:

    flyte run examples/deepinfer_offline_inference.py score --prompts '["Hello!", "Tell me a joke."]'

"""

import functools

import flyte

from flyteplugins.deepinfer import (
    create_llm,
    deepinfer_image_from_source,
    get_actual_num_gpus,
    sampling_params,
    setup_common_env,
    structured_output,
)

env = flyte.TaskEnvironment(
    name="deepinfer-batch",
    # Warm containers so repeated invocations skip a cold start. 2+ replicas avoid
    # starvation (1 replica can be busy handling the invoking task).
    reusable=flyte.ReusePolicy(
        replicas=(2, 4),
        idle_ttl=300,
    ),
    resources=flyte.Resources(
        # Integer GPU count (accelerator-agnostic); use your platform's SM100 / Ascend
        # accelerator token if one is exposed, e.g. gpu="SM100:8". One engine below.
        gpu=8,
    ),
    # default_deepinfer_image() has the toolchain but not DeepInfer itself; this builds
    # it from source. Use a prebuilt image to skip that (see deepinfer_multinode.py).
    image=deepinfer_image_from_source(),
)


@functools.lru_cache(maxsize=1)
def _build_llm(model: str) -> "object":
    """Construct the DeepInfer engine once per warm container.

    The keyed args define one engine; on a reused container (same Python process)
    subsequent calls return the already-loaded `LLM`, so model weights and the CUDA
    warm-up are not re-done for every task invocation.
    """
    # One engine over every visible GPU (context parallel), so it can live in this
    # process and be cached. For one engine per GPU, see deepinfer_data_parallel.py.
    setup_common_env()
    return create_llm(model, cp_size=get_actual_num_gpus(), gpu_memory_utilization=0.9)


@env.task
async def score(prompts: list[str], model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017") -> list[str]:
    # Env + visible devices are configured, the model resolved, and deepinfer imported
    # in the correct order inside create_llm -- which only runs on a cold container.
    from deepinfer.sampling_params import SamplingParams

    llm = _build_llm(model)
    params = SamplingParams(
        **sampling_params(
            temperature=0,
            max_tokens=256,
            so_args=structured_output("regex", r"[A-Z]{2}-[0-9]{3}"),
        )
    )
    outputs = llm.generate(prompts=prompts, sampling_params=[params] * len(prompts), use_tqdm=False)
    tokenizer = llm.get_tokenizer()
    # Each output is a list of per-token results; constrained outputs end with EOS.
    return [tokenizer.decode([tok.token_id for tok in output[:-1]], skip_special_tokens=False) for output in outputs]


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(score, prompts=["Hello!", "Tell me a joke."])
    print("Run URL:", run.url)
