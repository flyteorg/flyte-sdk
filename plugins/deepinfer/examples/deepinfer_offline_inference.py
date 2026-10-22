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
    default_deepinfer_image,
    sampling_params,
    structured_output,
)

env = flyte.TaskEnvironment(
    name="deepinfer-batch",
    # Warn containers so repeated invocations skip a cold start. 2+ replicas avoid
    # starvation (1 replica can be busy handling the invoking task).
    reusable=flyte.ReusePolicy(
        replicas=(2, 4),
        idle_ttl=300,
    ),
    resources=flyte.Resources(
        cpu="48",
        memory="512Gi",
        # Integer GPU count (accelerator-agnostic); use your platform's SM100 / Ascend
        # accelerator token if one is exposed, e.g. gpu="SM100:8". One engine below.
        gpu=8,
    ),
    image=default_deepinfer_image(),
)


@functools.lru_cache(maxsize=1)
def _build_llm(model: str) -> "object":
    """Construct the DeepInfer engine once per warm container.

    The keyed args define one engine; on a reused container (same Python process)
    subsequent calls return the already-loaded `LLM`, so model weights and the CUDA
    warm-up are not re-done for every task invocation.
    """
    return create_llm(
        model,
        engine_rank=0,
        ep_size=8,
        gpu_memory_utilization=0.9,
    )


@env.task
async def score(prompts: list[str]) -> list[str]:
    # Env + visible devices are configured, the model resolved, and deepinfer imported
    # in the correct order inside create_llm -- which only runs on a cold container.
    llm = _build_llm("deepseek-ai/DeepSeek-V4.1-Flash-1017")
    params = sampling_params(
        temperature=0,
        max_tokens=256,
        so_args=structured_output("regex", r"[A-Z]{2}-[0-9]{3}"),
    )
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    tokenizer = llm.get_tokenizer()
    return [
        tokenizer.decode(
            [tok.token_id for tok in output[:-1]],
            skip_special_tokens=False,
        )
        for output in outputs
    ]


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(score)
