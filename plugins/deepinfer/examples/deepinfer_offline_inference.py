"""Offline inference with DeepInfer on a reusable, warm container.

`DeepInfer` starts one engine per GPU on the first `generate` call and keeps them
running for the life of the container. With `reusable=`, the container (and therefore
the loaded model) survives between task invocations, so only the first call pays for
loading the weights and warming up CUDA.

    flyte run examples/deepinfer_offline_inference.py complete --prompts '["Hello!", "Tell me a joke."]'
"""

import flyte

from flyteplugins.deepinfer import DeepInfer, deepinfer_image

# --- Knobs ----------------------------------------------------------------------------
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
GPU = "B200:8"  # DeepInfer supports NVIDIA SM100 (B200 / GB200)

env = flyte.TaskEnvironment(
    name="deepinfer-offline",
    resources=flyte.Resources(gpu=GPU),
    # Reusable environments need `unionai-reuse` in the image.
    image=deepinfer_image().with_pip_packages("unionai-reuse"),
    reusable=flyte.ReusePolicy(replicas=(1, 2), idle_ttl=600),
)

# Cheap to construct: nothing starts until the first `generate`.
llm = DeepInfer(MODEL, gpu_memory_utilization=0.9)


@env.task
async def complete(prompts: list[str], temperature: float = 0.6, max_tokens: int = 512) -> list[str]:
    """One completion per prompt, in input order."""
    return await llm.generate.aio(prompts, temperature=temperature, max_tokens=max_tokens)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(complete, prompts=["Hello!", "Tell me a joke."])
    print("Run URL:", run.url)
