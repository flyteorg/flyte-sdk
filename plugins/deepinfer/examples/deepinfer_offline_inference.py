"""Run DeepSeek V4.1-Flash offline inference with DeepInfer as a Flyte task.

Demonstrates the flyteplugins-deepinfer plugin: create a task environment, build a
DeepInfer engine inside the task body (`create_llm`), sample with structured-output
support, and emit the generated texts.

Requires an image containing a built `deepinfer` package (see the plugin README) on
NVIDIA SM100 GPUs:

    flyte run examples/deepinfer_offline_inference.py score --prompts '["Hello!", "Tell me a joke."]'

"""

import flyte

from flyteplugins.deepinfer import (
    create_llm,
    default_deepinfer_image,
    sampling_params,
    structured_output,
)

env = flyte.TaskEnvironment(
    name="deepinfer-batch",
    image=default_deepinfer_image(),
)


@env.task(
    resources=flyte.Resources(
        cpu="48",
        memory="512Gi",
        # Integer GPU count (accelerator-agnostic); use your platform's SM100 / Ascend
        # accelerator token if one is exposed, e.g. gpu="SM100:8". One engine below.
        gpu=8,
    )
)
async def score(prompts: list[str]) -> list[str]:
    # Environment (visible devices, runtime flags) is configured, the model resolved,
    # and deepinfer imported — in the correct order, inside one call.
    llm = create_llm(
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        engine_rank=0,
        ep_size=8,
        gpu_memory_utilization=0.9,
    )
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
