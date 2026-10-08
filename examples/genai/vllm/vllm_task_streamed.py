"""
Run vLLM inside a task, with the weights streamed from object storage into the GPU.

The app examples in this directory (`vllm_app.py`, ...) serve a model behind an
OpenAI-compatible endpoint, and `VLLMAppEnvironment(stream_model=True)` streams its
weights. This is the same streaming for vLLM used *inside a task*: offline generation
with `vllm.LLM` in a reusable GPU container, where the engine is built once per
container by an `alru_cache`d bootstrap and shared by every task call it serves.

```python
from flyteplugins.vllm.model_streamer import engine_args

llm = LLM(**await engine_args(model.path), max_model_len=4096)
```

`engine_args()` downloads only the config and tokenizer and selects the
`flyte-streaming` load format. The engine then reads the safetensors from object
storage as parallel byte ranges and loads each tensor into GPU memory as it lands;
nothing is written to local disk. The load format is registered with vLLM through an
entry point of `flyteplugins-vllm`, in every process vLLM starts.

The model is `HuggingFaceTB/SmolLM2-135M-Instruct` (~270MB), small enough for any GPU.
For a batched, multi-replica version see `examples/ml/batch_inference_streamed.py`.

Run (prefetches the model into a model artifact, then generates):

```
python examples/genai/vllm/vllm_task_streamed.py
```
"""

import asyncio

from async_lru import alru_cache
from flyteplugins.vllm import DEFAULT_VLLM_IMAGE

import flyte
from flyte.io import Dir

MODEL_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
ARTIFACT_NAME = "SmolLM2-135M-Instruct"

env = flyte.TaskEnvironment(
    name="vllm_task_streamed",
    image=DEFAULT_VLLM_IMAGE.clone(name="vllm-task-streamed").with_pip_packages("async-lru", "unionai-reuse"),
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="L4:1"),
    # One warm container: the first call loads the engine, later calls reuse it.
    reusable=flyte.ReusePolicy(replicas=1, concurrency=4, idle_ttl=600),
)


@alru_cache(maxsize=1)
async def get_llm(model_path: str):
    """Build the engine once per container, streaming its weights into the GPU."""
    from flyteplugins.vllm.model_streamer import engine_args

    from vllm import LLM

    args = await engine_args(model_path)
    # LLM() blocks for the whole load; run it off the event loop.
    return await asyncio.to_thread(LLM, **args, gpu_memory_utilization=0.8, max_model_len=4096)


# Concurrent calls on one replica share the engine; vLLM's offline LLM is not re-entrant.
_generate_lock = asyncio.Lock()


@env.task
async def generate(model: Dir, prompts: list[str], max_tokens: int = 128) -> list[str]:
    from vllm import SamplingParams

    llm = await get_llm(model.path)
    messages = [[{"role": "user", "content": p}] for p in prompts]
    async with _generate_lock:
        outputs = await asyncio.to_thread(llm.chat, messages, SamplingParams(temperature=0.2, max_tokens=max_tokens))
    return [o.outputs[0].text for o in outputs]


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Artifact

    flyte.init_from_config()

    # Copy the model into the object store once and publish it as a model artifact.
    prefetch = flyte.prefetch.hf_model(repo=MODEL_REPO, artifact_name=ARTIFACT_NAME, hf_token_key=None)
    prefetch.wait()

    run = flyte.run(
        generate,
        model=Artifact.get(ARTIFACT_NAME),
        prompts=["What is object storage?", "Write a haiku about GPUs."],
    )
    print(run.url)
