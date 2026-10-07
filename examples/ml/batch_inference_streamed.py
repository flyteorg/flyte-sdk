"""Batch LLM inference on reusable GPU workers, with weights streamed straight into GPU memory.

This is the pattern of ``batch_inference_saturate.py`` (and the batch-inference guide,
https://www.union.ai/docs/v2/union/user-guide/run-scaling/batch-inference/): a pool of
reusable GPU containers, each loading its model once in an ``alru_cache``d bootstrap
function, and a ``TokenBatcher`` packing prompts from every concurrent task on that
replica into full GPU batches.

What changes is how the model loads. Instead of ``LLM(model="Qwen/...")``, which
downloads every weight file to local disk and then reads it back, each replica calls
``flyteplugins.vllm.model_streamer.engine_args()`` (from ``flyteplugins-vllm``). That
downloads only the config and tokenizer, and selects the ``flyte-streaming`` load format,
which fetches the safetensors from object storage as parallel byte ranges and hands each
tensor to vLLM's weight loader the moment it lands. Weights go from the bucket to the GPU
without touching local disk, loading overlaps the download, and the container needs
neither disk nor host memory for a full copy of the model.

The model comes from your object store, not the Hugging Face Hub:
``flyte.prefetch.hf_model()`` copies it there once and publishes it as a model artifact,
and the run binds that artifact to ``main``'s ``model`` input.

Run (prefetches the model, then launches the batch run):
    python batch_inference_streamed.py

Or, once the artifact exists:
    flyte run batch_inference_streamed.py main --model <artifact or directory uri>
"""

import asyncio
import logging
from dataclasses import dataclass

from async_lru import alru_cache
from flyteplugins.vllm import DEFAULT_VLLM_IMAGE

import flyte
from flyte.extras import TokenBatcher
from flyte.io import Dir

logger = logging.getLogger(__name__)

MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
ARTIFACT_NAME = "Qwen2-5-7B-Instruct"

# The vLLM plugin's image (a vLLM the streaming loader supports, plus matching FlashInfer
# kernels), extended with what the reusable workers need.
image = DEFAULT_VLLM_IMAGE.clone(name="vllm-batch-streamed").with_pip_packages("async-lru", "unionai-reuse")

gpu_env = flyte.TaskEnvironment(
    name="streamed_gpu_worker",
    # No disk for the weights and no host memory for a full copy: they stream to the GPU.
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="A10G:1"),
    image=image,
    reusable=flyte.ReusePolicy(
        replicas=2,  # 2 GPU replicas, each loading the model once
        concurrency=10,  # 10 concurrent infer_batch calls per replica, feeding one batcher
    ),
)

driver_env = flyte.TaskEnvironment(
    name="streamed_driver",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=image,
    depends_on=[gpu_env],
)


@dataclass
class Prompt:
    task_id: str
    index: int
    text: str

    def estimate_tokens(self) -> int:
        return len(self.text) // 4 + 1


@alru_cache(maxsize=1)
async def get_inference_fn(model_path: str):
    """Load the model once per container, streaming its weights into the GPU."""
    from flyteplugins.vllm.model_streamer import engine_args
    from vllm import LLM, SamplingParams

    # Config + tokenizer to a local dir; weights stay remote until the engine streams them.
    args = await engine_args(model_path)
    # LLM() blocks for the whole load; keep it off the event loop so the replica's
    # other concurrent tasks (and its heartbeats) keep running meanwhile.
    llm = await asyncio.to_thread(LLM, **args, gpu_memory_utilization=0.9, max_model_len=4096)
    params = SamplingParams(temperature=0.7, max_tokens=512)

    async def inference(batch: list[Prompt]) -> list[str]:
        # The batcher runs one batch at a time, so the engine is never entered concurrently.
        outputs = await asyncio.to_thread(llm.generate, [p.text for p in batch], params)
        return [o.outputs[0].text for o in outputs]

    return inference


@alru_cache(maxsize=1)
async def get_batcher(model_path: str) -> TokenBatcher[Prompt, str]:
    """One batcher per container, shared by every concurrent task on it."""
    batcher = TokenBatcher[Prompt, str](
        inference_fn=await get_inference_fn(model_path),
        target_batch_tokens=32_000,
        max_batch_size=256,
        batch_timeout_s=0.05,
        max_queue_size=5_000,
    )
    await batcher.start()
    return batcher


@gpu_env.task
async def infer_batch(model_path: str, prompts: list[str], task_id: str) -> list[str]:
    """Submit prompts to this replica's shared batcher and return the completions."""
    batcher = await get_batcher(model_path)
    futures = [await batcher.submit(Prompt(task_id=task_id, index=i, text=t)) for i, t in enumerate(prompts)]
    results = await asyncio.gather(*futures)
    logger.info(
        "[%s] completed %d records | utilization: %.1f%% | batches: %d",
        task_id,
        len(results),
        batcher.stats.utilization * 100,
        batcher.stats.total_batches,
    )
    return list(results)


QUESTIONS = [
    "Explain how a hash map handles collisions.",
    "What is the difference between a process and a thread?",
    "Summarize the causes of the French Revolution in three sentences.",
    "Write a haiku about object storage.",
    "Why does the sky look blue?",
    "Give two tips for writing readable Python.",
    "What does a GPU do better than a CPU, and why?",
    "Describe the water cycle to a ten-year-old.",
]


@driver_env.task
async def main(model: Dir, num_questions: int = 500, chunk_size: int = 50) -> dict[str, list[str]]:
    """Fan prompts out across the GPU replicas, each streaming `model` into its GPU once."""
    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(num_questions)]

    chunks = [questions[i : i + chunk_size] for i in range(0, len(questions), chunk_size)]
    task_ids = [f"chunk_{i:03d}" for i in range(len(chunks))]
    results = await asyncio.gather(*(infer_batch(model.path, chunk, tid) for chunk, tid in zip(chunks, task_ids)))
    return dict(zip(task_ids, results))


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Artifact

    flyte.init_from_config()

    # Copy the model into the object store once and publish it as a model artifact.
    prefetch = flyte.prefetch.hf_model(repo=MODEL_REPO, artifact_name=ARTIFACT_NAME, hf_token_key=None)
    prefetch.wait()

    run = flyte.run(main, model=Artifact.get(ARTIFACT_NAME))
    print(run.url)
