"""
Batch Inference with TokenBatcher
==================================

Demonstrates how to maximize GPU utilization when running large-scale batch
inference with vLLM inside reusable containers.

The pattern:
1. A **GPU worker** loads the vLLM model **once per process** (via
   ``alru_cache``) and creates a single :class:`~flyte.extras.TokenBatcher`
   that lives for the lifetime of the container.
2. Every invocation of ``infer_batch`` — which may run concurrently thanks
   to ``ReusePolicy(concurrency=10)`` — shares the same model and batcher.
   Records from all concurrent calls are batched together, keeping the GPU
   busy.
3. A **driver** task fans out many ``infer_batch`` calls across replicas.

Key Flyte concepts:
- ``flyte.ReusePolicy`` for persistent GPU workers with multiple replicas
- ``alru_cache`` for process-level singletons (model + batcher)
- ``flyte.extras.TokenBatcher`` for batched inference across concurrent calls
- ``asyncio.create_task`` for concurrent fan-out

Usage::

    flyte run batch_inference_saturate.py main
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx
from async_lru import alru_cache

import flyte
from flyte.extras import TokenBatcher

logger = logging.getLogger(__name__)

image = flyte.Image.from_debian_base().with_pip_packages("vllm", "unionai-reuse")

gpu_env = flyte.TaskEnvironment(
    name="gpu_worker",
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="A10G:1"),
    image=image,
    reusable=flyte.ReusePolicy(
        replicas=2,
        concurrency=10,
    ),
)

driver_env = flyte.TaskEnvironment(
    name="driver",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=image,
    depends_on=[gpu_env],
)


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    """A single inference request."""

    task_id: str
    index: int
    text: str

    def estimate_tokens(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        return len(self.text) // 4 + 1


# ---------------------------------------------------------------------------
# Process-level singletons — loaded once, reused across all task invocations
# ---------------------------------------------------------------------------


@alru_cache(maxsize=1)
async def get_inference_fn():
    """Load the vLLM model once per process and return an inference closure."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )
    params = SamplingParams(temperature=0.7, max_tokens=512)

    async def inference(batch: list[Prompt]) -> list[str]:
        texts = [p.text for p in batch]
        outputs = llm.generate(texts, params)
        return [o.outputs[0].text for o in outputs]

    logger.info("vLLM model loaded")
    return inference


@alru_cache(maxsize=1)
async def get_batcher() -> TokenBatcher[Prompt, str]:
    """Create a single TokenBatcher per process.

    The batcher is started once and shared across all concurrent
    ``infer_batch`` invocations on this replica.
    """
    inference_fn = await get_inference_fn()
    batcher = TokenBatcher[Prompt, str](
        inference_fn=inference_fn,
        target_batch_tokens=32_000,
        max_batch_size=256,
        batch_timeout_s=0.05,
        max_queue_size=5_000,
    )
    await batcher.start()
    logger.info("TokenBatcher started")
    return batcher


# ---------------------------------------------------------------------------
# GPU worker — each call shares the process-level batcher
# ---------------------------------------------------------------------------


@gpu_env.task
async def infer_batch(
    jsonl_url: str,
    task_id: str,
) -> list[str]:
    """Stream a JSONL file and submit each record to the shared batcher.

    Multiple ``infer_batch`` calls run concurrently on the same replica
    (``concurrency=10``).  All of them feed into the same
    :class:`TokenBatcher`, which assembles token-budgeted batches across
    every concurrent caller — maximizing GPU utilization.

    Args:
        jsonl_url: URL of a JSONL file where each line has a ``"prompt"`` field.
        task_id: Identifier for this logical task.

    Returns:
        List of generated outputs, one per JSONL line.
    """
    batcher = await get_batcher()

    futures: list[asyncio.Future[str]] = []
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", jsonl_url) as resp:
            resp.raise_for_status()
            idx = 0
            buffer = ""
            async for chunk in resp.aiter_bytes():
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    record = Prompt(
                        task_id=task_id,
                        index=idx,
                        text=data["prompt"],
                    )
                    future = await batcher.submit(record)
                    futures.append(future)
                    idx += 1

    results = await asyncio.gather(*futures)
    logger.info(
        "[%s] completed %d records | utilization: %.1f%% | Batches: %d",
        task_id,
        len(results),
        batcher.stats.utilization * 100,
        batcher.stats.total_batches,
    )
    return list(results)


# ---------------------------------------------------------------------------
# Orchestrator — fans out many infer_batch calls across GPU replicas
# ---------------------------------------------------------------------------


@driver_env.task
async def main(
    jsonl_urls: list[str] | None = None,
    task_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    """Launch GPU workers and feed them JSONL data.

    Each URL becomes a separate ``infer_batch`` call.  With
    ``ReusePolicy(replicas=2, concurrency=10)``, up to 20 calls run
    concurrently across 2 GPU replicas, all sharing their replica's
    model and batcher.

    Args:
        jsonl_urls: Remote JSONL file URLs (one per logical task).
            Each line must have a ``"prompt"`` field.
        task_ids: Identifiers for each URL (same length as *jsonl_urls*).

    Returns:
        Mapping from task_id to list of generated outputs.
    """
    if jsonl_urls is None:
        jsonl_urls = [
            "https://storage.example.com/prompts_001.jsonl",
            "https://storage.example.com/prompts_002.jsonl",
            "https://storage.example.com/prompts_003.jsonl",
        ]
    if task_ids is None:
        task_ids = [f"task_{i:03d}" for i in range(len(jsonl_urls))]

    all_results = await asyncio.gather(*(infer_batch(url, tid) for url, tid in zip(jsonl_urls, task_ids)))
    return dict(zip(task_ids, all_results))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
