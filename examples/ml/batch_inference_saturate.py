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
3. A **driver** task fetches problems from the HuggingFace ``gsm8k`` math
   dataset and fans them out across GPU replicas.

Key Flyte concepts:
- ``flyte.ReusePolicy`` for persistent GPU workers with multiple replicas
- ``alru_cache`` for process-level singletons (model + batcher)
- ``flyte.extras.TokenBatcher`` for batched inference across concurrent calls
- ``asyncio.create_task`` for concurrent fan-out

Startup optimizations:
- **hf-transfer** — Rust-based parallel downloader for 5-10x faster
  HuggingFace model downloads (enabled via ``HF_HUB_ENABLE_HF_TRANSFER``).
- **FlashInfer** — High-performance attention backend for vLLM, faster
  than the default FlashAttention for decode-heavy workloads.

Usage::

    flyte run batch_inference_saturate.py main
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from async_lru import alru_cache

import flyte
from flyte.extras import TokenBatcher

logger = logging.getLogger(__name__)

image = (
    flyte.Image.from_debian_base()
    .with_pip_packages("vllm", "hf-transfer", "unionai-reuse")
    .with_pip_packages("flashinfer-python==0.6.4", "flashinfer-cubin==0.6.4")
    .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu129")
    .with_env_vars({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

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

# HuggingFace Datasets API endpoint for gsm8k (math word problems).
# Public, no auth needed.  Each row has a "question" and "answer" field.
GSM8K_URL = "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset={offset}&length={length}"


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
    prompts: list[str],
    task_id: str,
) -> list[str]:
    """Submit prompts to the shared batcher and return completions.

    Multiple ``infer_batch`` calls run concurrently on the same replica
    (``concurrency=10``).  All of them feed into the same
    :class:`TokenBatcher`, which assembles token-budgeted batches across
    every concurrent caller — maximizing GPU utilization.

    Args:
        prompts: List of prompt strings to generate completions for.
        task_id: Identifier for this logical task.

    Returns:
        List of generated outputs, one per prompt.
    """
    batcher = await get_batcher()

    futures: list[asyncio.Future[str]] = []
    for idx, text in enumerate(prompts):
        record = Prompt(task_id=task_id, index=idx, text=text)
        future = await batcher.submit(record)
        futures.append(future)

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
# Orchestrator — fetches data and fans out across GPU replicas
# ---------------------------------------------------------------------------


async def fetch_gsm8k_questions(n: int = 100) -> list[str]:
    """Fetch math word problems from the HuggingFace gsm8k dataset.

    Uses the HuggingFace Datasets Server API (public, no auth required).

    Args:
        n: Number of questions to fetch (max 100 per request).

    Returns:
        List of question strings.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            GSM8K_URL.format(offset=0, length=min(n, 100)),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    return [row["row"]["question"] for row in data["rows"]]


@driver_env.task
async def main(
    num_questions: int = 60,
    chunk_size: int = 20,
) -> dict[str, list[str]]:
    """Fetch gsm8k math problems and solve them with batched LLM inference.

    Downloads questions from the HuggingFace ``openai/gsm8k`` dataset,
    splits them into chunks, and fans each chunk out as a separate
    ``infer_batch`` call.  With ``ReusePolicy(replicas=2, concurrency=10)``,
    up to 20 calls run concurrently across 2 GPU replicas, all sharing
    their replica's model and batcher.

    Args:
        num_questions: Total questions to fetch from gsm8k (max 100).
        chunk_size: Number of questions per ``infer_batch`` call.

    Returns:
        Mapping from task_id to list of generated answers.
    """
    questions = await fetch_gsm8k_questions(num_questions)
    logger.info("Fetched %d questions from gsm8k", len(questions))

    # Split into chunks → one infer_batch call per chunk
    chunks = [questions[i : i + chunk_size] for i in range(0, len(questions), chunk_size)]
    task_ids = [f"gsm8k_{i:03d}" for i in range(len(chunks))]

    all_results = await asyncio.gather(*(infer_batch(chunk, tid) for chunk, tid in zip(chunks, task_ids)))
    return dict(zip(task_ids, all_results))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
