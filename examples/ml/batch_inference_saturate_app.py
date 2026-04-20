"""
Batch Inference with FastAPI and TokenBatcher (Flyte App Environment)
====================================================================

Demonstrates how to expose a batched inference endpoint using `FastAPI` within a `flyte.App` environment, while still
using a `flyte.TaskEnvironment` for the main orchestrator.
This pattern is ideal for serving models where multiple concurrent requests can be
aggregated into a single large batch to maximize GPU throughput via `TokenBatcher`.

The pattern:
1. A **Flyte App** runs in a persistent container (via `flyte.App`) containing the FastAPI server and vLLM model.
2. The app initializes the vLLM model and `TokenBatcher` once on startup via FastAPI's lifespan.
3. An asynchronous FastAPI endpoint receives inference requests and submits them to the shared `TokenBatcher`.
4. A **Driver Task** (running in a `flyte.TaskEnvironment`) acts as the orchestrator, which could theoretically trigger
   other workflows or manage resources.

Key Flyte concepts:
- ``flyte.App`` for running long-lived, persistent services (e.g., FastAPI servers).
- ``flyte.TaskEnvironment`` to define the infrastructure for tasks.
- ``flyte.extras.TokenBatcher`` for efficient request batching across concurrent HTTP calls.

Usage:
    flyte run batch_inference_fastapi.py main
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment
from flyte.extras import TokenBatcher

logger = logging.getLogger(__name__)

# Image definition with necessary packages for vLLM and batching
image = (
    flyte.Image.from_debian_base()
    .with_pip_packages("vllm", "hf-transfer", "fastapi", "uvicorn")
    .with_pip_packages("flashinfer-python==0.6.4", "flashinfer-cubin==0.6.4")
    .with_pip_packages("flashinfer-jit-cache==0.6.4", index_url="https://flashinfer.ai/whl/cu129")
    .with_env_vars({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the vLLM model and start the TokenBatcher once at startup."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )
    params = SamplingParams(temperature=0.7, max_tokens=512)
    logger.info("vLLM model loaded")

    async def inference(batch: List[Prompt]) -> List[str]:
        texts = [p.text for p in batch]
        outputs = llm.generate(texts, params)
        return [o.outputs[0].text for o in outputs]

    batcher = TokenBatcher[Prompt, str](
        inference_fn=inference,
        target_batch_tokens=32_000,
        max_batch_size=256,
        batch_timeout_s=0.05,
        max_queue_size=5_000,
    )
    await batcher.start()
    logger.info("TokenBatcher started")

    app.state.batcher = batcher
    yield
    await batcher.stop()


app = FastAPI(title="Flyte Batched Inference Service", lifespan=lifespan)

# Define the environment for our FastAPI app (The GPU Worker)
app_env = FastAPIAppEnvironment(
    name="batch_inference_saturate_app",
    app=app,
    image=image,
    resources=flyte.Resources(cpu=6, memory="24Gi", gpu="L4:1", disk="64Gi"),
    scaling=flyte.app.Scaling(
        replicas=(0, 2),
        metric=flyte.app.Scaling.Concurrency(val=10),
        scaledown_after=300,
    ),
    requires_auth=False,
)

# Define the environment for the driver
driver_env = flyte.TaskEnvironment(
    name="batch_inference_saturate_app_driver",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=image,
    depends_on=[app_env],
)

# HuggingFace Datasets API endpoint for gsm8k (math word problems).
# Public, no auth needed.  Each row has a "question" and "answer" field.
GSM8K_URL = "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset={offset}&length={length}"

# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    """A single inference request record."""

    task_id: str
    index: int
    text: str


# ---------------------------------------------------------------------------
# FastAPI Application Setup (Runs inside the app_env)
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompts: List[str]
    task_id: str


@app.post("/generate")
async def generate(request_body: GenerateRequest, request: Request):
    """
    Endpoint to receive a list of prompts and return completions.
    Uses TokenBatcher to aggregate these requests with other concurrent HTTP calls.
    """
    prompts = request_body.prompts
    task_id = request_body.task_id

    if not prompts:
        raise HTTPException(status_code=400, detail="No prompts provided")

    batcher: TokenBatcher[Prompt, str] = request.app.state.batcher

    futures: List[asyncio.Future[str]] = []
    for idx, text in enumerate(prompts):
        record = Prompt(task_id=task_id, index=int(idx), text=text)
        future = await batcher.submit(record)
        futures.append(future)

    results = await asyncio.gather(*futures)

    logger.info(
        "[%s] Processed %d prompts | utilization: %.1f%% | Total Batches: %d",
        task_id,
        len(results),
        batcher.stats.utilization * 100,
        batcher.stats.total_batches,
    )
    return {"results": results}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Orchestrator Task (The driver logic in a task env)
# ---------------------------------------------------------------------------


@driver_env.task(retries=5)
async def infer_batch(
    endpoint: str,
    prompts: List[str],
    task_id: str,
) -> dict[str, List[str]]:
    """
    A driver task that calls the FastAPI inference service.
    In a real scenario, this would be triggered as part of a workflow.
    """
    url = f"{endpoint}/generate"
    print(f"Calling app at {url}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(1200.0)) as client:
        response = await client.post(
            url,
            json={"prompts": prompts, "task_id": task_id},
        )
        response.raise_for_status()
        return response.json()


async def fetch_gsm8k_questions(n: int = 500) -> list[str]:
    """Fetch math word problems from the HuggingFace gsm8k dataset.

    Uses the HuggingFace Datasets Server API (public, no auth required).
    Paginates automatically when more than 100 questions are requested.

    Args:
        n: Number of questions to fetch.

    Returns:
        List of question strings.
    """
    questions: list[str] = []
    async with httpx.AsyncClient() as client:
        offset = 0
        while offset < n:
            length = min(n - offset, 100)
            resp = await client.get(
                GSM8K_URL.format(offset=offset, length=length),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            questions.extend(row["row"]["question"] for row in data["rows"])
            offset += length
    return questions


@driver_env.task
async def main(
    num_questions: int = 500,
    chunk_size: int = 50,
) -> dict[str, list[str]]:
    """Fetch gsm8k math problems and solve them with batched LLM inference.

    Downloads questions from the HuggingFace ``openai/gsm8k`` dataset,
    splits them into chunks, and fans each chunk out as a separate
    ``infer_batch`` call.  With the AppEnvironment, up to 20 calls run
    concurrently across 2 GPU replicas, all sharing their replica's model and batcher.

    Args:
        num_questions: Total questions to fetch from gsm8k.
        chunk_size: Number of questions per ``infer_batch`` call.

    Returns:
        Mapping from task_id to list of generated answers.
    """
    questions = await fetch_gsm8k_questions(num_questions)
    logger.info("Fetched %d questions from gsm8k", len(questions))

    endpoint = app_env.endpoint
    logger.info("Resolved app endpoint: %s", endpoint)

    # Split into chunks → one infer_batch call per chunk
    chunks = [questions[i : i + chunk_size] for i in range(0, len(questions), chunk_size)]
    task_ids = [f"gsm8k_{i:03d}" for i in range(len(chunks))]

    all_results = await asyncio.gather(*(infer_batch(endpoint, chunk, tid) for chunk, tid in zip(chunks, task_ids)))
    return dict(zip(task_ids, all_results))


if __name__ == "__main__":
    flyte.init_from_config()
    # We run the 'main' task from the driver environment with some sample prompts.
    app = flyte.serve(app_env)
    print(app.url)
    print("activating app")
    app.activate(wait=True)
    run = flyte.run(main, num_questions=5000, chunk_size=50)
    print(run.url)
