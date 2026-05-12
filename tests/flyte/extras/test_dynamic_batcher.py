"""Tests for flyte.extras DynamicBatcher and TokenBatcher."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from flyte.extras._dynamic_batcher import (
    BatchStats,
    CostEstimator,
    DynamicBatcher,
    Prompt,
    TokenBatcher,
    TokenEstimator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def echo_process(batch: list[str]) -> list[str]:
    """Simple async process fn that echoes inputs."""
    return [f"result:{item}" for item in batch]


async def slow_process(batch: list[str]) -> list[str]:
    await asyncio.sleep(0.05)
    return [f"slow:{item}" for item in batch]


async def failing_process(batch: list[str]) -> list[str]:
    raise RuntimeError("processing exploded")


async def wrong_length_process(batch: list[str]) -> list[str]:
    return ["only_one"]


@dataclass
class CostRecord:
    """Record implementing the CostEstimator protocol."""

    payload: str

    def estimate_cost(self) -> int:
        return len(self.payload)


@dataclass
class TokenRecord:
    """Record implementing the TokenEstimator protocol."""

    text: str

    def estimate_tokens(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# DynamicBatcher tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_submit_and_result():
    """Submit a single record, await the future, get the right result."""
    async with DynamicBatcher(process_fn=echo_process) as batcher:
        future = await batcher.submit("hello")
        result = await future
    assert result == "result:hello"


@pytest.mark.asyncio
async def test_multiple_submits():
    """Submit several records and verify all results arrive correctly."""
    async with DynamicBatcher(
        process_fn=echo_process,
        batch_timeout_s=0.1,
    ) as batcher:
        futures = []
        for i in range(10):
            f = await batcher.submit(f"item-{i}")
            futures.append(f)
        results = await asyncio.gather(*futures)
    assert results == [f"result:item-{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_cost_estimation_explicit_override():
    """Explicit estimated_cost overrides all other estimators."""
    called = False

    def estimator(record: str) -> int:
        nonlocal called
        called = True
        return 999

    async with DynamicBatcher(
        process_fn=echo_process,
        cost_estimator=estimator,
    ) as batcher:
        future = await batcher.submit("hi", estimated_cost=42)
        await future
    assert not called, "estimator should not be called when explicit cost given"


@pytest.mark.asyncio
async def test_cost_estimation_estimator_fn():
    """cost_estimator function is used when no explicit cost given."""
    estimates: list[int] = []

    def estimator(record: str) -> int:
        est = len(record) * 10
        estimates.append(est)
        return est

    async with DynamicBatcher(
        process_fn=echo_process,
        cost_estimator=estimator,
    ) as batcher:
        future = await batcher.submit("hello")
        await future
    assert estimates == [50]


@pytest.mark.asyncio
async def test_cost_estimation_protocol():
    """Records implementing CostEstimator protocol are used automatically."""

    async def record_process(batch: list[CostRecord]) -> list[str]:
        return [r.payload.upper() for r in batch]

    async with DynamicBatcher(process_fn=record_process) as batcher:
        record = CostRecord(payload="hello")
        assert isinstance(record, CostEstimator)
        future = await batcher.submit(record)
        result = await future
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_cost_estimation_default():
    """When no estimator is available, default_cost is used."""
    async with DynamicBatcher(
        process_fn=echo_process,
        default_cost=7,
    ) as batcher:
        future = await batcher.submit("anything")
        await future
    assert batcher.stats.total_completed == 1


@pytest.mark.asyncio
async def test_batch_assembly_respects_target_cost():
    """Batches should not exceed target_batch_cost."""
    batch_sizes: list[int] = []

    async def tracking_process(batch: list[str]) -> list[str]:
        batch_sizes.append(len(batch))
        return [f"r:{x}" for x in batch]

    async with DynamicBatcher(
        process_fn=tracking_process,
        target_batch_cost=5,
        default_cost=2,
        max_batch_size=100,
        batch_timeout_s=0.05,
    ) as batcher:
        futures = []
        for i in range(10):
            f = await batcher.submit(f"item-{i}")
            futures.append(f)
        await asyncio.gather(*futures)

    # With 2 cost each and target of 5, batches should be at most 2-3 items
    for size in batch_sizes:
        assert size <= 3


@pytest.mark.asyncio
async def test_batch_assembly_respects_max_batch_size():
    """max_batch_size caps the batch regardless of cost budget."""
    batch_sizes: list[int] = []

    async def tracking_process(batch: list[str]) -> list[str]:
        batch_sizes.append(len(batch))
        return [f"r:{x}" for x in batch]

    async with DynamicBatcher(
        process_fn=tracking_process,
        target_batch_cost=1_000_000,
        max_batch_size=3,
        batch_timeout_s=0.05,
    ) as batcher:
        futures = []
        for i in range(9):
            f = await batcher.submit(f"item-{i}")
            futures.append(f)
        await asyncio.gather(*futures)

    for size in batch_sizes:
        assert size <= 3


@pytest.mark.asyncio
async def test_batch_timeout_dispatches_partial_batch():
    """A partial batch should be dispatched when the timeout expires."""
    batch_sizes: list[int] = []

    async def tracking_process(batch: list[str]) -> list[str]:
        batch_sizes.append(len(batch))
        return [f"r:{x}" for x in batch]

    async with DynamicBatcher(
        process_fn=tracking_process,
        target_batch_cost=1_000_000,
        max_batch_size=1000,
        batch_timeout_s=0.02,
    ) as batcher:
        # Submit 1 record and wait — timeout should dispatch it
        future = await batcher.submit("lonely")
        result = await future

    assert result == "r:lonely"
    assert 1 in batch_sizes


@pytest.mark.asyncio
async def test_backpressure():
    """submit() should block when the queue is full (backpressure)."""
    gate = asyncio.Event()

    async def gated_process(batch: list[str]) -> list[str]:
        await gate.wait()
        return [f"r:{x}" for x in batch]

    batcher = DynamicBatcher(
        process_fn=gated_process,
        max_queue_size=1,
        max_batch_size=1,
        prefetch_batches=1,
        batch_timeout_s=0.01,
    )
    async with batcher:
        try:
            await batcher.submit("a")
            await asyncio.sleep(0.05)

            await batcher.submit("b")
            await asyncio.sleep(0.05)

            await batcher.submit("c")
            await asyncio.sleep(0.05)

            await batcher.submit("d")

            submit_done = asyncio.Event()

            async def blocked_submit():
                await batcher.submit("e")
                submit_done.set()

            task = asyncio.create_task(blocked_submit())
            await asyncio.sleep(0.1)
            assert not submit_done.is_set(), "submit should be blocked by backpressure"
        finally:
            gate.set()

        await asyncio.wait_for(task, timeout=2.0)
        assert submit_done.is_set()


@pytest.mark.asyncio
async def test_error_propagation():
    """When process_fn raises, all futures in the batch get the exception."""
    async with DynamicBatcher(
        process_fn=failing_process,
        batch_timeout_s=0.02,
    ) as batcher:
        future = await batcher.submit("boom")
        with pytest.raises(RuntimeError, match="processing exploded"):
            await future


@pytest.mark.asyncio
async def test_result_count_mismatch():
    """process_fn returning wrong number of results raises ValueError."""
    async with DynamicBatcher(
        process_fn=wrong_length_process,
        batch_timeout_s=0.02,
    ) as batcher:
        futures = [await batcher.submit(f"item-{i}") for i in range(3)]
        for f in futures:
            with pytest.raises(ValueError, match="returned 1 results"):
                await f


@pytest.mark.asyncio
async def test_submit_batch_convenience():
    """submit_batch returns a list of futures, one per record."""
    async with DynamicBatcher(process_fn=echo_process) as batcher:
        futures = await batcher.submit_batch(["a", "b", "c"])
        results = await asyncio.gather(*futures)
    assert results == ["result:a", "result:b", "result:c"]


@pytest.mark.asyncio
async def test_batch_stats_utilization():
    """BatchStats.utilization is computed correctly."""
    stats = BatchStats(busy_time_s=3.0, idle_time_s=1.0)
    assert stats.utilization == pytest.approx(0.75)


def test_batch_stats_utilization_zero():
    """utilization returns 0 when no time has elapsed."""
    stats = BatchStats()
    assert stats.utilization == 0.0


@pytest.mark.asyncio
async def test_context_manager():
    """async with starts and stops the batcher."""
    batcher = DynamicBatcher(process_fn=echo_process)
    assert not batcher.is_running

    async with batcher:
        assert batcher.is_running
        future = await batcher.submit("test")
        await future

    assert not batcher.is_running


@pytest.mark.asyncio
async def test_double_start_raises():
    """Calling start() twice raises RuntimeError."""
    batcher = DynamicBatcher(process_fn=echo_process)
    await batcher.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await batcher.start()
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_submit_when_not_running_raises():
    """submit() before start() raises RuntimeError."""
    batcher = DynamicBatcher(process_fn=echo_process)
    with pytest.raises(RuntimeError, match="not running"):
        await batcher.submit("nope")


@pytest.mark.asyncio
async def test_stats_are_updated():
    """After processing, stats reflect the work done."""
    async with DynamicBatcher(
        process_fn=slow_process,
        batch_timeout_s=0.02,
    ) as batcher:
        futures = await batcher.submit_batch([f"item-{i}" for i in range(5)])
        await asyncio.gather(*futures)

    assert batcher.stats.total_submitted == 5
    assert batcher.stats.total_completed == 5
    assert batcher.stats.total_batches >= 1
    assert batcher.stats.busy_time_s > 0
    assert batcher.stats.avg_batch_size > 0


# ---------------------------------------------------------------------------
# TokenBatcher tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_batcher_inference_fn():
    """TokenBatcher accepts inference_fn as the processing function."""
    async with TokenBatcher(inference_fn=echo_process) as batcher:
        future = await batcher.submit("hello")
        result = await future
    assert result == "result:hello"


@pytest.mark.asyncio
async def test_token_batcher_process_fn_fallback():
    """TokenBatcher also accepts process_fn."""
    async with TokenBatcher(process_fn=echo_process) as batcher:
        future = await batcher.submit("hello")
        result = await future
    assert result == "result:hello"


@pytest.mark.asyncio
async def test_token_batcher_requires_fn():
    """TokenBatcher raises if neither inference_fn nor process_fn given."""
    with pytest.raises(TypeError, match="requires either"):
        TokenBatcher()


@pytest.mark.asyncio
async def test_token_batcher_estimated_tokens():
    """TokenBatcher.submit accepts estimated_tokens keyword."""
    async with TokenBatcher(inference_fn=echo_process) as batcher:
        future = await batcher.submit("hello", estimated_tokens=42)
        result = await future
    assert result == "result:hello"


@pytest.mark.asyncio
async def test_token_batcher_token_estimator_protocol():
    """TokenBatcher checks TokenEstimator protocol (estimate_tokens)."""

    async def record_process(batch: list[TokenRecord]) -> list[str]:
        return [r.text.upper() for r in batch]

    async with TokenBatcher(inference_fn=record_process) as batcher:
        record = TokenRecord(text="hello")
        assert isinstance(record, TokenEstimator)
        future = await batcher.submit(record)
        result = await future
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_token_batcher_token_estimator_fn():
    """TokenBatcher accepts token_estimator function."""
    estimates: list[int] = []

    def estimator(record: str) -> int:
        est = len(record) * 10
        estimates.append(est)
        return est

    async with TokenBatcher(
        inference_fn=echo_process,
        token_estimator=estimator,
    ) as batcher:
        future = await batcher.submit("hello")
        await future
    assert estimates == [50]


@pytest.mark.asyncio
async def test_token_batcher_target_batch_tokens():
    """TokenBatcher respects target_batch_tokens."""
    batch_sizes: list[int] = []

    async def tracking(batch: list[str]) -> list[str]:
        batch_sizes.append(len(batch))
        return [f"r:{x}" for x in batch]

    async with TokenBatcher(
        inference_fn=tracking,
        target_batch_tokens=5,
        default_token_estimate=2,
        max_batch_size=100,
        batch_timeout_s=0.05,
    ) as batcher:
        futures = []
        for i in range(10):
            f = await batcher.submit(f"item-{i}")
            futures.append(f)
        await asyncio.gather(*futures)

    for size in batch_sizes:
        assert size <= 3


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


def test_prompt_estimate_tokens():
    """Prompt.estimate_tokens returns len(text) // 4 + 1."""
    p = Prompt(text="Hello, world!")  # 13 chars → 13 // 4 + 1 = 4
    assert p.estimate_tokens() == 4


def test_prompt_implements_token_estimator():
    """Prompt implements the TokenEstimator protocol."""
    p = Prompt(text="test")
    assert isinstance(p, TokenEstimator)


@pytest.mark.asyncio
async def test_prompt_with_token_batcher():
    """Prompt works end-to-end with TokenBatcher."""

    async def inference(batch: list[Prompt]) -> list[str]:
        return [f"answer: {p.text}" for p in batch]

    async with TokenBatcher(inference_fn=inference) as batcher:
        future = await batcher.submit(Prompt(text="What is 2+2?"))
        result = await future
    assert result == "answer: What is 2+2?"
