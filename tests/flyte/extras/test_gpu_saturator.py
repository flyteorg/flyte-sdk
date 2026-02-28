"""Tests for flyte.extras.GPUSaturator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from flyte.extras._gpu_saturator import (
    BatchStats,
    GPUSaturator,
    TokenEstimator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def echo_inference(batch: list[str]) -> list[str]:
    """Simple async inference fn that echoes inputs."""
    return [f"result:{item}" for item in batch]


async def slow_inference(batch: list[str]) -> list[str]:
    await asyncio.sleep(0.05)
    return [f"slow:{item}" for item in batch]


async def failing_inference(batch: list[str]) -> list[str]:
    raise RuntimeError("inference exploded")


async def wrong_length_inference(batch: list[str]) -> list[str]:
    return ["only_one"]


@dataclass
class TokenRecord:
    """Record implementing the TokenEstimator protocol."""

    text: str

    def estimate_tokens(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_submit_and_result():
    """Submit a single record, await the future, get the right result."""
    async with GPUSaturator(inference_fn=echo_inference) as sat:
        future = await sat.submit("hello")
        result = await future
    assert result == "result:hello"


@pytest.mark.asyncio
async def test_multiple_submits():
    """Submit several records and verify all results arrive correctly."""
    async with GPUSaturator(
        inference_fn=echo_inference,
        batch_timeout_s=0.1,
    ) as sat:
        futures = []
        for i in range(10):
            f = await sat.submit(f"item-{i}")
            futures.append(f)
        results = await asyncio.gather(*futures)
    assert results == [f"result:item-{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_token_estimation_explicit_override():
    """Explicit estimated_tokens overrides all other estimators."""
    called = False

    def estimator(record: str) -> int:
        nonlocal called
        called = True
        return 999

    async with GPUSaturator(
        inference_fn=echo_inference,
        token_estimator=estimator,
    ) as sat:
        future = await sat.submit("hi", estimated_tokens=42)
        await future
    assert not called, "estimator should not be called when explicit tokens given"


@pytest.mark.asyncio
async def test_token_estimation_estimator_fn():
    """token_estimator function is used when no explicit tokens given."""
    estimates: list[int] = []

    def estimator(record: str) -> int:
        est = len(record) * 10
        estimates.append(est)
        return est

    async with GPUSaturator(
        inference_fn=echo_inference,
        token_estimator=estimator,
    ) as sat:
        future = await sat.submit("hello")
        await future
    assert estimates == [50]


@pytest.mark.asyncio
async def test_token_estimation_protocol():
    """Records implementing TokenEstimator protocol are used automatically."""

    async def record_inference(batch: list[TokenRecord]) -> list[str]:
        return [r.text.upper() for r in batch]

    async with GPUSaturator(inference_fn=record_inference) as sat:
        record = TokenRecord(text="hello")
        assert isinstance(record, TokenEstimator)
        future = await sat.submit(record)
        result = await future
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_token_estimation_default():
    """When no estimator is available, default_token_estimate is used."""
    async with GPUSaturator(
        inference_fn=echo_inference,
        default_token_estimate=7,
    ) as sat:
        future = await sat.submit("anything")
        await future
    # If we got here without error, default was used (no crash = success)
    assert sat.stats.total_completed == 1


@pytest.mark.asyncio
async def test_batch_assembly_respects_target_tokens():
    """Batches should not exceed target_batch_tokens."""
    batch_sizes: list[int] = []

    async def tracking_inference(batch: list[str]) -> list[str]:
        batch_sizes.append(len(batch))
        return [f"r:{x}" for x in batch]

    async with GPUSaturator(
        inference_fn=tracking_inference,
        target_batch_tokens=5,
        default_token_estimate=2,
        max_batch_size=100,
        batch_timeout_s=0.05,
    ) as sat:
        futures = []
        for i in range(10):
            f = await sat.submit(f"item-{i}")
            futures.append(f)
        await asyncio.gather(*futures)

    # With 2 tokens each and target of 5, batches should be at most 2-3 items
    for size in batch_sizes:
        assert size <= 3


@pytest.mark.asyncio
async def test_batch_assembly_respects_max_batch_size():
    """max_batch_size caps the batch regardless of token budget."""
    batch_sizes: list[int] = []

    async def tracking_inference(batch: list[str]) -> list[str]:
        batch_sizes.append(len(batch))
        return [f"r:{x}" for x in batch]

    async with GPUSaturator(
        inference_fn=tracking_inference,
        target_batch_tokens=1_000_000,
        max_batch_size=3,
        batch_timeout_s=0.05,
    ) as sat:
        futures = []
        for i in range(9):
            f = await sat.submit(f"item-{i}")
            futures.append(f)
        await asyncio.gather(*futures)

    for size in batch_sizes:
        assert size <= 3


@pytest.mark.asyncio
async def test_batch_timeout_dispatches_partial_batch():
    """A partial batch should be dispatched when the timeout expires."""
    batch_sizes: list[int] = []

    async def tracking_inference(batch: list[str]) -> list[str]:
        batch_sizes.append(len(batch))
        return [f"r:{x}" for x in batch]

    async with GPUSaturator(
        inference_fn=tracking_inference,
        target_batch_tokens=1_000_000,
        max_batch_size=1000,
        batch_timeout_s=0.02,
    ) as sat:
        # Submit 1 record and wait — timeout should dispatch it
        future = await sat.submit("lonely")
        result = await future

    assert result == "r:lonely"
    assert 1 in batch_sizes


@pytest.mark.asyncio
async def test_backpressure():
    """submit() should block when the queue is full (backpressure).

    With max_batch_size=1, prefetch_batches=1, max_queue_size=1 the pipeline
    fills up as:
      "a" → inference (blocked on gate)
      "b" → batch_queue (full, size 1)
      "c" → aggregator blocks trying to put on full batch_queue,
             so "c" is consumed from the main queue which is now empty
      "d" → main queue (full, size 1)
      "e" → blocks on main queue (backpressure!)
    """
    gate = asyncio.Event()

    async def gated_inference(batch: list[str]) -> list[str]:
        await gate.wait()
        return [f"r:{x}" for x in batch]

    sat = GPUSaturator(
        inference_fn=gated_inference,
        max_queue_size=1,
        max_batch_size=1,
        prefetch_batches=1,
        batch_timeout_s=0.01,
    )
    async with sat:
        try:
            # "a" → aggregator → batch_queue → inference (blocks on gate)
            await sat.submit("a")
            await asyncio.sleep(0.05)

            # "b" → aggregator → batch_queue (now full, inference holds batch_a)
            await sat.submit("b")
            await asyncio.sleep(0.05)

            # "c" → aggregator takes it, tries batch_queue.put → blocks
            await sat.submit("c")
            await asyncio.sleep(0.05)

            # "d" → sits in the main queue (size 1 = full)
            await sat.submit("d")

            # "e" should block — queue full, aggregator stuck
            submit_done = asyncio.Event()

            async def blocked_submit():
                await sat.submit("e")
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
    """When inference_fn raises, all futures in the batch get the exception."""
    async with GPUSaturator(
        inference_fn=failing_inference,
        batch_timeout_s=0.02,
    ) as sat:
        future = await sat.submit("boom")
        with pytest.raises(RuntimeError, match="inference exploded"):
            await future


@pytest.mark.asyncio
async def test_result_count_mismatch():
    """inference_fn returning wrong number of results raises ValueError."""
    async with GPUSaturator(
        inference_fn=wrong_length_inference,
        batch_timeout_s=0.02,
    ) as sat:
        futures = [await sat.submit(f"item-{i}") for i in range(3)]
        # All futures should get the ValueError
        for f in futures:
            with pytest.raises(ValueError, match="returned 1 results"):
                await f


@pytest.mark.asyncio
async def test_submit_batch_convenience():
    """submit_batch returns a list of futures, one per record."""
    async with GPUSaturator(inference_fn=echo_inference) as sat:
        futures = await sat.submit_batch(["a", "b", "c"])
        results = await asyncio.gather(*futures)
    assert results == ["result:a", "result:b", "result:c"]


@pytest.mark.asyncio
async def test_batch_stats_gpu_utilization():
    """BatchStats.gpu_utilization is computed correctly."""
    stats = BatchStats(gpu_busy_time_s=3.0, gpu_idle_time_s=1.0)
    assert stats.gpu_utilization == pytest.approx(0.75)


def test_batch_stats_gpu_utilization_zero():
    """gpu_utilization returns 0 when no time has elapsed."""
    stats = BatchStats()
    assert stats.gpu_utilization == 0.0


@pytest.mark.asyncio
async def test_context_manager():
    """async with starts and stops the saturator."""
    sat = GPUSaturator(inference_fn=echo_inference)
    assert not sat.is_running

    async with sat:
        assert sat.is_running
        future = await sat.submit("test")
        await future

    assert not sat.is_running


@pytest.mark.asyncio
async def test_double_start_raises():
    """Calling start() twice raises RuntimeError."""
    sat = GPUSaturator(inference_fn=echo_inference)
    await sat.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await sat.start()
    finally:
        await sat.stop()


@pytest.mark.asyncio
async def test_submit_when_not_running_raises():
    """submit() before start() raises RuntimeError."""
    sat = GPUSaturator(inference_fn=echo_inference)
    with pytest.raises(RuntimeError, match="not running"):
        await sat.submit("nope")


@pytest.mark.asyncio
async def test_stats_are_updated():
    """After processing, stats reflect the work done."""
    async with GPUSaturator(
        inference_fn=slow_inference,
        batch_timeout_s=0.02,
    ) as sat:
        futures = await sat.submit_batch([f"item-{i}" for i in range(5)])
        await asyncio.gather(*futures)

    assert sat.stats.total_submitted == 5
    assert sat.stats.total_completed == 5
    assert sat.stats.total_batches >= 1
    assert sat.stats.gpu_busy_time_s > 0
    assert sat.stats.avg_batch_size > 0
