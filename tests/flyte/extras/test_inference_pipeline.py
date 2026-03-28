"""Tests for flyte.extras InferencePipeline."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from flyte.extras._inference_pipeline import InferencePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class RawItem:
    value: int


@dataclass
class PreparedItem:
    doubled: int

    def estimate_cost(self) -> int:
        return 1


@dataclass
class Output:
    original: int
    result: int


def sync_preprocess(item: RawItem) -> PreparedItem:
    return PreparedItem(doubled=item.value * 2)


async def async_preprocess(item: RawItem) -> PreparedItem:
    await asyncio.sleep(0.001)
    return PreparedItem(doubled=item.value * 2)


async def inference_batch(batch: list[PreparedItem]) -> list[int]:
    """Batch inference: sum each prepared item's value with 100."""
    return [p.doubled + 100 for p in batch]


def sync_postprocess(item: RawItem, result: int) -> Output:
    return Output(original=item.value, result=result)


async def async_postprocess(item: RawItem, result: int) -> Output:
    return Output(original=item.value, result=result)


async def slow_inference(batch: list[PreparedItem]) -> list[int]:
    """Simulate slow GPU work."""
    await asyncio.sleep(0.1)
    return [p.doubled + 100 for p in batch]


async def failing_inference(batch: list[PreparedItem]) -> list[int]:
    raise RuntimeError("GPU exploded")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferencePipelineBasic:
    @pytest.mark.asyncio
    async def test_sync_preprocess_and_postprocess(self):
        async with InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=sync_postprocess,
            max_batch_size=4,
            target_batch_cost=4,
            batch_timeout_s=0.01,
        ) as pipeline:
            items = [RawItem(value=i) for i in range(10)]
            results = await pipeline.run_all(items)

        assert len(results) == 10
        for i, r in enumerate(results):
            assert r.original == i
            assert r.result == i * 2 + 100

    @pytest.mark.asyncio
    async def test_async_preprocess_and_postprocess(self):
        async with InferencePipeline(
            preprocess_fn=async_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=async_postprocess,
            max_batch_size=4,
            target_batch_cost=4,
            batch_timeout_s=0.01,
        ) as pipeline:
            results = await pipeline.run_all([RawItem(i) for i in range(5)])

        assert len(results) == 5
        assert results[0].result == 100  # 0 * 2 + 100

    @pytest.mark.asyncio
    async def test_with_executor(self):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            async with InferencePipeline(
                preprocess_fn=sync_preprocess,
                inference_fn=inference_batch,
                postprocess_fn=sync_postprocess,
                preprocess_executor=pool,
                postprocess_executor=pool,
                max_batch_size=4,
                target_batch_cost=4,
                batch_timeout_s=0.01,
            ) as pipeline:
                results = await pipeline.run_all([RawItem(i) for i in range(8)])

            assert len(results) == 8
        finally:
            pool.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_empty_input(self):
        async with InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=sync_postprocess,
            batch_timeout_s=0.01,
        ) as pipeline:
            results = await pipeline.run_all([])

        assert results == []


class TestInferencePipelineOrdering:
    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """Results must come back in input order."""
        async with InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=sync_postprocess,
            max_batch_size=3,
            target_batch_cost=3,
            batch_timeout_s=0.01,
        ) as pipeline:
            items = [RawItem(i) for i in range(20)]
            results = await pipeline.run_all(items)

        assert [r.original for r in results] == list(range(20))


class TestInferencePipelineAsyncIter:
    @pytest.mark.asyncio
    async def test_async_iterable_input(self):
        async def aiter_items():
            for i in range(5):
                yield RawItem(i)

        async with InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=sync_postprocess,
            max_batch_size=4,
            target_batch_cost=4,
            batch_timeout_s=0.01,
        ) as pipeline:
            results = await pipeline.run_all(aiter_items())

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_streaming_run(self):
        """Test the async generator interface."""
        async with InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=sync_postprocess,
            max_batch_size=4,
            target_batch_cost=4,
            batch_timeout_s=0.01,
        ) as pipeline:
            results = []
            async for output in pipeline.run([RawItem(i) for i in range(5)]):
                results.append(output)

        assert len(results) == 5


class TestInferencePipelineStats:
    @pytest.mark.asyncio
    async def test_stats_exposed(self):
        async with InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=sync_postprocess,
            max_batch_size=4,
            target_batch_cost=4,
            batch_timeout_s=0.01,
        ) as pipeline:
            await pipeline.run_all([RawItem(i) for i in range(10)])

            stats = pipeline.stats
            assert stats.total_completed == 10
            assert stats.total_batches >= 1
            assert stats.busy_time_s >= 0
            assert stats.utilization >= 0


class TestInferencePipelineBackpressure:
    @pytest.mark.asyncio
    async def test_backpressure_with_slow_inference(self):
        """With pipeline_depth=2 and slow inference, preprocess should not run far ahead."""
        preprocess_count = 0

        def counting_preprocess(item: RawItem) -> PreparedItem:
            nonlocal preprocess_count
            preprocess_count += 1
            return PreparedItem(doubled=item.value * 2)

        async with InferencePipeline(
            preprocess_fn=counting_preprocess,
            inference_fn=slow_inference,
            postprocess_fn=sync_postprocess,
            max_batch_size=2,
            target_batch_cost=2,
            batch_timeout_s=0.01,
            pipeline_depth=2,
            max_queue_size=2,
        ) as pipeline:
            results = await pipeline.run_all([RawItem(i) for i in range(10)])

        assert len(results) == 10
        # All items must eventually be processed
        assert preprocess_count == 10


class TestInferencePipelineErrors:
    @pytest.mark.asyncio
    async def test_inference_error_propagates(self):
        async with InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=failing_inference,
            postprocess_fn=sync_postprocess,
            max_batch_size=4,
            target_batch_cost=4,
            batch_timeout_s=0.01,
        ) as pipeline:
            with pytest.raises(RuntimeError, match="GPU exploded"):
                await pipeline.run_all([RawItem(1)])

    @pytest.mark.asyncio
    async def test_not_started_raises(self):
        pipeline = InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=sync_postprocess,
        )
        with pytest.raises(RuntimeError, match="not running"):
            await pipeline.run_all([RawItem(1)])


class TestInferencePipelineBatcher:
    @pytest.mark.asyncio
    async def test_batcher_property(self):
        pipeline = InferencePipeline(
            preprocess_fn=sync_preprocess,
            inference_fn=inference_batch,
            postprocess_fn=sync_postprocess,
        )
        assert pipeline.batcher is not None
        assert pipeline.stats is not None
