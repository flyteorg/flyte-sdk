"""
InferencePipeline — 3-stage async pipeline for batch inference.

Wires together preprocessing, batched GPU inference (via DynamicBatcher),
and postprocessing with proper thread pool dispatch and backpressure so
users only define the three stage functions.

Quick start::

    from flyte.extras import InferencePipeline

    pipeline = InferencePipeline(
        preprocess_fn=resize_and_normalize,   # sync or async, per-item
        inference_fn=gpu_forward_batch,        # async, batched
        postprocess_fn=decode_labels,          # sync or async, per-item
    )

    async with pipeline:
        results = await pipeline.run_all(image_urls)

Architecture::

    items ──► preprocess_fn  ──► DynamicBatcher  ──► postprocess_fn ──► outputs
              (executor)         (internal queues)    (executor/loop)

Multiple concurrent callers (e.g. from ReusePolicy concurrency) can share
one pipeline singleton so the DynamicBatcher sees items from many streams,
producing larger GPU batches and higher utilization.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from concurrent.futures import Executor
from typing import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    TypeVar,
    Union,
)

from ._dynamic_batcher import (
    BatchStats,
    CostEstimatorFn,
    DynamicBatcher,
    ProcessFn,
)

logger = logging.getLogger(__name__)

RawT = TypeVar("RawT")
PreparedT = TypeVar("PreparedT")
ResultT = TypeVar("ResultT")
OutputT = TypeVar("OutputT")

# Type aliases for the stage functions
PreprocessFn = Union[Callable[[RawT], PreparedT], Callable[[RawT], Awaitable[PreparedT]]]
PostprocessFn = Union[Callable[[RawT, ResultT], OutputT], Callable[[RawT, ResultT], Awaitable[OutputT]]]


class InferencePipeline(Generic[RawT, PreparedT, ResultT, OutputT]):
    """3-stage async pipeline: preprocess → batched inference → postprocess.

    Handles thread pool dispatch, bounded queues, and DynamicBatcher wiring
    so users only define the three stage functions.

    Type Parameters:
        RawT: The raw input type (e.g. image URL, JSONL line).
        PreparedT: The preprocessed type ready for inference (e.g. tensor).
        ResultT: The per-item result from the inference batch function.
        OutputT: The final output after postprocessing.

    Args:
        preprocess_fn:
            ``(raw: RawT) -> PreparedT`` or ``async (raw: RawT) -> PreparedT``
            Transforms a single raw item into inference-ready form.
            If synchronous and ``preprocess_executor`` is provided, runs on
            that executor to avoid blocking the event loop.

        inference_fn:
            ``async (batch: list[PreparedT]) -> list[ResultT]``
            Standard DynamicBatcher ``process_fn``. Receives a batch,
            returns results in the same order.

        postprocess_fn:
            ``(raw: RawT, result: ResultT) -> OutputT``
            Transforms inference output into the final result.  Receives
            the **original raw item** for full context (filename, metadata).
            If synchronous and ``postprocess_executor`` is provided, runs
            on that executor.

        target_batch_cost:
            Cost budget per batch for the internal DynamicBatcher.

        max_batch_size:
            Hard cap on items per GPU batch.

        batch_timeout_s:
            Max seconds to wait for a full batch.

        cost_estimator:
            Optional ``(PreparedT) -> int`` for cost estimation.

        max_queue_size:
            DynamicBatcher submission queue bound.

        preprocess_executor:
            Executor for sync ``preprocess_fn`` calls.  Pass a
            ``ThreadPoolExecutor`` for CPU-bound preprocessing.

        postprocess_executor:
            Executor for sync ``postprocess_fn`` calls.  ``None`` runs
            on the event loop (fine for lightweight postprocessing).

        pipeline_depth:
            Max items preprocessed-but-not-yet-submitted to the batcher.
            Controls memory pressure between preprocess and inference.

    Example::

        from concurrent.futures import ThreadPoolExecutor

        cpu_pool = ThreadPoolExecutor(4)
        gpu_pool = ThreadPoolExecutor(1)

        def preprocess(url: str) -> torch.Tensor:
            img = download(url)
            return normalize(resize(img))

        async def inference(batch: list[torch.Tensor]) -> list[torch.Tensor]:
            stacked = torch.stack(batch).cuda()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(gpu_pool, model, stacked)

        def postprocess(url: str, logits: torch.Tensor) -> dict:
            idx = logits.argmax().item()
            return {"url": url, "label": labels[idx]}

        pipeline = InferencePipeline(
            preprocess_fn=preprocess,
            inference_fn=inference,
            postprocess_fn=postprocess,
            preprocess_executor=cpu_pool,
            max_batch_size=32,
            target_batch_cost=32,
        )
    """

    def __init__(
        self,
        preprocess_fn: PreprocessFn[RawT, PreparedT],
        inference_fn: ProcessFn[PreparedT, ResultT],
        postprocess_fn: PostprocessFn[RawT, ResultT, OutputT],
        *,
        # DynamicBatcher config
        target_batch_cost: int = 32_000,
        max_batch_size: int = 256,
        min_batch_size: int = 1,
        batch_timeout_s: float = 0.05,
        cost_estimator: CostEstimatorFn[PreparedT] | None = None,
        max_queue_size: int = 5_000,
        prefetch_batches: int = 2,
        default_cost: int = 1,
        # Pipeline config
        preprocess_executor: Executor | None = None,
        postprocess_executor: Executor | None = None,
        pipeline_depth: int = 8,
    ):
        self._preprocess_fn = preprocess_fn
        self._postprocess_fn = postprocess_fn
        self._preprocess_executor = preprocess_executor
        self._postprocess_executor = postprocess_executor
        self._pipeline_depth = pipeline_depth

        self._preprocess_is_async = inspect.iscoroutinefunction(preprocess_fn)
        self._postprocess_is_async = inspect.iscoroutinefunction(postprocess_fn)

        self._batcher = DynamicBatcher[PreparedT, ResultT](
            process_fn=inference_fn,
            cost_estimator=cost_estimator,
            target_batch_cost=target_batch_cost,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            batch_timeout_s=batch_timeout_s,
            max_queue_size=max_queue_size,
            prefetch_batches=prefetch_batches,
            default_cost=default_cost,
        )

    # -- Public API --------------------------------------------------------

    @property
    def batcher(self) -> DynamicBatcher[PreparedT, ResultT]:
        """The underlying DynamicBatcher for advanced configuration."""
        return self._batcher

    @property
    def stats(self) -> BatchStats:
        """Shortcut for ``pipeline.batcher.stats``."""
        return self._batcher.stats

    async def start(self) -> None:
        """Start the internal DynamicBatcher."""
        await self._batcher.start()

    async def stop(self) -> None:
        """Stop the internal DynamicBatcher, processing remaining work."""
        await self._batcher.stop()

    async def __aenter__(self) -> InferencePipeline[RawT, PreparedT, ResultT, OutputT]:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    async def run(
        self,
        items: AsyncIterable[RawT] | Iterable[RawT],
    ) -> AsyncIterator[OutputT]:
        """Stream items through the pipeline, yielding results in order.

        Preprocessing and inference overlap: while the GPU processes batch N,
        the CPU preprocesses items for batch N+1.

        Args:
            items: Input items (sync or async iterable).

        Yields:
            Postprocessed results, one per input item, in order.
        """
        if not self._batcher.is_running:
            raise RuntimeError("InferencePipeline is not running. Call start() or use 'async with'.")

        # Channel between preprocess producer and result consumer
        prep_queue: asyncio.Queue[tuple[RawT, asyncio.Future[ResultT]] | None] = asyncio.Queue(
            maxsize=self._pipeline_depth,
        )
        producer_error: list[BaseException] = []

        async def _preprocess_and_submit():
            """Preprocess each item and submit to batcher."""
            try:
                async for raw in _as_async_iter(items):
                    prepared = await self._run_preprocess(raw)
                    future = await self._batcher.submit(prepared)
                    await prep_queue.put((raw, future))
            except Exception as exc:
                producer_error.append(exc)
            finally:
                await prep_queue.put(None)  # sentinel

        producer = asyncio.create_task(_preprocess_and_submit())

        try:
            while True:
                item = await prep_queue.get()
                if item is None:
                    # Check if producer failed
                    if producer_error:
                        raise producer_error[0]
                    break
                raw, future = item
                result = await future
                output = await self._run_postprocess(raw, result)
                yield output
        finally:
            # Ensure producer is cleaned up if consumer exits early
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except asyncio.CancelledError:
                    pass

    async def run_all(
        self,
        items: AsyncIterable[RawT] | Iterable[RawT],
    ) -> list[OutputT]:
        """Convenience: run all items and collect results as a list.

        Args:
            items: Input items (sync or async iterable).

        Returns:
            List of postprocessed results, in input order.
        """
        results = []
        async for output in self.run(items):
            results.append(output)
        return results

    # -- Internal helpers --------------------------------------------------

    async def _run_preprocess(self, raw: RawT) -> PreparedT:
        if self._preprocess_is_async:
            return await self._preprocess_fn(raw)
        if self._preprocess_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._preprocess_executor,
                self._preprocess_fn,
                raw,
            )
        return self._preprocess_fn(raw)

    async def _run_postprocess(self, raw: RawT, result: ResultT) -> OutputT:
        if self._postprocess_is_async:
            return await self._postprocess_fn(raw, result)
        if self._postprocess_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._postprocess_executor,
                self._postprocess_fn,
                raw,
                result,
            )
        return self._postprocess_fn(raw, result)


async def _as_async_iter(items: AsyncIterable[RawT] | Iterable[RawT]) -> AsyncIterator[RawT]:
    """Normalize sync/async iterables into an async iterator."""
    if isinstance(items, AsyncIterable):
        async for item in items:
            yield item
    else:
        for item in items:
            yield item


__all__ = [
    "InferencePipeline",
]
