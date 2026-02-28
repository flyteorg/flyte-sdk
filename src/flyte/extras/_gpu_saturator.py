"""
GPUSaturator — maximize GPU utilization by batching work from many concurrent
producers through a single async inference function.

This is useful for large-scale batch inference, especially with reusable
containers that have multiple parallel streams feeding a single GPU worker.

Quick start::

    from flyte.extras import GPUSaturator

    async def my_inference(batch: list[str]) -> list[str]:
        return [f"result for {item}" for item in batch]

    async with GPUSaturator(inference_fn=my_inference) as saturator:
        future = await saturator.submit("hello", estimated_tokens=5)
        result = await future

Typical usage with vLLM::

    from vllm import LLM, SamplingParams

    llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
    params = SamplingParams(temperature=0.7, max_tokens=512)

    async def inference(batch: list[Prompt]) -> list[str]:
        outputs = llm.generate([p.text for p in batch], params)
        return [o.outputs[0].text for o in outputs]

    async with GPUSaturator(
        inference_fn=inference,
        target_batch_tokens=32_000,
        max_batch_size=256,
    ) as saturator:
        # Many concurrent producers submit records
        future = await saturator.submit(Prompt(text="What is 2+2?"))
        answer = await future
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Awaitable,
    Callable,
    Generic,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

RecordT = TypeVar("RecordT")
ResultT = TypeVar("ResultT")


@runtime_checkable
class TokenEstimator(Protocol):
    """Protocol for records that can estimate their own token count.

    Implement this on your record type and the saturator will call it
    automatically when no explicit ``estimated_tokens`` is passed to
    :meth:`GPUSaturator.submit`.

    Example::

        @dataclass
        class Prompt:
            text: str

            def estimate_tokens(self) -> int:
                return len(self.text) // 4 + 1
    """

    def estimate_tokens(self) -> int: ...


InferenceFn = Callable[[list[RecordT]], Awaitable[list[ResultT]]]
"""Async function that takes a batch of records and returns results in the
same order.  Must be a native coroutine (``async def``)."""

TokenEstimatorFn = Callable[[RecordT], int]
"""Optional callable ``(record) -> int`` for token estimation."""


def _make_future() -> asyncio.Future:
    """Create a Future on the running event loop."""
    return asyncio.get_running_loop().create_future()


@dataclass
class _Envelope(Generic[RecordT, ResultT]):
    """Internal wrapper pairing a record with its result future."""

    record: RecordT
    estimated_tokens: int
    future: asyncio.Future[ResultT] = field(default_factory=_make_future)


@dataclass
class BatchStats:
    """Monitoring statistics exposed by :attr:`GPUSaturator.stats`.

    Attributes:
        total_submitted: Total records submitted via :meth:`submit`.
        total_completed: Total records whose futures have been resolved.
        total_batches: Number of inference batches dispatched.
        total_batch_tokens: Sum of estimated tokens across all batches.
        avg_batch_size: Running average records per batch.
        avg_batch_tokens: Running average tokens per batch.
        gpu_busy_time_s: Cumulative seconds spent inside ``inference_fn``.
        gpu_idle_time_s: Cumulative seconds the inference loop waited for
            a batch to be assembled.
    """

    total_submitted: int = 0
    total_completed: int = 0
    total_batches: int = 0
    total_batch_tokens: int = 0
    avg_batch_size: float = 0.0
    avg_batch_tokens: float = 0.0
    gpu_busy_time_s: float = 0.0
    gpu_idle_time_s: float = 0.0

    @property
    def gpu_utilization(self) -> float:
        """Fraction of wall-clock time spent in inference (0.0-1.0)."""
        total = self.gpu_busy_time_s + self.gpu_idle_time_s
        return self.gpu_busy_time_s / total if total > 0 else 0.0


class GPUSaturator(Generic[RecordT, ResultT]):
    """Batches records from many concurrent producers and runs them through
    a single async inference function, maximizing GPU utilization.

    The saturator runs two internal loops:

    1. **Aggregation loop** — drains the submission queue and assembles
       token-budgeted batches, respecting ``target_batch_tokens``,
       ``max_batch_size``, and ``batch_timeout_s``.
    2. **Inference loop** — pulls assembled batches and calls
       ``inference_fn``, resolving each record's :class:`asyncio.Future`.

    Type Parameters:
        RecordT: The input record type produced by your tasks.
        ResultT: The per-record output type returned by ``inference_fn``.

    Args:
        inference_fn:
            ``async def f(batch: list[RecordT]) -> list[ResultT]``
            Must return results in the **same order** as the input batch.

        token_estimator:
            Optional ``(RecordT) -> int`` function.  When provided, it is
            called to estimate the token count of each submitted record.
            Falls back to ``record.estimate_tokens()`` if the record
            implements :class:`TokenEstimator`, then to
            ``default_token_estimate``.

        target_batch_tokens:
            Token budget per batch.  The aggregator fills batches up to
            this limit before dispatching.

        max_batch_size:
            Hard cap on records per batch regardless of token budget.

        min_batch_size:
            Minimum records before dispatching.  Ignored when the timeout
            fires or shutdown is in progress.

        batch_timeout_s:
            Maximum seconds to wait for a full batch.  Lower values reduce
            GPU idle time but may produce smaller batches.

        max_queue_size:
            Bounded queue size.  When full, :meth:`submit` awaits
            (backpressure).

        prefetch_batches:
            Number of pre-assembled batches to buffer between the
            aggregation and inference loops.

        default_token_estimate:
            Fallback token count when no estimator is available.

    Example::

        async def inference(batch: list[dict]) -> list[str]:
            ...

        async with GPUSaturator(inference_fn=inference) as sat:
            futures = []
            for record in my_records:
                f = await sat.submit(record)
                futures.append(f)
            results = await asyncio.gather(*futures)
    """

    def __init__(
        self,
        inference_fn: InferenceFn[RecordT, ResultT],
        *,
        token_estimator: TokenEstimatorFn[RecordT] | None = None,
        target_batch_tokens: int = 32_000,
        max_batch_size: int = 256,
        min_batch_size: int = 1,
        batch_timeout_s: float = 0.05,
        max_queue_size: int = 5_000,
        prefetch_batches: int = 2,
        default_token_estimate: int = 1,
    ):
        self._inference_fn = inference_fn
        self._token_estimator = token_estimator
        self._target_batch_tokens = target_batch_tokens
        self._max_batch_size = max_batch_size
        self._min_batch_size = min_batch_size
        self._batch_timeout_s = batch_timeout_s
        self._prefetch_batches = prefetch_batches
        self._default_token_estimate = default_token_estimate

        self._queue: asyncio.Queue[_Envelope[RecordT, ResultT] | None] = asyncio.Queue(
            maxsize=max_queue_size,
        )
        self._batch_queue: asyncio.Queue[list[_Envelope[RecordT, ResultT]] | None] = asyncio.Queue(
            maxsize=prefetch_batches,
        )

        self._stats = BatchStats()
        self._running = False
        self._aggregator_task: asyncio.Task | None = None
        self._consumer_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    # -- Public API --------------------------------------------------------

    @property
    def stats(self) -> BatchStats:
        """Current :class:`BatchStats` snapshot."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Whether the aggregation and inference loops are active."""
        return self._running

    async def start(self) -> None:
        """Start the aggregation and inference loops.

        Raises:
            RuntimeError: If the saturator is already running.
        """
        if self._running:
            raise RuntimeError("GPUSaturator is already running")
        self._running = True
        self._shutdown_event.clear()
        self._aggregator_task = asyncio.create_task(
            self._aggregation_loop(),
            name="gpu-saturator-aggregator",
        )
        self._consumer_task = asyncio.create_task(
            self._inference_loop(),
            name="gpu-saturator-inference",
        )
        logger.info("GPUSaturator started")

    async def stop(self) -> None:
        """Graceful shutdown: process all enqueued work, then stop.

        Blocks until every pending future is resolved.
        """
        if not self._running:
            return
        self._shutdown_event.set()
        await self._queue.put(None)  # sentinel for aggregator
        if self._aggregator_task:
            await self._aggregator_task
        if self._consumer_task:
            await self._consumer_task
        self._running = False
        logger.info(
            "GPUSaturator stopped — %d records in %d batches, GPU util %.1f%%",
            self._stats.total_completed,
            self._stats.total_batches,
            self._stats.gpu_utilization * 100,
        )

    async def submit(
        self,
        record: RecordT,
        *,
        estimated_tokens: int | None = None,
    ) -> asyncio.Future[ResultT]:
        """Submit a single record for batched inference.

        Returns an :class:`asyncio.Future` that resolves once the batch
        containing this record has been processed.

        Args:
            record: The input record.
            estimated_tokens: Optional explicit token count.  When omitted
                the saturator tries ``token_estimator``, then
                ``record.estimate_tokens()``, then ``default_token_estimate``.

        Returns:
            A future whose result is the corresponding entry from the list
            returned by ``inference_fn``.

        Raises:
            RuntimeError: If the saturator is not running.

        Note:
            If the internal queue is full this coroutine awaits until space
            is available, providing natural backpressure to fast producers.

        Example::

            future = await saturator.submit(my_record, estimated_tokens=128)
            result = await future
        """
        if not self._running:
            raise RuntimeError("GPUSaturator is not running. Call start() or use 'async with'.")

        tokens = self._estimate_tokens(record, estimated_tokens)
        envelope: _Envelope[RecordT, ResultT] = _Envelope(record=record, estimated_tokens=tokens)
        await self._queue.put(envelope)
        self._stats.total_submitted += 1
        return envelope.future

    async def submit_batch(
        self,
        records: Sequence[RecordT],
        *,
        estimated_tokens: Sequence[int] | None = None,
    ) -> list[asyncio.Future[ResultT]]:
        """Convenience: submit multiple records and return their futures.

        Args:
            records: Iterable of input records.
            estimated_tokens: Optional per-record token estimates.  Length
                must match *records* when provided.

        Returns:
            List of futures, one per record.
        """
        futures = []
        for i, record in enumerate(records):
            tok = estimated_tokens[i] if estimated_tokens is not None else None
            f = await self.submit(record, estimated_tokens=tok)
            futures.append(f)
        return futures

    # -- Context manager ---------------------------------------------------

    async def __aenter__(self) -> GPUSaturator[RecordT, ResultT]:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # -- Token estimation --------------------------------------------------

    def _estimate_tokens(self, record: RecordT, override: int | None) -> int:
        if override is not None:
            return override
        if self._token_estimator is not None:
            return self._token_estimator(record)
        if isinstance(record, TokenEstimator):
            return record.estimate_tokens()
        return self._default_token_estimate

    # -- Aggregation loop --------------------------------------------------

    async def _aggregation_loop(self) -> None:
        """Drain the submission queue and assemble token-budgeted batches."""
        while True:
            batch: list[_Envelope[RecordT, ResultT]] = []
            token_count = 0

            # Block until the first item arrives
            envelope = await self._queue.get()
            if envelope is None:
                # Shutdown sentinel
                if batch:
                    await self._batch_queue.put(batch)
                await self._batch_queue.put(None)
                return

            batch.append(envelope)
            token_count += envelope.estimated_tokens

            # Fill the rest of the batch within the timeout window
            deadline = time.monotonic() + self._batch_timeout_s
            while token_count < self._target_batch_tokens and len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    envelope = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    break

                if envelope is None:
                    # Shutdown while filling a batch
                    if batch:
                        await self._batch_queue.put(batch)
                    await self._batch_queue.put(None)
                    return

                batch.append(envelope)
                token_count += envelope.estimated_tokens

            if len(batch) >= self._min_batch_size or self._shutdown_event.is_set():
                await self._batch_queue.put(batch)
            else:
                # Below min_batch_size and not shutting down — re-enqueue
                for env in batch:
                    await self._queue.put(env)
                await asyncio.sleep(self._batch_timeout_s)

    # -- Inference loop ----------------------------------------------------

    async def _inference_loop(self) -> None:
        """Pull assembled batches and run inference, resolving futures."""
        idle_start = time.monotonic()

        while True:
            batch = await self._batch_queue.get()

            # Track idle time (waiting for a batch)
            idle_end = time.monotonic()
            self._stats.gpu_idle_time_s += idle_end - idle_start

            if batch is None:
                return

            records = [env.record for env in batch]
            batch_tokens = sum(env.estimated_tokens for env in batch)

            busy_start = time.monotonic()
            try:
                results = await self._inference_fn(records)

                if len(results) != len(batch):
                    raise ValueError(f"inference_fn returned {len(results)} results for batch of {len(batch)} records")

                for envelope, result in zip(batch, results):
                    if not envelope.future.done():
                        envelope.future.set_result(result)

            except Exception as exc:
                logger.error("Inference batch failed: %s", exc)
                for envelope in batch:
                    if not envelope.future.done():
                        envelope.future.set_exception(exc)

            busy_end = time.monotonic()
            self._stats.gpu_busy_time_s += busy_end - busy_start
            self._stats.total_batches += 1
            self._stats.total_completed += len(batch)
            self._stats.total_batch_tokens += batch_tokens
            self._stats.avg_batch_size = self._stats.total_completed / self._stats.total_batches
            self._stats.avg_batch_tokens = self._stats.total_batch_tokens / self._stats.total_batches

            idle_start = time.monotonic()


__all__ = [
    "BatchStats",
    "GPUSaturator",
    "InferenceFn",
    "TokenEstimator",
    "TokenEstimatorFn",
]
