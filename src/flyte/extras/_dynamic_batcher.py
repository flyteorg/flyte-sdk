"""
DynamicBatcher — maximize resource utilization by batching work from many
concurrent producers through a single async processing function.

This is useful for any scenario where a shared, expensive resource (GPU, API
endpoint, external service) benefits from batched requests:

- **GPU inference** — assemble token-budgeted batches from concurrent streams
  and feed them through a single model.
- **API calls** — batch many small requests into fewer bulk calls to stay
  within rate limits and reduce overhead.
- **Generic processing** — any async function that is more efficient when
  operating on batches rather than individual items.

Quick start::

    from flyte.extras import DynamicBatcher

    async def process(batch: list[str]) -> list[str]:
        return [f"result for {item}" for item in batch]

    async with DynamicBatcher(process_fn=process) as batcher:
        future = await batcher.submit("hello", estimated_cost=5)
        result = await future

For the common LLM / token-budgeted use case, use :class:`TokenBatcher`::

    from flyte.extras import TokenBatcher

    async def inference(batch: list[Prompt]) -> list[str]:
        return [f"answer to {p.text}" for p in batch]

    async with TokenBatcher(inference_fn=inference) as batcher:
        future = await batcher.submit(Prompt(text="What is 2+2?"))
        result = await future
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


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class CostEstimator(Protocol):
    """Protocol for records that can estimate their own processing cost.

    Implement this on your record type and the batcher will call it
    automatically when no explicit ``estimated_cost`` is passed to
    :meth:`DynamicBatcher.submit`.

    Example::

        @dataclass
        class ApiRequest:
            payload: str

            def estimate_cost(self) -> int:
                return len(self.payload)
    """

    def estimate_cost(self) -> int: ...


@runtime_checkable
class TokenEstimator(Protocol):
    """Protocol for records that can estimate their own token count.

    Implement this on your record type and the :class:`TokenBatcher` will
    call it automatically when no explicit ``estimated_tokens`` is passed
    to :meth:`TokenBatcher.submit`.

    Example::

        @dataclass
        class Prompt:
            text: str

            def estimate_tokens(self) -> int:
                return len(self.text) // 4 + 1
    """

    def estimate_tokens(self) -> int: ...


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ProcessFn = Callable[[list[RecordT]], Awaitable[list[ResultT]]]
"""Async function that takes a batch of records and returns results in the
same order.  Must be a native coroutine (``async def``)."""

InferenceFn = ProcessFn
"""Alias for :data:`ProcessFn` — kept for LLM inference use cases."""

CostEstimatorFn = Callable[[RecordT], int]
"""Optional callable ``(record) -> int`` for cost estimation."""

TokenEstimatorFn = CostEstimatorFn
"""Alias for :data:`CostEstimatorFn` — kept for token estimation use cases."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_future() -> asyncio.Future:
    """Create a Future on the running event loop."""
    return asyncio.get_running_loop().create_future()


@dataclass
class _Envelope(Generic[RecordT, ResultT]):
    """Internal wrapper pairing a record with its result future."""

    record: RecordT
    estimated_cost: int
    future: asyncio.Future[ResultT] = field(default_factory=_make_future)


# ---------------------------------------------------------------------------
# BatchStats
# ---------------------------------------------------------------------------


@dataclass
class BatchStats:
    """Monitoring statistics exposed by :attr:`DynamicBatcher.stats`.

    Attributes:
        total_submitted: Total records submitted via :meth:`submit`.
        total_completed: Total records whose futures have been resolved.
        total_batches: Number of batches dispatched.
        total_batch_cost: Sum of estimated cost across all batches.
        avg_batch_size: Running average records per batch.
        avg_batch_cost: Running average cost per batch.
        busy_time_s: Cumulative seconds spent inside ``process_fn``.
        idle_time_s: Cumulative seconds the processing loop waited for
            a batch to be assembled.
    """

    total_submitted: int = 0
    total_completed: int = 0
    total_batches: int = 0
    total_batch_cost: int = 0
    avg_batch_size: float = 0.0
    avg_batch_cost: float = 0.0
    busy_time_s: float = 0.0
    idle_time_s: float = 0.0

    @property
    def utilization(self) -> float:
        """Fraction of wall-clock time spent processing (0.0-1.0)."""
        total = self.busy_time_s + self.idle_time_s
        return self.busy_time_s / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# DynamicBatcher
# ---------------------------------------------------------------------------


class DynamicBatcher(Generic[RecordT, ResultT]):
    """Batches records from many concurrent producers and runs them through
    a single async processing function, maximizing resource utilization.

    The batcher runs two internal loops:

    1. **Aggregation loop** — drains the submission queue and assembles
       cost-budgeted batches, respecting ``target_batch_cost``,
       ``max_batch_size``, and ``batch_timeout_s``.
    2. **Processing loop** — pulls assembled batches and calls
       ``process_fn``, resolving each record's :class:`asyncio.Future`.

    Type Parameters:
        RecordT: The input record type produced by your tasks.
        ResultT: The per-record output type returned by ``process_fn``.

    Args:
        process_fn:
            ``async def f(batch: list[RecordT]) -> list[ResultT]``
            Must return results in the **same order** as the input batch.

        cost_estimator:
            Optional ``(RecordT) -> int`` function.  When provided, it is
            called to estimate the cost of each submitted record.
            Falls back to ``record.estimate_cost()`` if the record
            implements :class:`CostEstimator`, then to ``default_cost``.

        target_batch_cost:
            Cost budget per batch.  The aggregator fills batches up to
            this limit before dispatching.

        max_batch_size:
            Hard cap on records per batch regardless of cost budget.

        min_batch_size:
            Minimum records before dispatching.  Ignored when the timeout
            fires or shutdown is in progress.

        batch_timeout_s:
            Maximum seconds to wait for a full batch.  Lower values reduce
            idle time but may produce smaller batches.

        max_queue_size:
            Bounded queue size.  When full, :meth:`submit` awaits
            (backpressure).

        prefetch_batches:
            Number of pre-assembled batches to buffer between the
            aggregation and processing loops.

        default_cost:
            Fallback cost when no estimator is available.

    Example::

        async def process(batch: list[dict]) -> list[str]:
            ...

        async with DynamicBatcher(process_fn=process) as batcher:
            futures = []
            for record in my_records:
                f = await batcher.submit(record)
                futures.append(f)
            results = await asyncio.gather(*futures)
    """

    def __init__(
        self,
        process_fn: ProcessFn[RecordT, ResultT],
        *,
        cost_estimator: CostEstimatorFn[RecordT] | None = None,
        target_batch_cost: int = 32_000,
        max_batch_size: int = 256,
        min_batch_size: int = 1,
        batch_timeout_s: float = 0.05,
        max_queue_size: int = 5_000,
        prefetch_batches: int = 2,
        default_cost: int = 1,
    ):
        self._process_fn = process_fn
        self._cost_estimator = cost_estimator
        self._target_batch_cost = target_batch_cost
        self._max_batch_size = max_batch_size
        self._min_batch_size = min_batch_size
        self._batch_timeout_s = batch_timeout_s
        self._prefetch_batches = prefetch_batches
        self._default_cost = default_cost

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
        """Whether the aggregation and processing loops are active."""
        return self._running

    async def start(self) -> None:
        """Start the aggregation and processing loops.

        Raises:
            RuntimeError: If the batcher is already running.
        """
        if self._running:
            raise RuntimeError(f"{type(self).__name__} is already running")
        self._running = True
        self._shutdown_event.clear()
        self._aggregator_task = asyncio.create_task(
            self._aggregation_loop(),
            name="dynamic-batcher-aggregator",
        )
        self._consumer_task = asyncio.create_task(
            self._processing_loop(),
            name="dynamic-batcher-consumer",
        )
        logger.info("%s started", type(self).__name__)

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
            "%s stopped — %d records in %d batches, utilization %.1f%%",
            type(self).__name__,
            self._stats.total_completed,
            self._stats.total_batches,
            self._stats.utilization * 100,
        )

    async def submit(
        self,
        record: RecordT,
        *,
        estimated_cost: int | None = None,
    ) -> asyncio.Future[ResultT]:
        """Submit a single record for batched processing.

        Returns an :class:`asyncio.Future` that resolves once the batch
        containing this record has been processed.

        Args:
            record: The input record.
            estimated_cost: Optional explicit cost.  When omitted the
                batcher tries ``cost_estimator``, then
                ``record.estimate_cost()``, then ``default_cost``.

        Returns:
            A future whose result is the corresponding entry from the list
            returned by ``process_fn``.

        Raises:
            RuntimeError: If the batcher is not running.

        Note:
            If the internal queue is full this coroutine awaits until space
            is available, providing natural backpressure to fast producers.

        Example::

            future = await batcher.submit(my_record, estimated_cost=128)
            result = await future
        """
        if not self._running:
            raise RuntimeError(f"{type(self).__name__} is not running. Call start() or use 'async with'.")

        cost = self._estimate_cost(record, estimated_cost)
        envelope: _Envelope[RecordT, ResultT] = _Envelope(record=record, estimated_cost=cost)
        await self._queue.put(envelope)
        self._stats.total_submitted += 1
        return envelope.future

    async def submit_batch(
        self,
        records: Sequence[RecordT],
        *,
        estimated_cost: Sequence[int] | None = None,
    ) -> list[asyncio.Future[ResultT]]:
        """Convenience: submit multiple records and return their futures.

        Args:
            records: Iterable of input records.
            estimated_cost: Optional per-record cost estimates.  Length
                must match *records* when provided.

        Returns:
            List of futures, one per record.
        """
        futures = []
        for i, record in enumerate(records):
            cost = estimated_cost[i] if estimated_cost is not None else None
            f = await self.submit(record, estimated_cost=cost)
            futures.append(f)
        return futures

    # -- Context manager ---------------------------------------------------

    async def __aenter__(self) -> DynamicBatcher[RecordT, ResultT]:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # -- Cost estimation ---------------------------------------------------

    def _estimate_cost(self, record: RecordT, override: int | None) -> int:
        if override is not None:
            return override
        if self._cost_estimator is not None:
            return self._cost_estimator(record)
        if isinstance(record, CostEstimator):
            return record.estimate_cost()
        return self._default_cost

    # -- Aggregation loop --------------------------------------------------

    async def _aggregation_loop(self) -> None:
        """Drain the submission queue and assemble cost-budgeted batches."""
        while True:
            batch: list[_Envelope[RecordT, ResultT]] = []
            cost_count = 0

            # Block until the first item arrives
            envelope = await self._queue.get()
            if envelope is None:
                # Shutdown sentinel
                if batch:
                    await self._batch_queue.put(batch)
                await self._batch_queue.put(None)
                return

            batch.append(envelope)
            cost_count += envelope.estimated_cost

            # Fill the rest of the batch within the timeout window
            deadline = time.monotonic() + self._batch_timeout_s
            while cost_count < self._target_batch_cost and len(batch) < self._max_batch_size:
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
                cost_count += envelope.estimated_cost

            if len(batch) >= self._min_batch_size or self._shutdown_event.is_set():
                await self._batch_queue.put(batch)
            else:
                # Below min_batch_size and not shutting down — re-enqueue
                for env in batch:
                    await self._queue.put(env)
                await asyncio.sleep(self._batch_timeout_s)

    # -- Processing loop ---------------------------------------------------

    async def _processing_loop(self) -> None:
        """Pull assembled batches and run processing, resolving futures."""
        idle_start = time.monotonic()

        while True:
            batch = await self._batch_queue.get()

            # Track idle time (waiting for a batch)
            idle_end = time.monotonic()
            self._stats.idle_time_s += idle_end - idle_start

            if batch is None:
                return

            records = [env.record for env in batch]
            batch_cost = sum(env.estimated_cost for env in batch)

            busy_start = time.monotonic()
            try:
                results = await self._process_fn(records)

                if len(results) != len(batch):
                    raise ValueError(f"process_fn returned {len(results)} results for batch of {len(batch)} records")

                for envelope, result in zip(batch, results):
                    if not envelope.future.done():
                        envelope.future.set_result(result)

            except Exception as exc:
                logger.error("%s batch failed: %s", type(self).__name__, exc)
                for envelope in batch:
                    if not envelope.future.done():
                        envelope.future.set_exception(exc)

            busy_end = time.monotonic()
            self._stats.busy_time_s += busy_end - busy_start
            self._stats.total_batches += 1
            self._stats.total_completed += len(batch)
            self._stats.total_batch_cost += batch_cost
            self._stats.avg_batch_size = self._stats.total_completed / self._stats.total_batches
            self._stats.avg_batch_cost = self._stats.total_batch_cost / self._stats.total_batches

            idle_start = time.monotonic()


# ---------------------------------------------------------------------------
# Prompt — shipped convenience record for LLM use cases
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    """Simple prompt record with built-in token estimation.

    This is a convenience type for common LLM use cases.  For richer
    prompt types (e.g. with system messages, metadata), define your own
    dataclass implementing :class:`TokenEstimator`.

    Attributes:
        text: The prompt text.
    """

    text: str

    def estimate_tokens(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        return len(self.text) // 4 + 1


# ---------------------------------------------------------------------------
# TokenBatcher — thin subclass for token-budgeted LLM inference
# ---------------------------------------------------------------------------


class TokenBatcher(DynamicBatcher[RecordT, ResultT]):
    """Token-aware batcher for LLM inference workloads.

    A thin convenience wrapper around :class:`DynamicBatcher` that accepts
    token-specific parameter names (``inference_fn``, ``token_estimator``,
    ``target_batch_tokens``, etc.) and maps them to the base class.

    Also checks the :class:`TokenEstimator` protocol (``estimate_tokens()``)
    in addition to :class:`CostEstimator` (``estimate_cost()``).

    Example::

        async def inference(batch: list[Prompt]) -> list[str]:
            ...

        async with TokenBatcher(inference_fn=inference) as batcher:
            future = await batcher.submit(Prompt(text="Hello"))
            result = await future
    """

    def __init__(
        self,
        inference_fn: ProcessFn[RecordT, ResultT] | None = None,
        *,
        process_fn: ProcessFn[RecordT, ResultT] | None = None,
        token_estimator: CostEstimatorFn[RecordT] | None = None,
        cost_estimator: CostEstimatorFn[RecordT] | None = None,
        target_batch_tokens: int | None = None,
        target_batch_cost: int = 32_000,
        default_token_estimate: int | None = None,
        default_cost: int = 1,
        max_batch_size: int = 256,
        min_batch_size: int = 1,
        batch_timeout_s: float = 0.05,
        max_queue_size: int = 5_000,
        prefetch_batches: int = 2,
    ):
        fn = inference_fn or process_fn
        if fn is None:
            raise TypeError("TokenBatcher requires either 'inference_fn' or 'process_fn'")

        super().__init__(
            process_fn=fn,
            cost_estimator=token_estimator or cost_estimator,
            target_batch_cost=target_batch_tokens if target_batch_tokens is not None else target_batch_cost,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            batch_timeout_s=batch_timeout_s,
            max_queue_size=max_queue_size,
            prefetch_batches=prefetch_batches,
            default_cost=default_token_estimate if default_token_estimate is not None else default_cost,
        )

    def _estimate_cost(self, record: RecordT, override: int | None) -> int:
        if override is not None:
            return override
        if self._cost_estimator is not None:
            return self._cost_estimator(record)
        if isinstance(record, TokenEstimator):
            return record.estimate_tokens()
        if isinstance(record, CostEstimator):
            return record.estimate_cost()
        return self._default_cost

    async def submit(
        self,
        record: RecordT,
        *,
        estimated_tokens: int | None = None,
        estimated_cost: int | None = None,
    ) -> asyncio.Future[ResultT]:
        """Submit a single record for batched inference.

        Accepts either ``estimated_tokens`` or ``estimated_cost``.

        Args:
            record: The input record.
            estimated_tokens: Optional explicit token count.
            estimated_cost: Optional explicit cost (base class parameter).

        Returns:
            A future whose result is the corresponding entry from the list
            returned by the inference function.
        """
        cost = estimated_tokens if estimated_tokens is not None else estimated_cost
        return await super().submit(record, estimated_cost=cost)


__all__ = [
    "BatchStats",
    "CostEstimator",
    "CostEstimatorFn",
    "DynamicBatcher",
    "InferenceFn",
    "ProcessFn",
    "Prompt",
    "TokenBatcher",
    "TokenEstimator",
    "TokenEstimatorFn",
]
