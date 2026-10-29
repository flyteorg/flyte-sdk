from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import flyte.errors
from flyte._context import internal_ctx
from flyte._task import AsyncFunctionTaskTemplate
from flyte._timeout import Timeout, timeout_from_request

from ._provider import CapacityProvider, Reservation
from ._shape import PodSet


class NoCapacityError(flyte.errors.RuntimeUserError):
    """No candidate cluster reserved the requested capacity within the search timeout."""

    def __init__(self, message: str):
        super().__init__("NoCapacity", message, "user")


@dataclass(frozen=True)
class AutoPlaceSettings:
    """Snapshot of an AutoPlaceEnvironment's placement settings, carried by its task templates."""

    clusters: Tuple[str, ...]
    podsets: Tuple[PodSet, ...]
    search_timeout: timedelta
    booking_window: timedelta
    provider: CapacityProvider
    max_placements: int = 2


@dataclass(kw_only=True)
class AutoPlaceTaskTemplate(AsyncFunctionTaskTemplate):
    """
    A Python task whose cluster is chosen at call time.

    Called from a parent task, it asks the provider to reserve capacity in every cluster in
    `autoplace.clusters` at once, takes the first reservation that comes through, cancels the rest,
    and submits itself to that cluster's queue with a pod template that consumes the reserved
    machines. Called outside a task context it is an ordinary task (local runs are untouched).
    `override(queue=...)` pins a cluster and skips the search.

    Serialization is unchanged: the backend sees a plain task with a queue and a pod template.
    """

    autoplace: Optional[AutoPlaceSettings] = None
    # Set on the template that comes out of the search (or out of `override(queue=)`), so the
    # placed submit does not search again.
    placed: bool = False

    def _searches(self) -> bool:
        if self.autoplace is None or self.placed:
            return False
        ctx = internal_ctx()
        return ctx.is_task_context() and not ctx.is_in_trace()

    def override(self, **kwargs: Any) -> AutoPlaceTaskTemplate:  # type: ignore[override]
        tmpl = super().override(**kwargs)
        if kwargs.get("queue") is not None:
            tmpl.placed = True
        return tmpl  # type: ignore[return-value]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._searches():
            return super().__call__(*args, **kwargs)
        if self._call_as_synchronous:
            raise flyte.errors.RuntimeUserError(
                "AutoPlaceSyncCall",
                f"Sync task '{self.name}' in an AutoPlaceEnvironment must be awaited with "
                f"`.aio(...)`: the capacity search runs on the event loop.",
                "user",
            )
        return self._autoplace_submit(*args, **kwargs)

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        if not self._searches():
            return await super().aio(*args, **kwargs)
        return await self._autoplace_submit(*args, **kwargs)

    async def _autoplace_submit(self, *args: Any, **kwargs: Any) -> Any:
        assert self.autoplace is not None
        settings = self.autoplace
        misses = []
        for _ in range(settings.max_placements):
            reservation = await self._reserve()
            placed = self.override(
                queue=reservation.queue,
                pod_template=settings.provider.consume(self.pod_template, reservation),  # type: ignore[arg-type]
                timeout=self._booking_timeout(),
            )
            try:
                return await placed.aio(*args, **kwargs)
            except flyte.errors.MaxQueuedTimeExceededError as e:
                # Not scheduled inside the cloud's hold on the machines: they are gone, and a pod pinned
                # to this reservation would wait forever. Search again rather than retry in place.
                misses.append(f"{reservation.queue}: {e}")
        raise NoCapacityError(
            f"Task {self.name}: capacity was reserved but the task was not scheduled within the booking "
            f"window {settings.max_placements} time(s): " + "; ".join(misses)
        )

    def _booking_timeout(self) -> Timeout:
        """
        The task's own timeout with `max_queued_time` capped at the booking window, so a placement
        that misses the window fails fast instead of pinning a pod to released machines.
        """
        assert self.autoplace is not None
        base = timeout_from_request(self.timeout) if self.timeout else Timeout()
        window = self.autoplace.booking_window
        queued = base.max_queued_time
        if isinstance(queued, int):
            queued = timedelta(seconds=queued)
        if queued and queued < window:
            window = queued
        return Timeout(max_runtime=base.max_runtime, max_queued_time=window, deadline=base.deadline)

    async def _reserve(self) -> Reservation:
        """
        Ask every cluster at once; return the first reservation that is `ok`. Losing probes are
        cancelled, which aborts their child actions and lets the provider withdraw their requests.
        Raises NoCapacityError when every cluster said no.
        """
        assert self.autoplace is not None
        s = self.autoplace
        probes: Dict[asyncio.Future, str] = {}
        for queue in s.clusters:
            probes[asyncio.ensure_future(s.provider.reserve(queue, s.podsets, s.search_timeout))] = queue
        misses: Dict[str, str] = {}
        try:
            while probes:
                done, _ = await asyncio.wait(probes, return_when=asyncio.FIRST_COMPLETED)
                for fut in done:
                    queue = probes.pop(fut)
                    try:
                        reservation: Reservation = fut.result()
                    except Exception as e:
                        misses[queue] = f"probe failed: {e!r}"
                        continue
                    if reservation.ok:
                        return reservation
                    misses[queue] = reservation.reason or "no capacity"
            raise NoCapacityError(
                f"Task {self.name}: no cluster in {list(s.clusters)} could reserve {list(s.podsets)} "
                f"within {s.search_timeout}: " + "; ".join(f"{q}: {r}" for q, r in misses.items())
            )
        finally:
            for fut in probes:
                fut.cancel()
            if probes:
                await asyncio.gather(*probes, return_exceptions=True)
