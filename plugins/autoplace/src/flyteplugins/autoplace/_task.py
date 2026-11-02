from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List, Optional, Tuple

import flyte
import flyte.errors
from flyte._context import internal_ctx
from flyte._task import AsyncFunctionTaskTemplate
from flyte._timeout import Timeout, timeout_from_request

from ._provider import CapacityProvider, Reservation
from ._shape import PodSet

logger = logging.getLogger(__name__)

# A reservation is placed on only if at least this much of the cloud's hold is left for the task to be
# scheduled; less than that and it is released and the search runs again.
_MIN_QUEUE_BUDGET = timedelta(minutes=1)
# Slack kept between the placed task's queue bound and the end of the hold, for clock skew between the
# cloud (which stamps `provisioned_at`) and the driver, and for the pod to bind once scheduled.
_HOLD_SAFETY = timedelta(seconds=30)


class NoCapacityError(flyte.errors.RuntimeUserError):
    """No candidate cluster reserved the requested capacity within the search timeout."""

    def __init__(self, message: str):
        super().__init__("NoCapacity", message, "user")


def queue_budget(provisioned_at: float, now: float, hold: timedelta, window: timedelta) -> timedelta:
    """
    How long the placed task may wait in the queue: the booking window, cut to what is left of the
    cloud's hold on a reservation provisioned at `provisioned_at` (epoch seconds; 0 means unknown,
    treated as just now). Zero when too little is left to be worth placing on.
    """
    if not provisioned_at:
        return window
    remaining = hold - timedelta(seconds=max(0.0, now - provisioned_at)) - _HOLD_SAFETY
    budget = min(window, remaining)
    return budget if budget >= _MIN_QUEUE_BUDGET else timedelta(0)


@flyte.trace
async def placement_budget(reservation: str, provisioned_at: float, hold_seconds: float, window_seconds: float) -> int:
    """
    `queue_budget` for one reservation, recorded: the decision reads the wall clock, so a driver that
    is retried after the hold has passed would otherwise decide differently about a placement it already
    made. Recorded, the retry replays "place it" and re-attaches to the task it submitted.
    """
    budget = queue_budget(
        provisioned_at, time.time(), timedelta(seconds=hold_seconds), timedelta(seconds=window_seconds)
    )
    return int(budget.total_seconds())


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

    Called from a parent task, it has the provider search every cluster in `autoplace.clusters` at once
    (a durable child task, so a retried parent gets the same answer), releases any extra cluster that
    also came through, and submits itself to the winner's queue with a pod template that consumes the
    reserved machines and a queue bound that fits inside what is left of the cloud's hold. The winner's
    reservation is released when the task finishes. Called outside a task context it is an ordinary
    task (local runs are untouched). `override(queue=...)` pins a cluster and skips the search.

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
        s = self.autoplace
        hold = s.provider.capacity_hold.total_seconds()
        misses: List[str] = []
        for _ in range(s.max_placements):
            selection = await s.provider.select(s.clusters, s.podsets, s.search_timeout)
            winner = selection.winner
            # Extra winners are handed back while the task is placed, not after: their machines are
            # provisioned and idle, and they hold room in their pools that the next search may need.
            releases = [asyncio.ensure_future(self._release(r)) for r in selection.extras]
            try:
                if winner is None:
                    raise NoCapacityError(
                        f"Task {self.name}: no cluster in {list(s.clusters)} could reserve {list(s.podsets)} "
                        f"within {s.search_timeout}: " + "; ".join(f"{q}: {r}" for q, r in selection.misses.items())
                    )
                budget = await placement_budget(
                    winner.name, winner.provisioned_at, hold, s.booking_window.total_seconds()
                )
                if budget <= 0:
                    misses.append(f"{winner.queue}: reservation {winner.name} expired before the task was placed")
                    continue
                placed = self.override(
                    queue=winner.queue,
                    pod_template=s.provider.consume(self.pod_template, winner),  # type: ignore[arg-type]
                    timeout=self._booking_timeout(timedelta(seconds=budget)),
                )
                try:
                    return await placed.aio(*args, **kwargs)
                except flyte.errors.MaxQueuedTimeExceededError as e:
                    # Not scheduled inside the cloud's hold on the machines: they are gone, and a pod pinned
                    # to this reservation would wait forever. Search again rather than retry in place.
                    misses.append(f"{winner.queue}: {e}")
            finally:
                if winner is not None:
                    releases.append(asyncio.ensure_future(self._release(winner)))
                if releases:
                    await asyncio.gather(*releases)
        raise NoCapacityError(
            f"Task {self.name}: capacity was reserved but the task was not scheduled within the booking "
            f"window {s.max_placements} time(s): " + "; ".join(misses)
        )

    async def _release(self, reservation: Reservation) -> None:
        """Release one reservation; a failure is logged, never raised over the task's own outcome."""
        assert self.autoplace is not None
        try:
            await self.autoplace.provider.release(reservation)
        except Exception as e:
            logger.warning(f"Could not release reservation {reservation.name} in {reservation.queue}: {e!r}")

    def _booking_timeout(self, budget: timedelta) -> Timeout:
        """
        The task's own timeout with `max_queued_time` capped at `budget`, so a placement that misses
        the cloud's hold fails fast instead of pinning a pod to released machines.
        """
        base = timeout_from_request(self.timeout) if self.timeout else Timeout()
        queued = base.max_queued_time
        if isinstance(queued, int):
            queued = timedelta(seconds=queued)
        if queued and queued < budget:
            budget = queued
        return Timeout(max_runtime=base.max_runtime, max_queued_time=budget, deadline=base.deadline)
