from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import flyte
import flyte.errors
from flyte import Timeout
from flyte._context import internal_ctx
from flyte._timeout import timeout_from_request
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry

from ._provider import CapacityProvider, Reservation, reservation_name
from ._shape import PodSet

logger = logging.getLogger(__name__)

# Slack between the cloud's hold on reserved machines and the placed task's queue bound: time for the
# scheduler to place the action and the image to pull before the hold expires.
BOOKING_MARGIN = timedelta(minutes=2)

# Searches started per parent action, so each search in a parent gets its own reservation names and a
# retried parent, calling in the same order, gets the same ones.
_searches: Dict[str, int] = {}


class NoCapacityError(flyte.errors.RuntimeUserError):
    """No candidate cluster reserved the requested capacity, or the task missed the reservation."""

    def __init__(self, message: str):
        super().__init__("NoCapacity", message, "user")


@dataclass(frozen=True)
class AutoPlaceSettings:
    """An AutoPlaceEnvironment's placement settings, carried by its tasks as their `plugin_config`."""

    cluster_queues: Tuple[str, ...]
    podsets: Tuple[PodSet, ...]
    search_deadline: timedelta
    service_account: str
    provider: CapacityProvider


@dataclass(kw_only=True)
class AutoPlaceTaskTemplate(AsyncFunctionTaskTemplate):
    """
    A Python task whose cluster is chosen at call time.

    Called from a parent task, it asks every queue in `cluster_queues` for capacity at once, takes the
    first cluster that reserves it, releases every other queue, and submits itself to the winner's queue
    with a pod template that consumes the reserved machines. The winner is released when the task
    finishes. Called outside a task context it is an ordinary task (local runs are untouched).
    `override(queue=...)` pins a cluster and skips the search.

    Serialization is unchanged: the backend sees a plain task with a queue and a pod template.
    """

    # Set on the template that comes out of the search (or out of `override(queue=)`), so the
    # placed submit does not search again.
    placed: bool = False

    @property
    def autoplace(self) -> Optional[AutoPlaceSettings]:
        return self.plugin_config if isinstance(self.plugin_config, AutoPlaceSettings) else None

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
        s = self.autoplace
        assert s is not None
        winner = await self._search(s)
        # Everything after the search is inside the `try`, so nothing can skip the winner's release. A parent
        # that dies here is covered when it is retried: it replays the same search and releases the winner.
        try:
            timeout = self._placed_timeout(s)
            placed = self.override(
                queue=winner.queue,
                pod_template=s.provider.consume(self.pod_template, winner),  # type: ignore[arg-type]
                timeout=timeout,
            )
            return await placed.aio(*args, **kwargs)
        except flyte.errors.MaxQueuedTimeExceededError as e:
            # The cloud gave the machines back before the task was scheduled on them.
            raise NoCapacityError(
                f"Task {self.name}: capacity was reserved in {winner.queue} but the task was not scheduled within "
                f"{timeout.max_queued_time}: {e}"
            ) from e
        finally:
            await self._release(s, winner)

    async def _search(self, s: AutoPlaceSettings) -> Reservation:
        """
        Ask every queue at once and return the first reservation; every other queue is released by name,
        whatever its probe did (said no, was cancelled, or reserved at the same moment).
        """
        ctx = flyte.ctx()
        action = ctx.action if ctx else None
        parent = f"{action.run_name}/{action.name}" if action else ""
        seq = _searches.get(parent, 0)
        _searches[parent] = seq + 1
        search = f"{parent}/{self.name}/{seq}"

        names = {q: reservation_name(search, q) for q in s.cluster_queues}
        pending = {
            asyncio.ensure_future(s.provider.reserve(q, names[q], s.podsets, s.search_deadline, s.service_account)): q
            for q in names
        }
        winner: Optional[Reservation] = None
        misses: Dict[str, str] = {}
        try:
            # A probe that says no does not end the search; the first reservation does.
            while pending and winner is None:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for probe in done:
                    queue = pending.pop(probe)
                    try:
                        reservation = probe.result()
                    except flyte.errors.DeadlineExceededError:
                        misses[queue] = f"no capacity within {s.search_deadline}"
                    except Exception as e:
                        misses[queue] = f"probe failed: {e!r}"
                    else:
                        winner = winner or reservation
        finally:
            for probe in pending:
                probe.cancel()

        losers = [Reservation(queue=q, name=n) for q, n in names.items() if winner is None or q != winner.queue]
        await asyncio.gather(*(self._release(s, r) for r in losers))
        if winner is None:
            raise NoCapacityError(
                f"Task {self.name}: no cluster in {list(s.cluster_queues)} could reserve {list(s.podsets)}: "
                + "; ".join(f"{q}: {r}" for q, r in misses.items())
            )
        return winner

    async def _release(self, s: AutoPlaceSettings, reservation: Reservation) -> None:
        """Release one reservation; a failure is logged, never raised over the task's own outcome."""
        try:
            await s.provider.release(reservation, s.service_account)
        except Exception as e:
            logger.warning(f"Could not release reservation {reservation.name} in {reservation.queue}: {e!r}")

    def _placed_timeout(self, s: AutoPlaceSettings) -> Timeout:
        """
        The task's own timeout, with `max_queued_time` capped at the cloud's hold (less a margin), so a task
        that misses the reserved machines fails fast instead of waiting on capacity that is gone, and
        `deadline` capped at how long the machines live.
        """
        base = timeout_from_request(self.timeout) if self.timeout else Timeout()
        return Timeout(
            max_runtime=base.max_runtime,
            max_queued_time=_tighter(base.max_queued_time, s.provider.capacity_hold - BOOKING_MARGIN),
            deadline=_tighter(base.deadline, s.provider.max_run),
        )


def _tighter(own: timedelta | int | None, cap: timedelta) -> timedelta:
    if isinstance(own, int):
        own = timedelta(seconds=own)
    return own if own and own < cap else cap


TaskPluginRegistry.register(AutoPlaceSettings, AutoPlaceTaskTemplate)
