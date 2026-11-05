from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, Mapping, Optional

from ._provider import Reservation, Selection

CANCELLED = "cancelled: another cluster won"


async def race(
    probes: Mapping[str, Callable[[], Awaitable[Reservation]]], handles: Optional[Mapping[str, str]] = None
) -> Selection:
    """
    Start every probe at once (keyed by queue) and return the first reservation that comes through,
    plus every other one that also did.

    More than one can: clusters answer independently, and a probe that has already returned a win
    cannot be cancelled into giving it back. So once the winner is known the still-running probes are
    cancelled, but the ones that completed in the same instant, or that finish before the cancel
    lands, are collected as `extras` for the caller to release rather than dropped. Every cluster that
    said no is in `misses` with its reason. A probe that raises counts as a miss, never ends the race.

    A probe whose cancellation went through can still have won on the cloud's side; its outcome is lost
    with the cancel. When `handles` names the reservation each queue's probe creates (providers that
    pick the names up front can), such probes are returned in `unresolved` with that handle, so the
    caller can release whatever they may have left behind.

    Providers call this from inside their `select` task, with probes that are child tasks in the
    candidate clusters.
    """
    pending: Dict[asyncio.Future, str] = {asyncio.ensure_future(start()): queue for queue, start in probes.items()}
    selection = Selection()

    def collect(fut: asyncio.Future, queue: str) -> None:
        if fut.cancelled():
            selection.misses.setdefault(queue, CANCELLED)
            if handles and handles.get(queue):
                selection.unresolved.append(Reservation(queue=queue, name=handles[queue], reason=CANCELLED))
            return
        try:
            reservation: Reservation = fut.result()
        except Exception as e:
            selection.misses[queue] = f"probe failed: {e!r}"
            return
        if not reservation.ok:
            selection.misses[queue] = reservation.reason or "no capacity"
        elif selection.winner is None:
            selection.winner = reservation
        else:
            selection.extras.append(reservation)

    try:
        while pending and selection.winner is None:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # Sorted so that when several probes complete together, which one wins does not depend on
            # set iteration order.
            for fut in sorted(done, key=lambda f: pending[f]):
                collect(fut, pending.pop(fut))
    finally:
        for fut in pending:
            fut.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            for fut, queue in pending.items():
                collect(fut, queue)
    return selection
