"""The run that performs a backfill.

``flyte backfill`` does not create the backfilled runs from your machine. It
launches one small driver run that creates them from inside the cluster, so a
long backfill does not depend on a laptop staying connected, and so the work is
itself observable, retryable, and abortable like any other run.

Every run the driver creates is linked back to it as a spawned child, so the
backfill shows up as one tree rather than a scatter of unrelated runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flyte.remote import Run

    from ._plan import BackfillPlan

__all__ = ["backfill_driver", "encode_plan", "launch_backfill"]


@dataclass
class _EncodedPlan:
    """The approved plan, in the form the driver receives it.

    The exact slot list is carried across rather than recomputed, so the driver
    creates precisely what was shown and approved -- no re-expansion, no drift if
    the trigger's schedule changes in between.
    """

    org: str
    project: str
    domain: str
    task_name: str
    trigger_name: str
    force: bool
    salt: str | None
    slots: list[dict]


def encode_plan(plan: "BackfillPlan") -> str:
    payload = _EncodedPlan(
        org=plan.org,
        project=plan.project,
        domain=plan.domain,
        task_name=plan.task_name,
        trigger_name=plan.trigger_name,
        force=plan.force,
        salt=plan.salt,
        slots=[
            {
                "at": s.scheduled_at.isoformat(),
                "name": s.run_name,
                "already_ran": s.already_ran,
            }
            for s in plan.to_create
        ],
    )
    return json.dumps(payload.__dict__)


def _decode_plan(encoded: str) -> tuple["BackfillPlan", list]:
    from ._plan import BackfillPlan, BackfillSlot

    raw = json.loads(encoded)
    slots = [
        BackfillSlot(
            scheduled_at=datetime.fromisoformat(s["at"]),
            run_name=s["name"],
            already_ran=s["already_ran"],
        )
        for s in raw["slots"]
    ]
    plan = BackfillPlan(
        trigger_name=raw["trigger_name"],
        task_name=raw["task_name"],
        project=raw["project"],
        domain=raw["domain"],
        org=raw["org"],
        schedule="",
        start=slots[0].scheduled_at if slots else datetime.now(timezone.utc),
        end=slots[-1].scheduled_at if slots else datetime.now(timezone.utc),
        force=raw["force"],
        salt=raw["salt"],
        queue=None,
        max_runs=len(slots),
        slots=slots,
    )
    return plan, slots


async def backfill_driver(encoded_plan: str) -> str:
    """Create every run in the encoded plan. Runs inside the cluster.

    Returns a short summary. Individual slot failures are reported rather than
    raised, so one rejected slot does not abandon the rest of the backfill.
    """
    from flyte.remote import Trigger

    from ._execute import execute_plan

    plan, _ = _decode_plan(encoded_plan)
    details = await Trigger.get.aio(name=plan.trigger_name, task_name=plan.task_name)

    created = 0
    deduped = 0
    failed: list[str] = []

    def _record(result) -> None:
        nonlocal created, deduped
        if result.error:
            failed.append(f"{result.slot.scheduled_at.isoformat()}: {result.error}")
        elif result.created:
            created += 1
        else:
            deduped += 1

    await execute_plan(plan, details, on_slot=_record)

    summary = f"backfill {plan.trigger_name}: {created} created, {deduped} already existed, {len(failed)} failed"
    if failed:
        summary += "\n" + "\n".join(failed[:20])
    print(summary, flush=True)
    return summary


def launch_backfill(plan: "BackfillPlan", name: str | None = None) -> "Run":
    """Launch the driver run that performs ``plan``."""
    import flyte

    env = flyte.TaskEnvironment(
        name="flyte-backfill",
        resources=flyte.Resources(cpu=1, memory="500Mi"),
    )
    driver = env.task(backfill_driver)
    run = flyte.with_runcontext(
        mode="remote",
        name=name,
        # The driver has no local source file of its own; ship it as a code bundle.
        interactive_mode=True,
        queue=plan.queue,
    ).run(driver, encode_plan(plan))
    from typing import cast

    from flyte.remote import Run as _Run

    return cast(_Run, run)
