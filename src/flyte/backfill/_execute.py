"""Create the runs a backfill plan describes.

A backfilled slot is created the same way the scheduler creates a real fire: the
run is addressed by ``TriggerName`` (so the control plane resolves the task,
inputs and run spec from the trigger itself), named by the deterministic slot
hash, and marked as schedule-sourced.

Using the schedule-trigger source matters. The control plane rewrites the run
name's prefix for automation-sourced runs, and de-duplication happens against the
rewritten name. Creating these runs under any other source would store them under
a different name than a real fire, and the de-duplication that makes backfill safe
to re-run would silently stop working.
"""

from __future__ import annotations

from datetime import timezone
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ._plan import BackfillPlan, BackfillSlot

__all__ = ["SlotResult", "execute_plan"]


class SlotResult:
    """Outcome of creating one slot's run."""

    __slots__ = ("created", "error", "run_name", "slot")

    def __init__(self, slot: "BackfillSlot", run_name: str, created: bool, error: str | None = None):
        self.slot = slot
        self.run_name = run_name
        #: False when the control plane returned a pre-existing run instead of creating one.
        self.created = created
        self.error = error


def _kickoff_arg_name(details) -> str:
    """The input variable a trigger binds its scheduled time to, if any."""
    schedule = details.pb2.automation_spec.schedule
    return schedule.kickoff_time_input_arg or ""


def _build_request(
    plan: "BackfillPlan",
    slot: "BackfillSlot",
    kickoff_arg: str,
):
    from flyteidl2.common import identifier_pb2
    from flyteidl2.core import literals_pb2
    from flyteidl2.task import common_pb2 as task_common_pb2
    from flyteidl2.workflow import run_definition_pb2, run_service_pb2
    from google.protobuf import timestamp_pb2

    scheduled = slot.scheduled_at
    if scheduled.tzinfo is None:
        scheduled = scheduled.replace(tzinfo=timezone.utc)
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(scheduled.astimezone(timezone.utc))

    req = run_service_pb2.CreateRunRequest(
        run_id=identifier_pb2.RunIdentifier(
            org=plan.org,
            project=plan.project,
            domain=plan.domain,
            name=slot.run_name,
        ),
        trigger_name=identifier_pb2.TriggerName(
            org=plan.org,
            project=plan.project,
            domain=plan.domain,
            task_name=plan.task_name,
            name=plan.trigger_name,
        ),
        source=run_definition_pb2.RUN_SOURCE_SCHEDULE_TRIGGER,
        run_start_time=ts,
    )

    # Only triggers that bind their scheduled time to an input need the literal;
    # everything else reads the time from run_start_time.
    if kickoff_arg:
        req.inputs.CopyFrom(
            task_common_pb2.Inputs(
                literals=[
                    task_common_pb2.NamedLiteral(
                        name=kickoff_arg,
                        value=literals_pb2.Literal(
                            scalar=literals_pb2.Scalar(
                                primitive=literals_pb2.Primitive(datetime=ts),
                            )
                        ),
                    )
                ]
            )
        )
    return req


async def execute_plan(
    plan: "BackfillPlan",
    details,
    on_slot: Callable[[SlotResult], None] | None = None,
) -> list[SlotResult]:
    """Create a run for every slot in ``plan.to_create``.

    Slots the plan marked as already run are skipped unless the plan is forced.
    Creating a run whose name already exists is not an error -- the control plane
    returns the existing run -- so a re-run of the same backfill is a no-op.
    """
    from flyte._initialize import ensure_client, get_client

    ensure_client()
    kickoff_arg = _kickoff_arg_name(details)
    results: list[SlotResult] = []

    for slot in plan.to_create:
        req = _build_request(plan, slot, kickoff_arg)
        try:
            resp = await get_client().run_service.create_run(req)
            returned = resp.run.action.id.run.name or slot.run_name
            # The server hands back the pre-existing run when the name is taken,
            # which is the de-duplication path rather than a failure.
            created = returned == slot.run_name or not slot.already_ran
            result = SlotResult(slot, returned, created)
        except Exception as exc:  # surfaced per slot; one bad slot must not sink the rest
            result = SlotResult(slot, slot.run_name, False, error=str(exc))
        results.append(result)
        if on_slot is not None:
            on_slot(result)
    return results


async def probe_existing(
    plan_names: list[str],
) -> set[str]:
    """Return which of ``plan_names`` already exist as runs.

    Used to mark slots as already run before anything is created, so the preview
    and the confirmation prompt reflect what will actually happen.
    """
    from flyteidl2.workflow import run_service_pb2

    from flyte._initialize import ensure_client, get_client

    ensure_client()
    from flyte._initialize import get_init_config

    cfg = get_init_config()
    found: set[str] = set()
    for name in plan_names:
        try:
            from flyteidl2.common import identifier_pb2

            await get_client().run_service.get_run_details(
                run_service_pb2.GetRunDetailsRequest(
                    run_id=identifier_pb2.RunIdentifier(
                        org=cfg.org,
                        project=cfg.project,
                        domain=cfg.domain,
                        name=name,
                    )
                )
            )
            found.add(name)
        except Exception:
            # Anything other than a hit is treated as "not there". A lookup failure
            # only costs us a redundant create, which the server de-duplicates.
            continue
    return found
