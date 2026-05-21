"""Map remote actions into the core ``ActionTracker`` model."""

from __future__ import annotations

import json
import time
from datetime import timezone
from typing import Any

from flyte.cli._tui._tracker import ActionStatus, ActionTracker
from flyte.models import ActionPhase

_PHASE_TO_STATUS: dict[ActionPhase, ActionStatus] = {
    ActionPhase.QUEUED: ActionStatus.RUNNING,
    ActionPhase.WAITING_FOR_RESOURCES: ActionStatus.RUNNING,
    ActionPhase.INITIALIZING: ActionStatus.RUNNING,
    ActionPhase.RUNNING: ActionStatus.RUNNING,
    ActionPhase.SUCCEEDED: ActionStatus.SUCCEEDED,
    ActionPhase.FAILED: ActionStatus.FAILED,
    ActionPhase.ABORTED: ActionStatus.FAILED,
    ActionPhase.TIMED_OUT: ActionStatus.FAILED,
}


def _phase_status(phase: ActionPhase | str) -> ActionStatus:
    if isinstance(phase, str):
        try:
            phase = ActionPhase(phase.lower())
        except ValueError:
            return ActionStatus.RUNNING
    return _PHASE_TO_STATUS.get(phase, ActionStatus.RUNNING)


def _mono_ts(dt) -> float:
    if dt is None:
        return time.monotonic()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _safe_json(obj: Any) -> Any:
    if obj is None:
        return None
    try:
        json.dumps(obj, default=repr)
        return obj
    except (TypeError, ValueError, OverflowError):
        if isinstance(obj, dict):
            return {str(k): _safe_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe_json(v) for v in obj]
        return repr(obj)


def _attempt_records(action) -> list[dict[str, Any]]:
    attempts = []
    pb2 = action.pb2 if hasattr(action, "pb2") else action
    status = pb2.status
    for att in getattr(pb2, "attempts", []) or []:
        phase_name = att.phase
        if hasattr(phase_name, "name"):
            phase_str = phase_name.name.replace("ACTION_PHASE_", "").lower()
        else:
            phase_str = str(phase_name)
        attempts.append(
            {
                "attempt_num": int(att.attempt),
                "status": phase_str.replace("action_phase_", "").lower(),
                "logs_available": bool(getattr(att, "logs_available", False)),
            }
        )
    if not attempts and status.attempts:
        attempts.append(
            {
                "attempt_num": int(status.attempts),
                "status": _phase_status(ActionPhase.from_protobuf(status.phase)).value,
            }
        )
    return attempts


def _fetch_action_details(action) -> Any:
    """Fetch action details synchronously (safe outside running event loop)."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(action.details())
    else:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, action.details()).result()


def build_action_tree(actions: list) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (parent_by_id, children_map) keyed by action name."""
    names = {a.name for a in actions}
    parent_by_id: dict[str, str | None] = {}
    children: dict[str, list[str]] = {}

    for action in actions:
        parent = (action.pb2.metadata.parent or "").strip() or None
        if parent and parent not in names:
            parent = None
        parent_by_id[action.name] = parent
        if parent:
            children.setdefault(parent, []).append(action.name)

    return parent_by_id, children


def load_run_into_tracker(
    tracker: ActionTracker,
    actions: list,
    *,
    fetch_io: bool = True,
) -> None:
    """Populate *tracker* from remote ``Action`` list (replaces prior state)."""
    import flyte.remote as remote

    tracker._lock.acquire()
    try:
        tracker._nodes.clear()
        tracker._root_ids.clear()
        tracker._children.clear()
        tracker._version = 0
    finally:
        tracker._lock.release()

    parent_by_id, _ = build_action_tree(actions)
    details_cache: dict[str, remote.ActionDetails] = {}

    for action in actions:
        phase = action.phase
        status = _phase_status(phase)
        start = _mono_ts(action.start_time)
        end = None
        if action.done():
            end_pb = action.pb2.status
            if end_pb.HasField("end_time"):
                end = _mono_ts(end_pb.end_time.ToDatetime().replace(tzinfo=timezone.utc))

        task_name = action.task_name or action.name
        short_name = None
        if action.pb2.metadata.HasField("task"):
            short_name = action.pb2.metadata.task.short_name or None

        inputs = None
        outputs = None
        error = None
        log_links = None
        attempt_count = int(action.pb2.status.attempts or 0)
        attempts = _attempt_records(action)

        if fetch_io and action.done():
            try:
                details = _fetch_action_details(action)
                details_cache[action.name] = details
                if details.error_info:
                    error = f"{details.error_info.kind}: {details.error_info.message}"
                try:
                    inp = details.inputs()
                    if inp:
                        inputs = _safe_json(dict(inp))
                except Exception:
                    pass
                try:
                    out = details.outputs()
                    if out:
                        outputs = _safe_json(out.named_outputs if hasattr(out, "named_outputs") else tuple(out))
                except Exception:
                    pass
            except Exception:
                pass

        parent_id = parent_by_id.get(action.name)
        group = (action.pb2.metadata.group or "").strip() or None

        tracker.record_start(
            action_id=action.name,
            task_name=task_name,
            parent_id=parent_id,
            short_name=short_name,
            inputs=inputs,
            output_path=None,
            has_report=False,
            context=None,
            group=group,
            log_links=log_links,
            attempt_count=attempt_count,
            attempts=attempts,
            start_time=start,
        )

        if status == ActionStatus.SUCCEEDED:
            tracker.record_complete(action_id=action.name, outputs=outputs, end_time=end)
        elif status == ActionStatus.FAILED:
            tracker.record_failure(action_id=action.name, error=error or phase.value, end_time=end)
