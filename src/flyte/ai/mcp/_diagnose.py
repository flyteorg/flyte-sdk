"""Helpers for the diagnose-and-recover options of the run/action/condition/log tools.

Split out of `_tools.py` for size. The tools themselves stay there, because FastMCP builds each
tool's input schema from its signature and docstring; this module holds the filter and
validation builders, the response shaping and the rerun plumbing they call.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import flyte
import flyte.remote
from flyte.ai.mcp._tools import ToolError, _is_task_allowed, _pb_to_dict, _seconds, _ts

if TYPE_CHECKING:
    from flyte.ai.mcp._flyte_mcp_app import FlyteMCPAppEnvironment

#: Longest error message `get_run` returns, so a huge traceback cannot blow the context window.
#: `get_action` returns the full `error_info` when the whole text is needed.
MAX_ERROR_CHARS = 2000


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated, {len(text) - limit} more characters]"


def parse_phases(in_phase: list[str] | None) -> tuple[Any, ...] | None:
    """Validate phase names against the SDK's `ActionPhase`, so a typo is a ToolError, not a silent empty list."""
    if not in_phase:
        return None
    from flyte.models import ActionPhase

    valid = [p.value for p in ActionPhase]
    phases = []
    for raw in in_phase:
        try:
            phases.append(ActionPhase(raw.strip().lower()))
        except ValueError:
            raise ToolError(f"Unknown phase '{raw}'. Valid phases: {', '.join(valid)}.") from None
    return tuple(phases)


def parse_iso_time(param: str, value: str | None) -> datetime | None:
    """Parse an ISO-8601 string into an aware datetime (naive input is taken as UTC, like the CLI)."""
    if value is None:
        return None
    try:
        # `fromisoformat` only accepts a trailing "Z" from Python 3.11; this package supports 3.10.
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ToolError(
            f"Could not parse {param}='{value}' as an ISO-8601 timestamp; "
            f"use a form like '2025-01-31T12:00:00Z' or '2025-01-31'."
        ) from None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def time_filter(param_prefix: str, after: str | None, before: str | None) -> Any:
    from flyte.remote._common import TimeFilter

    after_dt = parse_iso_time(f"{param_prefix}_after", after)
    before_dt = parse_iso_time(f"{param_prefix}_before", before)
    if after_dt is None and before_dt is None:
        return None
    return TimeFilter(after=after_dt, before=before_dt)


def check_kv(param: str, items: dict[str, str] | None) -> dict[str, str] | None:
    """Empty means unset (as in the CLI); an empty key is rejected the way the CLI rejects `=value`."""
    if not items:
        return None
    if any(not k for k in items):
        raise ToolError(f"{param} keys must not be empty.")
    return items


# ------------------------------
# list_runs
# ------------------------------


def run_filters(
    in_phase: list[str] | None,
    paused_only: bool,
    task_version: str | None,
    created_after: str | None,
    created_before: str | None,
    updated_after: str | None,
    updated_before: str | None,
    with_labels: dict[str, str] | None,
    with_label_keys: list[str] | None,
) -> dict[str, Any]:
    """Validate the `list_runs` filters and return the `Run.listall` keyword arguments they select."""
    phases = parse_phases(in_phase)
    created_at = time_filter("created", created_after, created_before)
    updated_at = time_filter("updated", updated_after, updated_before)
    labels = check_kv("with_labels", with_labels)

    # Only the filters actually set are passed, so an unfiltered call reaches the SDK exactly
    # as it did before these parameters existed.
    filters: dict[str, Any] = {}
    if phases:
        filters["in_phase"] = phases
    if paused_only:
        filters["paused_actions_only"] = True
    if task_version:
        filters["task_version"] = task_version
    if created_at:
        filters["created_at"] = created_at
    if updated_at:
        filters["updated_at"] = updated_at
    if labels:
        filters["with_labels"] = labels
    if with_label_keys:
        filters["with_label_keys"] = with_label_keys
    return filters


async def filter_to_caller(filters: dict[str, Any]) -> None:
    """Add the `only_mine` filter to `filters`, for the `list_*` tools that offer it."""
    # Same lookup as `flyte get {run,app} --only-mine`: filter on the caller's subject.
    user = await flyte.remote.User.get.aio()
    filters["created_by_subject"] = user.subject()


def run_entry(r: Any) -> dict:
    if hasattr(r, "to_dict"):
        entry = r.to_dict()
    else:
        entry = {"name": getattr(r, "name", None), "url": getattr(r, "url", None)}
    action = getattr(r, "action", None)
    if action is not None:
        entry["task_name"] = action.task_name
        entry["started_at"] = _ts(action.pb2.status, "start_time")
        entry["ended_at"] = _ts(action.pb2.status, "end_time")
    return entry


# ------------------------------
# get_run / get_run_io
# ------------------------------


async def add_run_details(run: Any, result: dict[str, Any]) -> None:
    """Add timing, labels and the error to a `get_run` result.

    Best-effort: a failure to read them must not hide the phase, which is the one thing
    `get_run` exists to answer.
    """
    try:
        details = await run.details.aio()
        action = details.action_details
        started_at = _ts(action.pb2.status, "start_time")
        result["task_name"] = details.task_name
        result["started_at"] = started_at
        result["ended_at"] = _ts(action.pb2.status, "end_time")
        # `runtime` is measured from the start time, which is unset (epoch) until it starts.
        result["duration_s"] = _seconds(action.runtime) if started_at else None
        result["labels"] = dict(details.pb2.run_spec.labels.values)
        error = _pb_to_dict(action.error_info)
        if error is not None:
            result["error"] = {
                "kind": error.get("kind"),
                "message": truncate(error.get("message", ""), MAX_ERROR_CHARS),
            }
    except Exception as e:
        result["details_error"] = f"{type(e).__name__}: {e}"


def check_io_selection(inputs_only: bool, outputs_only: bool) -> None:
    if inputs_only and outputs_only:
        raise ToolError("inputs_only and outputs_only cannot both be true; omit both to get inputs and outputs.")


async def io_target(name: str, action_name: str | None) -> tuple[Any, str]:
    """The run, or the named action of it, whose data `get_run_io` reports, with the run's name."""
    # A `Run` and an `ActionDetails` expose the same inputs()/outputs()/done().
    if action_name is None:
        target: Any = await flyte.remote.Run.get.aio(name=name)
        return target, target.name
    target = await flyte.remote.ActionDetails.get.aio(run_name=name, name=action_name)
    return target, target.run_name


def io_result(
    run_name: str,
    action_name: str | None,
    inputs_only: bool,
    outputs_only: bool,
    inputs: Any,
    outputs: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": run_name}
    if action_name is not None:
        result["action_name"] = action_name
    if not outputs_only:
        result["inputs"] = inputs
    if not inputs_only:
        result["outputs"] = outputs
    return result


# ------------------------------
# rerun_run
# ------------------------------


def check_rerun_args(
    recover: bool,
    action_name: str | None,
    force_rerun_actions: list[str] | None,
    labels: dict[str, str] | None,
    env_vars: dict[str, str] | None,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """Argument errors, raised before any network call. Returns the checked `labels` and `env`."""
    if force_rerun_actions and not recover:
        raise ToolError("force_rerun_actions requires recover=true.")
    if action_name and recover:
        raise ToolError(
            "action_name cannot be combined with recover: recovery matches succeeded actions "
            "by name in the source run, and a run rooted at one action has a different action "
            "tree. Re-run the action on its own, or recover the whole run."
        )
    return check_kv("labels", labels), check_kv("env", env_vars)


async def check_rerun_allowed(
    env: FlyteMCPAppEnvironment, run_name: str, action_name: str | None, domain: str, project: str
) -> None:
    """A rerun launches the prior run's task, so it has to clear the same gate as run_task.

    Only pays for the lookup when there is an allowlist to check against.
    """
    if env.task_allowlist is None:
        return
    # With `action_name` the task that runs is that action's, not the root's.
    if action_name:
        prior_action = await flyte.remote.Action.get.aio(run_name=run_name, name=action_name)
        source = f"action '{action_name}' of run '{run_name}'"
    else:
        prior_action = (await flyte.remote.Run.get.aio(name=run_name)).action
        source = f"run '{run_name}'"
    task_name = getattr(prior_action, "task_name", None)
    if task_name is None:
        raise ToolError(
            f"Cannot determine which task {source} launched, so it cannot be checked against the task allowlist."
        )
    if not _is_task_allowed(env.task_allowlist, domain, project, task_name):
        raise ToolError(f"Task {domain}/{project}/{task_name} is not allowlisted.")


async def rerun(
    run_name: str,
    recover: bool,
    action_name: str | None,
    force_rerun_actions: list[str] | None,
    allow_missing_outputs: bool,
    name: str | None,
    labels: dict[str, str] | None,
    env_vars: dict[str, str] | None,
) -> Any:
    # Same arguments `flyte rerun` passes, and only the ones set, so a plain rerun is still
    # the one-argument call it was.
    rerun_args: dict[str, Any] = {}
    if recover:
        rerun_args["recover"] = True
    if action_name:
        rerun_args["action_name"] = action_name
    if force_rerun_actions:
        rerun_args["force_rerun_actions"] = force_rerun_actions
    if allow_missing_outputs:
        rerun_args["allow_missing_source_outputs"] = True

    if name or labels or env_vars:
        # Name, labels and env are run-context overrides; `flyte.rerun` has no such arguments.
        runner = flyte.with_runcontext(mode="remote", name=name, env_vars=env_vars, labels=labels)
        return await runner.rerun.aio(run_name, **rerun_args)
    return await flyte.rerun.aio(run_name, **rerun_args)


# ------------------------------
# list_conditions
# ------------------------------


async def add_condition_prompt(entry: dict[str, Any], run_name: str, action_name: str) -> None:
    """Add the run author's prompt to a `list_conditions` entry, or `prompt_error` if the lookup fails."""
    from flyteidl2.workflow import run_definition_pb2

    entry["prompt"] = None
    entry["prompt_type"] = None
    # The prompt is on the action's details, not on the listed action (as `flyte signal
    # condition` does).
    try:
        details = await flyte.remote.ActionDetails.get.aio(run_name=run_name, name=action_name)
        if details.pb2.HasField("condition"):
            cond = details.pb2.condition
            entry["prompt"] = cond.prompt or None
            prompt_type = (
                run_definition_pb2.ConditionPromptType.Name(cond.prompt_type)
                .removeprefix("CONDITION_PROMPT_TYPE_")
                .lower()
            )
            entry["prompt_type"] = None if prompt_type == "unspecified" else prompt_type
    except Exception as e:
        entry["prompt_error"] = f"{type(e).__name__}: {e}"
