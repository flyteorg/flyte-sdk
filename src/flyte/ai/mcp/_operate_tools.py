"""Helpers and tools for the launch-and-operate side of `flyte.ai.mcp.FlyteMCPAppEnvironment`.

Split out of `_tools.py` for size: the trigger and app tools that manage things rather than
read them, and the validation behind the new options of `run_task` and `list_apps`. The
existing tools stay in `_tools.py`, because FastMCP builds each tool's input schema from its
signature and docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import flyte
import flyte.remote
from flyte.ai.mcp._diagnose import check_kv
from flyte.ai.mcp._tools import (
    MCPContext,
    ToolError,
    _is_app_allowed,
    _is_task_allowed,
    _is_trigger_allowed,
)

if TYPE_CHECKING:
    from flyte.ai.mcp._flyte_mcp_app import FlyteMCPAppEnvironment


def check_app_statuses(status: list[str] | None) -> tuple[str, ...] | None:
    """Validate `list_apps(status=...)` with the SDK's own parser, so it accepts exactly what `App.listall` does."""
    if not status:
        return None
    from flyteidl2.app import app_definition_pb2

    from flyte.remote._app import _status_filter

    try:
        _status_filter(tuple(status))
    except ValueError:
        valid = [
            n.removeprefix("DEPLOYMENT_STATUS_").lower()
            for n in cast(Any, app_definition_pb2.Status.DeploymentStatus).keys()
            if n != "DEPLOYMENT_STATUS_UNSPECIFIED"
        ]
        raise ToolError(f"Unknown app status in {status}. Valid statuses: {', '.join(valid)}.") from None
    return tuple(status)


def run_context(
    run_name: str | None,
    labels: dict[str, str] | None,
    env_vars: dict[str, str] | None,
    queue: str | None,
    service_account: str | None,
    max_action_concurrency: int | None,
    raw_data_path: str | None,
    custom_context: dict[str, str] | None,
    run_project: str | None,
    run_domain: str | None,
) -> Any:
    """Validate the `run_task` run settings; returns the runner that applies them, or None if none were given."""
    # Only the settings actually given are passed on, so a plain call still makes the bare
    # `flyte.run.aio(tk, **inputs)` call it always did.
    context: dict[str, Any] = {}
    env_vars = check_kv("env", env_vars)
    run_labels = check_kv("labels", labels)
    context_values = check_kv("custom_context", custom_context)
    if context_values:
        from flyte._internal.runtime.convert import RESERVED_CONTEXT_KEYS

        reserved = sorted(set(context_values) & RESERVED_CONTEXT_KEYS)
        if reserved:
            raise ToolError(f"custom_context key(s) {reserved} are reserved by Flyte.")
    for key, value in (
        ("name", run_name),
        ("env_vars", env_vars),
        ("labels", run_labels),
        ("queue", queue),
        ("service_account", service_account),
        ("raw_data_path", raw_data_path),
        ("custom_context", context_values),
        ("project", run_project),
        ("domain", run_domain),
    ):
        if value:
            context[key] = value
    if max_action_concurrency is not None:
        context["max_action_concurrency"] = max_action_concurrency
    if not context:
        return None
    try:
        return flyte.with_runcontext(mode="remote", **context)
    except ValueError as e:  # e.g. max_action_concurrency=1, which would deadlock the run
        raise ToolError(str(e)) from None


def build_operate_tools(env: FlyteMCPAppEnvironment) -> dict[str, Any]:
    """Build the tools that create or delete triggers and apps, bound to `env`, keyed by tool name."""

    async def delete_app(
        name: str, project: str | None = None, domain: str | None = None, ctx: MCPContext | None = None
    ) -> dict:
        """Permanently delete a deployed app. Its endpoint stops serving and it cannot be restored.

        To stop an app but keep it, use `deactivate_app` instead. Deleting an app that does
        not exist succeeds, so this is safe to retry.
        """
        if not _is_app_allowed(env.app_allowlist, name):
            raise ToolError(f"App {name} is not allowlisted.")
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            await cast(Any, flyte.remote.App.delete).aio(name=name, project=resolved_project, domain=resolved_domain)
            return {"name": name, "deleted": True}

    async def create_trigger(
        task_name: str,
        trigger_name: str,
        schedule: str,
        description: str | None = None,
        auto_activate: bool = True,
        trigger_time_input: str | None = None,
        task_version: str | None = None,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Create a trigger that runs a registered task on a cron schedule.

        `schedule` is a five-field cron expression in UTC (`minute hour day-of-month month
        day-of-week`), e.g. "0 0 * * *" for daily at midnight. The task launched must be
        allowlisted, exactly as for `run_task`. The trigger is active straight away unless
        `auto_activate` is false (then use `activate_trigger` later). Set
        `trigger_time_input` to the name of one of the task's datetime inputs to receive the
        scheduled fire time. Omit `task_version` to follow the latest registered version of
        the task. Returns the trigger's name and whether it is active. The server checks the
        cron expression; a bad one comes back as an error.
        """
        if not _is_trigger_allowed(env.trigger_allowlist, task_name, trigger_name):
            raise ToolError(f"Trigger {task_name}/{trigger_name} is not allowlisted.")
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            # A trigger launches its task, so it has to clear the same gate as run_task.
            if not _is_task_allowed(env.task_allowlist, resolved_domain, resolved_project, task_name):
                raise ToolError(f"Task {resolved_domain}/{resolved_project}/{task_name} is not allowlisted.")
            # `flyte.Cron` does not check the expression (the CLI does not either; the control
            # plane does), so the only client-side error left is an unusable trigger definition.
            try:
                trigger = flyte.Trigger(
                    name=trigger_name,
                    automation=flyte.Cron(schedule),
                    description=description or "",
                    auto_activate=auto_activate,
                    inputs={trigger_time_input: flyte.TriggerTime} if trigger_time_input else None,
                )
                created = await cast(Any, flyte.remote.Trigger.create).aio(
                    trigger, task_name=task_name, task_version=task_version
                )
            except ValueError as e:  # a bad trigger definition, or the task does not exist
                raise ToolError(str(e)) from None
            except ConnectError as e:
                if e.code == Code.INVALID_ARGUMENT:
                    raise ToolError(f"The server rejected the trigger: {e.message}") from None
                raise
            return {"task_name": task_name, "name": created.name, "created": True, "is_active": created.is_active}

    async def delete_trigger(
        task_name: str,
        trigger_name: str,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Permanently delete a trigger. Its schedule stops firing and it cannot be restored.

        To pause a trigger but keep it, use `deactivate_trigger` instead. Find the exact
        names with `list_triggers`. Runs already started by the trigger are unaffected.
        """
        if not _is_trigger_allowed(env.trigger_allowlist, task_name, trigger_name):
            raise ToolError(f"Trigger {task_name}/{trigger_name} is not allowlisted.")
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            await cast(Any, flyte.remote.Trigger.delete).aio(
                name=trigger_name, task_name=task_name, project=resolved_project, domain=resolved_domain
            )
            return {"task_name": task_name, "name": trigger_name, "deleted": True}

    return {
        "delete_app": delete_app,
        "create_trigger": create_trigger,
        "delete_trigger": delete_trigger,
    }
