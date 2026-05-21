"""Thin wrappers around ``flyte.remote`` for the TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import flyte.remote as remote
    from flyte.models import ActionPhase


@dataclass(frozen=True)
class ClusterContext:
    """Resolved project/domain/org for the active session."""

    project: str
    domain: str
    org: str
    endpoint: str | None = None


def init_cluster(
    *,
    project: str | None = None,
    domain: str | None = None,
) -> ClusterContext:
    """Initialize flyte client from CLI config and return resolved context."""
    import flyte
    from flyte._initialize import get_init_config

    flyte.init(project=project, domain=domain)
    cfg = get_init_config()
    endpoint = None
    try:
        from flyte.remote._settings import get_endpoint

        endpoint = get_endpoint()
    except Exception:
        pass
    return ClusterContext(
        project=cfg.project,
        domain=cfg.domain,
        org=cfg.org,
        endpoint=endpoint,
    )


def list_runs(
    *,
    limit: int = 200,
    task_name: str | None = None,
    in_phase: tuple[ActionPhase, ...] | None = None,
) -> list[remote.Run]:
    import flyte.remote as remote

    return list(
        remote.Run.listall(
            limit=limit,
            task_name=task_name,
            in_phase=in_phase,
            sort_by=("created_at", "desc"),
        )
    )


def list_actions_for_run(run_name: str) -> list[remote.Action]:
    import flyte.remote as remote

    return list(remote.Action.listall(for_run_name=run_name, sort_by=("created_at", "asc")))


def get_run(run_name: str) -> remote.Run:
    import flyte.remote as remote

    return remote.Run.get(name=run_name)


def list_tasks(*, limit: int = 200, task_name: str | None = None) -> list[remote.Task]:
    import flyte.remote as remote

    if task_name:
        return list(remote.Task.listall(by_task_name=task_name, limit=limit))
    return list(remote.Task.listall(limit=limit))


def list_apps(*, limit: int = 200) -> list[remote.App]:
    import flyte.remote as remote

    return list(remote.App.listall(limit=limit))


def list_triggers(*, limit: int = 200, task_name: str | None = None) -> list[remote.Trigger]:
    import flyte.remote as remote

    return list(remote.Trigger.listall(limit=limit, task_name=task_name))


def abort_run(run_name: str, reason: str = "Aborted from remote TUI.") -> None:
    run = get_run(run_name)
    run.abort(reason=reason)


def fetch_log_tail(
    run_name: str,
    action_name: str | None = None,
    *,
    max_lines: int = 200,
    show_ts: bool = False,
    filter_system: bool = False,
) -> list[str]:
    """Return up to *max_lines* recent log lines (non-blocking tail)."""
    import flyte.remote as remote

    if action_name:
        log_source = remote.Action.get(run_name=run_name, name=action_name)
    else:
        log_source = get_run(run_name)
    lines: list[str] = []
    for line in log_source.get_logs(show_ts=show_ts, filter_system=filter_system):
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines[-max_lines:]
