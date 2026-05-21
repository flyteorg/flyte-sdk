"""Thin wrappers around ``flyte.remote`` for the TUI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

T = TypeVar("T")

MIN_PAGE_SIZE = 10
PAGE_SIZE = MIN_PAGE_SIZE  # default when caller does not pass page_size


@dataclass(frozen=True)
class PagedResult(Generic[T]):
    """One page of list results."""

    items: list[T]
    page: int
    page_size: int
    has_next: bool


def _slice_page(all_items: list[T], page: int, page_size: int) -> PagedResult[T]:
    start = page * page_size
    end = start + page_size
    return PagedResult(
        items=all_items[start:end],
        page=page,
        page_size=page_size,
        has_next=len(all_items) > end,
    )


if TYPE_CHECKING:
    import flyte.remote as remote
    from flyte.models import ActionPhase


@dataclass(frozen=True)
class ClusterContext:
    """Resolved domain/org/endpoint for the active cluster session."""

    domain: str
    org: str
    endpoint: str | None = None
    default_project: str | None = None


def init_cluster(*, config: str | Path | None = None) -> ClusterContext:
    """Initialize flyte remote client from config (same path as ``flyte get`` / CLI)."""
    import flyte
    import flyte.config as flyte_config
    from flyte._initialize import ensure_client, get_init_config

    cfg = flyte_config.auto(config)
    if not cfg.task.domain:
        config_hint = f" ({config})" if config else " (~/.flyte/config.yaml)"
        raise RuntimeError(f"Domain is required in the config file{config_hint}.")
    project = cfg.task.project or ""
    updated_config = _config_for_scope(
        config=config,
        project=project,
        domain=cfg.task.domain or "",
        org=cfg.task.org,
    )

    if not updated_config.platform.endpoint and not os.getenv("FLYTE_API_KEY"):
        config_hint = f" in {config}" if config else ""
        raise RuntimeError(
            f"No Flyte endpoint configured{config_hint}. Run `flyte create config --endpoint <url>` "
            "or set FLYTE_API_KEY before starting the remote TUI."
        )

    flyte.init_from_config(updated_config)
    ensure_client()

    resolved = get_init_config()

    return ClusterContext(
        domain=resolved.domain,
        org=resolved.org,
        endpoint=updated_config.platform.endpoint,
        default_project=resolved.project,
    )


def list_projects(*, archived: bool = False) -> list[remote.Project]:
    import flyte.remote as remote

    return list(remote.Project.listall(archived=archived))


def list_runs(
    *,
    project: str,
    domain: str,
    limit: int = 200,
    task_name: str | None = None,
    in_phase: tuple[ActionPhase, ...] | None = None,
) -> list[remote.Run]:
    import flyte.remote as remote

    return list(
        remote.Run.listall(
            limit=limit,
            project=project,
            domain=domain,
            task_name=task_name,
            in_phase=in_phase,
            sort_by=("created_at", "desc"),
        )
    )


def list_runs_paginated(
    *,
    project: str,
    domain: str,
    page: int = 0,
    page_size: int = PAGE_SIZE,
    task_name: str | None = None,
    in_phase: tuple[ActionPhase, ...] | None = None,
) -> PagedResult[remote.Run]:
    fetch_limit = (page + 1) * page_size + 1
    return _slice_page(
        list_runs(
            limit=fetch_limit,
            project=project,
            domain=domain,
            task_name=task_name,
            in_phase=in_phase,
        ),
        page,
        page_size,
    )


def list_actions_for_run(
    run_name: str,
    *,
    project: str,
    domain: str,
) -> list[remote.Action]:
    import flyte.remote as remote

    # Actions are listed under the run; project/domain come from global init.
    _ = project, domain
    return list(remote.Action.listall(for_run_name=run_name, sort_by=("created_at", "asc")))


def get_run(run_name: str) -> remote.Run:
    import flyte.remote as remote

    return remote.Run.get(name=run_name)


def _config_for_scope(
    *,
    config: str | Path | None,
    project: str,
    domain: str,
    org: str | None = None,
):
    """Build a ``Config`` for the given project/domain (shared by init and activate)."""
    import flyte.config as flyte_config
    from flyte.config._config import TaskConfig

    cfg = flyte_config.auto(config)
    task_cfg = TaskConfig(
        org=org or cfg.task.org,
        project=project,
        domain=domain,
    )
    platform_kwargs: dict[str, Any] = {}
    api_key = os.getenv("FLYTE_API_KEY")
    if api_key:
        from flyte._utils import sanitize_endpoint
        from flyte.remote._client.auth._auth_utils import decode_api_key

        endpoint, client_id, client_secret, key_org = decode_api_key(api_key)
        platform_kwargs["endpoint"] = sanitize_endpoint(endpoint)
        platform_kwargs["client_id"] = client_id
        platform_kwargs["client_credentials_secret"] = client_secret
        platform_kwargs["auth_mode"] = "ClientSecret"
        if key_org and key_org != "None":
            task_cfg = TaskConfig(org=key_org, project=project, domain=domain)
    platform_cfg = cfg.platform.replace(**platform_kwargs)
    return cfg.with_params(platform_cfg, task_cfg)


def activate_project(*, config: str | Path | None, project: str, domain: str, org: str) -> None:
    """Re-initialize the Flyte client so run/action APIs target *project*."""
    import flyte
    from flyte._initialize import ensure_client

    flyte.init_from_config(_config_for_scope(config=config, project=project, domain=domain, org=org))
    ensure_client()


def list_tasks(
    *,
    project: str,
    domain: str,
    limit: int = 200,
    task_name: str | None = None,
) -> list[remote.Task]:
    import flyte.remote as remote

    if task_name:
        return list(
            remote.Task.listall(
                by_task_name=task_name,
                limit=limit,
                project=project,
                domain=domain,
            )
        )
    return list(remote.Task.listall(limit=limit, project=project, domain=domain))


def list_tasks_paginated(
    *,
    project: str,
    domain: str,
    page: int = 0,
    page_size: int = PAGE_SIZE,
    task_name: str | None = None,
) -> PagedResult[remote.Task]:
    fetch_limit = (page + 1) * page_size + 1
    return _slice_page(
        list_tasks(limit=fetch_limit, project=project, domain=domain, task_name=task_name),
        page,
        page_size,
    )


def list_apps(*, limit: int = 200) -> list[remote.App]:
    """List apps for the active project (from ``activate_project`` / init config)."""
    import flyte.remote as remote

    return list(remote.App.listall(limit=limit))


def list_apps_paginated(*, page: int = 0, page_size: int = PAGE_SIZE) -> PagedResult[remote.App]:
    fetch_limit = (page + 1) * page_size + 1
    return _slice_page(list_apps(limit=fetch_limit), page, page_size)


def list_triggers(*, limit: int = 200, task_name: str | None = None) -> list[remote.Trigger]:
    """List triggers for the active project (from ``activate_project`` / init config)."""
    import flyte.remote as remote

    return list(remote.Trigger.listall(limit=limit, task_name=task_name))


def list_triggers_paginated(
    *,
    page: int = 0,
    page_size: int = PAGE_SIZE,
    search: str | None = None,
) -> PagedResult[remote.Trigger]:
    if search:
        search_l = search.lower()
        filtered = [
            tr for tr in list_triggers(limit=500) if search_l in tr.name.lower() or search_l in tr.task_name.lower()
        ]
        return _slice_page(filtered, page, page_size)
    fetch_limit = (page + 1) * page_size + 1
    return _slice_page(list_triggers(limit=fetch_limit), page, page_size)


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
