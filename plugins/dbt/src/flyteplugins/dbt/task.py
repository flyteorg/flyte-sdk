from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from flyte.extend import RuntimeTaskTemplate
from flyte.models import NativeInterface

from flyteplugins.dbt.runner import (
    DbtEventCallback,
    DbtNodeResult,
    invoke_dbt,
)

if TYPE_CHECKING:
    from flyte import TaskEnvironment


_MANAGED_DBT_FLAGS = {
    "--project-dir",
    "--profiles-dir",
    "--profile",
    "--target",
    "--target-path",
    "--select",
    "--exclude",
}


def _validate_project_dir(project_dir: str | None) -> None:
    if project_dir is None:
        return
    if not (Path(project_dir) / "dbt_project.yml").exists():
        raise ValueError(f"dbt project_dir {project_dir!r} must contain a dbt_project.yml file.")


def _validate_extra_args(extra_args: list[str] | None) -> list[str]:
    args = list(extra_args or [])
    managed_flags = sorted(set(args) & _MANAGED_DBT_FLAGS)
    if managed_flags:
        raise ValueError("dbt extra_args cannot include flags managed by DbtTask: " + ", ".join(managed_flags))
    return args


def _build_cli_args(
    *,
    command: str,
    project_dir: str | None,
    profiles_dir: str | None,
    profile: str | None,
    target_path: str | None,
    select: list[str] | None,
    exclude: list[str] | None,
    target: str | None,
    extra_args: list[str] | None,
) -> list[str]:
    args = [command]
    if project_dir:
        args.extend(["--project-dir", project_dir])
    if profiles_dir:
        args.extend(["--profiles-dir", profiles_dir])
    if profile:
        args.extend(["--profile", profile])
    if target:
        args.extend(["--target", target])
    if target_path:
        args.extend(["--target-path", target_path])
    if select:
        args.extend(["--select", *select])
    if exclude:
        args.extend(["--exclude", *exclude])
    args.extend(_validate_extra_args(extra_args))
    return args


@dataclass(kw_only=True)
class DbtTask(RuntimeTaskTemplate):
    """A Flyte task that maps one dbtRunner.invoke(...) call to one task."""

    project_dir: str | None = None
    profiles_dir: str | None = None
    profile: str | None = None
    target_path: str | None = None
    callbacks: list[DbtEventCallback | str] = field(default_factory=list)
    trace_node_events: bool = True

    def __init__(
        self,
        *,
        name: str,
        task_environment: Optional[TaskEnvironment] = None,
        project_dir: str | None = None,
        profiles_dir: str | None = None,
        profile: str | None = None,
        target_path: str | None = None,
        callbacks: list[DbtEventCallback | str] | None = None,
        trace_node_events: bool = True,
        **kwargs: Any,
    ):
        project_dir = kwargs.pop("project_dir", project_dir)
        profiles_dir = kwargs.pop("profiles_dir", profiles_dir)
        profile = kwargs.pop("profile", profile)
        target_path = kwargs.pop("target_path", target_path)
        _validate_project_dir(project_dir)
        self.project_dir = project_dir
        self.profiles_dir = profiles_dir
        self.profile = profile
        self.target_path = target_path
        self.callbacks = list(callbacks or [])
        self.trace_node_events = trace_node_events
        from flyteplugins.dbt.resolver import DbtTaskResolver

        task_name = f"{task_environment.name}.{name}" if task_environment else name
        interface = kwargs.pop(
            "interface",
            NativeInterface(
                inputs={
                    "command": (str, inspect.Parameter.empty),
                    "select": (Optional[list[str]], None),
                    "exclude": (Optional[list[str]], None),
                    "target": (Optional[str], None),
                    "extra_args": (Optional[list[str]], None),
                },
                outputs={"results": list[DbtNodeResult]},
            ),
        )
        parent_env = kwargs.pop("parent_env", weakref.ref(task_environment) if task_environment else None)
        parent_env_name = kwargs.pop("parent_env_name", task_environment.name if task_environment else None)

        super().__init__(
            name=task_name,
            interface=interface,
            image=kwargs.pop("image", task_environment.image if task_environment else "auto"),
            resources=kwargs.pop("resources", task_environment.resources if task_environment else None),
            cache=kwargs.pop("cache", task_environment.cache if task_environment else "disable"),
            reusable=kwargs.pop("reusable", task_environment.reusable if task_environment else None),
            env_vars=kwargs.pop("env_vars", task_environment.env_vars if task_environment else None),
            secrets=kwargs.pop("secrets", task_environment.secrets if task_environment else None),
            service_account=kwargs.pop(
                "service_account", task_environment.service_account if task_environment else None
            ),
            pod_template=kwargs.pop("pod_template", task_environment.pod_template if task_environment else None),
            queue=kwargs.pop("queue", task_environment.queue if task_environment else None),
            interruptible=kwargs.pop("interruptible", task_environment.interruptible if task_environment else False),
            short_name=kwargs.pop("short_name", name if task_environment else ""),
            task_type=kwargs.pop("task_type", "dbt"),
            _call_as_synchronous=kwargs.pop("_call_as_synchronous", True),
            parent_env=parent_env,
            parent_env_name=parent_env_name,
            task_resolver=kwargs.pop("task_resolver", DbtTaskResolver()),
            **kwargs,
        )

        if task_environment is not None:
            task_environment._tasks[task_name] = self

    def forward(self, *args: Any, **kwargs: Any) -> list[DbtNodeResult]:
        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)
        cli_args = _build_cli_args(
            command=kwargs["command"],
            project_dir=self.project_dir,
            profiles_dir=self.profiles_dir,
            profile=self.profile,
            target_path=self.target_path,
            select=kwargs.get("select"),
            exclude=kwargs.get("exclude"),
            target=kwargs.get("target"),
            extra_args=kwargs.get("extra_args"),
        )
        return invoke_dbt(
            cli_args,
            callbacks=self.callbacks,
            trace_node_events=self.trace_node_events,
        )

    async def aio(
        self,
        command: str,
        *,
        select: list[str] | None = None,
        exclude: list[str] | None = None,
        target: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[DbtNodeResult]:
        return await super().aio(
            command=command,
            select=select,
            exclude=exclude,
            target=target,
            extra_args=extra_args,
        )

    async def execute(self, *args: Any, **kwargs: Any) -> list[DbtNodeResult]:
        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)
        cli_args = _build_cli_args(
            command=kwargs["command"],
            project_dir=self.project_dir,
            profiles_dir=self.profiles_dir,
            profile=self.profile,
            target_path=self.target_path,
            select=kwargs.get("select"),
            exclude=kwargs.get("exclude"),
            target=kwargs.get("target"),
            extra_args=kwargs.get("extra_args"),
        )

        from flyte._utils.asyncify import run_sync_in_thread

        return await run_sync_in_thread(
            invoke_dbt,
            cli_args,
            self.callbacks,
            trace_node_events=self.trace_node_events,
        )
