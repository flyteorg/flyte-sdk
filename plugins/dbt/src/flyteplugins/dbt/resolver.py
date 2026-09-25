from __future__ import annotations

import json
import pathlib

from flyte._task import TaskTemplate


class DbtTaskResolver:
    """Reconstructs a DbtTask in the remote task container."""

    @property
    def import_path(self) -> str:
        return "flyteplugins.dbt.resolver.DbtTaskResolver"

    def load_task(self, loader_args: list[str]) -> TaskTemplate:
        from flyteplugins.dbt.task import DbtTask

        it = iter(loader_args)
        args_dict: dict[str, str] = {}
        for key in it:
            try:
                args_dict[key] = next(it)
            except StopIteration:
                raise ValueError(f"Odd number of loader args: missing value for key '{key}'")

        callbacks = json.loads(args_dict.get("callbacks_json", "[]"))
        trace_node_events = args_dict.get("trace_node_events", "true").lower() == "true"

        return DbtTask(
            name=args_dict["name"],
            task_environment=None,
            project_dir=args_dict.get("project_dir") or None,
            profiles_dir=args_dict.get("profiles_dir") or None,
            profile=args_dict.get("profile") or None,
            target_path=args_dict.get("target_path") or None,
            callbacks=callbacks,
            trace_node_events=trace_node_events,
        )

    def loader_args(self, task: TaskTemplate, root_dir: pathlib.Path | None = None) -> list[str]:
        from flyteplugins.dbt.runner import callback_import_paths
        from flyteplugins.dbt.task import DbtTask

        if not isinstance(task, DbtTask):
            raise TypeError(f"DbtTaskResolver only handles DbtTask, got {type(task)}")

        callback_paths = callback_import_paths(task.callbacks)

        return [
            "name",
            task.name,
            "project_dir",
            task.project_dir or "",
            "profiles_dir",
            task.profiles_dir or "",
            "profile",
            task.profile or "",
            "target_path",
            task.target_path or "",
            "trace_node_events",
            str(task.trace_node_events).lower(),
            "callbacks_json",
            json.dumps(callback_paths),
        ]
