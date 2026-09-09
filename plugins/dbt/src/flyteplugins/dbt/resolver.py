from __future__ import annotations

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

        callbacks = args_dict.get("callbacks")
        include_default_callback = args_dict.get("include_default_callback", "true").lower() == "true"

        return DbtTask(
            name=args_dict["name"],
            task_environment=None,
            callbacks=callbacks.split(",") if callbacks else None,
            include_default_callback=include_default_callback,
        )

    def loader_args(self, task: TaskTemplate, root_dir: pathlib.Path | None = None) -> list[str]:
        from flyteplugins.dbt.task import DbtTask

        if not isinstance(task, DbtTask):
            raise TypeError(f"DbtTaskResolver only handles DbtTask, got {type(task)}")

        callback_paths = task.callback_import_paths()

        return [
            "name",
            task.name,
            "include_default_callback",
            str(task.include_default_callback).lower(),
            "callbacks",
            ",".join(callback_paths),
        ]
