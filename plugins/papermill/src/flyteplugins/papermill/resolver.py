from __future__ import annotations

import importlib
import os
import pathlib
import sys

from flyte._task import TaskTemplate


class NotebookTaskResolver:
    """Resolver for NotebookTask instances.

    Locates a NotebookTask by its module-level variable name so that the
    Flyte runtime can reconstruct the task in a remote container.
    """

    @property
    def import_path(self) -> str:
        return "flyteplugins.papermill.resolver.NotebookTaskResolver"

    def load_task(self, loader_args: list[str]) -> TaskTemplate:
        _, task_module, _, task_name = loader_args
        mod = importlib.import_module(name=task_module)
        task = getattr(mod, task_name, None)
        if task is None:
            raise ValueError(
                f"Could not find NotebookTask '{task_name}' in module '{task_module}'. "
                "Make sure the NotebookTask is assigned to a module-level variable."
            )
        return task

    def loader_args(
        self, task: TaskTemplate, root_dir: pathlib.Path | None
    ) -> list[str]:
        from flyteplugins.papermill.task import NotebookTask

        if not isinstance(task, NotebookTask):
            raise TypeError(
                f"NotebookTaskResolver only handles NotebookTask, got {type(task)}"
            )

        caller_file = task._caller_file
        if not caller_file:
            raise ValueError(
                f"NotebookTask '{task.name}' has no source file recorded. "
                "It must be defined at module level in a .py file."
            )

        if root_dir is None:
            raise ValueError("root_dir is required for NotebookTask serialization.")

        # Convert the caller's file path to a dotted module name
        file_path = pathlib.Path(caller_file).resolve()
        root = pathlib.Path(root_dir).resolve()

        try:
            relative = file_path.relative_to(root)
        except ValueError:
            # Fall back to checking sys.modules
            module_name = self._find_module_for_file(caller_file)
            if module_name is None:
                raise ValueError(
                    f"Cannot determine module for '{caller_file}' relative to root '{root_dir}'."
                )
            relative = None

        if relative is not None:
            module_name = os.path.splitext(str(relative))[0].replace(os.sep, ".")

        # Find the variable name by scanning the module
        mod = self._get_module(module_name, caller_file)
        var_name = None
        for attr_name in dir(mod):
            if getattr(mod, attr_name, None) is task:
                var_name = attr_name
                break

        if var_name is None:
            raise ValueError(
                f"NotebookTask '{task.name}' is not assigned to a module-level variable "
                f"in '{module_name}'. The resolver needs a named reference to load it."
            )

        return ["mod", module_name, "instance", var_name]

    @staticmethod
    def _find_module_for_file(file_path: str) -> str | None:
        abs_path = os.path.abspath(file_path)
        for name, mod in sys.modules.items():
            mod_file = getattr(mod, "__file__", None)
            if mod_file and os.path.abspath(mod_file) == abs_path:
                return name
        return None

    @staticmethod
    def _get_module(module_name: str, file_path: str):
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        return importlib.import_module(module_name)
