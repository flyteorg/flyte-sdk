"""Generic resolver for internal Flyte tasks.

Stores an import path to a task-builder function and arbitrary keyword
arguments.  At runtime `load_task` dynamically imports the builder and
calls it with the stored kwargs, recreating a lightweight task without
pickling.  This is the same mechanism used by `run_python_script` and
can be reused for prefetch, custom bundling, and other internal tasks.
"""

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from flyte._internal.resolvers.common import Resolver
from flyte._task import TaskTemplate


class InternalTaskResolver(Resolver):
    """Resolve an internal task by dynamically importing its builder.

    During serialization the resolver stores:

    * `task_builder` - fully-qualified import path of a callable that
      returns a `TaskTemplate` (e.g.
      `"flyte._run_python_script._build_script_runner_task"`).
    * Arbitrary keyword arguments forwarded to the builder.

    At runtime `load_task` re-imports the builder and calls it with
    the stored kwargs.
    """

    def __init__(self, task_builder: str = "", **kwargs: Any):
        self._task_builder = task_builder
        self._kwargs = kwargs

    @property
    def import_path(self) -> str:
        return "flyte._internal.resolvers.internal.InternalTaskResolver"

    def load_task(self, loader_args: List[str]) -> TaskTemplate:
        args_iter = iter(loader_args)
        parsed: Dict[str, str] = dict(zip(args_iter, args_iter))

        builder_path = parsed.pop("task_builder")
        module_path, func_name = builder_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        builder = getattr(module, func_name)

        return builder(**parsed)

    def loader_args(self, task: TaskTemplate, root_dir: Optional[Path] = None) -> List[str]:
        args = ["task_builder", self._task_builder]
        for key, value in self._kwargs.items():
            if value is not None:
                args.extend([key, str(value)])
        return args
