import importlib
from pathlib import Path
from typing import List

from flyte._internal.resolvers._task_module import extract_task_module
from flyte._internal.resolvers.common import Resolver
from flyte._task import TaskTemplate


class DefaultTaskResolver(Resolver):
    """
    Please see the notes in the TaskResolverMixin as it describes this default behavior.
    """

    @property
    def import_path(self) -> str:
        return "flyte._internal.resolvers.default.DefaultTaskResolver"

    def load_task(self, loader_args: List[str]) -> TaskTemplate:
        _, task_module, _, task_name, *_ = loader_args

        task_module = importlib.import_module(name=task_module)  # type: ignore
        task_def = getattr(task_module, task_name)
        return task_def

    def loader_args(self, task: TaskTemplate, root_dir: Path) -> List[str]:  # type:ignore
        t, m = extract_task_module(task, root_dir)
        return ["mod", m, "instance", t]
