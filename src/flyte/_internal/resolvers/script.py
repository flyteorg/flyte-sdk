"""Resolver for python-script tasks.

Bundles a plain Python script into the code bundle and recreates
the ``execute_script`` task at runtime so the Flyte entrypoint can
run it without pickling.
"""

from pathlib import Path
from typing import List, Optional

from flyte._internal.resolvers.common import Resolver
from flyte._task import TaskTemplate


class ScriptTaskResolver(Resolver):
    """Resolve a bundled Python script into an executable task.

    During serialization the resolver stores the script filename and timeout
    as loader args.  At runtime ``load_task`` recreates a lightweight task
    that executes the script via ``subprocess``.
    """

    def __init__(self, script_name: str = "", output_dir: Optional[str] = None, timeout: int = 3600):
        self._script_name = script_name
        self._output_dir = output_dir
        self._timeout = timeout

    @property
    def import_path(self) -> str:
        return "flyte._internal.resolvers.script.ScriptTaskResolver"

    def load_task(self, loader_args: List[str]) -> TaskTemplate:
        args_iter = iter(loader_args)
        parsed = dict(zip(args_iter, args_iter))
        script_name = parsed["script"]
        output_dir = parsed.get("output_dir", None)
        timeout = int(parsed.get("timeout", "3600"))

        from flyte._run_python_script import _build_script_runner_task

        return _build_script_runner_task(script_name, output_dir, timeout)

    def loader_args(self, task: TaskTemplate, root_dir: Optional[Path] = None) -> List[str]:
        args = ["script", self._script_name]
        if self._output_dir is not None:
            args.extend(["output_dir", self._output_dir])
        args.extend(["timeout", str(self._timeout)])
        return args
