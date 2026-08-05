from __future__ import annotations

import json
import os
import pathlib
import sys

from flyte._task import TaskTemplate


class NotebookTaskResolver:
    """Resolver for NotebookTask instances.

    Serializes all task state (notebook path, type schemas, execution config)
    into the loader args at serialization time so the task can be reconstructed
    in the container without importing the user's module. This lets NotebookTask
    be defined inline (e.g. inside a task) rather than only at module level.

    Loader args format (key-value pairs):
        notebook  <relative-or-absolute notebook path>
        name      <task name>
        input-schema   <JSON: {field: LiteralType dict}>
        output-schema  <JSON: {field: LiteralType dict}>
        config         <JSON: execution params>
    """

    @property
    def import_path(self) -> str:
        return "flyteplugins.papermill.resolver.NotebookTaskResolver"

    def load_task(self, loader_args: list[str]) -> TaskTemplate:
        # Ensure all IO type transformers are registered before attempting
        # guess_python_type — they are registered as side-effects of import.
        import flyte.io  # noqa: F401
        from flyte.types import TypeEngine
        from flyteidl2.core.types_pb2 import LiteralType
        from google.protobuf import json_format

        from flyteplugins.papermill.task import NotebookTask

        # Parse flat key-value list
        it = iter(loader_args)
        args_dict: dict[str, str] = {}
        for key in it:
            try:
                args_dict[key] = next(it)
            except StopIteration:
                raise ValueError(f"Odd number of loader args — missing value for key '{key}'")

        notebook_path = args_dict["notebook"]
        name = args_dict["name"]

        # Relative paths are stored relative to the bundle root (root_dir at
        # serialization time). At execution time, the bundle extraction dir is
        # prepended to sys.path by download_code_bundle, so we search there.
        if not os.path.isabs(notebook_path):
            for p in sys.path:
                candidate = os.path.abspath(os.path.join(p, notebook_path))
                if os.path.exists(candidate):
                    notebook_path = candidate
                    break
            else:
                # CWD fallback: covers the common destination="." case where
                # the bundle is extracted into the current working directory.
                notebook_path = os.path.abspath(notebook_path)

        def _schema_to_types(schema_json: str) -> dict | None:
            schema = json.loads(schema_json)
            if not schema:
                return None
            result: dict = {}
            for field_name, lt_dict in schema.items():
                lt = LiteralType()
                json_format.ParseDict(lt_dict, lt)
                result[field_name] = TypeEngine.guess_python_type(lt)
            return result

        inputs = _schema_to_types(args_dict.get("input-schema", "{}"))
        outputs = _schema_to_types(args_dict.get("output-schema", "{}"))
        config: dict = json.loads(args_dict.get("config", "{}"))

        return NotebookTask(
            name=name,
            notebook_path=notebook_path,
            task_environment=None,
            inputs=inputs,
            outputs=outputs,
            **config,
        )

    def loader_args(self, task: TaskTemplate, root_dir: pathlib.Path | None) -> list[str]:
        from flyte.types import TypeEngine
        from google.protobuf import json_format

        from flyteplugins.papermill.task import NotebookTask

        if not isinstance(task, NotebookTask):
            raise TypeError(f"NotebookTaskResolver only handles NotebookTask, got {type(task)}")

        # If the notebook is inside the bundle root, store a relative path so
        # it resolves correctly wherever the bundle is extracted in the container.
        # If outside root_dir (or no root_dir), preserve the path exactly as the
        # user wrote it: absolute paths are expected to exist in the container
        # image; relative paths remain relative and must be resolvable from CWD
        # or sys.path at execution time.
        nb_path = pathlib.Path(task._resolved_notebook_path)
        if root_dir is not None:
            try:
                notebook_arg = str(nb_path.relative_to(pathlib.Path(root_dir).resolve()))
            except ValueError:
                notebook_arg = str(nb_path)
        else:
            # No bundle root: use the original path as the user wrote it.
            notebook_arg = task.notebook_path

        def _types_to_schema(types_dict: dict) -> dict:
            schema: dict = {}
            for field_name, typ in types_dict.items():
                lt = TypeEngine.to_literal_type(typ)
                schema[field_name] = json_format.MessageToDict(lt)
            return schema

        # Inputs
        input_types = {name: typ for name, (typ, _) in (task.interface.inputs or {}).items()}
        input_schema = _types_to_schema(input_types)

        # Outputs: strip the auto-added notebook File outputs so the reconstructed
        # task can re-add them via output_notebooks=True in the config.
        skip_outputs = {"output_notebook", "output_notebook_executed"} if task.output_notebooks else set()
        output_types = {name: typ for name, typ in (task.interface.outputs or {}).items() if name not in skip_outputs}
        output_schema = _types_to_schema(output_types)

        # Execution config (plugin_config is serialization-only; not needed at execution time)
        config: dict = {
            "log_output": task.log_output,
            "start_timeout": task.start_timeout,
            "report_mode": task.report_mode,
            "request_save_on_cell_execute": task.request_save_on_cell_execute,
            "progress_bar": task.progress_bar,
            "output_notebooks": task.output_notebooks,
        }
        if task.kernel_name is not None:
            config["kernel_name"] = task.kernel_name
        if task.engine_name is not None:
            config["engine_name"] = task.engine_name
        if task.execution_timeout is not None:
            config["execution_timeout"] = task.execution_timeout
        if task.language is not None:
            config["language"] = task.language
        if task.engine_kwargs:
            config["engine_kwargs"] = task.engine_kwargs

        return [
            "notebook",
            notebook_arg,
            "name",
            task.name,
            "input-schema",
            json.dumps(input_schema),
            "output-schema",
            json.dumps(output_schema),
            "config",
            json.dumps(config),
        ]
