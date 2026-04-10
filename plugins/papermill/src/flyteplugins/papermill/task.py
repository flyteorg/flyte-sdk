from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import sys
import tempfile
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from flyte import TaskEnvironment
from flyte.extend import TaskTemplate
from flyte.models import NativeInterface, SerializationContext

logger = logging.getLogger(__name__)

# Environment variable used to pass the parent task context to the notebook kernel.
_FLYTE_NB_CTX_ENV = "_FLYTE_NB_CTX"

# Cell injected at the start of the notebook to initialize the Flyte runtime context
# in the kernel subprocess. This allows tasks called from within the notebook to be
# submitted through the Flyte controller.
_FLYTE_NB_SETUP_CELL = """\
# Flyte notebook context setup — initializes runtime context in the kernel
# so that task calls are submitted through the Flyte controller.
# IMPORTANT: This cell must be fully synchronous (no `await`). IPython wraps
# cells containing `await` in an async coroutine, which creates a ContextVar
# copy. Changes to ContextVar inside a copy don't persist to later cells.
from flyteplugins.papermill._setup import initialize_context as _flyte_init_ctx
_flyte_init_ctx()
del _flyte_init_ctx
"""

# Maps plugin_config types to their task_type strings and plugin task classes.
# This allows NotebookTask to delegate pre/post/custom_config to the right plugin.
_PLUGIN_REGISTRY: dict[type, tuple[str, type]] = {}


def _register_plugin(config_type: type, task_type: str, task_class: type) -> None:
    _PLUGIN_REGISTRY[config_type] = (task_type, task_class)


@functools.lru_cache(maxsize=None)
def _auto_register_plugins() -> None:
    try:
        from flyteplugins.spark import Spark
        from flyteplugins.spark.task import PysparkFunctionTask

        _register_plugin(Spark, "spark", PysparkFunctionTask)
    except ImportError:
        pass


def _resolve_task_type(plugin_config: Any) -> str:
    if plugin_config is None:
        return "notebook"
    config_type = type(plugin_config)
    if config_type in _PLUGIN_REGISTRY:
        return _PLUGIN_REGISTRY[config_type][0]
    raise ValueError(f"Unsupported plugin_config type: {config_type}. No plugin registered for it.")


def _build_interface(
    inputs: Optional[dict[str, Type]],
    outputs: Optional[dict[str, Type]],
) -> NativeInterface:
    """Build a NativeInterface from simple type dicts.

    Inputs format: {"name": type, ...} — all required (no defaults).
    Outputs format: {"name": type, ...}
    """
    input_spec: dict[str, tuple] = {}
    if inputs:
        for name, typ in inputs.items():
            # (type, default value)
            input_spec[name] = (typ, inspect.Parameter.empty)

    output_spec: dict[str, Type] = outputs or {}
    return NativeInterface(inputs=input_spec, outputs=output_spec)


@dataclass(kw_only=True)
class NotebookTask(TaskTemplate):
    """A Flyte task that executes a Jupyter notebook via Papermill.

    The notebook receives task inputs as parameters (injected into the cell
    tagged ``parameters``) and produces outputs via ``record_outputs()``
    called inside the notebook.

    Example::

        from flyteplugins.papermill import NotebookTask

        analyze = NotebookTask(
            name="analyze",
            notebook_path="notebooks/analyze.ipynb",
            task_environment=env,
            inputs={"x": int, "y": float},
            outputs={"result": int},
        )

    Inside *notebooks/analyze.ipynb*::

        from flyteplugins.papermill import record_outputs

        result = x + y  # x, y injected by papermill
        record_outputs(result=int(result))

    You can also call other Flyte tasks from within the notebook — just
    import and call them as usual::

        from my_tasks import expensive_task

        intermediate = await expensive_task(data=x)  # submitted to Flyte when running remotely
        record_outputs(result=intermediate)

    Spark example::

        from flyteplugins.papermill import NotebookTask
        from flyteplugins.spark import Spark

        spark_nb = NotebookTask(
            name="spark_analyze",
            notebook_path="notebooks/spark_analysis.ipynb",
            task_environment=env,
            plugin_config=Spark(spark_conf={...}),
            inputs={"path": str},
            outputs={"count": int},
        )

    Args:
        name: Task name.
        notebook_path: Path to the ``.ipynb`` file (relative to the caller's
            file or absolute).
        task_environment: The ``TaskEnvironment`` this task belongs to.
            Required for remote execution.
        plugin_config: Plugin configuration (e.g. ``Spark(...)``). Sets
            the task type and execution environment accordingly.
        inputs: Mapping of input names to Python types.
        outputs: Mapping of output names to Python types.
        kernel_name: Jupyter kernel to use. Defaults to the kernel
            specified in the notebook metadata.
        engine_name: Papermill engine name. Defaults to the standard
            ``nbclient`` engine. Custom engines registered via the
            ``papermill.engine`` entry point are also available.
        log_output: Stream cell outputs to the task log.
        start_timeout: Seconds to wait for the kernel to start.
        execution_timeout: Per-cell execution timeout in seconds.
            ``None`` means no timeout.
        report_mode: Hide input cells in the output notebook.
        request_save_on_cell_execute: Save the notebook after every cell
            execution. Useful for inspecting partial progress on failure.
        progress_bar: Show a progress bar during execution.
        language: Override the notebook language.
        engine_kwargs: Extra keyword arguments forwarded to the
            papermill engine (e.g. ``autosave_cell_every``).
        output_notebooks: When ``True``, the actual and executed ``.ipynb`` files
            are uploaded to remote storage and returned as `Files`s in the task output,
            making it accessible to downstream tasks.
    """

    notebook_path: str
    plugin_config: Optional[Any] = None

    # Papermill execute_notebook parameters
    kernel_name: Optional[str] = None
    engine_name: Optional[str] = None
    log_output: bool = False
    start_timeout: int = 60
    execution_timeout: Optional[int] = None
    report_mode: bool = False
    request_save_on_cell_execute: bool = True
    progress_bar: bool = True
    language: Optional[str] = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    output_notebooks: bool = False

    # Internal
    _resolved_notebook_path: str = field(init=False, repr=False, default="")

    def __init__(
        self,
        *,
        name: str,
        notebook_path: str,
        task_environment: TaskEnvironment,
        plugin_config: Optional[Any] = None,
        inputs: Optional[dict[str, Type]] = None,
        outputs: Optional[dict[str, Type]] = None,
        kernel_name: Optional[str] = None,
        engine_name: Optional[str] = None,
        log_output: bool = False,
        start_timeout: int = 60,
        execution_timeout: Optional[int] = None,
        report_mode: bool = False,
        request_save_on_cell_execute: bool = True,
        progress_bar: bool = True,
        language: Optional[str] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
        output_notebooks: bool = False,
        **kwargs: Any,
    ):
        # Resolve relative notebook paths against CWD (the project root when
        # running normally). Absolute paths are used as-is.
        if not os.path.isabs(notebook_path):
            resolved = os.path.normpath(os.path.join(os.getcwd(), notebook_path))
        else:
            resolved = os.path.normpath(notebook_path)

        if output_notebooks:
            from flyte.io import File

            outputs = dict(outputs or {})
            outputs["output_notebook"] = File
            outputs["output_notebook_executed"] = File

        interface = _build_interface(inputs, outputs)

        # Auto-register known plugins on first use
        _auto_register_plugins()

        # Resolve task_type from plugin_config
        task_type = _resolve_task_type(plugin_config)

        super().__init__(
            name=name,
            interface=interface,
            task_type=task_type,
            report=True,
            _call_as_synchronous=True,
            parent_env=weakref.ref(task_environment) if task_environment else None,
            parent_env_name=task_environment.name if task_environment else None,
            **kwargs,
        )
        self.notebook_path = notebook_path
        self.plugin_config = plugin_config
        self.kernel_name = kernel_name
        self.engine_name = engine_name
        self.log_output = log_output
        self.start_timeout = start_timeout
        self.execution_timeout = execution_timeout
        self.report_mode = report_mode
        self.request_save_on_cell_execute = request_save_on_cell_execute
        self.progress_bar = progress_bar
        self.language = language
        self.engine_kwargs = engine_kwargs or {}
        self.output_notebooks = output_notebooks
        self._resolved_notebook_path = resolved

        if task_environment is not None:
            task_environment._tasks[name] = self

        if log_output:
            pm_logger = logging.getLogger("papermill")
            if not pm_logger.handlers:
                pm_logger.addHandler(logging.StreamHandler(sys.stdout))
                pm_logger.setLevel(logging.INFO)

    @property
    def resolved_notebook_path(self) -> str:
        return self._resolved_notebook_path

    @property
    def output_notebook_path(self) -> str:
        base, ext = os.path.splitext(self._resolved_notebook_path)
        return f"{base}-out{ext}"

    @property
    def _plugin_task_class(self) -> Optional[type]:
        if self.plugin_config is None:
            return None
        entry = _PLUGIN_REGISTRY.get(type(self.plugin_config))
        return entry[1] if entry else None

    def custom_config(self, sctx: SerializationContext) -> dict[str, Any]:
        # For Spark: serializes the SparkJob spec (sparkConf, hadoopConf, pod
        # templates, etc.) consumed by the K8s Spark operator to configure the
        # driver and executor pods. This is the sole Spark initialization path
        # for notebook tasks — there is no pre()/post() lifecycle here.
        #
        # NOTE: SparkContext.addPyFile() is not called for notebook tasks.
        # Regular PysparkFunctionTask uses addPyFile() to distribute the code
        # bundle to executors, but since the notebook runs in a kernel subprocess
        # that cannot share state with the parent process, that approach does not
        # work. In practice this is fine for K8s Spark: executors use the same
        # Docker image as the driver, so any packages needed in UDFs must be
        # installed in the image. Dynamic code distribution via addPyFile() is
        # not supported for notebook tasks.
        plugin_cls = self._plugin_task_class
        if plugin_cls is not None and hasattr(plugin_cls, "custom_config"):
            return plugin_cls.custom_config(self, sctx)
        return {}

    @staticmethod
    def _serialize_params(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert complex Flyte types to JSON-serializable papermill parameters.

        Papermill parameters must be JSON-serializable primitives.  Complex
        types like ``File``, ``Dir``, and ``DataFrame`` are converted to
        their path/URI strings.  Use ``load_file()`` / ``load_dir()`` /
        ``load_dataframe()`` inside the notebook to reconstruct them.
        """
        from flyte.io import DataFrame, Dir, File

        _PRIMITIVES = (int, float, str, bool, list, dict, type(None))
        out: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, File) or isinstance(v, Dir):
                out[k] = str(v.path)
            elif isinstance(v, DataFrame):
                out[k] = str(v.uri)
            elif isinstance(v, _PRIMITIVES):
                out[k] = v
            else:
                raise TypeError(
                    f"NotebookTask input '{k}' has unsupported type {type(v)!r}. "
                    f"Papermill only supports primitives (int, float, str, bool, list, dict, None) "
                    f"and flyte.io types (File, Dir, DataFrame)."
                )
        return out

    @staticmethod
    def _serialize_task_context() -> Optional[str]:
        """Serialize the current TaskContext to JSON for the notebook kernel.

        Only returns context for remote/hybrid execution.
        """
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if tctx is None or tctx.mode == "local":
            return None

        ctx_data: dict[str, Any] = {
            "action_name": tctx.action.name,
            "run_name": tctx.action.run_name,
            "project": tctx.action.project,
            "domain": tctx.action.domain,
            "org": tctx.action.org,
            "version": tctx.version,
            "output_path": tctx.output_path,
            "run_base_dir": tctx.run_base_dir,
            "raw_data_path": tctx.raw_data_path.path,
            "mode": tctx.mode,
            "interactive_mode": tctx.interactive_mode,
        }

        if tctx.code_bundle:
            ctx_data["code_bundle"] = {
                "tgz": tctx.code_bundle.tgz,
                "pkl": tctx.code_bundle.pkl,
                "destination": tctx.code_bundle.destination,
                "computed_version": tctx.code_bundle.computed_version,
            }

        if tctx.compiled_image_cache:
            ctx_data["image_cache"] = tctx.compiled_image_cache.to_transport

        return json.dumps(ctx_data)

    @staticmethod
    def _serialize_local_context() -> str:
        """Produce a minimal context JSON for local notebook execution.

        The notebook kernel is a fresh subprocess that has not had
        ``flyte.init()`` called.  Injecting this context triggers the
        setup cell to call ``flyte.init()`` (no controller) and sets a
        local temp directory as the ``raw_data_path``, which is required
        for APIs like ``File.new_remote()`` to work inside the notebook.
        """
        raw_data_dir = tempfile.mkdtemp(prefix="flyte_nb_")
        return json.dumps(
            {
                "action_name": "local",
                "run_name": "local",
                "project": "local",
                "domain": "development",
                "org": "",
                "version": "local",
                "output_path": raw_data_dir,
                "run_base_dir": "",
                "raw_data_path": raw_data_dir,
                "mode": "local",
                "interactive_mode": False,
            }
        )

    @staticmethod
    def _inject_setup_cell(notebook_path: str) -> str:
        """Inject the Flyte context setup cell into the notebook.

        Returns the path to a temporary notebook file with the setup cell
        injected at position 0. The caller is responsible for cleanup.
        """
        import nbformat

        nb = nbformat.read(notebook_path, as_version=4)

        setup_cell = nbformat.v4.new_code_cell(source=_FLYTE_NB_SETUP_CELL)
        setup_cell.metadata["tags"] = ["flyte-setup"]
        nb.cells.insert(0, setup_cell)

        # Normalize to fix missing cell IDs
        nbformat.validator.normalize(nb)

        fd, tmp_path = tempfile.mkstemp(suffix=".ipynb")
        try:
            with os.fdopen(fd, "w") as f:
                nbformat.write(nb, f)
        except Exception:
            # os.fdopen takes ownership of fd and closes it on exit;
            # only the temp file needs cleanup here.
            os.unlink(tmp_path)
            raise

        return tmp_path

    @staticmethod
    def _extract_outputs(notebook_path: str) -> Optional[Any]:
        """Extract recorded outputs from the executed notebook.

        Looks for a cell tagged ``"outputs"`` in the executed notebook and
        parses the protobuf text that ``record_outputs()`` returned as the
        cell's ``text/plain`` output.
        """
        from flyteidl2.core.literals_pb2 import LiteralMap
        from google.protobuf import text_format

        with open(notebook_path) as f:
            nb = json.load(f)

        for cell in nb.get("cells", []):
            metadata = cell.get("metadata", {})
            tags = metadata.get("tags", [])
            if "outputs" not in tags:
                continue

            # Find the text/plain output from the cell
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                text = data.get("text/plain")
                if text is None:
                    continue

                # text can be a list of lines or a single string
                if isinstance(text, list):
                    text = "".join(text)

                # Strip surrounding quotes that Jupyter adds to repr strings
                text = text.strip()
                if text.startswith("'") and text.endswith("'"):
                    text = text[1:-1]
                elif text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]

                # Unescape escaped newlines from the repr
                text = text.replace("\\n", "\n")

                literal_map = LiteralMap()
                text_format.Parse(text, literal_map)
                return literal_map

        return None

    def _execute_notebook(self, *, _inject_context: bool = False, **kwargs: Any) -> Optional[Any]:
        """Run the notebook via papermill and collect outputs.

        When *_inject_context* is ``True`` and the ``_FLYTE_NB_CTX`` env var
        is set, a setup cell is injected at the start of the notebook to
        initialize the Flyte runtime context in the kernel subprocess.
        This allows tasks called from within the notebook to be submitted
        through the controller.
        """
        import papermill as pm

        kwargs = self._serialize_params(kwargs)

        input_path = self._resolved_notebook_path
        tmp_notebook: Optional[str] = None

        if _inject_context and os.environ.get(_FLYTE_NB_CTX_ENV):
            tmp_notebook = self._inject_setup_cell(self._resolved_notebook_path)
            input_path = tmp_notebook

        try:
            pm_kwargs: dict[str, Any] = {
                "input_path": input_path,
                "output_path": self.output_notebook_path,
                "parameters": kwargs,
                "kernel_name": self.kernel_name,
                "engine_name": self.engine_name,
                "log_output": self.log_output,
                "start_timeout": self.start_timeout,
                "report_mode": self.report_mode,
                "request_save_on_cell_execute": self.request_save_on_cell_execute,
                "progress_bar": self.progress_bar,
                "language": self.language,
            }
            # Remove None values so papermill uses its own defaults
            pm_kwargs = {k: v for k, v in pm_kwargs.items() if v is not None}

            # execution_timeout is an engine kwarg, not a top-level pm param
            extra = dict(self.engine_kwargs)
            if self.execution_timeout is not None:
                extra["execution_timeout"] = self.execution_timeout
            pm_kwargs.update(extra)

            pm.execute_notebook(**pm_kwargs)
        finally:
            if tmp_notebook and os.path.exists(tmp_notebook):
                os.unlink(tmp_notebook)

        # Extract outputs from the cell tagged "outputs" in the executed notebook
        return self._extract_outputs(self.output_notebook_path)

    async def _render_and_upload_report(self) -> tuple[Optional[Any], Optional[Any]]:
        """Render the notebook to HTML, log to Flyte report, and optionally
        upload the source and executed ``.ipynb`` as ``File`` artifacts.

        Returns:
            ``(source_file, executed_file)`` when ``output_notebooks=True``
            and running in a task context, otherwise ``(None, None)``.
        """
        import flyte.report
        import flyte.storage as storage
        import nbconvert
        import nbformat
        from flyte._context import internal_ctx
        from flyte.io import File

        nb = nbformat.read(self.output_notebook_path, as_version=4)

        # Strip the injected setup cell
        nb.cells = [c for c in nb.cells if "flyte-setup" not in c.get("metadata", {}).get("tags", [])]

        if self.report_mode:
            for cell in nb.cells:
                if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
                    cell["source"] = ""

        # Write back to disk so the uploaded .ipynb reflects the same
        # cell filtering (setup cell stripped, report_mode cells stripped).
        nbformat.write(nb, self.output_notebook_path)

        exporter = nbconvert.HTMLExporter()
        html_body, _ = exporter.from_notebook_node(nb)

        # Log rendered HTML to Flyte report
        await flyte.report.log.aio(html_body, do_flush=True)

        if not self.output_notebooks:
            return None, None

        ctx = internal_ctx()
        if not ctx.is_task_context():
            return None, None

        output_base = ctx.data.task_context.output_path

        source_remote = storage.join(output_base, f"{self.name}-source.ipynb")
        source_file = await File.from_local(self._resolved_notebook_path, source_remote)
        logger.info(f"Uploaded source notebook to {source_remote}")

        executed_remote = storage.join(output_base, f"{self.name}-executed.ipynb")
        executed_file = await File.from_local(self.output_notebook_path, executed_remote)
        logger.info(f"Uploaded executed notebook to {executed_remote}")

        return source_file, executed_file

    async def _build_return(self, literal_map: Any, extra_outputs: Optional[dict[str, Any]] = None) -> Any:
        """Convert Flyte Literal outputs back to Python values.

        Args:
            literal_map: Outputs recorded by ``record_outputs()`` in the notebook.
            extra_outputs: Additional Python values (e.g. ``File`` objects created
                in-process) to merge in without going through the literal map.
        """
        from flyte.types import TypeEngine

        output_types = self.interface.outputs
        if not output_types:
            return None

        # Outputs provided directly as Python values don't need literal map lookup.
        extra = extra_outputs or {}
        literal_output_types = {k: v for k, v in output_types.items() if k not in extra}

        kwargs: dict[str, Any] = {}
        if literal_output_types:
            if literal_map is None:
                missing = list(literal_output_types.keys())
                raise TypeError(
                    f"Notebook did not produce expected outputs: {missing}. "
                    "Make sure to call record_outputs() in your notebook."
                )
            for name in literal_output_types:
                if name not in literal_map.literals:
                    raise TypeError(
                        f"Notebook did not produce expected output '{name}'. "
                        f"Make sure to call record_outputs({name}=...) in your notebook."
                    )
            kwargs = await TypeEngine.literal_map_to_kwargs(literal_map, literal_output_types)

        kwargs.update(extra)
        if len(kwargs) == 1:
            return next(iter(kwargs.values()))
        return tuple(kwargs[k] for k in output_types)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the notebook locally (outside of a Flyte run context).

        Called when the task is invoked directly as a Python function
        (e.g. in a test or script) rather than through the Flyte runner.
        Runs the notebook in-process and returns Python values.
        """
        import asyncio

        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)

        os.environ[_FLYTE_NB_CTX_ENV] = self._serialize_local_context()
        try:
            literal_map = self._execute_notebook(_inject_context=True, **kwargs)
        finally:
            os.environ.pop(_FLYTE_NB_CTX_ENV, None)

        source_path = self._resolved_notebook_path
        executed_path = self.output_notebook_path

        async def _build():
            extra: dict[str, Any] = {}
            if self.output_notebooks:
                from flyte.io import File

                extra = {
                    "output_notebook": File(path=source_path),
                    "output_notebook_executed": File(path=executed_path),
                }
            return await self._build_return(literal_map, extra_outputs=extra or None)

        return asyncio.run(_build())

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the notebook within a Flyte task context."""
        from flyte._context import internal_ctx
        from flyte._utils.asyncify import run_sync_with_loop

        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)

        ctx = internal_ctx()
        ctx_manager = ctx.replace_task_context(ctx.data.task_context) if ctx.data.task_context else nullcontext()
        with ctx_manager:
            # For local mode, _serialize_task_context returns None, so fall back to
            # a minimal local context so the setup cell still runs and initializes
            # Flyte in the kernel subprocess (required for File.new_remote() etc.).
            ctx_json = self._serialize_task_context() or self._serialize_local_context()
            os.environ[_FLYTE_NB_CTX_ENV] = ctx_json

            def _run():
                return self._execute_notebook(_inject_context=True, **kwargs)

            literal_map = None
            execution_error: Optional[BaseException] = None
            try:
                literal_map = await run_sync_with_loop(_run)
            except Exception as exc:
                execution_error = exc
            finally:
                os.environ.pop(_FLYTE_NB_CTX_ENV, None)

            # Always render before re-raising — papermill writes the output
            # notebook cell-by-cell, so the partial notebook is on disk even
            # after a failure and the report is still populated.
            source_file, executed_file = None, None
            if self.report and os.path.exists(self.output_notebook_path):
                source_file, executed_file = await self._render_and_upload_report()

            if execution_error is not None:
                raise execution_error

            extra: dict[str, Any] = {}
            if self.output_notebooks and source_file is not None:
                extra = {
                    "output_notebook": source_file,
                    "output_notebook_executed": executed_file,
                }

            result = await self._build_return(literal_map, extra_outputs=extra or None)

        return result

    def container_args(self, serialize_context: SerializationContext) -> list[str]:
        from .resolver import NotebookTaskResolver

        resolver = NotebookTaskResolver()

        args = [
            "a0",
            "--inputs",
            serialize_context.input_path,
            "--outputs-path",
            serialize_context.output_path,
            "--version",
            serialize_context.version,
            "--raw-data-path",
            "{{.rawOutputDataPrefix}}",
            "--checkpoint-path",
            "{{.checkpointOutputPrefix}}",
            "--prev-checkpoint",
            "{{.prevCheckpointPrefix}}",
            "--run-name",
            "{{.runName}}",
            "--name",
            "{{.actionName}}",
        ]

        if serialize_context.image_cache and serialize_context.image_cache.serialized_form:
            args = [
                *args,
                "--image-cache",
                serialize_context.image_cache.serialized_form,
            ]
        elif serialize_context.image_cache:
            args = [
                *args,
                "--image-cache",
                serialize_context.image_cache.to_transport,
            ]

        if serialize_context.code_bundle:
            if serialize_context.code_bundle.tgz:
                args = [*args, "--tgz", f"{serialize_context.code_bundle.tgz}"]
            elif serialize_context.code_bundle.pkl:
                args = [*args, "--pkl", f"{serialize_context.code_bundle.pkl}"]
            args = [
                *args,
                "--dest",
                f"{serialize_context.code_bundle.destination or '.'}",
            ]

        args = [
            *args,
            "--resolver",
            resolver.import_path,
            *resolver.loader_args(task=self, root_dir=serialize_context.root_dir),
        ]

        return args
