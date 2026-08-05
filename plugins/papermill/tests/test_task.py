"""Tests for flyteplugins.papermill.task and flyteplugins.papermill.notebook."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import nbformat
import pytest

TESTS_DIR = Path(__file__).parent
NOTEBOOK_PATH = str(TESTS_DIR / "test_notebook.ipynb")
NO_OUTPUTS_NOTEBOOK_PATH = str(TESTS_DIR / "test_notebook_no_outputs.ipynb")
PRIMITIVES_NOTEBOOK_PATH = str(TESTS_DIR / "test_notebook_primitives.ipynb")
PRIMITIVE_TYPES_NOTEBOOK_PATH = str(TESTS_DIR.parent / "examples" / "notebooks" / "primitive_types.ipynb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_UNSET = object()


def make_task(notebook_path=NOTEBOOK_PATH, inputs=_UNSET, outputs=_UNSET, **kwargs):
    """Create a NotebookTask without requiring a real TaskEnvironment."""
    from flyteplugins.papermill.task import NotebookTask

    with patch("flyte.TaskEnvironment") as mock_env_cls:
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        mock_env._tasks = {}
        return NotebookTask(
            name="test_task",
            notebook_path=notebook_path,
            task_environment=mock_env,
            inputs={"x": int, "y": float} if inputs is _UNSET else inputs,
            outputs={"result": int} if outputs is _UNSET else outputs,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# _build_interface
# ---------------------------------------------------------------------------


def test_build_interface_inputs_and_outputs():
    from flyteplugins.papermill.task import _build_interface

    iface = _build_interface({"x": int, "y": float}, {"result": int})
    assert "x" in iface.inputs
    assert "y" in iface.inputs
    assert "result" in iface.outputs


def test_build_interface_empty():
    from flyteplugins.papermill.task import _build_interface

    iface = _build_interface(None, None)
    assert iface.inputs == {}
    assert iface.outputs == {}


# ---------------------------------------------------------------------------
# NotebookTask construction
# ---------------------------------------------------------------------------


def test_notebook_path_resolved_relative():
    task = make_task()
    # Resolved path should be absolute and exist
    assert os.path.isabs(task.resolved_notebook_path)
    assert task.resolved_notebook_path.endswith(".ipynb")


def test_output_notebook_path():
    task = make_task()
    base = os.path.splitext(task.resolved_notebook_path)[0]
    assert task.output_notebook_path == f"{base}-out.ipynb"


def test_output_notebooks_adds_file_outputs():
    from flyte.io import File

    task = make_task(outputs={"result": int}, output_notebooks=True)
    assert "output_notebook" in task.interface.outputs
    assert "output_notebook_executed" in task.interface.outputs
    assert task.interface.outputs["output_notebook"] is File
    assert task.interface.outputs["output_notebook_executed"] is File


def test_output_notebooks_false_no_extra_outputs():
    task = make_task(outputs={"result": int}, output_notebooks=False)
    assert "output_notebook" not in task.interface.outputs
    assert "output_notebook_executed" not in task.interface.outputs


def test_notebook_task_accepts_bool_list_dict():
    task = make_task(inputs={"flag": bool, "items": list, "metadata": dict}, outputs={"result": int})
    assert "flag" in task.interface.inputs
    assert "items" in task.interface.inputs
    assert "metadata" in task.interface.inputs


# ---------------------------------------------------------------------------
# _serialize_params
# ---------------------------------------------------------------------------


def test_serialize_params_primitives():
    task = make_task()
    result = task._serialize_params({"x": 1, "y": 2.5, "s": "hello"})
    assert result == {"x": 1, "y": 2.5, "s": "hello"}


def test_serialize_params_file():
    from flyte.io import File

    task = make_task()
    f = File(path="s3://bucket/file.txt")
    result = task._serialize_params({"f": f})
    assert result["f"] == "s3://bucket/file.txt"


def test_serialize_params_dir():
    from flyte.io import Dir

    task = make_task()
    d = Dir(path="s3://bucket/dir/")
    result = task._serialize_params({"d": d})
    assert result["d"] == "s3://bucket/dir/"


def test_serialize_params_dataframe():
    from flyte.io import DataFrame

    task = make_task()
    df = DataFrame(uri="s3://bucket/data.parquet", format="parquet")
    result = task._serialize_params({"df": df})
    assert result["df"] == "s3://bucket/data.parquet"


def test_serialize_params_bool_list_dict_none():
    task = make_task()
    result = task._serialize_params(
        {
            "flag": True,
            "items": [1, "two", 3.0],
            "config": {"key": "val", "count": 5},
            "nothing": None,
        }
    )
    assert result == {
        "flag": True,
        "items": [1, "two", 3.0],
        "config": {"key": "val", "count": 5},
        "nothing": None,
    }


def test_serialize_params_unsupported_type_raises():
    import dataclasses

    @dataclasses.dataclass
    class Foo:
        x: int = 0

    task = make_task()
    with pytest.raises(TypeError, match="unsupported type"):
        task._serialize_params({"bad": Foo()})


# ---------------------------------------------------------------------------
# _serialize_local_context
# ---------------------------------------------------------------------------


def test_serialize_local_context_structure():
    task = make_task()
    raw = task._serialize_local_context()
    data = json.loads(raw)
    assert data["mode"] == "local"
    assert data["action_name"] == "local"
    assert os.path.isdir(data["raw_data_path"])


# ---------------------------------------------------------------------------
# _inject_setup_cell
# ---------------------------------------------------------------------------


def test_inject_setup_cell_inserts_at_position_zero():
    task = make_task()
    tmp = task._inject_setup_cell(NOTEBOOK_PATH)
    try:
        nb = nbformat.read(tmp, as_version=4)
        first_cell = nb.cells[0]
        assert "flyte-setup" in first_cell.metadata.get("tags", [])
        assert "initialize_context" in first_cell.source
    finally:
        os.unlink(tmp)


def test_inject_setup_cell_original_cells_follow():
    task = make_task()
    original_nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    original_count = len(original_nb.cells)
    tmp = task._inject_setup_cell(NOTEBOOK_PATH)
    try:
        nb = nbformat.read(tmp, as_version=4)
        assert len(nb.cells) == original_count + 1
    finally:
        os.unlink(tmp)


def test_inject_setup_cell_cleans_up_on_error():
    task = make_task()
    created_paths = []

    original_mkstemp = tempfile.mkstemp

    def capturing_mkstemp(**kwargs):
        fd, path = original_mkstemp(**kwargs)
        created_paths.append(path)
        return fd, path

    with patch("tempfile.mkstemp", side_effect=capturing_mkstemp):
        with patch("nbformat.write", side_effect=RuntimeError("write failed")):
            with pytest.raises(RuntimeError):
                task._inject_setup_cell(NOTEBOOK_PATH)

    # Temp file must have been deleted on error
    for path in created_paths:
        assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# _extract_outputs
# ---------------------------------------------------------------------------


def _make_executed_notebook(outputs_text: str, tag: str = "outputs") -> str:
    """Write a minimal executed notebook with one tagged outputs cell."""
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell(source="pass")
    cell.metadata["tags"] = [tag]
    cell.outputs = [
        nbformat.v4.new_output(
            output_type="execute_result",
            data={"text/plain": repr(outputs_text)},
            execution_count=1,
        )
    ]
    nb.cells.append(cell)
    nbformat.validator.normalize(nb)
    fd, path = tempfile.mkstemp(suffix=".ipynb")
    with os.fdopen(fd, "w") as f:
        nbformat.write(nb, f)
    return path


def test_extract_outputs_returns_none_without_outputs_tag():
    task = make_task()
    path = _make_executed_notebook("some text", tag="not-outputs")
    try:
        result = task._extract_outputs(path)
        assert result is None
    finally:
        os.unlink(path)


def test_extract_outputs_parses_literal_map():
    from flyteplugins.papermill.notebook import record_outputs

    task = make_task(outputs={"result": int})
    proto_text = record_outputs(result=42)

    path = _make_executed_notebook(proto_text)
    try:
        literal_map = task._extract_outputs(path)
        assert literal_map is not None
        assert "result" in literal_map.literals
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# _render_and_upload_report — report_mode cell stripping
# ---------------------------------------------------------------------------


def _make_executed_notebook_with_source_hidden() -> str:
    """Notebook with one cell marked source_hidden (as papermill report_mode sets)."""
    nb = nbformat.v4.new_notebook()
    visible_cell = nbformat.v4.new_code_cell(source="result = 1")
    hidden_cell = nbformat.v4.new_code_cell(source="x = 0  # this should be hidden")
    hidden_cell.metadata["jupyter"] = {"source_hidden": True}
    nb.cells = [visible_cell, hidden_cell]
    nbformat.validator.normalize(nb)
    fd, path = tempfile.mkstemp(suffix="-out.ipynb")
    with os.fdopen(fd, "w") as f:
        nbformat.write(nb, f)
    return path


@pytest.mark.asyncio
async def test_render_and_upload_report_strips_source_hidden():
    task = make_task(report_mode=True)

    executed_path = _make_executed_notebook_with_source_hidden()
    task._resolved_notebook_path = executed_path.replace("-out.ipynb", ".ipynb")

    # Redirect output_notebook_path to our temp file
    with patch.object(
        type(task),
        "output_notebook_path",
        new_callable=lambda: property(lambda self: executed_path),
    ):
        mock_log = MagicMock()
        mock_log.aio = AsyncMock()
        with patch("flyte.report.log", mock_log):
            with patch("flyte._context.internal_ctx") as mock_ctx:
                mock_ctx.return_value.is_task_context.return_value = False
                await task._render_and_upload_report()

    nb = nbformat.read(executed_path, as_version=4)
    for cell in nb.cells:
        if cell.metadata.get("jupyter", {}).get("source_hidden"):
            assert cell["source"] == ""

    os.unlink(executed_path)


# ---------------------------------------------------------------------------
# resolver — schema-based self-contained loading
# ---------------------------------------------------------------------------


def test_resolver_loader_args_schema_format():
    """loader_args() produces the new schema-based flat key-value format."""
    from flyteplugins.papermill.resolver import NotebookTaskResolver

    task = make_task(inputs={"x": int, "y": float}, outputs={"result": int})
    resolver = NotebookTaskResolver()
    args = resolver.loader_args(task=task, root_dir=None)

    args_dict = dict(zip(args[0::2], args[1::2]))
    assert args_dict["notebook"].endswith(".ipynb")
    assert args_dict["name"] == "test_task"

    import json

    input_schema = json.loads(args_dict["input-schema"])
    assert "x" in input_schema
    assert "y" in input_schema

    output_schema = json.loads(args_dict["output-schema"])
    assert "result" in output_schema


def test_resolver_round_trip_basic_types():
    """loader_args() + load_task() round-trips type schemas correctly."""
    from flyteplugins.papermill.resolver import NotebookTaskResolver
    from flyteplugins.papermill.task import NotebookTask

    task = make_task(inputs={"x": int, "y": float}, outputs={"result": int})
    resolver = NotebookTaskResolver()
    args = resolver.loader_args(task=task, root_dir=None)

    reconstructed = resolver.load_task(args)
    assert isinstance(reconstructed, NotebookTask)
    assert "x" in reconstructed.interface.inputs
    assert "y" in reconstructed.interface.inputs
    assert "result" in reconstructed.interface.outputs


def test_resolver_round_trip_complex_types():
    """DataFrame, File, and Dir types survive the loader_args/load_task round-trip."""
    from flyte.io import DataFrame, Dir, File

    from flyteplugins.papermill.resolver import NotebookTaskResolver
    from flyteplugins.papermill.task import NotebookTask

    task = make_task(
        inputs={"df": DataFrame, "f": File, "d": Dir},
        outputs={"out_df": DataFrame},
    )
    resolver = NotebookTaskResolver()
    args = resolver.loader_args(task=task, root_dir=None)
    reconstructed = resolver.load_task(args)

    assert isinstance(reconstructed, NotebookTask)
    # inputs are stored as (type, default) tuples
    assert reconstructed.interface.inputs["df"][0] is DataFrame
    assert reconstructed.interface.inputs["f"][0] is File
    assert reconstructed.interface.inputs["d"][0] is Dir
    assert reconstructed.interface.outputs["out_df"] is DataFrame


def test_resolver_round_trip_output_notebooks():
    """output_notebooks=True: auto-added File outputs are stripped before serialization
    and re-added by the reconstructed task's __init__."""
    from flyte.io import File

    from flyteplugins.papermill.resolver import NotebookTaskResolver

    task = make_task(outputs={"result": int}, output_notebooks=True)
    resolver = NotebookTaskResolver()
    args = resolver.loader_args(task=task, root_dir=None)

    import json

    args_dict = dict(zip(args[0::2], args[1::2]))
    output_schema = json.loads(args_dict["output-schema"])
    # File outputs are stripped from schema so the reconstructed task can re-add them
    assert "output_notebook" not in output_schema
    assert "output_notebook_executed" not in output_schema
    assert "result" in output_schema

    config = json.loads(args_dict["config"])
    assert config.get("output_notebooks") is True

    reconstructed = resolver.load_task(args)
    assert "output_notebook" in reconstructed.interface.outputs
    assert reconstructed.interface.outputs["output_notebook"] is File


def test_resolver_round_trip_no_outputs():
    """Tasks with no outputs are serialized and reconstructed correctly."""
    from flyteplugins.papermill.resolver import NotebookTaskResolver
    from flyteplugins.papermill.task import NotebookTask

    task = make_task(outputs=None)
    resolver = NotebookTaskResolver()
    args = resolver.loader_args(task=task, root_dir=None)

    reconstructed = resolver.load_task(args)
    assert isinstance(reconstructed, NotebookTask)
    assert reconstructed.interface.outputs == {}


def test_resolver_no_root_dir_preserves_original_path():
    """Without root_dir, notebook_path is stored as the user originally wrote it,
    not as the developer-machine absolute path."""
    from flyteplugins.papermill.resolver import NotebookTaskResolver

    # Use a relative path (as a user would write it)
    task = make_task(notebook_path=NOTEBOOK_PATH)  # abs path stands in for the user's path
    resolver = NotebookTaskResolver()
    args = resolver.loader_args(task=task, root_dir=None)

    args_dict = dict(zip(args[0::2], args[1::2]))
    # When root_dir is None, we use task.notebook_path (the original string)
    assert args_dict["notebook"] == task.notebook_path


def test_resolver_notebook_relative_to_root_dir(tmp_path):
    """When root_dir is provided, notebook path is stored relative to it."""
    import shutil

    from flyteplugins.papermill.resolver import NotebookTaskResolver

    # Copy the test notebook into a temp "root" dir to simulate a code bundle
    root = tmp_path / "bundle"
    nb_dir = root / "notebooks"
    nb_dir.mkdir(parents=True)
    dest = nb_dir / "test_notebook.ipynb"
    shutil.copy(NOTEBOOK_PATH, dest)

    task = make_task(notebook_path=str(dest))
    resolver = NotebookTaskResolver()
    args = resolver.loader_args(task=task, root_dir=root)

    args_dict = dict(zip(args[0::2], args[1::2]))
    # Notebook path should be relative to the bundle root
    assert not args_dict["notebook"].startswith("/")
    assert "notebooks" in args_dict["notebook"]


def test_resolver_load_task_resolves_relative_notebook(tmp_path):
    """load_task() finds the notebook when the path is relative and CWD is the bundle root."""
    import shutil
    import sys

    from flyteplugins.papermill.resolver import NotebookTaskResolver
    from flyteplugins.papermill.task import NotebookTask

    # Set up bundle structure
    root = tmp_path / "bundle"
    nb_dir = root / "notebooks"
    nb_dir.mkdir(parents=True)
    dest = nb_dir / "test_notebook.ipynb"
    shutil.copy(NOTEBOOK_PATH, dest)

    task = make_task(notebook_path=str(dest))
    resolver = NotebookTaskResolver()
    args = resolver.loader_args(task=task, root_dir=root)

    # Simulate container: add bundle root to sys.path
    old_sys_path = sys.path[:]
    try:
        sys.path.insert(0, str(root))
        reconstructed = resolver.load_task(args)
        assert isinstance(reconstructed, NotebookTask)
        assert os.path.isabs(reconstructed.resolved_notebook_path)
        assert os.path.exists(reconstructed.resolved_notebook_path)
    finally:
        sys.path[:] = old_sys_path


def test_container_args_no_root_dir_required():
    """container_args() no longer raises when root_dir is absent."""
    from flyte.models import SerializationContext

    task = make_task()
    sctx = SerializationContext(version="1", input_path="/in", output_path="/out", root_dir=None)
    args = task.container_args(sctx)
    assert "--resolver" in args


# ---------------------------------------------------------------------------
# custom_config — Spark plugin delegation
# ---------------------------------------------------------------------------


def test_custom_config_no_plugin():
    from flyte.models import SerializationContext

    task = make_task()
    sctx = SerializationContext(version="1")
    assert task.custom_config(sctx) == {}


def test_custom_config_spark_delegates():
    from flyte.models import SerializationContext

    with patch.dict("sys.modules", {"pyspark": MagicMock(), "pyspark.sql": MagicMock()}):
        try:
            from flyteplugins.spark import Spark

            sctx = SerializationContext(version="1")
            task = make_task(plugin_config=Spark(spark_conf={"spark.executor.instances": "2"}))
            result = task.custom_config(sctx)
            assert result.get("sparkConf", {}).get("spark.executor.instances") == "2"
        except ImportError:
            pytest.skip("flyteplugins-spark not installed")


# ---------------------------------------------------------------------------
# notebook helpers: load_file, load_dir, load_dataframe
# ---------------------------------------------------------------------------


def test_load_file_returns_file():
    from flyte.io import File

    from flyteplugins.papermill.notebook import load_file

    f = load_file("s3://bucket/data.txt")
    assert isinstance(f, File)
    assert str(f.path) == "s3://bucket/data.txt"


def test_load_dir_returns_dir():
    from flyte.io import Dir

    from flyteplugins.papermill.notebook import load_dir

    d = load_dir("s3://bucket/dir/")
    assert isinstance(d, Dir)
    assert str(d.path) == "s3://bucket/dir/"


def test_load_dataframe_returns_dataframe():
    from flyte.io import DataFrame

    from flyteplugins.papermill.notebook import load_dataframe

    df = load_dataframe("s3://bucket/data.parquet")
    assert isinstance(df, DataFrame)
    assert str(df.uri) == "s3://bucket/data.parquet"


def test_load_dataframe_custom_format():
    from flyteplugins.papermill.notebook import load_dataframe

    df = load_dataframe("s3://bucket/data.csv", fmt="csv")
    assert df.format == "csv"


# ---------------------------------------------------------------------------
# record_outputs
# ---------------------------------------------------------------------------


def test_record_outputs_returns_parseable_proto():
    from flyteidl2.core.literals_pb2 import LiteralMap
    from google.protobuf import text_format

    from flyteplugins.papermill.notebook import record_outputs

    text = record_outputs(x=1, y=2.5)
    assert isinstance(text, str)
    lm = LiteralMap()
    text_format.Parse(text, lm)
    assert "x" in lm.literals
    assert "y" in lm.literals


def test_record_outputs_multiple_types():
    from flyteidl2.core.literals_pb2 import LiteralMap
    from google.protobuf import text_format

    from flyteplugins.papermill.notebook import record_outputs

    text = record_outputs(count=10, name="hello", ratio=0.5)
    lm = LiteralMap()
    text_format.Parse(text, lm)
    assert set(lm.literals.keys()) == {"count", "name", "ratio"}


# ---------------------------------------------------------------------------
# forward() — integration: actually runs the test notebook via papermill
# ---------------------------------------------------------------------------


def test_forward_basic():
    task = make_task(
        notebook_path=NOTEBOOK_PATH,
        inputs={"x": int, "y": float},
        outputs={"result": int},
    )
    result = task.forward(x=3, y=1.5)
    assert result == 4  # int(3 + 1.5) == 4


def test_forward_no_outputs():
    task = make_task(
        notebook_path=NO_OUTPUTS_NOTEBOOK_PATH,
        inputs={"x": int},
        outputs=None,
    )
    result = task.forward(x=7)
    assert result is None


def test_forward_bool_list_dict():
    task = make_task(
        notebook_path=PRIMITIVES_NOTEBOOK_PATH,
        inputs={"flag": bool, "items": list, "metadata": dict},
        outputs={"item_count": int, "key_count": int},
    )
    item_count, key_count = task.forward(flag=True, items=[1, 2, 3], metadata={"a": 1, "b": 2})
    assert item_count == 3
    assert key_count == 2


def test_forward_primitive_types_example_notebook():
    """Integration test against the example notebook: bool/list/dict inputs, int/float/str outputs."""
    task = make_task(
        notebook_path=PRIMITIVE_TYPES_NOTEBOOK_PATH,
        inputs={"enabled": bool, "values": list, "options": dict},
        outputs={"count": int, "total": float, "label": str},
    )
    count, total, label = task.forward(
        enabled=True,
        values=[1, 5, 10, 15, 20],
        options={"threshold": 8, "label": "demo"},
    )
    assert count == 3  # values above threshold 8: [10, 15, 20]
    assert total == 51.0  # sum([1, 5, 10, 15, 20])
    assert label == "demo"


def test_forward_primitive_types_disabled():
    """When enabled=False the notebook skips computation and returns zeros."""
    task = make_task(
        notebook_path=PRIMITIVE_TYPES_NOTEBOOK_PATH,
        inputs={"enabled": bool, "values": list, "options": dict},
        outputs={"count": int, "total": float, "label": str},
    )
    count, total, label = task.forward(
        enabled=False,
        values=[1, 5, 10, 15, 20],
        options={"threshold": 8, "label": "skipped"},
    )
    assert count == 0
    assert total == 0.0
    assert label == "skipped"


def test_forward_output_notebook_path_written():
    task = make_task(
        notebook_path=NOTEBOOK_PATH,
        inputs={"x": int, "y": float},
        outputs={"result": int},
    )
    output_path = task.output_notebook_path
    if os.path.exists(output_path):
        os.unlink(output_path)
    task.forward(x=1, y=0.0)
    assert os.path.exists(output_path)


def test_forward_output_notebooks_returns_local_files():
    from flyte.io import File

    task = make_task(
        notebook_path=NOTEBOOK_PATH,
        inputs={"x": int, "y": float},
        outputs={"result": int},
        output_notebooks=True,
    )
    result = task.forward(x=2, y=0.0)
    # result is (result_val, source_file, executed_file)
    assert isinstance(result, tuple)
    assert len(result) == 3
    result_val, source_file, executed_file = result
    assert result_val == 2
    assert isinstance(source_file, File)
    assert isinstance(executed_file, File)


# ---------------------------------------------------------------------------
# forward() — partial output notebook written on failure
# ---------------------------------------------------------------------------

PARTIAL_FAILURE_NOTEBOOK_PATH = str(Path(__file__).parent.parent / "examples" / "notebooks" / "partial_failure.ipynb")


def test_forward_failure_writes_partial_output_notebook():
    """When a notebook raises, the output notebook is still written to disk.

    Papermill writes cells incrementally, so the partial notebook exists even
    after a failure. This is the same notebook that gets rendered into the
    report in remote execution.
    """

    task = make_task(
        notebook_path=PARTIAL_FAILURE_NOTEBOOK_PATH,
        inputs={"n": int},
        outputs={"result": int},
    )
    output_path = task.output_notebook_path
    if os.path.exists(output_path):
        os.unlink(output_path)

    with pytest.raises(Exception):
        task.forward(n=5)  # n=5 triggers the ValueError in step3

    # Partial output notebook must exist despite the failure
    assert os.path.exists(output_path), "Output notebook not written after failure"

    # The partial notebook should contain outputs from cells that ran before
    # the failure (step1 and step2) but not from step3.
    import nbformat as _nbf

    nb = _nbf.read(output_path, as_version=4)
    cell_ids_with_output = [c.get("id") for c in nb.cells if c.get("outputs")]
    assert "step1" in cell_ids_with_output
    assert "step2" in cell_ids_with_output


# ---------------------------------------------------------------------------
# Inline NotebookTask — defined inside a function
# ---------------------------------------------------------------------------


def test_forward_inline_definition():
    """NotebookTask defined inside a function works the same as module-level."""
    from flyteplugins.papermill.task import NotebookTask

    def make_inline_task():
        with patch("flyte.TaskEnvironment") as mock_env_cls:
            mock_env = MagicMock()
            mock_env_cls.return_value = mock_env
            mock_env._tasks = {}
            return NotebookTask(
                name="inline_task",
                notebook_path=NOTEBOOK_PATH,
                task_environment=mock_env,
                inputs={"x": int, "y": float},
                outputs={"result": int},
            )

    task = make_inline_task()
    result = task.forward(x=3, y=1.5)
    assert result == 4
