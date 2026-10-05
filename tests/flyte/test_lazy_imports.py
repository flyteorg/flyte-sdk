"""Starting a task must not pay for deploy/build machinery or the dataframe
engine. Each check runs in a fresh interpreter, since this test session has
long since imported everything."""

import subprocess
import sys

import pytest

HEAVY = [
    "flyte._deploy",
    "flyte._build",
    "flyte.io._dataframe",
    "mashumaro.jsonschema",
    "markdown_it",
]


def _loaded_after(code: str) -> set:
    out = subprocess.run(
        [sys.executable, "-c", f"{code}\nimport sys\nprint('\\n'.join(sys.modules))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return set(out.split())


@pytest.mark.parametrize(
    "code",
    [
        "import flyte",
        "import flyte.io",
        "from flyte.io import File, Dir",
        "import flyte.types",
    ],
)
def test_light_imports_skip_heavy_modules(code):
    loaded = _loaded_after(code)
    assert not [m for m in HEAVY if m in loaded]


@pytest.mark.parametrize(
    "code,forbidden",
    [
        # Lineage authoring is opt-in: `import flyte` loads none of it.
        (
            "import flyte",
            ["flyte.artifacts", "flyte.artifacts._handle", "flyte.artifacts._lineage", "flyte._materialize"],
        ),
        # Handles are importable from a module that only declares them, without the deploy-time
        # extraction, the planner delegator, or a dataframe engine.
        (
            "import flyte.artifacts as a\na.Artifact('x', partitions={'date': a.Daily})",
            ["flyte.artifacts._lineage", "flyte._materialize", "flyte._deploy", "pandas", "pyarrow", "jsonschema"],
        ),
    ],
)
def test_lineage_imports_stay_light(code, forbidden):
    loaded = _loaded_after(code)
    assert not [m for m in forbidden if m in loaded]


def test_lineage_lazy_names_resolve():
    loaded = _loaded_after(
        "import flyte\n"
        "assert flyte.TimeRange('2026-08-01', '2026-08-02').is_absolute\n"
        "assert callable(flyte.materialize)"
    )
    assert "flyte._materialize" in loaded and "flyte.artifacts._handle" in loaded


def test_lazy_names_still_resolve():
    loaded = _loaded_after(
        "import flyte, flyte.io\n"
        "assert flyte.deploy and flyte.build and flyte.build_images and flyte.ImageBuild\n"
        "from flyte.io import DataFrame, PARQUET\n"
        "assert DataFrame is flyte.io.DataFrame and PARQUET\n"
        "from flyte.types._pydantic_schema_plugin import PydanticSchemaPlugin\n"
        "assert PydanticSchemaPlugin is not None"
    )
    assert "flyte._deploy" in loaded and "flyte.io._dataframe" in loaded


def test_unknown_names_still_raise():
    import flyte
    import flyte.io

    with pytest.raises(AttributeError):
        flyte.not_a_thing
    with pytest.raises(AttributeError):
        flyte.io.not_a_thing


def test_plugin_dataframe_types_still_resolve():
    """polars & co. register through their "flyte.plugins.types" entry point,
    which imports the dataframe engine itself — no eager import needed."""
    pytest.importorskip("polars")
    pytest.importorskip("flyteplugins.polars")
    loaded = _loaded_after(
        "import polars as pl\n"
        "from flyte.types import TypeEngine\n"
        "assert TypeEngine.to_literal_type(pl.DataFrame).HasField('structured_dataset_type')\n"
        "assert TypeEngine.to_literal_type(pl.LazyFrame).HasField('structured_dataset_type')"
    )
    assert "flyte.io._dataframe" in loaded


def test_polars_task_end_to_end_without_importing_the_plugin(tmp_path):
    """`@env.task async def t(foo: pl.DataFrame)` with only `import polars` and
    `import flyte`: the interface serializes as a structured dataset and a value
    round-trips through the transformer as a task pod converts it, with
    flyteplugins-polars reached only through its entry point."""
    pytest.importorskip("polars")
    pytest.importorskip("flyteplugins.polars")
    (tmp_path / "pl_task_mod.py").write_text(
        "import polars as pl\n"
        "import flyte\n"
        "env = flyte.TaskEnvironment('x')\n"
        "@env.task\n"
        "async def my_task(foo: pl.DataFrame) -> pl.DataFrame:\n"
        "    return foo\n"
    )
    loaded = _loaded_after(
        "import asyncio, pathlib, sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import polars as pl\n"
        "from pl_task_mod import my_task\n"
        "assert 'flyteplugins.polars' not in sys.modules\n"
        "from flyte._internal.runtime.task_serde import translate_task_to_wire\n"
        "from flyte.models import SerializationContext\n"
        "spec = translate_task_to_wire(my_task, SerializationContext(version='v', project='p', domain='d', "
        f"org='o', root_dir=pathlib.Path({str(tmp_path)!r})))\n"
        "lt = {v.key: v.value for v in spec.task_template.interface.inputs.variables}['foo'].type\n"
        "assert lt.WhichOneof('type') == 'structured_dataset_type'\n"
        "from flyte._context import RawDataPath, internal_ctx\n"
        "from flyte.types import TypeEngine\n"
        "df = pl.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})\n"
        "async def go():\n"
        "    with internal_ctx().new_raw_data_path(raw_data_path=RawDataPath.from_local_folder()):\n"
        "        lit = await TypeEngine.to_literal(df, pl.DataFrame, lt)\n"
        "        return await TypeEngine.to_python_value(lit, pl.DataFrame)\n"
        "assert asyncio.run(go()).equals(df)"
    )
    assert "flyteplugins.polars.df_transformer" in loaded


@pytest.mark.parametrize(
    ("lib", "annotation", "make", "same"),
    [
        ("pandas as pd", "pd.DataFrame", "pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})", "back.equals(val)"),
        ("pyarrow as pa", "pa.Table", "pa.table({'a': [1, 2], 'b': ['x', 'y']})", "back.equals(val)"),
    ],
)
def test_pandas_and_arrow_tasks_end_to_end(tmp_path, lib, annotation, make, same):
    """A task annotated with a pandas DataFrame or an Arrow Table, with only that
    library and flyte imported: the interface is a structured dataset and a
    value round-trips the way a task pod converts it."""
    pytest.importorskip(lib.split()[0])
    (tmp_path / "df_task_mod.py").write_text(
        f"import {lib}\n"
        "import flyte\n"
        "env = flyte.TaskEnvironment('x')\n"
        "@env.task\n"
        f"async def t(foo: {annotation}) -> {annotation}:\n"
        "    return foo\n"
    )
    _loaded_after(
        "import asyncio, pathlib, sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        f"import {lib}\n"
        "from df_task_mod import t\n"
        "from flyte._internal.runtime.task_serde import translate_task_to_wire\n"
        "from flyte.models import SerializationContext\n"
        "spec = translate_task_to_wire(t, SerializationContext(version='v', project='p', domain='d', "
        f"org='o', root_dir=pathlib.Path({str(tmp_path)!r})))\n"
        "lt = {v.key: v.value for v in spec.task_template.interface.inputs.variables}['foo'].type\n"
        "assert lt.WhichOneof('type') == 'structured_dataset_type'\n"
        "from flyte._context import RawDataPath, internal_ctx\n"
        "from flyte.types import TypeEngine\n"
        f"val = {make}\n"
        "async def go():\n"
        "    with internal_ctx().new_raw_data_path(raw_data_path=RawDataPath.from_local_folder()):\n"
        f"        lit = await TypeEngine.to_literal(val, {annotation}, lt)\n"
        f"        return await TypeEngine.to_python_value(lit, {annotation})\n"
        "back = asyncio.run(go())\n"
        f"assert {same}"
    )
