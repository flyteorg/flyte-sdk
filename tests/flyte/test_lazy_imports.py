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


def test_lazy_names_still_resolve():
    loaded = _loaded_after(
        "import flyte, flyte.io\n"
        "assert flyte.deploy and flyte.build and flyte.build_images and flyte.ImageBuild\n"
        "from flyte.io import DataFrame, PARQUET\n"
        "assert DataFrame is flyte.io.DataFrame and PARQUET\n"
        "from flyte.types._type_engine import PydanticSchemaPlugin\n"
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
