import pathlib

import pytest
from click.testing import CliRunner

from flyte.cli._build import build

TEST_CODE_PATH = pathlib.Path(__file__).parent
HELLO_WORLD_PY = TEST_CODE_PATH / "run_testdata" / "hello_world.py"


@pytest.fixture
def runner():
    return CliRunner()


def test_build_exposes_copy_style_and_root_dir(runner):
    """flyte build must accept --copy-style and --root-dir so CI can split
    image builds from deploys without hash drift."""
    result = runner.invoke(build, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--copy-style" in result.output
    assert "--root-dir" in result.output


def test_build_rejects_invalid_copy_style(runner):
    """--copy-style is a Click.Choice constrained to the CopyFiles values;
    unknown values must fail loudly (not silently build with the wrong hash)."""
    result = runner.invoke(
        build,
        ["--copy-style", "bogus", str(HELLO_WORLD_PY), "hello_world"],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output.lower() or "invalid" in result.output.lower()
