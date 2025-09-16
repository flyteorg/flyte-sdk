# test/cli/test_run.py
import json
import pathlib

import pytest
from click.testing import CliRunner

from flyte import Image
from flyte._initialize import _get_init_config
from flyte.cli._run import run

TEST_CODE_PATH = pathlib.Path(__file__).parent
RUN_TESTDATA = TEST_CODE_PATH / "run_testdata"
HELLO_WORLD_PY = RUN_TESTDATA / "hello_world.py"
COMPLEX_INPUTS_PY = RUN_TESTDATA / "complex_inputs.py"
PARQUET_FILE = RUN_TESTDATA / "df.parquet"


@pytest.fixture
def runner():
    return CliRunner()


def test_run_command(runner):
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Run a task from a python file" in result.output


def test_run_hello_world(runner):
    try:
        cmd = ["--local", str(HELLO_WORLD_PY), "say_hello", "--name", "World"]
        result = runner.invoke(run, cmd)
        assert result.exit_code == 0, result.output
    except ValueError as ve:
        if "I/O operation on closed file" in str(ve):
            # Can't figure out around this error
            # https://github.com/pallets/click/issues/824
            return
        else:
            raise ve


@pytest.mark.integration
def test_run_complex_inputs(runner):
    result = runner.invoke(
        run,
        [
            "--local",
            str(COMPLEX_INPUTS_PY),
            "print_all",
            "--a",
            "1",
            "--b",
            "Hello",
            "--c",
            "1.1",
            "--d",
            '{"i":1,"a":["h","e"]}',
            "--e",
            "[1,2,3]",
            "--f",
            '{"x":1.0, "y":2.0}',
            "--g",
            str(PARQUET_FILE),
            "--i",
            "2020-05-01",
            "--j",
            "P1D",
            "--k",
            "RED",
            "--h",
            "--m",
            '{"hello": "world"}',
            # "--n",
            # json.dumps([{"x": str(PARQUET_FILE)}]),
            # "--o",
            # json.dumps({"x": [str(PARQUET_FILE)]}),
            "--p",
            "Any",
            "--q",
            str(RUN_TESTDATA),
            "--r",
            json.dumps([{"i": 1, "a": ["h", "e"]}]),
            "--s",
            json.dumps({"x": {"i": 1, "a": ["h", "e"]}}),
            "--t",
            json.dumps({"i": [{"i": 1, "a": ["h", "e"]}]}),
        ],
    )
    assert result.exit_code == 0, result.output


def test_run_with_multiple_images_from_name(runner):
    """Test that multiple --image parameters work correctly with Image.from_name()"""
    from pathlib import Path

    import flyte

    # Initialize flyte to set up the config
    flyte.init(root_dir=Path.cwd())

    # Test with multiple images
    cmd = [
        "--local",
        "--image",
        "custom=my-custom-registry/custom-image:v2.0",
        "--image",
        "my-default-registry/default-image:v3.0",  # will assign name "default" to it
        str(HELLO_WORLD_PY),
        "say_hello",
        "--name",
        "World",
    ]
    result = runner.invoke(run, cmd)
    assert result.exit_code == 0, result.output

    # Verify that both images were registered in the config
    cfg = _get_init_config()
    assert cfg is not None, "Config should be initialized"
    assert cfg.images is not None, "Images dict should exist"
    assert "custom" in cfg.images
    assert "default" in cfg.images

    # Test that Image.from_name() returns the correct images
    custom_image = Image.from_name("custom")
    default_image = Image.from_name("default")

    assert custom_image.base_image == "my-custom-registry/custom-image:v2.0"
    assert default_image.base_image == "my-default-registry/default-image:v3.0"
