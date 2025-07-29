# test/cli/test_run.py
import json
import pathlib

import pytest
from click.testing import CliRunner

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
