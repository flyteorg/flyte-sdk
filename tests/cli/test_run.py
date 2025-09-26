# test/cli/test_run.py
import asyncio
import json
import pathlib
from pathlib import Path

import pytest
from click.testing import CliRunner

import flyte
from flyte import Image
from flyte._deploy import DeploymentPlan, _build_images
from flyte._initialize import _get_init_config
from flyte._task_environment import TaskEnvironment
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


def test_run_with_multiple_images_and_build_images_cache(runner):
    """Test that multiple --image parameters work correctly with Image.from_name() and _build_images uses config URIs"""

    custom_env_name = "env_with_custom_img"
    default_env_name = "env_with_default_img"
    default_image_uri = "my-default-registry/default-image:v3.0"
    custom_image_uri = "my-custom-registry/custom-image:v2.0"

    flyte.init(root_dir=Path.cwd())

    # Test with multiple images
    cmd = [
        "--local",
        "--image",
        f"custom={custom_image_uri}",
        "--image",
        default_image_uri,  # will assign name "default" to it
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

    # Test that Image.from_name() set the image name
    custom_image = Image.from_name("custom")
    assert custom_image.name == "custom"

    # Test that _build_images uses the config URIs instead of building
    custom_env = TaskEnvironment(name=custom_env_name, image=custom_image)
    default_env = TaskEnvironment(name=default_env_name)
    deployment_plan = DeploymentPlan(envs={custom_env_name: custom_env, default_env_name: default_env})

    image_cache = asyncio.run(_build_images(deployment_plan))

    assert image_cache.image_lookup[custom_env_name] == custom_image_uri
    # use default image set in CLI
    assert image_cache.image_lookup[default_env_name] == default_image_uri


def test_build_images_image_name_not_found_error(runner):
    """Test if _build_images raises error when image name is not found in config"""

    flyte.init(root_dir=Path.cwd())

    cmd = [
        "--local",
        "--image",
        "custom=some-registry/existing-image:v1.0",
        str(HELLO_WORLD_PY),
        "say_hello",
        "--name",
        "World",
    ]
    runner.invoke(run, cmd)

    # Create a TaskEnvironment with an image name that doesn't exist in config
    invalid_image = Image.from_name("invalid")
    env_name = "test_env"
    task_env = TaskEnvironment(name=env_name, image=invalid_image)
    deployment_plan = DeploymentPlan(envs={env_name: task_env})

    # Check if _build_images raises ValueError for missing image name
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(_build_images(deployment_plan))

    error_msg = str(exc_info.value)
    assert "Image name 'invalid' not found in config" in error_msg
    assert "Available: ['custom']" in error_msg
