import asyncio
import os
import pathlib
import subprocess
from typing import List, Optional, Tuple

import pytest

import flyte
import flyte.errors
from flyte._bin.runtime import ACTION_NAME, DOMAIN_NAME, ORG_NAME, PROJECT_NAME, RUN_NAME
from flyte._internal.runtime import io, taskrunner
from flyte._internal.runtime.convert import Error, Outputs, convert_error_to_native, convert_outputs_to_native
from flyte._logging import logger
from flyte.models import ActionID

env = flyte.TaskEnvironment("test")


@env.task
async def task1(v: str) -> str:
    return f"Hello, world {v}!"


@env.task
async def task2(v: str) -> str:
    raise ValueError(f"Hello, world {v}!")


async def run_subprocess(args: List[str], working_dir: pathlib.Path) -> None:
    """
    Run the subprocess with the given arguments and working directory.
    """
    logger.info("Task Invocation ==============")
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=working_dir,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            ACTION_NAME: "test",
            PROJECT_NAME: "test",
            ORG_NAME: "testorg",
            DOMAIN_NAME: "test",
            RUN_NAME: "test_run",
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    logger.info("Task Invocation Complete ==============")
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, args, output=stdout, stderr=stderr)


async def load_task_results(working_dir: pathlib.Path) -> Tuple[Optional[Outputs], Optional[Error]]:
    """
    Load task outputs or errors from the working directory.
    """
    outputs_path = io.outputs_path(str(working_dir.absolute()))
    err_path = io.error_path(str(working_dir.absolute()))
    outputs = None
    err = None

    if pathlib.Path(outputs_path).exists():
        outputs = await io.load_outputs(outputs_path)
    elif pathlib.Path(err_path).exists():
        err = await io.load_error(err_path)
    else:
        pytest.fail("Failed to load outputs and errors")

    return outputs, err


async def execute_task(
    args: List[str], working_dir: pathlib.Path, metadata_dir: pathlib.Path
) -> Tuple[Optional[Outputs], Optional[Error]]:
    """
    Execute a task and return its outputs or errors.
    """
    print(
        f"Executing task with args: {args} in working directory: {working_dir} and metadata directory: {metadata_dir}"
    )
    await run_subprocess(args, working_dir)
    return await load_task_results(metadata_dir)


async def setup_dry_run(task, tmp_path, pkl=False):
    """
    Helper function to perform a dry run and return task_spec, inputs, code_bundle, and replaced_args.
    """
    result = await flyte.with_runcontext(
        mode="remote",
        dry_run=True,
        copy_bundle_to=tmp_path,
        interactive_mode=pkl,
        disable_run_cache=True,
    ).run.aio(task, "test")
    task_spec = result.task_spec
    inputs = result.inputs  # type: ignore
    code_bundle = result.code_bundle  # type: ignore

    # Validate task spec, inputs, and code bundle
    assert task_spec is not None
    template = task_spec.task_template
    assert template is not None
    assert template.container is not None
    assert template.container.args is not None
    args = template.container.args

    assert inputs is not None

    files = list(tmp_path.iterdir())
    # Include code bundle file and SQLite DB file for local cache
    assert len(files) == 2

    if not pkl:
        assert code_bundle.tgz, "Code bundle should be a tgz file"
        gz_files = [f for f in files if f.suffix == ".gz"]
        assert str(gz_files[0]) == code_bundle.tgz
    else:
        assert code_bundle.pkl, "Code bundle should be a pkl file"
        gz_files = [f for f in files if f.suffix == ".gz"]
        assert str(gz_files[0]) == code_bundle.pkl

    replaced_args = taskrunner.replace_task_cli(args, inputs, tmp_path, ActionID.create_random())
    assert replaced_args is not None

    return task_spec, inputs, code_bundle, replaced_args


async def assert_task_outputs(outputs, expected_value):
    """
    Helper function to validate task outputs.
    """
    assert outputs is not None
    assert outputs.proto_outputs is not None
    v = await convert_outputs_to_native(task1.native_interface, outputs)
    assert v is not None
    assert v == expected_value


async def assert_task_error(err, expected_error_message):
    """
    Helper function to validate task errors.
    """
    assert err is not None
    exc = convert_error_to_native(err)
    assert exc is not None
    assert isinstance(exc, flyte.errors.RuntimeUserError)
    assert str(exc) == expected_error_message


@pytest.mark.asyncio
async def test_launch_remote_success(tmp_path):
    await flyte.init.aio(api_key="")
    _, _, _, replaced_args = await setup_dry_run(task1, tmp_path)
    working_dir = tmp_path / "working_dir"
    working_dir.mkdir(parents=True, exist_ok=True)
    outputs, err = await execute_task(replaced_args, working_dir, tmp_path)
    assert err is None
    await assert_task_outputs(outputs, "Hello, world test!")


@pytest.mark.asyncio
async def test_launch_remote_err(tmp_path):
    await flyte.init.aio(api_key="")
    _, _, _, replaced_args = await setup_dry_run(task2, tmp_path)
    working_dir = tmp_path / "working_dir"
    working_dir.mkdir(parents=True, exist_ok=True)
    outputs, err = await execute_task(replaced_args, working_dir, tmp_path)
    assert outputs is None
    await assert_task_error(err, "Hello, world test!")


@pytest.mark.asyncio
async def test_launch_remote_success_interactive(tmp_path):
    await flyte.init.aio(api_key="")
    _, _, _, replaced_args = await setup_dry_run(task1, tmp_path, pkl=True)
    working_dir = tmp_path / "working_dir"
    working_dir.mkdir(parents=True, exist_ok=True)
    outputs, err = await execute_task(replaced_args, working_dir, tmp_path)
    assert err is None
    await assert_task_outputs(outputs, "Hello, world test!")


@pytest.mark.asyncio
async def test_launch_remote_err_interactive(tmp_path):
    await flyte.init.aio(api_key="")
    _, _, _, replaced_args = await setup_dry_run(task2, tmp_path, pkl=True)
    working_dir = tmp_path / "working_dir"
    working_dir.mkdir(parents=True, exist_ok=True)
    outputs, err = await execute_task(replaced_args, working_dir, tmp_path)
    assert outputs is None
    await assert_task_error(err, "Hello, world test!")


@pytest.mark.asyncio
async def test_success_interactive(tmp_path):
    await flyte.init.aio(api_key="")
    _, _, _, replaced_args = await setup_dry_run(task1, tmp_path, pkl=True)
    working_dir = tmp_path / "working_dir"
    working_dir.mkdir(parents=True, exist_ok=True)
    outputs, err = await execute_task(replaced_args, working_dir, tmp_path)
    assert err is None
    await assert_task_outputs(outputs, "Hello, world test!")
