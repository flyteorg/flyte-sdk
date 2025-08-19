import asyncio
import inspect
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable, List, Optional

import click
import fsspec

from flyte._context import internal_ctx
from flyte._debug.constants import EXIT_CODE_SUCCESS, MAX_IDLE_SECONDS
from flyte._debug.utils import (
    execute_command,
)
from flyte._debug.vscode_lib.config import VscodeConfig
from flyte._debug.vscode_lib.constants import (
    DOWNLOAD_DIR,
    EXECUTABLE_NAME,
    HEARTBEAT_PATH,
    INTERACTIVE_DEBUGGING_FILE_NAME,
    RESUME_TASK_FILE_NAME,
    TASK_FUNCTION_SOURCE_PATH, DEFAULT_CODE_SERVER_REMOTE_PATHS, DEFAULT_CODE_SERVER_DIR_NAMES,
    DEFAULT_CODE_SERVER_EXTENSIONS,
)
from flyte._logging import logger
from flyte._task import ClassDecorator, TaskTemplate
from flyte._tools import is_in_cluster
from flyte._utils.module_loader import _load_module_from_file


async def exit_handler(
    child_process: multiprocessing.Process,
    task_function,
    args,
    kwargs,
    max_idle_seconds: int = 180,
    post_execute: Optional[Callable] = None,
):
    """
    1. Check the modified time of ~/.local/share/code-server/heartbeat.
       If it is older than max_idle_second seconds, kill the container.
       Otherwise, check again every HEARTBEAT_CHECK_SECONDS.
    2. Wait for user to resume the task. If resume_task is set, terminate the VSCode server,
     reload the task function, and run it with the input of the task.

    Args:
        child_process (multiprocessing.Process, optional): The process to be terminated.
        max_idle_seconds (int, optional): The duration in seconds to live after no activity detected.
        post_execute (function, optional): The function to be executed before the vscode is self-terminated.
    """

    def terminate_process():
        if post_execute is not None:
            post_execute()
            logger.info("Post execute function executed successfully!")
        if child_process.is_alive():
            child_process.terminate()
        child_process.join()

    start_time = time.time()
    check_interval = 60  # Interval for heartbeat checking in seconds
    last_heartbeat_check = time.time() - check_interval

    logger.info("waiting for task to resume...")
    while child_process.is_alive():
        current_time = time.time()
        if current_time - last_heartbeat_check >= check_interval:
            last_heartbeat_check = current_time
            if not os.path.exists(HEARTBEAT_PATH):
                delta = current_time - start_time
                logger.info(f"Code server has not been connected since {delta} seconds ago.")
                logger.info("Please open the browser to connect to the running server.")
            else:
                delta = current_time - os.path.getmtime(HEARTBEAT_PATH)
                logger.info(f"The latest activity on code server is {delta} seconds ago.")

            # If the time from last connection is longer than max idle seconds, terminate the vscode server.
            if delta > max_idle_seconds:
                logger.info(f"VSCode server is idle for more than {max_idle_seconds} seconds. Terminating...")
                terminate_process()
                sys.exit()

        await asyncio.sleep(1)

    logger.info("User has resumed the task.")
    terminate_process()

    # Reload the task function since it may be modified.
    ctx = internal_ctx()
    if ctx.data.task_context is None:
        raise RuntimeError("Task context was not provided.")
    task_function_source_path = ctx.data.task_context.data[TASK_FUNCTION_SOURCE_PATH]
    _, module = _load_module_from_file(Path(task_function_source_path))
    task = getattr(
        module,
        task_function.__name__,
    )

    # Get the actual function from the task.
    task_function = task.func.__wrapped__
    return await task_function(*args, **kwargs)


async def download_file(url, target_dir: str = "."):
    """
    Download a file from a given URL using fsspec.

    Args:
        url (str): The URL of the file to download.
        target_dir (str, optional): The directory where the file should be saved. Defaults to current directory.

    Returns:
        str: The path to the downloaded file.
    """
    if not url.startswith("http"):
        raise ValueError(f"URL {url} is not valid. Only http/https is supported.")

    # Derive the local filename from the URL
    local_file_name = os.path.join(target_dir, os.path.basename(url))

    fs = fsspec.filesystem("http")

    # Use fsspec to get the remote file and save it locally
    logger.info(f"Downloading {url}... to {os.path.abspath(local_file_name)}")
    fs.get(url, local_file_name)
    logger.info("File downloaded successfully!")

    return local_file_name


def get_code_server_info(code_server_info_dict: dict) -> str:
    """
    Returns the code server information based on the system's architecture.

    This function checks the system's architecture and returns the corresponding
    code server information from the provided dictionary. The function currently
    supports AMD64 and ARM64 architectures.

    Args:
        code_server_info_dict (dict): A dictionary containing code server information.
            The keys should be the architecture type ('amd64' or 'arm64') and the values
            should be the corresponding code server information.

    Returns:
        str: The code server information corresponding to the system's architecture.

    Raises:
        ValueError: If the system's architecture is not AMD64 or ARM64.
    """
    machine_info = platform.machine()
    logger.info(f"machine type: {machine_info}")

    if "aarch64" == machine_info:
        return code_server_info_dict["arm64"]
    elif "x86_64" == machine_info:
        return code_server_info_dict["amd64"]
    else:
        raise ValueError(
            "Automatic download is only supported on AMD64 and ARM64 architectures."
            " If you are using a different architecture, please visit the code-server official website to"
            " manually download the appropriate version for your image."
        )


def get_installed_extensions() -> List[str]:
    """
    Get the list of installed extensions.

    Returns:
        List[str]: The list of installed extensions.
    """
    installed_extensions = subprocess.run(
        ["code-server", "--list-extensions"], check=False, capture_output=True, text=True
    )
    if installed_extensions.returncode != EXIT_CODE_SUCCESS:
        logger.info(f"Command code-server --list-extensions failed with error: {installed_extensions.stderr}")
        return []

    return installed_extensions.stdout.splitlines()


def is_extension_installed(extension: str, installed_extensions: List[str]) -> bool:
    return any(installed_extension in extension for installed_extension in installed_extensions)


async def download_vscode():
    """
    Download vscode server and extension from remote to local and add the directory of binary executable to $PATH.
    """
    # If the code server already exists in the container, skip downloading
    executable_path = shutil.which(EXECUTABLE_NAME)
    if executable_path is not None or os.path.exists(DOWNLOAD_DIR):
        logger.info(f"Code server binary already exists at {executable_path}")
        logger.info("Skipping downloading code server...")
    else:
        logger.info("Code server is not in $PATH, start downloading code server...")
        # Create DOWNLOAD_DIR if not exist
        logger.info(f"DOWNLOAD_DIR: {DOWNLOAD_DIR}")
        os.makedirs(DOWNLOAD_DIR)

        logger.info(f"Start downloading files to {DOWNLOAD_DIR}")
        # Download remote file to local
        code_server_remote_path = get_code_server_info(DEFAULT_CODE_SERVER_REMOTE_PATHS)
        code_server_tar_path = await download_file(code_server_remote_path, str(DOWNLOAD_DIR))

        # Extract the tarball
        with tarfile.open(code_server_tar_path, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR)

    if os.path.exists(DOWNLOAD_DIR):
        code_server_dir_name = get_code_server_info(DEFAULT_CODE_SERVER_DIR_NAMES)
        code_server_bin_dir = os.path.join(DOWNLOAD_DIR, code_server_dir_name, "bin")
        # Add the directory of code-server binary to $PATH
        os.environ["PATH"] = code_server_bin_dir + os.pathsep + os.environ["PATH"]

    # If the extension already exists in the container, skip downloading
    installed_extensions = get_installed_extensions()
    coros = []

    for extension in DEFAULT_CODE_SERVER_EXTENSIONS:
        if not is_extension_installed(extension, installed_extensions):
            coros.append(download_file(extension, str(DOWNLOAD_DIR)))
    extension_paths = await asyncio.gather(*coros)

    coros = []
    for p in extension_paths:
        logger.info(f"Execute extension installation command to install extension {p}")
        coros.append(execute_command(f"code-server --install-extension {p}"))

    await asyncio.gather(*coros)


def prepare_interactive_python(task_function):
    """
    1. Copy the original task file to the context working directory.
     This ensures that the inputs.pb can be loaded, as loading requires the original task interface.
       By doing so, even if users change the task interface in their code,
        we can use the copied task file to load the inputs as native Python objects.
    2. Generate a Python script and a launch.json for users to debug interactively.

    Args:
        task_function (function): User's task function.
    """

    ctx = internal_ctx()
    task_function_source_path = ctx.data.task_context.data[TASK_FUNCTION_SOURCE_PATH]
    context_working_dir = os.getcwd()

    # Copy the user's Python file to the working directory.
    shutil.copy(
        task_function_source_path,
        os.path.join(context_working_dir, os.path.basename(task_function_source_path)),
    )

    # Generate a Python script
    task_module_name, task_name = task_function.__module__, task_function.__name__
    python_script = f"""# This file is auto-generated by flytekit

from {task_module_name} import {task_name}
from flytekit.interactive import get_task_inputs

if __name__ == "__main__":
    inputs = get_task_inputs(
        task_module_name="{task_module_name.split(".")[-1]}",
        task_name="{task_name}",
        context_working_dir="{context_working_dir}",
    )
    # You can modify the inputs! Ex: inputs['a'] = 5
    print({task_name}(**inputs))
"""

    task_function_source_dir = os.path.dirname(task_function_source_path)
    with open(os.path.join(task_function_source_dir, INTERACTIVE_DEBUGGING_FILE_NAME), "w") as file:
        file.write(python_script)


def prepare_resume_task_python(pid: int):
    """
    Generate a Python script for users to resume the task.
    """

    python_script = f"""import os
import signal

if __name__ == "__main__":
    print("Terminating server and resuming task.")
    answer = input("This operation will kill the server. All unsaved data will be lost, and you will no longer be able to connect to it. Do you really want to terminate? (Y/N): ").strip().upper()
    if answer == 'Y':
        os.kill({pid}, signal.SIGTERM)
        print(f"The server has been terminated and the task has been resumed.")
    else:
        print("Operation canceled.")
"""  # noqa: E501
    ctx = internal_ctx()
    if ctx.data.task_context is None:
        raise RuntimeError("Task context was not provided.")
    task_function_source_dir = os.path.dirname(ctx.data.task_context.data[TASK_FUNCTION_SOURCE_PATH])
    with open(os.path.join(task_function_source_dir, RESUME_TASK_FILE_NAME), "w") as file:
        file.write(python_script)


def prepare_launch_json(ctx: click.Context, pid: int):
    """
    Generate the launch.json and settings.json for users to easily launch interactive debugging and task resumption.
    """

    # ctx = internal_ctx()
    # task_function_source_dir = os.path.dirname(ctx.data.task_context.data[TASK_FUNCTION_SOURCE_PATH])
    virtual_venv = os.getenv("VIRTUAL_ENV", "/opt/venv")
    # task_module_name, task_name = task_func.__module__, task_func.__name__

    launch_json = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Resume Task v2",
                "type": "python",
                "request": "resume",
                "program": f"{virtual_venv}/bin/debug.py",
                "console": "integratedTerminal",
                "justMyCode": True,
                "args": ["debug", "--pid", str(pid)],
            },
            {
                "name": "Interactive Debugging v2",
                "type": "python",
                "request": "launch",
                "program": f"{virtual_venv}/bin/runtime.py",
                "console": "integratedTerminal",
                "justMyCode": True,
                "args": [
                    "a0",
                    "--inputs",
                    ctx.params["inputs"],
                    "--outputs-path",
                    ctx.params["outputs_path"],
                    "--version",
                    ctx.params["version"],
                    "--run-base-dir",
                    ctx.params["run_base_dir"],
                    "--name",
                    ctx.params["name"],
                    "--run-name",
                    ctx.params["run_name"],
                    "--project",
                    ctx.params["project"],
                    "--domain",
                    ctx.params["domain"],
                    "--org",
                    ctx.params["org"],
                    "--image-cache",
                    ctx.params["image_cache"],
                    "--tgz",
                    ctx.params["tgz"],
                    # "--pkl",
                    # ctx.data.task_context.pkl,
                    "--dest",
                    ctx.params["dest"],
                    "--resolver",
                    ctx.params["resolver"],
                    *ctx.params["resolver-args"]
                ],
            },
        ],
    }

    vscode_directory = os.path.join(os.getcwd(), ".vscode")
    if not os.path.exists(vscode_directory):
        os.makedirs(vscode_directory)

    with open(os.path.join(vscode_directory, "launch.json"), "w") as file:
        json.dump(launch_json, file, indent=4)

    settings_json = {"python.defaultInterpreterPath": sys.executable}
    with open(os.path.join(vscode_directory, "settings.json"), "w") as file:
        json.dump(settings_json, file, indent=4)


async def _start_vscode_server(ctx: click.Context):
    await download_vscode()
    child_process = multiprocessing.Process(
        target=lambda cmd: asyncio.run(asyncio.run(execute_command(cmd))),
        kwargs={
            "cmd": f"code-server --bind-addr 0.0.0.0:8080"
                   f" --disable-workspace-trust --auth none {os.getcwd()}"
        },
    )
    child_process.start()
    prepare_launch_json(ctx, child_process.pid)

    start_time = time.time()
    check_interval = 60  # Interval for heartbeat checking in seconds
    last_heartbeat_check = time.time() - check_interval

    logger.info("waiting for task to resume...")
    while child_process.is_alive():
        current_time = time.time()
        if current_time - last_heartbeat_check >= check_interval:
            last_heartbeat_check = current_time
            if not os.path.exists(HEARTBEAT_PATH):
                delta = current_time - start_time
                logger.info(f"Code server has not been connected since {delta} seconds ago.")
                logger.info("Please open the browser to connect to the running server.")
            else:
                delta = current_time - os.path.getmtime(HEARTBEAT_PATH)
                logger.info(f"The latest activity on code server is {delta} seconds ago.")

            # If the time from last connection is longer than max idle seconds, terminate the vscode server.
            if delta > max_idle_seconds:
                logger.info(f"VSCode server is idle for more than {max_idle_seconds} seconds. Terminating...")
                terminate_process()
                sys.exit()

        await asyncio.sleep(1)
