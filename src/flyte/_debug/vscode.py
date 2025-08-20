import asyncio
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from typing import List

import click
import fsspec

from flyte._debug.constants import (
    DEFAULT_CODE_SERVER_DIR_NAMES,
    DEFAULT_CODE_SERVER_EXTENSIONS,
    DEFAULT_CODE_SERVER_REMOTE_PATHS,
    DOWNLOAD_DIR,
    EXECUTABLE_NAME,
    EXIT_CODE_SUCCESS,
    HEARTBEAT_PATH,
    MAX_IDLE_SECONDS,
)
from flyte._debug.utils import (
    execute_command,
)
from flyte._logging import logger


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


def prepare_launch_json(ctx: click.Context, pid: int):
    """
    Generate the launch.json and settings.json for users to easily launch interactive debugging and task resumption.
    """

    # ctx = internal_ctx()
    # task_function_source_dir = os.path.dirname(ctx.data.task_context.data[TASK_FUNCTION_SOURCE_PATH])
    virtual_venv = os.getenv("VIRTUAL_ENV", "/opt/venv")
    # task_module_name, task_name = task_func.__module__, task_func.__name__
    print("ctx", ctx.params)
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
                    *ctx.params["resolver-args"],
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
        kwargs={"cmd": f"code-server --bind-addr 0.0.0.0:8080 --disable-workspace-trust --auth none {os.getcwd()}"},
    )
    child_process.start()
    prepare_launch_json(ctx, child_process.pid)

    start_time = time.time()
    check_interval = 60  # Interval for heartbeat checking in seconds
    last_heartbeat_check = time.time() - check_interval

    def terminate_process():
        if child_process.is_alive():
            child_process.terminate()
        child_process.join()

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
            if delta > MAX_IDLE_SECONDS:
                logger.info(f"VSCode server is idle for more than {MAX_IDLE_SECONDS} seconds. Terminating...")
                terminate_process()
                sys.exit()

        await asyncio.sleep(1)

    logger.info("User has resumed the task.")
    terminate_process()
    return
