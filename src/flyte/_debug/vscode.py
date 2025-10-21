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
from pathlib import Path
from typing import List

import aiofiles
import click
import httpx

from flyte import storage
from flyte._debug.constants import (
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
from flyte._internal.runtime.rusty import download_tgz
from flyte._logging import logger


async def download_file(url: str, target_dir: str) -> str:
    """
    Downloads a file from a given URL using HTTPX and saves it locally.

    Args:
        url (str): The URL of the file to download.
        target_dir (str): The directory where the file should be saved. Defaults to current directory.
    """
    try:
        filename = os.path.join(target_dir, os.path.basename(url))
        if url.startswith("http"):
            response = httpx.get(url, follow_redirects=True)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            async with aiofiles.open(filename, "wb") as f:
                await f.write(response.content)
        else:
            await storage.get(url, filename)
        logger.info(f"File '{filename}' downloaded successfully from '{url}'.")
        return filename

    except httpx.RequestError as e:
        raise RuntimeError(f"An error occurred while requesting '{url}': {e}")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def get_default_extensions() -> List[str]:
    extensions = os.getenv("_F_CS_E")
    if extensions is not None:
        return extensions.split(",")
    return DEFAULT_CODE_SERVER_EXTENSIONS


def get_code_server_info() -> str:
    """
    Returns the code server information based on the system's architecture.

    This function checks the system's architecture and returns the corresponding
    code server information from the provided dictionary. The function currently
    supports AMD64 and ARM64 architectures.

    Returns:
        str: The code server information corresponding to the system's architecture.

    Raises:
        ValueError: If the system's architecture is not AMD64 or ARM64.
    """
    code_server_path = os.getenv("_F_CS_RP")
    if code_server_path is not None:
        return code_server_path

    machine_info = platform.machine()
    logger.info(f"machine type: {machine_info}")
    code_server_info_dict = DEFAULT_CODE_SERVER_REMOTE_PATHS

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
        code_server_remote_path = get_code_server_info()
        code_server_tar_path = await download_file(code_server_remote_path, str(DOWNLOAD_DIR))

        # Extract the tarball
        with tarfile.open(code_server_tar_path, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR)

    if os.path.exists(DOWNLOAD_DIR):
        code_server_dir_name = os.path.basename(get_code_server_info()).removesuffix(".tar.gz")
        code_server_bin_dir = os.path.join(DOWNLOAD_DIR, code_server_dir_name, "bin")
        # Add the directory of code-server binary to $PATH
        os.environ["PATH"] = code_server_bin_dir + os.pathsep + os.environ["PATH"]

    # If the extension already exists in the container, skip downloading
    installed_extensions = get_installed_extensions()
    coros = []

    for extension in get_default_extensions():
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

    virtual_venv = os.getenv("VIRTUAL_ENV", str(Path(sys.executable).parent.parent))
    if virtual_venv is None:
        raise RuntimeError("VIRTUAL_ENV is not found in environment variables.")

    run_name = ctx.params["run_name"]
    name = ctx.params["name"]
    # TODO: Executor should pass correct name.
    if run_name.startswith("{{"):
        run_name = os.getenv("RUN_NAME", "")
    if name.startswith("{{"):
        name = os.getenv("ACTION_NAME", "")

    launch_json = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Interactive Debugging",
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
                    name,
                    "--run-name",
                    run_name,
                    "--project",
                    ctx.params["project"],
                    "--domain",
                    ctx.params["domain"],
                    "--org",
                    ctx.params["org"],
                    "--image-cache",
                    ctx.params["image_cache"],
                    "--debug",
                    "False",
                    "--interactive-mode",
                    "True",
                    "--tgz",
                    ctx.params["tgz"],
                    "--dest",
                    ctx.params["dest"],
                    "--resolver",
                    ctx.params["resolver"],
                    *ctx.params["resolver_args"],
                ],
            },
            {
                "name": "Resume Task",
                "type": "python",
                "request": "launch",
                "program": f"{virtual_venv}/bin/debug.py",
                "console": "integratedTerminal",
                "justMyCode": True,
                "args": ["resume", "--pid", str(pid)],
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
    if ctx.params["tgz"] is None:
        await download_vscode()
    else:
        await asyncio.gather(
            download_tgz(ctx.params["dest"], ctx.params["version"], ctx.params["tgz"]), download_vscode()
        )
    child_process = multiprocessing.Process(
        target=lambda cmd: asyncio.run(asyncio.run(execute_command(cmd))),
        kwargs={"cmd": f"code-server --bind-addr 0.0.0.0:6060 --disable-workspace-trust --auth none {os.getcwd()}"},
    )
    child_process.start()
    if child_process.pid is None:
        raise RuntimeError("Failed to start vscode server.")

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
