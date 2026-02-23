"""VSCode App Environment for running code-server as a Flyte App."""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import rich.repr

import flyte.app
from flyte._debug.constants import (
    HEARTBEAT_PATH,
    MAX_IDLE_SECONDS,
)
from flyte._debug.utils import execute_command
from flyte._debug.vscode import download_vscode
from flyte._logging import logger
from flyte.app._types import Link, Port


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class VSCodeAppEnvironment(flyte.app.AppEnvironment):
    """
    An AppEnvironment that runs code-server (VS Code in the browser).

    This environment downloads and runs code-server, providing a web-based
    VS Code IDE that can be accessed through the browser.

    :param idle_timeout_seconds: Maximum idle time in seconds before the server
        terminates. Defaults to MAX_IDLE_SECONDS (10 hours).
    :param bind_addr: Address to bind the code-server to. Defaults to "0.0.0.0".
    :param disable_workspace_trust: Whether to disable workspace trust prompts.
        Defaults to True.
    :param auth: Authentication mode for code-server. Defaults to "none".
    :param working_dir: Working directory for code-server. Defaults to current
        working directory.
    """

    type: str = "VSCode"
    port: int | Port = 6060
    idle_timeout_seconds: int = MAX_IDLE_SECONDS
    bind_addr: str = "0.0.0.0"
    disable_workspace_trust: bool = True
    auth: str = "none"
    working_dir: str | None = None
    _caller_frame: inspect.FrameInfo | None = None

    def __post_init__(self):
        super().__post_init__()

        self.links = [
            Link(path="/", title="VS Code IDE", is_relative=True),
            *self.links,
        ]
        self._server = self._vscode_server

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            if caller_frame and caller_frame.f_back:
                self._caller_frame = inspect.getframeinfo(caller_frame.f_back)

    def _build_code_server_command(self) -> str:
        """Build the code-server command with all configured options."""
        port = self.port.port if isinstance(self.port, Port) else self.port
        working_dir = self.working_dir or os.getcwd()

        cmd_parts = [
            "code-server",
            f"--bind-addr {self.bind_addr}:{port}",
            f"--idle-timeout-seconds {self.idle_timeout_seconds}",
            f"--auth {self.auth}",
        ]

        if self.disable_workspace_trust:
            cmd_parts.append("--disable-workspace-trust")

        cmd_parts.append(working_dir)

        return " ".join(cmd_parts)

    async def _vscode_server(self):
        """
        Start the VS Code server (code-server) and monitor its lifecycle.

        This method:
        1. Downloads code-server if not already installed
        2. Starts code-server in a subprocess
        3. Monitors the heartbeat file to detect idle timeouts
        4. Terminates the server when idle for too long
        """
        await download_vscode()

        cmd = self._build_code_server_command()
        child_process = multiprocessing.Process(
            target=lambda c: asyncio.run(execute_command(c)),
            kwargs={"c": cmd},
        )
        child_process.start()

        if child_process.pid is None:
            raise RuntimeError("Failed to start VS Code server.")

        logger.info(f"VS Code server started with PID {child_process.pid}")

        start_time = time.time()
        check_interval = 60

        def terminate_process():
            if child_process.is_alive():
                child_process.terminate()
            child_process.join()

        logger.info("VS Code server is running. Waiting for connections...")

        while child_process.is_alive():
            current_time = time.time()
            await asyncio.sleep(check_interval)

            if not os.path.exists(HEARTBEAT_PATH):
                delta = current_time - start_time
                logger.info(f"Code server has not been connected for {delta:.0f} seconds.")
            else:
                delta = current_time - os.path.getmtime(HEARTBEAT_PATH)
                logger.info(f"Last activity on code server was {delta:.0f} seconds ago.")

            if delta > self.idle_timeout_seconds:
                logger.info(f"VS Code server idle for more than {self.idle_timeout_seconds} seconds. Terminating...")
                terminate_process()
                sys.exit()

        logger.info("VS Code server process has ended.")
        terminate_process()

    def container_args(self, serialize_context) -> List[str]:
        """Return container arguments for the VS Code environment."""
        if self.args is None:
            return []
        return super().container_args(serialize_context)


vscode_env = VSCodeAppEnvironment(
    name="vscode-dev",
    requires_auth=False,
    resources=flyte.Resources(cpu="2", memory="1Gi"),
)


if __name__ == "__main__":
    flyte.init_from_config()
    remote_app = flyte.serve(vscode_env)
    print(f"Remote app URL: {remote_app.url}")
