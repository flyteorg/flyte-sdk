"""Subprocess-based helpers for building/running Flyte UV scripts.

The functions in this module are designed to be invoked from inside a Flyte
task (see :mod:`flyte.ai.mcp.tasks`) so that the MCP server's
``build_uv_script_image_remote`` and ``run_uv_script_remote`` tools can
dispatch a user-supplied script to a clean execution environment.

Each helper:

1. Writes the supplied script to a temporary file inside the task's working
   directory.
2. Executes it with the current interpreter (the script template's ``--build``
   flag selects build-vs-run behavior).
3. Captures stdout/stderr, trims long output, and returns a :class:`RunResult`.

The script template (see ``flyte.ai.mcp._flyte_mcp_app.UV_SCRIPT_FORMAT``)
expects an ``FLYTE_PASSTHROUGH_API_KEY`` env var to be present so it can
authenticate back to the Flyte cluster. The helper task is responsible for
mounting a Flyte secret under that name; this module simply inherits
``os.environ`` into the subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass

from flyte._utils.asyncify import run_sync_with_loop


@dataclass
class RunResult:
    """Captured result of a UV-script build or run subprocess.

    :param stdout: Tail of the captured stdout (line-truncated on success).
    :param stderr: Full stderr on failure, otherwise a line-truncated tail.
    :param returncode: Subprocess exit code (``0`` on success).
    :param next_step: Human-readable hint for the calling MCP tool / agent.
    """

    stdout: str
    stderr: str
    returncode: int
    next_step: str


def _python_interpreter() -> str:
    """Return the Python interpreter the helper task should invoke.

    Honors ``FLYTE_MCP_HELPER_PYTHON`` (useful when the helper image keeps the
    Flyte venv at a non-default location) and otherwise falls back to the
    current process' interpreter.
    """
    return os.environ.get("FLYTE_MCP_HELPER_PYTHON") or sys.executable


def _write_script_file(filename: str, script: str) -> None:
    """Synchronous helper that writes the script to disk.

    Kept separate so the async callers can dispatch the blocking ``open`` call
    via :func:`run_sync_with_loop`.
    """
    with open(filename, "w") as f:
        f.write(script)


async def build_script_image_(script: str, tail: int = 200) -> RunResult:
    """Execute the UV script with ``--build`` to trigger a remote image build.

    :param script: The full UV-script source to write to disk and execute.
    :param tail: Number of trailing stdout/stderr lines to retain on success.
    :return: A :class:`RunResult` describing the subprocess outcome.
    """
    filename = f"__build_script_{uuid.uuid4().hex[:8]}__.py"
    await run_sync_with_loop(_write_script_file, filename, script)

    try:
        proc = await run_sync_with_loop(
            subprocess.run,
            [_python_interpreter(), filename, "--build"],
            capture_output=True,
            env=os.environ,
            text=True,
        )
        full_stderr = proc.stderr if proc.returncode != 0 else "\n".join(proc.stderr.splitlines()[-tail:])
        return RunResult(
            stdout="\n".join(proc.stdout.splitlines()[-tail:]),
            stderr=full_stderr,
            returncode=proc.returncode,
            next_step=(
                "if the image build is successful, run the script with the "
                "run_uv_script_remote tool. if the image build fails, check the "
                "build run details and debug the issue."
            ),
        )
    finally:
        if os.path.exists(filename):
            os.remove(filename)


async def run_script_remote_(script: str, tail: int = 200) -> RunResult:
    """Execute the UV script (no ``--build``) to dispatch a remote Flyte run.

    :param script: The full UV-script source to write to disk and execute.
    :param tail: Number of trailing stdout/stderr lines to retain on success.
    :return: A :class:`RunResult` describing the subprocess outcome.
    """
    filename = f"__run_script_{uuid.uuid4().hex[:16]}__.py"
    await run_sync_with_loop(_write_script_file, filename, script)

    try:
        proc = await run_sync_with_loop(
            subprocess.run,
            [_python_interpreter(), filename],
            capture_output=True,
            env=os.environ,
            text=True,
        )
        full_stderr = proc.stderr if proc.returncode != 0 else "\n".join(proc.stderr.splitlines()[-tail:])
        return RunResult(
            stdout="\n".join(proc.stdout.splitlines()[-tail:]),
            stderr=full_stderr,
            returncode=proc.returncode,
            next_step=(
                "if the script run is successful, use the get_run_io tool to "
                "get the inputs and outputs of the run. if the script run "
                "fails, check the run details and debug the issue."
            ),
        )
    finally:
        if os.path.exists(filename):
            os.remove(filename)
