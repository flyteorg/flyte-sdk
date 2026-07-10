import asyncio
import os
from typing import Dict, Optional

from flyte._debug.constants import EXIT_CODE_SUCCESS
from flyte._logging import logger


async def execute_command(cmd: str, env: Optional[Dict[str, str]] = None):
    """
    Execute a command in the shell.

    Args:
        cmd (str): The command to execute.
        env (Optional[Dict[str, str]]): Environment variables to set for the subprocess. These are merged on top of
            the current process environment, so callers can override specific variables (e.g. ``PORT``) without
            dropping the rest of the inherited environment.

    Raises:
        RuntimeError: If the command exits with a non-zero return code.
    """
    subprocess_env = None
    if env is not None:
        subprocess_env = {**os.environ, **env}

    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=subprocess_env
    )
    logger.info(f"cmd: {cmd}")
    stdout, stderr = await process.communicate()
    if process.returncode != EXIT_CODE_SUCCESS:
        raise RuntimeError(f"Command {cmd} failed with error: {stderr!r}")
    logger.info(f"stdout: {stdout!r}")
    logger.info(f"stderr: {stderr!r}")
