import asyncio

from flyte._debug.constants import EXIT_CODE_SUCCESS
from flyte._logging import logger


async def execute_command(cmd: str):
    """
    Execute a command in the shell.
    """
    process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    logger.info(f"cmd: {cmd}")
    stdout, stderr = await process.communicate()
    if process.returncode != EXIT_CODE_SUCCESS:
        raise RuntimeError(f"Command {cmd} failed with error: {stderr!r}")
    logger.info(f"stdout: {stdout!r}")
    logger.info(f"stderr: {stderr!r}")
