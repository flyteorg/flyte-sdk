from __future__ import annotations

import logging
import os, time
from typing import Optional

from ._tools import ipython_check, is_in_cluster

DEFAULT_LOG_LEVEL = logging.WARNING


def make_hyperlink(label: str, url: str):
    """
    Create a hyperlink in the terminal output.
    """
    BLUE = "\033[94m"
    RESET = "\033[0m"
    OSC8_BEGIN = f"\033]8;;{url}\033\\"
    OSC8_END = "\033]8;;\033\\"
    return f"{BLUE}{OSC8_BEGIN}{label}{RESET}{OSC8_END}"


def is_rich_logging_disabled() -> bool:
    """
    Check if rich logging is enabled
    """
    return os.environ.get("DISABLE_RICH_LOGGING") is not None


def get_env_log_level() -> int:
    return int(os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL))

def get_logger_default_loc() -> str:
    return os.getcwd()


def log_format_from_env() -> str:
    """
    Get the log format from the environment variable.
    """
    return os.environ.get("LOG_FORMAT", "json")


def get_rich_handler(log_level: int) -> Optional[logging.Handler]:
    """
    Upgrades the global loggers to use Rich logging.
    """
    if is_in_cluster():
        return None
    if not ipython_check() and is_rich_logging_disabled():
        return None

    import click
    from rich.console import Console
    from rich.logging import RichHandler

    try:
        width = os.get_terminal_size().columns
    except Exception as e:
        logger.debug(f"Failed to get terminal size: {e}")
        width = 160

    handler = RichHandler(
        tracebacks_suppress=[click],
        rich_tracebacks=True,
        omit_repeated_times=False,
        show_path=False,
        log_time_format="%H:%M:%S.%f",
        console=Console(width=width),
        level=log_level,
    )

    formatter = logging.Formatter(fmt="%(filename)s:%(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    return handler


def get_default_handler(log_level: int) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter(fmt="[%(name)s] %(message)s")
    if log_format_from_env() == "json":
        pass
        # formatter = jsonlogger.JsonFormatter(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    return handler

def get_file_handler(log_level: int,filename:str) -> logging.Handler:
    handler = logging.FileHandler(filename=filename, encoding='utf-8')
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    return handler


def initialize_logger(log_level: int = get_env_log_level(), enable_rich: bool = False,stream_logs: bool = True, log_file_dir: str = os.getcwd()):
    """
    Initializes the global loggers to the default configuration.
    """
    global logger  # noqa: PLW0603
    timestamp = time.time()
    logger = _create_logger(f"flyte_{timestamp}", log_level, enable_rich, stream_logs, log_file_dir)


def _create_logger(name: str, log_level: int = DEFAULT_LOG_LEVEL, enable_rich: bool = False, stream_logs: bool = True, log_file_dir: str = os.getcwd() ) -> logging.Logger:
    """
    Creates a logger with the given name and log level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = None
    logger.handlers = []
    if stream_logs:
        handler = get_rich_handler(log_level) if enable_rich else get_default_handler(log_level)
    else:
        handler = get_file_handler(log_level,f"{log_file_dir}{os.sep}{name}")
            
    logger.addHandler(handler)
    return logger


def log(fn=None, *, level=logging.DEBUG, entry=True, exit=True):
    """
    Decorator to log function calls.
    """

    def decorator(func):
        if logger.isEnabledFor(level):

            def wrapper(*args, **kwargs):
                if entry:
                    logger.log(level, f"[{func.__name__}] with args: {args} and kwargs: {kwargs}")
                try:
                    return func(*args, **kwargs)
                finally:
                    if exit:
                        logger.log(level, f"[{func.__name__}] completed")

            return wrapper
        return func

    if fn is None:
        return decorator
    return decorator(fn)


logger = _create_logger("flyte", get_env_log_level())
