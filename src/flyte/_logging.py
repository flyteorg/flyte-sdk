from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Literal, Optional

import flyte

from ._tools import ipython_check

LogFormat = Literal["console", "json"]

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


def log_format_from_env() -> LogFormat:
    """
    Get the log format from the environment variable.
    """
    format_str = os.environ.get("LOG_FORMAT", "console")
    if format_str not in ("console", "json"):
        return "console"
    return format_str  # type: ignore[return-value]


def _get_console():
    """
    Get the console.
    """
    from rich.console import Console

    try:
        width = os.get_terminal_size().columns
    except Exception as e:
        logger.debug(f"Failed to get terminal size: {e}")
        width = 160

    return Console(width=width)


def get_rich_handler(log_level: int) -> Optional[logging.Handler]:
    """
    Upgrades the global loggers to use Rich logging.
    """
    ctx = flyte.ctx()
    if ctx and ctx.is_in_cluster():
        return None
    if not ipython_check() and is_rich_logging_disabled():
        return None

    import click
    from rich.highlighter import NullHighlighter
    from rich.logging import RichHandler

    handler = RichHandler(
        tracebacks_suppress=[click],
        rich_tracebacks=False,
        omit_repeated_times=False,
        show_path=False,
        log_time_format="%H:%M:%S.%f",
        console=_get_console(),
        level=log_level,
        highlighter=NullHighlighter(),
        markup=True,
    )

    formatter = logging.Formatter(fmt="%(filename)s:%(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    return handler


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for each log record.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
        }

        # Add context fields if present
        if getattr(record, "run_name", None):
            log_data["run_name"] = record.run_name  # type: ignore[attr-defined]
        if getattr(record, "action_name", None):
            log_data["action_name"] = record.action_name  # type: ignore[attr-defined]
        if getattr(record, "is_flyte_internal", False):
            log_data["is_flyte_internal"] = True

        # Add exception info if present
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def initialize_logger(log_level: int | None = None, log_format: LogFormat | None = None, enable_rich: bool = False):
    """
    Initializes the global loggers to the default configuration.
    When enable_rich=True, upgrades to Rich handler for local CLI usage.
    """
    global logger  # noqa: PLW0603

    if log_level is None:
        log_level = get_env_log_level()
    if log_format is None:
        log_format = log_format_from_env()

    # Clear existing handlers to reconfigure
    root = logging.getLogger()
    root.handlers.clear()

    flyte_logger = logging.getLogger("flyte")
    flyte_logger.handlers.clear()

    # Determine log format (JSON takes precedence over Rich)
    use_json = log_format == "json"
    use_rich = enable_rich and not use_json

    # Set up root logger handler
    root_handler: logging.Handler | None = None
    if use_json:
        root_handler = logging.StreamHandler()
        root_handler.setFormatter(JSONFormatter())
    elif use_rich:
        root_handler = get_rich_handler(log_level)

    if root_handler is None:
        root_handler = logging.StreamHandler()

    # Add context filter to root handler for all logging
    root_handler.addFilter(ContextFilter())
    root_handler.setLevel(logging.DEBUG)
    root.addHandler(root_handler)

    # Set up Flyte logger handler
    flyte_handler: logging.Handler | None = None
    if use_json:
        flyte_handler = logging.StreamHandler()
        flyte_handler.setLevel(log_level)
        flyte_handler.setFormatter(JSONFormatter())
    elif use_rich:
        flyte_handler = get_rich_handler(log_level)

    if flyte_handler is None:
        flyte_handler = logging.StreamHandler()
        flyte_handler.setLevel(log_level)
        formatter = logging.Formatter(fmt="%(message)s")
        flyte_handler.setFormatter(formatter)

    # Add both filters to Flyte handler
    flyte_handler.addFilter(FlyteInternalFilter())
    flyte_handler.addFilter(ContextFilter())

    flyte_logger.addHandler(flyte_handler)
    flyte_logger.setLevel(log_level)
    flyte_logger.propagate = False  # Prevent double logging

    logger = flyte_logger


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


class ContextFilter(logging.Filter):
    """
    A logging filter that adds the current action's run name and name to all log records.
    Applied globally to capture context for both user and Flyte internal logging.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        from flyte._context import ctx

        c = ctx()
        if c:
            action = c.action
            # Add as attributes for structured logging (JSON)
            record.run_name = action.run_name
            record.action_name = action.name
            # Also modify message for console/Rich output
            record.msg = f"[{action.run_name}][{action.name}] {record.msg}"
        else:
            record.run_name = None
            record.action_name = None
        return True


class FlyteInternalFilter(logging.Filter):
    """
    A logging filter that adds [flyte] prefix to internal Flyte logging only.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        is_internal = record.name.startswith("flyte")
        # Add as attribute for structured logging (JSON)
        record.is_flyte_internal = is_internal
        # Also modify message for console/Rich output
        if is_internal:
            record.msg = f"[flyte] {record.msg}"
        return True


def _setup_root_logger():
    """
    Configure the root logger to capture all logging with context information.
    This ensures both user code and Flyte internal logging get the context.
    """
    root = logging.getLogger()
    root.handlers.clear()  # Remove any existing handlers to prevent double logging

    # Create a basic handler for the root logger
    handler = logging.StreamHandler()
    # Add context filter to ALL logging
    handler.addFilter(ContextFilter())
    handler.setLevel(logging.DEBUG)

    # Simple formatter since filters handle prefixes
    root.addHandler(handler)


def _create_flyte_logger() -> logging.Logger:
    """
    Create the internal Flyte logger with [flyte] prefix.
    """
    flyte_logger = logging.getLogger("flyte")
    flyte_logger.setLevel(get_env_log_level())

    # Add a handler specifically for flyte logging with the prefix filter
    handler = logging.StreamHandler()
    handler.setLevel(get_env_log_level())
    handler.addFilter(FlyteInternalFilter())
    handler.addFilter(ContextFilter())

    formatter = logging.Formatter(fmt="%(message)s")
    handler.setFormatter(formatter)

    # Prevent propagation to root to avoid double logging
    flyte_logger.propagate = False
    flyte_logger.addHandler(handler)

    return flyte_logger


# Initialize root logger for global context
_setup_root_logger()

# Create the Flyte internal logger
logger = _create_flyte_logger()
