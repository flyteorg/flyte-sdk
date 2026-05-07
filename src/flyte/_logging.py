from __future__ import annotations

import contextvars
import logging
import os
from datetime import datetime
from typing import Any, Literal, Optional

import flyte

from ._tools import ipython_check

LogFormat = Literal["console", "json"]

# Registry of contextvars to stamp onto every LogRecord.
# Map of attribute name -> ContextVar. Populated via register_log_context().
_LOG_CONTEXT_VARS: dict[str, contextvars.ContextVar] = {}


def register_log_context(name: str, var: contextvars.ContextVar[Any]) -> None:
    """
    Register a contextvar to be stamped on every LogRecord as `record.<name>`.

    The value is pulled at record-creation time, so it reflects the context the
    log call was issued in (not the context the handler runs in). JSON output
    will include the attribute when its value is not None; console output can
    reference it via `%(<name>)s` in a Formatter string.
    """
    _LOG_CONTEXT_VARS[name] = var


def unregister_log_context(name: str) -> None:
    """Remove a previously-registered contextvar from the log record factory."""
    _LOG_CONTEXT_VARS.pop(name, None)


_orig_record_factory = logging.getLogRecordFactory()


def _flyte_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    record = _orig_record_factory(*args, **kwargs)

    for attr, var in _LOG_CONTEXT_VARS.items():
        if not hasattr(record, attr):
            setattr(record, attr, var.get(None))

    # Stamp the active flyte action context, if any. Imported lazily because
    # this factory runs on every record, including during flyte's own import.
    try:
        from flyte._context import ctx as _flyte_ctx

        c = _flyte_ctx()
    except Exception:
        c = None
    if c is not None:
        record.run_name = c.action.run_name
        record.action_name = c.action.name
    else:
        record.run_name = None
        record.action_name = None

    record.is_flyte_internal = record.name == "flyte" or (
        record.name.startswith("flyte.") and not record.name.startswith("flyte.user")
    )
    return record


logging.setLogRecordFactory(_flyte_record_factory)


_LOG_LEVEL_MAP = {
    "critical": logging.CRITICAL,  # 50
    "error": logging.ERROR,  # 40
    "warning": logging.WARNING,  # 30
    "warn": logging.WARNING,  # 30
    "info": logging.INFO,  # 20
    "debug": logging.DEBUG,  # 10
}
DEFAULT_LOG_LEVEL = logging.WARNING
DEFAULT_USER_LOG_LEVEL = logging.INFO


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
    value = os.getenv("LOG_LEVEL")
    if value is None:
        return DEFAULT_LOG_LEVEL
    # Case 1: numeric value ("10", "20", "5", etc.)
    if value.isdigit():
        return int(value)

    # Case 2: named log level ("info", "debug", ...)
    if value.lower() in _LOG_LEVEL_MAP:
        return _LOG_LEVEL_MAP[value.lower()]

    return DEFAULT_LOG_LEVEL


def get_env_user_log_level() -> int:
    value = os.getenv("USER_LOG_LEVEL")
    if value is None:
        return DEFAULT_USER_LOG_LEVEL
    if value.isdigit():
        return int(value)
    if value.lower() in _LOG_LEVEL_MAP:
        return _LOG_LEVEL_MAP[value.lower()]
    return DEFAULT_USER_LOG_LEVEL


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

    formatter = ContextFormatter(fmt="%(filename)s:%(lineno)d - %(message)s", internal_prefix=True)
    handler.setFormatter(formatter)
    return handler


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for each log record.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

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

        # Add any user-registered contextvars stamped by the record factory.
        for attr in _LOG_CONTEXT_VARS:
            if attr in log_data:
                continue
            val = getattr(record, attr, None)
            if val is not None:
                log_data[attr] = val

        # Add metric fields if present
        if getattr(record, "metric_type", None):
            log_data["metric_type"] = record.metric_type  # type: ignore[attr-defined]
            log_data["metric_name"] = record.metric_name  # type: ignore[attr-defined]
            log_data["duration_seconds"] = record.duration_seconds  # type: ignore[attr-defined]

        # Add exception info if present
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def initialize_logger(
    log_level: int | None = None,
    log_format: LogFormat | None = None,
    enable_rich: bool = False,
    reset_root_logger: bool = False,
    user_log_level: int | None = None,
):
    """
    Initializes the global loggers to the default configuration.
    When enable_rich=True, upgrades to Rich handler for local CLI usage.
    """
    global logger  # noqa: PLW0603

    if log_level is None:
        log_level = get_env_log_level()
    if log_format is None:
        log_format = log_format_from_env()

    flyte_logger = logging.getLogger("flyte")
    flyte_logger.handlers.clear()

    # Determine log format (JSON takes precedence over Rich)
    use_json = log_format == "json"
    use_rich = enable_rich and not use_json

    reset_root_logger = reset_root_logger or os.environ.get("FLYTE_RESET_ROOT_LOGGER") == "1"
    if reset_root_logger:
        _setup_root_logger(use_json=use_json, use_rich=use_rich, log_level=log_level)

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
        flyte_handler.setFormatter(ContextFormatter(fmt="%(message)s", internal_prefix=True))

    flyte_logger.addHandler(flyte_handler)
    flyte_logger.setLevel(log_level)
    flyte_logger.propagate = False  # Prevent double logging

    logger = flyte_logger

    # Reconfigure the user-facing logger with the same format, but its own level
    global user_logger  # noqa: PLW0603
    user_log_level = user_log_level if user_log_level is not None else get_env_user_log_level()
    user_flyte_logger = logging.getLogger("flyte.user")
    user_flyte_logger.handlers.clear()

    user_handler: logging.Handler
    if use_json:
        user_handler = logging.StreamHandler()
        user_handler.setLevel(user_log_level)
        user_handler.setFormatter(JSONFormatter())
    elif use_rich:
        rich_handler = get_rich_handler(user_log_level)
        user_handler = rich_handler if rich_handler is not None else logging.StreamHandler()
        user_handler.setLevel(user_log_level)
        if not rich_handler:
            user_handler.setFormatter(ContextFormatter(fmt="%(message)s"))
    else:
        user_handler = logging.StreamHandler()
        user_handler.setLevel(user_log_level)
        user_handler.setFormatter(ContextFormatter(fmt="%(message)s"))

    user_flyte_logger.addHandler(user_handler)
    user_flyte_logger.setLevel(user_log_level)
    user_flyte_logger.propagate = False

    user_logger = user_flyte_logger


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


class ContextFormatter(logging.Formatter):
    """
    Console formatter that prefixes records with action context and an optional
    [flyte] marker, both pulled from attributes stamped by _flyte_record_factory.
    Does not mutate record state, so the same record can be formatted by
    multiple handlers without compounding prefixes.
    """

    def __init__(self, fmt: str = "%(message)s", *, internal_prefix: bool = False, **kwargs: Any) -> None:
        super().__init__(fmt=fmt, **kwargs)
        self._internal_prefix = internal_prefix

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        parts: list[str] = []
        run_name = getattr(record, "run_name", None)
        action_name = getattr(record, "action_name", None)
        if run_name and action_name:
            parts.append(f"[{run_name}][{action_name}]")
        if self._internal_prefix and getattr(record, "is_flyte_internal", False):
            parts.append("[flyte]")
        if not parts:
            return base
        return f"{' '.join(parts)} {base}"


def _setup_root_logger(use_json: bool, use_rich: bool, log_level: int):
    """
    Wipe all handlers from the root logger and reconfigure. This ensures
    both user/library logging and Flyte internal logging get context information and look the same.
    """
    root = logging.getLogger()
    root.handlers.clear()  # Remove any existing handlers to prevent double logging

    root_handler: logging.Handler | None = None
    if use_json:
        root_handler = logging.StreamHandler()
        root_handler.setFormatter(JSONFormatter())
    elif use_rich:
        root_handler = get_rich_handler(log_level)

    # get_rich_handler can return None in some environments
    if not root_handler:
        root_handler = logging.StreamHandler()
        root_handler.setFormatter(ContextFormatter(fmt="%(message)s"))

    root_handler.setLevel(log_level)
    root.addHandler(root_handler)
    root.setLevel(log_level)


def _create_user_logger() -> logging.Logger:
    """
    Create the user-facing logger. Defaults to INFO so user logs are visible by default.
    No [flyte] prefix on user messages.
    """
    user_flyte_logger = logging.getLogger("flyte.user")
    user_log_level = get_env_user_log_level()
    user_flyte_logger.setLevel(user_log_level)

    handler = logging.StreamHandler()
    handler.setLevel(user_log_level)
    handler.setFormatter(ContextFormatter(fmt="%(message)s"))

    user_flyte_logger.propagate = False
    user_flyte_logger.addHandler(handler)

    return user_flyte_logger


def _create_flyte_logger() -> logging.Logger:
    """
    Create the internal Flyte logger with [flyte] prefix.
    """
    flyte_logger = logging.getLogger("flyte")
    flyte_logger.setLevel(get_env_log_level())

    handler = logging.StreamHandler()
    handler.setLevel(get_env_log_level())
    handler.setFormatter(ContextFormatter(fmt="%(message)s", internal_prefix=True))

    # Prevent propagation to root to avoid double logging
    flyte_logger.propagate = False
    flyte_logger.addHandler(handler)

    return flyte_logger


# Create the Flyte internal logger
logger = _create_flyte_logger()

# Create the user-facing logger
user_logger = _create_user_logger()
