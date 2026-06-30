import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import cast

from flyte._serve import ServeMode

_raw_data_path_var: ContextVar[str | None] = ContextVar("raw_data_path", default=None)
_RAW_DATA_PATH_ENV = "_FLYTE_APP_RAW_DATA_PATH"


@dataclass(frozen=True)
class AppContext:
    mode: ServeMode = "remote"
    project: str = ""
    raw_data_path: str = ""
    domain: str = ""


def ctx() -> AppContext:
    """
    Returns the current app context.
    Returns: AppContext
    """
    from flyte._serve import ServeMode

    mode = os.getenv("_RUN_MODE", "remote")
    project = os.getenv("FLYTE_INTERNAL_EXECUTION_PROJECT", "")
    domain = os.getenv("FLYTE_INTERNAL_EXECUTION_DOMAIN", "")
    raw_data_path = _raw_data_path_var.get() or os.getenv(_RAW_DATA_PATH_ENV, "") or ""
    return AppContext(
        mode=cast(ServeMode, mode),
        project=project,
        domain=domain,
        raw_data_path=raw_data_path,
    )


def set_raw_data_path(raw_data_path: str | None) -> None:
    """Set the raw data path in the current context and as env var for thread propagation."""
    value = raw_data_path or ""
    _raw_data_path_var.set(value)
    os.environ[_RAW_DATA_PATH_ENV] = value
