import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AppContext:
    mode: Literal["local", "remote"] = "remote"
    project: str = ""
    domain: str = ""


def ctx() -> AppContext:
    """
    Returns the current app context.
    Returns: AppContext
    """
    mode = os.getenv("_RUN_MODE", "remote")
    project = os.getenv("FLYTE_INTERNAL_EXECUTION_PROJECT", "")
    domain = os.getenv("FLYTE_INTERNAL_EXECUTION_DOMAIN", "")
    return AppContext(
        mode=mode,
        project=project,
        domain=domain,
    )
