import os
from dataclasses import dataclass
from typing import cast

from flyte._serve import ServeMode


@dataclass(frozen=True)
class AppContext:
    mode: ServeMode = "remote"
    project: str = ""
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
    return AppContext(
        mode=cast(ServeMode, mode),
        project=project,
        domain=domain,
    )
