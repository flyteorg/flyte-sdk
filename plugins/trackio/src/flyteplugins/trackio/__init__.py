"""
Flyte Trackio plugin.

Provides seamless Trackio experiment tracking for Flyte tasks through the
`@trackio_init` decorator.

Basic usage:

    from flyteplugins.trackio import (
        trackio_init,
        get_trackio_run,
    )

    @trackio_init(project="my-project")
    @env.task
    async def train():
        run = get_trackio_run()
        run.log({"loss": 0.123})
        return run.id

Configuration can also be provided via `trackio_config()`:

    r = flyte.with_runcontext(
        custom_context=trackio_config(
            project="my-project",
            tags=["baseline"],
        )
    ).run(train)
"""

from __future__ import annotations

import flyte
import trackio

from ._context import (
    get_trackio_context,
    get_trackio_run,
    trackio_config,
)
from ._decorator import (
    trackio_init,
)
from ._link import Trackio

__version__ = "0.1.0"

__all__ = [
    "Trackio",
    "trackio_init",
    "trackio_config",
    "get_trackio_context",
    "get_trackio_run",
]
