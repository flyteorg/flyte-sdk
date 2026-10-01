from __future__ import annotations

import functools
import importlib.metadata
from typing import cast


@functools.cache
def _installed_entry_points() -> importlib.metadata.EntryPoints:
    # Python 3.10 and 3.11 return a `SelectableGroups` here, which offers the same `select`.
    return cast("importlib.metadata.EntryPoints", importlib.metadata.entry_points())


def entry_points(*, group: str) -> importlib.metadata.EntryPoints:
    """Entry points registered under `group`.

    `importlib.metadata.entry_points` reads the metadata of every installed distribution on each
    call, which takes several milliseconds and grows with the size of the environment. Flyte looks
    up a handful of plugin groups during startup, so the scan is done once per process and shared.
    """
    return _installed_entry_points().select(group=group)
