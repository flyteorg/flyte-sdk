"""Shared base for the per-provider event constants."""

from __future__ import annotations

import enum


class EventType(str, enum.Enum):
    """Base for event constants: a real `str`, usable anywhere a pattern is."""

    # Without these, Python 3.11+ renders members as "Class.MEMBER" in f-strings
    # and str(), rather than the wire value handlers match on — which would leak
    # enum names into the dashboard and /api/status.
    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]
