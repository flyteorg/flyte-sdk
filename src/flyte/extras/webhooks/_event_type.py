"""Base for the typed event constants each provider plugin ships."""

from __future__ import annotations

import enum


class EventType(str, enum.Enum):
    """Base for event constants: a real `str`, usable anywhere a pattern is.

    Subclass this in a provider plugin's `events` module, one class per event
    type, with `ANY` as the bare type when the product splits type and action:

    ```python
    class PullRequest(EventType):
        ANY = "pull_request"
        OPENED = "pull_request.opened"
    ```
    """

    # Without these, Python 3.11+ renders members as "Class.MEMBER" in f-strings
    # and str(), rather than the wire value handlers match on — which would leak
    # enum names into the dashboard and /api/status.
    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]
