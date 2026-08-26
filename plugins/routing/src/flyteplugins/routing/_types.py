"""The shapes this plugin's policy is written against.

Defined here, not imported from the SDK: routing is entirely a plugin concern. A fork owns these
shapes and can change them freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

__all__ = ["RoutingContext", "RoutingDecision"]


@dataclass(frozen=True)
class RoutingContext:
    """What the policy decides on: the task, its arguments, and the session's config.

    Attributes:
        profiles: Profile names declared by the config file.
        active_profile: The profile already in effect (`--profile` / `FLYTE_PROFILE`), or None.
        resources: Resources the task requests, falling back to its environment's.
        inputs: The run's arguments by name.
        run_name: A run name the caller asked for explicitly, or None.
    """

    profiles: Tuple[str, ...] = ()
    active_profile: Optional[str] = None
    project: Optional[str] = None
    domain: Optional[str] = None
    task_name: Optional[str] = None
    resources: Optional[Any] = None
    inputs: Mapping[str, Any] = field(default_factory=dict)
    run_name: Optional[str] = None


@dataclass(frozen=True)
class RoutingDecision:
    """What the policy returns.

    Attributes:
        profile: Profile to submit under. None keeps the profile already in effect.
        run_name: Run name to submit under, replacing the control plane's. This is what makes the
            decision recoverable -- the read path decodes it. Ignored if the caller named the run.
        labels: Why this route was chosen. Caller labels win on conflict.
    """

    profile: Optional[str] = None
    run_name: Optional[str] = None
    labels: Optional[Mapping[str, str]] = None
