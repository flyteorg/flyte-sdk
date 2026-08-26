"""The routing policy: pick a profile for a run, locally, before anything is submitted."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ._tags import _digest, make_run_name
from ._types import RoutingContext, RoutingDecision

__all__ = ["route", "select_profile"]


def _describe(inputs: Mapping[str, Any]) -> str:
    """A stable description of a run's arguments, for hashing.

    `repr` so unhashable and unorderable arguments still produce a key. An argument whose `repr`
    is not stable (an object at a default memory address) will not route consistently -- a real
    deployment should hash the fields it cares about, not the whole argument set.
    """
    return "\x00".join(f"{k}={v!r}" for k, v in sorted(inputs.items()))


def select_profile(profiles: Sequence[str], key: str) -> str:
    """Rendezvous ("highest random weight") hashing of `key` onto `profiles`.

    Preferred over `hash(key) % len(profiles)`: modulo reshuffles nearly every key when the
    profile set changes, whereas rendezvous only moves the keys that belonged to the profile that
    came or went. That matters here because a run's cluster determines where its cached outputs
    and its data live -- reshuffling on an unrelated config edit throws both away.
    """
    return max(sorted(profiles), key=lambda p: _digest("rendezvous", p, key))


def route(ctx: RoutingContext) -> Optional[RoutingDecision]:
    """Route a run to a profile by hashing its task and arguments.

    Declines when no profiles are declared, and when `--profile` already pinned one -- an explicit
    choice is not a policy's to override.

    The same task on the same arguments always lands on the same cluster, so repeat work returns
    to the cluster holding its cache and its data. That is locality, not just load spreading.

    One policy, not the policy: `ctx` also carries `resources`, and nothing stops a fork consulting
    the local user or an external service.
    """
    if not ctx.profiles or ctx.active_profile:
        return None

    key = "\x00".join([ctx.task_name or "", ctx.project or "", ctx.domain or "", _describe(ctx.inputs)])
    profile = select_profile(ctx.profiles, key)

    return RoutingDecision(
        profile=profile,
        # Tag the name so the read path can find this run later with no stored state and no
        # search. Skipped when the caller named the run themselves.
        run_name=None if ctx.run_name else make_run_name(profile),
        labels={"routed-by": "consistent-hash", "routed-to": profile},
    )
