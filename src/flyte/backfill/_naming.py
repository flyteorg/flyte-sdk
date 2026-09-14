"""Deterministic run names for scheduled trigger fires.

A scheduled fire's run name is a hash of the trigger's identity plus the exact
second it was scheduled for. That is what makes fires idempotent: re-firing the
same slot produces the same name, and the control plane returns the existing run
instead of creating a second one.

Backfill reuses that scheme so a backfilled slot collides with the slot the
scheduler already ran -- which is exactly the behaviour we want by default, and
what ``force`` opts out of by salting the name.

The construction here mirrors the scheduler's ``runName`` byte for byte. Any
divergence silently breaks de-duplication (the backfill would create a parallel
run rather than recognising the existing one), so ``tests/backfill/test_naming.py``
pins the expected hashes.
"""

from __future__ import annotations

from datetime import datetime

__all__ = ["candidate_run_names", "fnv1_64", "scheduled_run_name"]

_FNV1_64_OFFSET = 0xCBF29CE484222325
_FNV1_64_PRIME = 0x100000001B3
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF

# The scheduler emits names prefixed "r". For automation-sourced runs routed to
# the actions engine the control plane swaps that prefix to "u", keeping the hash.
# A client cannot see the routing decision, so both spellings are candidates when
# checking whether a slot has already run.
_SCHEDULER_PREFIX = "r"
_ACTIONS_PREFIX = "u"


def fnv1_64(data: bytes) -> int:
    """FNV-1 (not FNV-1a) 64-bit hash, matching Go's ``hash/fnv.New64()``.

    Note the operation order: FNV-1 multiplies then XORs; FNV-1a is the reverse.
    Using the wrong variant produces plausible-looking but non-matching names.
    """
    h = _FNV1_64_OFFSET
    for byte in data:
        h = (h * _FNV1_64_PRIME) & _UINT64_MASK
        h ^= byte
    return h


def _identity(
    org: str,
    project: str,
    domain: str,
    task_name: str,
    trigger_name: str,
    at: datetime,
    salt: str | None = None,
) -> str:
    """Build the string that gets hashed.

    Time components are the *wall clock* fields of ``at`` in whatever timezone it
    carries -- the scheduler hashes the schedule's local time, not UTC -- and are
    formatted as unpadded integers.
    """
    base = (
        f"{org}:{project}:{domain}:{task_name}:{trigger_name}:"
        f"{at.year}:{at.month}:{at.day}:{at.hour}:{at.minute}:{at.second}"
    )
    # A salt is prepended rather than appended, following the artifact-trigger
    # naming precedent, so salted names occupy a disjoint namespace.
    return f"{salt}:{base}" if salt else base


def scheduled_run_name(
    org: str,
    project: str,
    domain: str,
    task_name: str,
    trigger_name: str,
    at: datetime,
    salt: str | None = None,
) -> str:
    """Return the run name a scheduled fire of ``trigger_name`` at ``at`` produces.

    With ``salt`` set the name lands in a separate namespace, so it will not
    collide with -- and therefore will not be de-duplicated against -- the run the
    scheduler created for the same slot. That is how ``--force`` re-runs a slot.
    """
    digest = fnv1_64(_identity(org, project, domain, task_name, trigger_name, at, salt).encode())
    return f"{_SCHEDULER_PREFIX}{digest:x}"


def candidate_run_names(name: str) -> tuple[str, ...]:
    """Both spellings a scheduled run may be stored under.

    The control plane rewrites the leading "r" to "u" for automation-sourced runs
    routed to the actions engine. The routing decision is not visible from a
    client, so an existence check has to consider both.
    """
    if name.startswith(_SCHEDULER_PREFIX):
        return (name, _ACTIONS_PREFIX + name[1:])
    return (name,)
