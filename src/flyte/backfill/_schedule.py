"""Expand a trigger's schedule into the individual times it fires.

Backfill needs the same list of fire times the scheduler would have produced, so
it can name each slot the way a real fire would. Expansion happens in the
schedule's own timezone because the run name hashes local wall-clock fields.

Only the five-field cron form is supported, which is what triggers accept:
``minute hour day-of-month month day-of-week``, each field being ``*``, a value,
a ``lo-hi`` range, a comma-separated list, or any of those with a ``/step``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Iterator, Sequence

__all__ = ["CronParseError", "cron_occurrences", "fixed_rate_occurrences", "parse_cron"]

_CRON_TZ_PREFIX = re.compile(r"^\s*CRON_TZ=(?P<tz>\S+)\s+(?P<expr>.*)$")

_FIELD_BOUNDS = (
    (0, 59),  # minute
    (0, 23),  # hour
    (1, 31),  # day of month
    (1, 12),  # month
    (0, 6),  # day of week, Sunday = 0
)

_MONTH_ALIASES = {
    m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
}
_DOW_ALIASES = {m: i for i, m in enumerate(["sun", "mon", "tue", "wed", "thu", "fri", "sat"])}


class CronParseError(ValueError):
    """Raised when a cron expression cannot be understood."""


def _alias(token: str, index: int) -> str:
    lowered = token.lower()
    if index == 3 and lowered in _MONTH_ALIASES:
        return str(_MONTH_ALIASES[lowered])
    if index == 4 and lowered in _DOW_ALIASES:
        return str(_DOW_ALIASES[lowered])
    return token


def _parse_field(raw: str, index: int) -> frozenset[int] | None:
    """Parse one cron field. ``None`` means "unrestricted" (a bare ``*``)."""
    lo_bound, hi_bound = _FIELD_BOUNDS[index]
    if raw == "*":
        return None

    values: set[int] = set()
    for part in raw.split(","):
        body, _, step_raw = part.partition("/")
        try:
            step = int(step_raw) if step_raw else 1
        except ValueError as exc:
            raise CronParseError(f"invalid step in cron field {raw!r}") from exc
        if step < 1:
            raise CronParseError(f"step must be positive in cron field {raw!r}")

        if body == "*":
            lo, hi = lo_bound, hi_bound
        else:
            bits = [_alias(b, index) for b in body.split("-")]
            try:
                lo = int(bits[0])
                hi = int(bits[1]) if len(bits) > 1 else (hi_bound if step_raw else lo)
            except (ValueError, IndexError) as exc:
                raise CronParseError(f"invalid range in cron field {raw!r}") from exc

        # Sunday is expressible as both 0 and 7; normalise 7 down.
        if index == 4:
            lo, hi = (0 if lo == 7 else lo), (0 if hi == 7 else hi)
            if hi < lo:
                lo, hi = hi, lo
        if lo < lo_bound or hi > hi_bound or hi < lo:
            raise CronParseError(f"cron field {raw!r} out of range {lo_bound}-{hi_bound}")
        values.update(range(lo, hi + 1, step))

    return frozenset(values)


def parse_cron(expression: str) -> tuple[tuple[frozenset[int] | None, ...], str | None]:
    """Parse a cron expression into per-field value sets plus an optional timezone.

    Returns ``(fields, timezone)`` where ``fields`` is
    ``(minute, hour, day_of_month, month, day_of_week)`` and each entry is either a
    set of matching values or ``None`` for unrestricted.
    """
    timezone_name = None
    match = _CRON_TZ_PREFIX.match(expression)
    if match:
        timezone_name = match.group("tz")
        expression = match.group("expr")

    parts = expression.split()
    if len(parts) != 5:
        raise CronParseError(f"expected a 5-field cron expression, got {len(parts)} field(s): {expression!r}")
    return tuple(_parse_field(p, i) for i, p in enumerate(parts)), timezone_name


def _day_matches(
    day: datetime, dom: frozenset[int] | None, month: frozenset[int] | None, dow: frozenset[int] | None
) -> bool:
    if month is not None and day.month not in month:
        return False
    # Standard cron: when BOTH day-of-month and day-of-week are restricted, a day
    # matches if EITHER does. Treating it as AND silently drops fire times.
    py_dow = (day.weekday() + 1) % 7  # Python: Monday=0; cron: Sunday=0
    if dom is None and dow is None:
        return True
    if dom is not None and dow is not None:
        return day.day in dom or py_dow in dow
    if dom is not None:
        return day.day in dom
    return py_dow in dow


def cron_occurrences(
    expression: str,
    start: datetime,
    end: datetime,
    limit: int | None = None,
) -> list[datetime]:
    """Every time ``expression`` fires in ``[start, end]``, inclusive.

    ``start`` and ``end`` are used as given -- pass times already localised to the
    schedule's timezone, since the resulting datetimes are what get hashed into
    run names.
    """
    return list(_iter_cron(expression, start, end, limit))


def _iter_cron(expression: str, start: datetime, end: datetime, limit: int | None) -> Iterator[datetime]:
    fields, _ = parse_cron(expression)
    minutes, hours, dom, month, dow = fields
    minute_values: Sequence[int] = sorted(minutes) if minutes is not None else range(60)
    hour_values: Sequence[int] = sorted(hours) if hours is not None else range(24)

    # Walk day by day and only expand hours/minutes on days that match, rather
    # than testing every minute in the range.
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    emitted = 0
    while day <= end:
        if _day_matches(day, dom, month, dow):
            for hour in hour_values:
                for minute in minute_values:
                    moment = day.replace(hour=hour, minute=minute)
                    if moment < start:
                        continue
                    if moment > end:
                        return
                    yield moment
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return
        day += timedelta(days=1)


def fixed_rate_occurrences(
    interval_minutes: int,
    start: datetime,
    end: datetime,
    anchor: datetime | None = None,
    limit: int | None = None,
) -> list[datetime]:
    """Every fire time of a fixed-rate schedule in ``[start, end]``.

    ``anchor`` is the schedule's own start time, which sets the phase; without one
    the window start is used.
    """
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")
    step = timedelta(minutes=interval_minutes)
    moment = anchor or start
    if moment < start:
        # Advance to the first tick at or after the window, without looping.
        gap = (start - moment) // step
        moment += step * gap
        while moment < start:
            moment += step

    out: list[datetime] = []
    while moment <= end and (limit is None or len(out) < limit):
        out.append(moment)
        moment += step
    return out
