"""Cron expansion has to agree with the scheduler's cron library.

Expected counts and first fire times below come from running robfig/cron v3 --
the parser the scheduler itself uses -- over May 2026.
"""

from datetime import datetime, timedelta, timezone

import pytest

from flyte.backfill._schedule import (
    CronParseError,
    cron_occurrences,
    fixed_rate_occurrences,
    parse_cron,
)

MAY_START = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
MAY_END = datetime(2026, 5, 31, 23, 59, 59, tzinfo=timezone.utc)

# expression -> (count over May 2026, first three fire times as MM-DDTHH:MM)
ROBFIG_REFERENCE = {
    "0 2 * * *": (31, ["05-01T02:00", "05-02T02:00", "05-03T02:00"]),
    "0 6 * * 1": (4, ["05-04T06:00", "05-11T06:00", "05-18T06:00"]),
    "*/15 * * * *": (2976, ["05-01T00:00", "05-01T00:15", "05-01T00:30"]),
    "30 3 1,15 * *": (2, ["05-01T03:30", "05-15T03:30"]),
    # Both day-of-month and day-of-week restricted: standard cron ORs them.
    "0 0 1 * 1": (5, ["05-01T00:00", "05-04T00:00", "05-11T00:00"]),
    "0 9-17/4 * * MON-FRI": (63, ["05-01T09:00", "05-01T13:00", "05-01T17:00"]),
}


@pytest.mark.parametrize(("expression", "expected"), sorted(ROBFIG_REFERENCE.items()))
def test_matches_robfig_cron(expression, expected):
    count, first = expected
    got = cron_occurrences(expression, MAY_START, MAY_END)
    assert len(got) == count
    assert [d.strftime("%m-%dT%H:%M") for d in got[: len(first)]] == first


def test_dom_and_dow_are_ored_not_anded():
    """A day matches if either field matches, when both are restricted."""
    got = cron_occurrences("0 0 1 * 1", MAY_START, MAY_END)
    days = sorted({d.day for d in got})
    assert days == [1, 4, 11, 18, 25]  # the 1st, plus every Monday


def test_window_bounds_are_inclusive():
    exact = datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc)
    assert cron_occurrences("0 2 * * *", exact, exact) == [exact]


def test_limit_stops_expansion():
    got = cron_occurrences("*/15 * * * *", MAY_START, MAY_END, limit=10)
    assert len(got) == 10


def test_timezone_prefix_is_stripped():
    fields, tz = parse_cron("CRON_TZ=America/New_York 0 2 * * *")
    assert tz == "America/New_York"
    assert fields[1] == frozenset({2})


def test_sunday_accepts_both_zero_and_seven():
    as_zero = cron_occurrences("0 0 * * 0", MAY_START, MAY_END)
    as_seven = cron_occurrences("0 0 * * 7", MAY_START, MAY_END)
    assert as_zero == as_seven
    assert all(d.weekday() == 6 for d in as_zero)


@pytest.mark.parametrize("bad", ["0 2 * *", "", "0 2 * * * *", "99 2 * * *", "0 2 * * 1/0"])
def test_invalid_expressions_raise(bad):
    with pytest.raises(CronParseError):
        cron_occurrences(bad, MAY_START, MAY_END)


def test_fixed_rate_respects_the_anchor_phase():
    anchor = datetime(2026, 5, 1, 0, 7, tzinfo=timezone.utc)
    got = fixed_rate_occurrences(30, MAY_START, MAY_START + timedelta(hours=2), anchor=anchor)
    assert [d.strftime("%H:%M") for d in got] == ["00:07", "00:37", "01:07", "01:37"]


def test_fixed_rate_skips_forward_to_the_window():
    anchor = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    got = fixed_rate_occurrences(60, MAY_START, MAY_START + timedelta(hours=3), anchor=anchor)
    assert got[0] == MAY_START
    assert len(got) == 4


def test_fixed_rate_rejects_a_nonpositive_interval():
    with pytest.raises(ValueError):
        fixed_rate_occurrences(0, MAY_START, MAY_END)
