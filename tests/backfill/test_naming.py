"""The run-name hash must match the scheduler's byte for byte.

If it drifts, a backfilled slot no longer collides with the run the scheduler
already created, so de-duplication silently stops working and the backfill
quietly doubles up on work. The expected values below were produced by running
the scheduler's own Go implementation (hash/fnv New64 + the same format string),
so they pin the contract rather than the current Python behaviour.
"""

from datetime import datetime, timezone

import pytest

from flyte.backfill._naming import candidate_run_names, fnv1_64, scheduled_run_name

# (org, project, domain, task, trigger, (Y, M, D, h, m, s)) -> name from the Go scheduler
GO_REFERENCE = {
    (
        "acme",
        "ml-platform",
        "production",
        "evals.weekly.run",
        "nightly_eval",
        (2026, 5, 20, 2, 0, 0),
    ): "r64640bf1ae650449",
    (
        "acme",
        "ml-platform",
        "production",
        "evals.weekly.run",
        "nightly_eval",
        (2026, 6, 1, 2, 0, 0),
    ): "r5f16c7b4a799ddf7",
    ("", "p", "d", "t", "n", (2026, 1, 1, 0, 0, 0)): "rf3c00da918b55b3",
    ("o", "p", "d", "t", "n", (2026, 12, 31, 23, 59, 59)): "rae982f539aa14d00",
    (
        "union",
        "search",
        "staging",
        "search.rerank.rebuild",
        "weekly_retraining",
        (2026, 6, 1, 6, 0, 0),
    ): "rd8c65d1b71ce83af",
}


@pytest.mark.parametrize(("key", "expected"), sorted(GO_REFERENCE.items()))
def test_matches_go_scheduler(key, expected):
    org, project, domain, task, trigger, parts = key
    at = datetime(*parts, tzinfo=timezone.utc)
    assert scheduled_run_name(org, project, domain, task, trigger, at) == expected


def test_hash_is_fnv1_not_fnv1a():
    # FNV-1 and FNV-1a differ only in operation order; picking the wrong one still
    # yields a plausible name, so assert against a known FNV-1 value.
    assert fnv1_64(b"") == 0xCBF29CE484222325
    assert fnv1_64(b"a") == 0xAF63BD4C8601B7BE
    # FNV-1a of "a" would be 0xAF63DC4C8601EC8C -- close enough to be worth pinning.
    assert fnv1_64(b"a") != 0xAF63DC4C8601EC8C


def test_hex_is_unpadded():
    # Go's %x drops leading zeros, so names are not a fixed width.
    name = scheduled_run_name("", "p", "d", "t", "n", datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert name == "rf3c00da918b55b3"
    assert len(name) == 16  # 'r' + 15 hex digits, i.e. one short of the usual 16


def test_same_slot_is_stable():
    at = datetime(2026, 5, 20, 2, 0, tzinfo=timezone.utc)
    first = scheduled_run_name("o", "p", "d", "t", "n", at)
    second = scheduled_run_name("o", "p", "d", "t", "n", at)
    assert first == second


def test_different_second_is_a_different_name():
    base = datetime(2026, 5, 20, 2, 0, 0, tzinfo=timezone.utc)
    later = datetime(2026, 5, 20, 2, 0, 1, tzinfo=timezone.utc)
    assert scheduled_run_name("o", "p", "d", "t", "n", base) != scheduled_run_name("o", "p", "d", "t", "n", later)


def test_salt_moves_the_name_into_a_separate_namespace():
    at = datetime(2026, 5, 20, 2, 0, tzinfo=timezone.utc)
    plain = scheduled_run_name("o", "p", "d", "t", "n", at)
    salted = scheduled_run_name("o", "p", "d", "t", "n", at, salt="bf1")
    other = scheduled_run_name("o", "p", "d", "t", "n", at, salt="bf1:second")
    assert len({plain, salted, other}) == 3


def test_wall_clock_fields_are_used_verbatim():
    """The scheduler hashes local wall-clock fields, not an instant.

    Two datetimes describing the same instant in different offsets therefore hash
    differently -- which is the scheduler's behaviour, not a bug here.
    """
    from datetime import timedelta

    utc = datetime(2026, 5, 20, 2, 0, tzinfo=timezone.utc)
    plus_two = datetime(2026, 5, 20, 4, 0, tzinfo=timezone(timedelta(hours=2)))
    assert utc == plus_two  # same instant
    assert scheduled_run_name("o", "p", "d", "t", "n", utc) != scheduled_run_name("o", "p", "d", "t", "n", plus_two)


def test_candidate_names_cover_the_actions_prefix_rewrite():
    # Automation-sourced runs routed to the actions engine are stored under "u".
    assert candidate_run_names("r64640bf1ae650449") == ("r64640bf1ae650449", "u64640bf1ae650449")
    assert candidate_run_names("xdeadbeef") == ("xdeadbeef",)
