from datetime import timedelta

import pytest

import flyte
from flyte._timeout import Timeout, timeout_from_request


def test_timeout_with_int():
    t = Timeout(max_runtime=300)
    assert t.max_runtime == 300
    assert t.max_queued_time is None
    assert t.deadline is None


def test_timeout_with_timedelta():
    t = Timeout(max_runtime=timedelta(minutes=5))
    assert t.max_runtime == timedelta(minutes=5)


def test_timeout_with_queued_time():
    t = Timeout(max_runtime=300, max_queued_time=600)
    assert t.max_runtime == 300
    assert t.max_queued_time == 600
    assert t.deadline is None


def test_timeout_with_timedelta_queued():
    t = Timeout(max_runtime=timedelta(hours=1), max_queued_time=timedelta(minutes=30))
    assert t.max_runtime == timedelta(hours=1)
    assert t.max_queued_time == timedelta(minutes=30)


def test_timeout_with_deadline():
    t = Timeout(deadline=timedelta(hours=2))
    assert t.max_runtime is None
    assert t.max_queued_time is None
    assert t.deadline == timedelta(hours=2)


def test_timeout_deadline_int():
    t = Timeout(deadline=7200)
    assert t.deadline == 7200


def test_timeout_all_three_bounds():
    t = Timeout(
        max_runtime=timedelta(minutes=30),
        max_queued_time=timedelta(minutes=15),
        deadline=timedelta(hours=2),
    )
    assert t.max_runtime == timedelta(minutes=30)
    assert t.max_queued_time == timedelta(minutes=15)
    assert t.deadline == timedelta(hours=2)


def test_timeout_default_unset():
    t = Timeout()
    assert t.max_runtime is None
    assert t.max_queued_time is None
    assert t.deadline is None


def test_timeout_from_request_int():
    t = timeout_from_request(300)
    assert isinstance(t, Timeout)
    assert t.max_runtime == timedelta(seconds=300)
    assert t.max_queued_time is None
    assert t.deadline is None


def test_timeout_from_request_timedelta():
    td = timedelta(minutes=10)
    t = timeout_from_request(td)
    assert isinstance(t, Timeout)
    assert t.max_runtime == td


def test_timeout_from_request_timeout_object():
    original = Timeout(max_runtime=100, max_queued_time=200, deadline=300)
    result = timeout_from_request(original)
    assert result is original


def test_timeout_from_request_invalid_type():
    with pytest.raises(ValueError, match="Timeout must be an instance of"):
        timeout_from_request("invalid")


def test_flyte_timeout_importable():
    assert flyte.Timeout is Timeout


# A value that is neither an int nor a timedelta used to survive construction and only fail
# inside `_to_timeout_duration`, which calls `.total_seconds()` on it -- an AttributeError from
# SDK frames naming neither the field nor the task. Each bound is now rejected where the mistake
# was made. `timeout_from_request` already behaved this way for a bare `timeout=`; these pin the
# per-field arms to the same contract.


@pytest.mark.parametrize("field", ["max_runtime", "max_queued_time", "deadline"])
@pytest.mark.parametrize("bad", [1.5, "30", "30s", [30], object()])
def test_timeout_rejects_non_duration_bound(field, bad):
    with pytest.raises(ValueError, match=rf"Timeout\.{field} must be an int"):
        Timeout(**{field: bad})


def test_timeout_rejects_float_naming_the_field_that_was_wrong():
    """The message must identify the offending field, which the old AttributeError did not."""
    with pytest.raises(ValueError, match=r"Timeout\.deadline must be an int \(seconds\), a timedelta, or None"):
        Timeout(max_runtime=60, deadline=1.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"max_runtime": 0},
        {"max_runtime": 300},
        {"max_runtime": timedelta(minutes=5)},
        {"max_runtime": None, "max_queued_time": None, "deadline": None},
        {"max_runtime": 300, "max_queued_time": 600, "deadline": timedelta(hours=2)},
        {"deadline": timedelta(0)},
        {"max_queued_time": True},  # bool is an int; not narrowed by the guard
    ],
)
def test_timeout_accepted_surface_is_unchanged(kwargs):
    """The guard must not narrow what already worked: unset, zero, int, timedelta, all three combined."""
    t = Timeout(**kwargs)
    for name, value in kwargs.items():
        assert getattr(t, name) == value


def test_timeout_does_not_coerce_int_bounds():
    """Validation only -- an int bound is still readable as the int the user passed."""
    t = Timeout(max_runtime=300)
    assert t.max_runtime == 300
    assert not isinstance(t.max_runtime, timedelta)


def test_bad_bound_serializes_cleanly_once_rejected():
    """End-to-end: the value that used to crash serde is now refused at construction."""
    from flyte._internal.runtime.task_serde import get_proto_max_runtime

    with pytest.raises(ValueError, match=r"Timeout\.max_runtime must be an int"):
        get_proto_max_runtime(Timeout(max_runtime=1.5))

    # the control: the same bare value on the `timeout=` parameter already reported cleanly
    with pytest.raises(ValueError, match=r"Timeout must be an instance of Timeout, int, or timedelta"):
        get_proto_max_runtime(1.5)
