from datetime import timedelta

import pytest

from flyte.app import Timeouts


def test_timeouts_default():
    t = Timeouts()
    assert t.request is None


def test_timeouts_int_seconds():
    t = Timeouts(request=30)
    assert t.request == timedelta(seconds=30)


def test_timeouts_timedelta():
    td = timedelta(minutes=5)
    t = Timeouts(request=td)
    assert t.request == td


def test_timeouts_max_one_hour():
    t = Timeouts(request=3600)
    assert t.request == timedelta(hours=1)


def test_timeouts_exceeds_one_hour():
    with pytest.raises(ValueError, match="must not exceed 1 hour"):
        Timeouts(request=3601)


def test_timeouts_negative():
    with pytest.raises(ValueError, match="must be non-negative"):
        Timeouts(request=-1)


def test_timeouts_invalid_type():
    with pytest.raises(TypeError, match="Expected request to be of type int or timedelta"):
        Timeouts(request="invalid")


def test_timeouts_zero():
    t = Timeouts(request=0)
    assert t.request == timedelta(0)
