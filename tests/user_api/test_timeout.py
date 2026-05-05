from datetime import timedelta

import pytest

import flyte
from flyte._timeout import Timeout, timeout_from_request


def test_timeout_with_int():
    t = Timeout(max_runtime=300)
    assert t.max_runtime == 300
    assert t.max_queued_time is None


def test_timeout_with_timedelta():
    t = Timeout(max_runtime=timedelta(minutes=5))
    assert t.max_runtime == timedelta(minutes=5)


def test_timeout_with_queued_time():
    t = Timeout(max_runtime=300, max_queued_time=600)
    assert t.max_runtime == 300
    assert t.max_queued_time == 600


def test_timeout_with_timedelta_queued():
    t = Timeout(max_runtime=timedelta(hours=1), max_queued_time=timedelta(minutes=30))
    assert t.max_runtime == timedelta(hours=1)
    assert t.max_queued_time == timedelta(minutes=30)


def test_timeout_from_request_int():
    t = timeout_from_request(300)
    assert isinstance(t, Timeout)
    assert t.max_runtime == timedelta(seconds=300)
    assert t.max_queued_time is None


def test_timeout_from_request_timedelta():
    td = timedelta(minutes=10)
    t = timeout_from_request(td)
    assert isinstance(t, Timeout)
    assert t.max_runtime == td


def test_timeout_from_request_timeout_object():
    original = Timeout(max_runtime=100, max_queued_time=200)
    result = timeout_from_request(original)
    assert result is original


def test_timeout_from_request_invalid_type():
    with pytest.raises(ValueError, match="Timeout must be an instance of"):
        timeout_from_request("invalid")


def test_flyte_timeout_importable():
    assert flyte.Timeout is Timeout
