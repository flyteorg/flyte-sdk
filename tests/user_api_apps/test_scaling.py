from datetime import timedelta

import pytest

from flyte.app import Scaling


def test_scaling_defaults():
    s = Scaling()
    assert s.replicas == (0, 1)
    assert s.metric is None
    assert s.scaledown_after is None


def test_scaling_int_replicas():
    s = Scaling(replicas=3)
    assert s.replicas == (3, 3)


def test_scaling_tuple_replicas():
    s = Scaling(replicas=(1, 5))
    assert s.replicas == (1, 5)


def test_scaling_zero_replicas():
    s = Scaling(replicas=0)
    assert s.replicas == (0, 0)


def test_scaling_negative_replicas():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        Scaling(replicas=-1)


def test_scaling_invalid_tuple_length():
    with pytest.raises(ValueError, match="tuple must be of length 2"):
        Scaling(replicas=(1, 2, 3))


def test_scaling_min_greater_than_max():
    with pytest.raises(ValueError, match="max_replicas must be greater than or equal to 1 and min_replicas"):
        Scaling(replicas=(5, 2))


def test_scaling_max_zero():
    with pytest.raises(ValueError, match="max_replicas must be greater than or equal to 1"):
        Scaling(replicas=(0, 0))


def test_scaling_invalid_replicas_type():
    with pytest.raises(TypeError, match="replicas must be an int or a tuple"):
        Scaling(replicas="invalid")


def test_scaling_concurrency_metric():
    s = Scaling(replicas=(1, 10), metric=Scaling.Concurrency(val=5))
    assert isinstance(s.metric, Scaling.Concurrency)
    assert s.metric.val == 5


def test_scaling_request_rate_metric():
    s = Scaling(replicas=(1, 10), metric=Scaling.RequestRate(val=100))
    assert isinstance(s.metric, Scaling.RequestRate)
    assert s.metric.val == 100


def test_scaling_concurrency_min_value():
    with pytest.raises(ValueError, match="Concurrency must be greater than or equal to 1"):
        Scaling.Concurrency(val=0)


def test_scaling_request_rate_min_value():
    with pytest.raises(ValueError, match="Request rate must be greater than or equal to 1"):
        Scaling.RequestRate(val=0)


def test_scaling_invalid_metric_type():
    with pytest.raises(TypeError, match="metric must be an instance"):
        Scaling(metric="invalid")


def test_scaling_scaledown_after_int():
    s = Scaling(scaledown_after=300)
    assert s.scaledown_after == timedelta(seconds=300)


def test_scaling_scaledown_after_timedelta():
    td = timedelta(minutes=5)
    s = Scaling(scaledown_after=td)
    assert s.scaledown_after == td


def test_scaling_scaledown_after_invalid_type():
    with pytest.raises(TypeError, match="scaledown_after must be an int or a timedelta"):
        Scaling(scaledown_after="invalid")


def test_scaling_get_replicas():
    s = Scaling(replicas=(2, 8))
    assert s.get_replicas() == (2, 8)


def test_scaling_concurrency_frozen():
    c = Scaling.Concurrency(val=10)
    with pytest.raises(AttributeError):
        c.val = 20
